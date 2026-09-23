"""Frozen CRISPRi supervision with disjoint feature/reference NTC pools."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'experiments/exp002-response-transfer-validation/src'))
sys.path.insert(0, str(ROOT / 'scripts/dossier'))

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from scipy import sparse

from features import build_priors, canonical_map, relation_matrices
from profile_responses import write_json
from rna import hash_file
from vcc_baseline import validate_sources
from vcc_submission import official_inputs


def target_reserved(target, percent):
    return int.from_bytes(hashlib.sha256(('context-relation-v1|' + target).encode()).digest()[:8], 'little') % 100 < percent


def safe_axis(symbols, aliases):
    pairs = [(i, aliases[g]) for i, g in enumerate(symbols) if g in aliases]
    count = {}
    for _, dest in pairs:
        count[dest] = count.get(dest, 0) + 1
    pairs = [(i, j) for i, j in pairs if count[j] == 1]
    return np.asarray([x[0] for x in pairs], int), np.asarray([x[1] for x in pairs], int)


def choose_controls(frame, maximum, seed):
    """Uniform selection across eligible cells; persist all selected identities."""
    rng = np.random.default_rng(seed)
    return frame.iloc[np.sort(rng.choice(len(frame), min(len(frame), maximum), replace=False))].copy()


def control_state(log, batches, depth, measured, graph, embedding, config, directory):
    mean, std = log.mean(0), log.std(0)
    detection = (log > 0).mean(0).astype(np.float32)
    mean[~measured], std[~measured], detection[~measured] = np.nan, np.nan, np.nan
    weight = np.asarray(graph @ measured.astype(np.float32)).ravel()
    neighbor_mean = np.divide(graph @ np.nan_to_num(mean), weight, out=np.full(len(mean), np.nan), where=weight > 0)
    neighbor_detection = np.divide(graph @ np.nan_to_num(detection), weight, out=np.full(len(mean), np.nan), where=weight > 0)
    # Only common measured genes contribute, making platform coverage explicit.
    common = np.load(directory.parents[1] / 'common-measured.npy')
    known = measured & common
    global_state = (np.nan_to_num(mean)[known] @ embedding[known] / max(known.sum(), 1)).astype(np.float32)
    np.savez_compressed(directory / 'state.npz', mean=mean, std=std, detection=detection,
                        measured=measured, neighbor_mean=neighbor_mean, neighbor_detection=neighbor_detection,
                        global_state=global_state)
    report = relation_matrices(log, batches, np.log1p(depth), measured, config, directory)
    write_json(directory / 'relation-summary.json', report)


def source_paths(panel):
    base = ROOT / 'data/assessments'
    if panel.startswith('H1:'):
        split = panel.split(':')[1]
        return base / 'h1-response-20260919' / ('cache_' + split), base / 'h1-annotation-20260919/cells.parquet', split
    family, name = panel.split(':')
    return base / f'{family}-response-{name}-20260919/cache', base / f'{family}-annotation-{name}-20260919/cells.parquet', None


def prepare(config, output):
    prepared = output / 'prepared'
    done = prepared / 'preparation.json'
    if done.exists():
        return json.loads(done.read_text())
    prepared.mkdir(exist_ok=True)
    official, manifest, genes, official_targets, official_identities = official_inputs(config)
    collection = ROOT / config['collection']
    reports, identities = validate_sources(collection, config['panels'])
    # Verify additionally consumed sidecars, not just response-moment files.
    for r in reports:
        for name in ['cell-splits.parquet', 'native-gene-mapping.parquet']:
            path = collection / r['context_id'] / name
            assert hash_file(path) == r['artifacts'][name], str(path)
            identities.append({'path': str(path.relative_to(ROOT)), 'sha256': r['artifacts'][name]})
    network = ROOT / 'data/raw/networks'
    prior_identity = json.loads((network / 'SOURCE.json').read_text())
    for item in prior_identity['files']:
        if item['name'] == 'goa_human.gaf.gz':
            continue
        path = network / item['name']
        assert hash_file(path) == item['sha256'] and path.stat().st_size == item['bytes']
        identities.append({'path': str(path.relative_to(ROOT)), 'sha256': item['sha256']})
    hgnc = pd.read_csv(network / 'hgnc_complete_set.txt', sep='\t', low_memory=False)
    aliases = canonical_map(hgnc, genes)
    prior_dir = prepared / 'priors'; prior_dir.mkdir(exist_ok=True)
    if not (prior_dir / 'prior-summary.json').exists():
        build_priors(network, genes, config, prior_dir)
    graph = sparse.load_npz(prior_dir / 'functional.npz')
    embedding = np.load(prior_dir / 'embedding.npy')
    common = np.ones(len(genes), bool)
    for r in reports:
        with h5py.File(collection / r['context_id'] / 'moments.h5') as h:
            _, dest = safe_axis(h['safe_symbol'].asstr()[:], aliases)
        mask = np.zeros(len(genes), bool); mask[dest] = True
        common &= mask
    np.save(prepared / 'common-measured.npy', common)
    contexts, labels, ledgers, excluded, seen_source = [], {}, [], [], set()
    for r in reports:
        line = r['source_metadata']['cell_line']
        folder = collection / r['context_id']
        cache, annotation, split_name = source_paths(r['panel_id'])
        cache_identity = json.loads((cache / 'identity.json').read_text())
        cells = pd.read_parquet(folder / 'cell-splits.parquet')
        assert cells.input_sha256.nunique() == 1
        assert cache_identity['identity']['input_sha256'] == cells.input_sha256.iloc[0]
        assert cache_identity['identity']['parameters']['normalization_total'] == 10000
        cache_path = cache / 'logcp.npy'
        if str(cache_path) not in seen_source:
            print(json.dumps({'stage': 'verify_source_cache', 'panel': r['panel_id'], 'bytes': cache_path.stat().st_size}), flush=True)
            assert hash_file(cache_path) == cache_identity['hashes']['logcp.npy']
            identities.append({'path': str(cache_path.relative_to(ROOT)), 'sha256': cache_identity['hashes']['logcp.npy']})
            seen_source.add(str(cache_path))
        log = np.load(cache_path, mmap_mode='r')
        half = f"half_{config['control_split_seed']}"
        feature = cells.loc[cells.is_NTC & cells[half].eq(0)].copy()
        reference = cells.loc[cells.is_NTC & cells[half].eq(1)].copy()
        assert not set(feature.source_barcode) & set(reference.source_barcode)
        with h5py.File(folder / 'moments.h5') as h:
            src, dest = safe_axis(h['safe_symbol'].asstr()[:], aliases)
            assert log.shape[1] == h['target'].shape[-1]
            measured = np.zeros(len(genes), bool); measured[dest] = True
            target_table = pd.read_parquet(folder / 'tasks.parquet')
            supported = pd.read_parquet(folder / 'task-support.parquet')
            supported = supported.loc[supported.variant.eq('full')].set_index('task_uid')
            if line not in contexts:
                contexts.append(line); labels[line] = {}
                context_dir = prepared / 'contexts' / line; context_dir.mkdir(parents=True, exist_ok=True)
                selected = choose_controls(feature, config['maximum_feature_controls'], config['seed'])
                selected.to_parquet(context_dir / 'feature-controls.parquet', index=False)
                reference.to_parquet(context_dir / 'reference-controls.parquet', index=False)
                expected_annotation = next(line.split()[0] for line in (annotation.parent / 'SHA256SUMS').read_text().splitlines()
                                           if line.split()[-1] == 'cells.parquet')
                assert hash_file(annotation) == expected_annotation
                identities.append({'path': str(annotation.relative_to(ROOT)), 'sha256': expected_annotation})
                annotated = pd.read_parquet(annotation, columns=['row_index', 'computed_total_counts'] + (['split'] if split_name else []))
                if split_name:
                    annotated = annotated.loc[annotated['split'].eq(split_name)]
                depth = annotated.set_index('row_index').loc[selected.row_index, 'computed_total_counts'].to_numpy(float)
                sample = np.zeros((len(selected), len(genes)), np.float32)
                sample[:, dest] = log[selected.row_index.to_numpy()][:, src]
                if not (context_dir / 'relation-summary.json').exists():
                    control_state(sample, selected.source_batch.to_numpy(str), depth, measured, graph, embedding, config, context_dir)
                print(json.dumps({'stage': 'context_features', 'context': line, 'sampled_NTC': len(sample), 'reference_NTC': len(reference)}), flush=True)
                del sample, annotated
            # Reconstruct independent reference means with the original full-target
            # technical-stratum weights, rather than sharing feature NTC noise.
            reference_means = {}
            for batch, group in reference.groupby('source_batch', sort=False):
                sums = np.zeros(len(src), np.float64)
                rows = group.row_index.to_numpy()
                for start in range(0, len(rows), 512):
                    sums += log[rows[start:start + 512]][:, src].sum(0, dtype=np.float64)
                reference_means[batch] = sums / len(rows)
            ntc_batches = set(cells.loc[cells.is_NTC, 'source_batch'])
            target_groups = {uid: group for uid, group in cells.loc[cells.task_uid.notna()].groupby('task_uid', sort=False)}
            for target, group in target_table.groupby('canonical_target', sort=True):
                if target not in aliases:
                    excluded.append({'panel': r['panel_id'], 'target': target, 'reason': 'target_not_safely_on_official_axis'})
                    continue
                p = aliases[target]
                estimates, total_cells = [], 0
                for idx, task in group.iterrows():
                    support = supported.loc[task.task_uid]
                    if support.status != 'completed' or support.n_target < config['minimum_target_cells']:
                        excluded.append({'panel': r['panel_id'], 'target': target, 'task_uid': task.task_uid,
                                         'reason': 'insufficient_matched_target_cells', 'cells': int(support.n_target)})
                        continue
                    observed = target_groups[task.task_uid]
                    observed = observed.loc[observed.source_batch.isin(ntc_batches)]
                    assert len(observed) == support.n_target
                    weights = observed.source_batch.value_counts(normalize=True)
                    if not set(weights.index) <= set(reference_means):
                        excluded.append({'panel': r['panel_id'], 'target': target, 'task_uid': task.task_uid,
                                         'reason': 'missing_independent_reference_batch'})
                        continue
                    control = sum(weight * reference_means[b] for b, weight in weights.items())
                    estimates.append(h['target'][idx, 0][src].astype(float) - control)
                    total_cells += len(observed)
                if estimates:
                    value = np.full(len(genes), np.nan, np.float32)
                    value[dest] = np.mean(estimates, axis=0)
                    value[p] = np.nan  # fixed on-target prior is evaluated separately
                    if p in labels[line]:
                        raise ValueError('duplicate canonical target across supposedly disjoint H1 panels')
                    labels[line][p] = value
                    ledgers.append({'context': line, 'target': genes[p], 'target_index': p, 'target_cells': total_cells,
                                    'constructs': len(estimates), 'measured_readouts': int(np.isfinite(value).sum()),
                                    'reserved_target': target_reserved(genes[p], config['unseen_target_percent'])})
        del log
    for line in contexts:
        rows = sorted(labels[line])
        np.savez_compressed(prepared / 'contexts' / line / 'labels.npz', targets=np.asarray(rows),
                            response=np.asarray([labels[line][p] for p in rows]))
    pd.DataFrame(ledgers).to_parquet(prepared / 'tasks.parquet', index=False)
    pd.DataFrame(excluded).to_parquet(prepared / 'excluded-tasks.parquet', index=False)
    for ci, context in enumerate(manifest['contexts']):
        context_dir = prepared / 'contexts' / context; context_dir.mkdir(parents=True, exist_ok=True)
        if (context_dir / 'relation-summary.json').exists():
            continue
        control = ad.read_h5ad(official / f'context_{context}.h5ad')
        assert control.var_names.tolist() == genes and set(control.obs.context.astype(str)) == {context}
        rng = np.random.default_rng(config['seed'] + ci)
        rows = np.sort(rng.choice(control.n_obs, min(control.n_obs, config['maximum_feature_controls']), replace=False))
        x = control.X[rows].astype(np.float32)
        depth = np.asarray(x.sum(1)).ravel()
        x.data = np.log1p(x.data * np.repeat(10000. / depth, np.diff(x.indptr)))
        pd.DataFrame({'row_index': rows, 'cell_id': control.obs_names[rows]}).to_parquet(context_dir / 'feature-controls.parquet', index=False)
        control_state(x.toarray(), control.obs.ntc_id.astype(str).to_numpy()[rows], depth,
                      np.ones(len(genes), bool), graph, embedding, config, context_dir)
        print(json.dumps({'stage': 'context_features', 'context': context, 'sampled_NTC': len(rows)}), flush=True)
        del control, x
    write_json(prepared / 'input-identities.json', {'training_and_priors': identities, 'official': official_identities})
    result = {'contexts': sorted(contexts), 'official_contexts': manifest['contexts'], 'genes': genes,
              'official_targets': official_targets, 'tasks': len(ledgers), 'common_genes': int(common.sum()),
              'label': 'mean(log1p(CP10K)) target minus disjoint, batch-matched reference NTC',
              'feature_NTC_and_label_reference_disjoint': True,
              'on_target_excluded_from_supervised_downstream_training': True}
    write_json(done, result)
    return result
