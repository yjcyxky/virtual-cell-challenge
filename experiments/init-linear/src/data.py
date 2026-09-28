"""Rebuild frozen physical splits; expose only allowed training pseudobulks."""
from pathlib import Path
import json
import sys
import gc
import shutil
import yaml
import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'scripts/dossier'))
from challenge import ChallengeIdentity, inspect_context, reference_indices, NTC
from rna import RNAFile, hash_file, value_hash


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    tmp.replace(path)


def ref(path):
    return {'path': str(Path(path).relative_to(ROOT)), 'sha256': hash_file(Path(path))}


def normalized(matrix, target_sum):
    matrix = matrix.astype(np.float64)
    total = np.asarray(matrix.sum(axis=1)).ravel()
    if (total <= 0).any():
        raise ValueError('zero measured library')
    matrix = sparse.diags(target_sum / total) @ matrix
    matrix.data = np.log1p(matrix.data)
    return matrix.tocsr()


def prepare(output, config):
    directory = output / 'cache'
    if config.get('reuse'):
        import_cached_inputs(output, config)
    audit = json.loads((ROOT / config['benchmark']['data_audit']['path']).read_text())
    splits = json.loads((ROOT / config['benchmark']['split_manifest']['path']).read_text())
    split = next(s for s in splits if s['id'] == config['fit_scope']['outer_split'])
    if split['id'] != 'S2-H1':
        raise ValueError('this registered implementation supports S2-H1 only')
    axis = pd.read_csv(ROOT / config['benchmark']['gene_axis']['path']).gene_name.tolist()
    identity = ChallengeIdentity(axis, pd.read_csv(ROOT / 'data/raw/networks/hgnc_complete_set.txt', sep='\t', low_memory=False))
    tasks = pd.read_csv(ROOT / 'docs/research/challenge_2026/tasks.csv')
    source_hashes = {r['path']: r['sha256'] for r in audit['inputs']}
    prepared = {}
    for context in split['training_contexts'] + ['H1']:
        done = directory / f'{context}-prepared.json'
        if done.exists():
            result = json.loads(done.read_text())
            for record in result['files']:
                if hash_file(ROOT / record['path']) != record['sha256']:
                    raise ValueError('prepared cache checksum changed')
            prepared[context] = result
            continue
        inputs = []
        frame, mapping, summary = inspect_context(ROOT, context, identity, audit['config'], inputs)
        if summary != audit['contexts'][context] or any(source_hashes.get(r['path']) != r['sha256'] for r in inputs):
            raise ValueError(f'frozen data audit mismatch: {context}')
        frame.to_parquet(directory / f'{context}-identities.parquet', index=False)
        training_targets = (sorted(set(frame.loc[frame.eligible, 'target']) & set(split['training_targets']))
                            if context in split['training_contexts'] else [])
        labels = [NTC] + training_targets
        label_index = {name: i for i, name in enumerate(labels)}
        positions = np.asarray(summary['official_gene_positions'])
        sums = np.zeros((len(labels), len(positions)), dtype=np.float64)
        counts = np.zeros(len(labels), dtype=np.int64)
        input_blocks, input_obs, reference_blocks, reference_obs = [], [], [], []
        chosen = set()
        if context == 'H1':
            targets = split['evaluation_targets'][context]
            for target in targets:
                members = frame.loc[frame.eligible & frame.target.eq(target)]
                ids = reference_indices(members, audit['config']['split_seed'])
                task = tasks.loc[tasks.context.eq(context) & tasks.target.eq(target)].iloc[0]
                if value_hash(sorted(frame.loc[ids, 'physical_id'])) != task.reference_identity_sha256:
                    raise ValueError('reference identities differ from frozen task')
                chosen.update(ids)
            chosen.update(frame.index[frame.ntc_pool.eq('score')])
        for relative, members in frame.groupby('file', sort=False):
            feature_map = mapping.loc[mapping.file.eq(relative) & mapping.measured].sort_values('official_position')
            cols = feature_map.source_position.to_numpy()
            with RNAFile(ROOT / 'data/raw' / relative) as source:
                row_to_frame = np.full(source.shape[0], -1, dtype=np.int64)
                row_to_frame[members.source_row.to_numpy()] = members.index.to_numpy()
                for start, block in source.blocks(config['data']['chunk_rows']):
                    indices = row_to_frame[start:start + block.shape[0]]
                    keep = indices >= 0
                    if not keep.any():
                        continue
                    rows = frame.loc[indices[keep]]
                    block = block[keep][:, cols].tocsr()
                    inp = rows.ntc_pool.eq('input').to_numpy()
                    fit = (rows.eligible & rows.target.isin(training_targets)).to_numpy() | inp
                    if fit.any():
                        group = np.array([label_index[t] for t in rows.loc[fit, 'target']])
                        present, inverse = np.unique(group, return_inverse=True)
                        selector = sparse.csr_matrix((np.ones(len(group)), (inverse, np.arange(len(group)))), shape=(len(present), len(group)))
                        sums[present] += (selector @ normalized(block[fit], config['data']['target_sum'])).toarray()
                        counts += np.bincount(group, minlength=len(labels))
                    if inp.any():
                        input_blocks.append(block[inp].astype(np.float32))
                        input_obs.append(rows.loc[inp, ['physical_id', 'target', 'batch']])
                    take = rows.index.isin(chosen)
                    if take.any():
                        reference_blocks.append(block[take].astype(np.float32))
                        reference_obs.append(rows.loc[take, ['physical_id', 'target', 'batch']])
        if (counts <= 0).any():
            raise ValueError('empty training pseudobulk')
        mean = (sums / counts[:, None]).astype(np.float32)
        statistics = directory / f'{context}-statistics.npz'
        np.savez_compressed(statistics, labels=np.asarray(labels), positions=positions, mean=mean, counts=counts)
        obs = pd.concat(input_obs).set_index('physical_id').rename(columns={'target': 'target_gene'})
        ntc_path = directory / f'{context}-input.h5ad'
        var = pd.DataFrame(index=np.asarray(axis)[positions])
        ad.AnnData(sparse.vstack(input_blocks, format='csr'), obs=obs, var=var).write_h5ad(ntc_path, compression='lzf')
        files = [ref(statistics), ref(ntc_path), ref(directory / f'{context}-identities.parquet')]
        if context == 'H1':
            obs = pd.concat(reference_obs).set_index('physical_id').rename(columns={'target': 'target_gene'})
            real_path = directory / 'H1-reference.h5ad'
            real = ad.AnnData(sparse.vstack(reference_blocks, format='csr'), obs=obs, var=var)
            real.write_h5ad(real_path, compression='lzf')
            files.append(ref(real_path))
            del real
        result = {'context': context, 'training_tasks': len(labels)-1, 'training_cells': int(counts[1:].sum()),
                  'input_ntc': int(counts[0]), 'files': files, 'source_refs': inputs,
                  'evaluation_only_reference': context == 'H1'}
        write_json(done, result)
        prepared[context] = result
        print(f'Prepared {context}: {result["training_tasks"]} training tasks, {result["training_cells"]} training cells', flush=True)
        del frame, mapping, sums, mean, input_blocks, reference_blocks
        gc.collect()
    write_json(directory / 'preparation.json', prepared)
    return axis, split, prepared


def import_cached_inputs(output, config):
    """Copy frozen inputs/bundles, never weights or model predictions, into this run."""
    manifest_ref = config['reuse']
    if ref(ROOT / manifest_ref['path']) != manifest_ref:
        raise ValueError('reuse manifest checksum mismatch')
    manifest = json.loads((ROOT / manifest_ref['path']).read_text())
    for key in ('source_metrics', 'source_config'):
        record = manifest[key]
        if ref(ROOT / record['path']) != record:
            raise ValueError(f'{key} checksum mismatch')
    metrics = json.loads((ROOT / manifest['source_metrics']['path']).read_text())
    previous = yaml.safe_load((ROOT / manifest['source_config']['path']).read_text())
    if (metrics['status'] != 'completed' or not metrics['evaluation_completed'] or
            metrics['research']['node_id'] != manifest['source_run_id'] or
            previous['research'] != metrics['research']):
        raise ValueError('reuse source is not the registered completed run')
    for key in ('benchmark', 'data', 'fit_scope', 'seed', 'generation'):
        if previous[key] != config[key]:
            raise ValueError(f'reuse changes frozen {key}')
    # Inputs and reference bundles are independent of which prediction arms are scored.
    old_evaluation, new_evaluation = dict(previous['evaluation']), dict(config['evaluation'])
    old_arms, new_arms = old_evaluation.pop('arms', []), new_evaluation.pop('arms', [])
    if old_evaluation != new_evaluation or not set(new_arms) <= set(old_arms):
        raise ValueError('reuse changes frozen evaluation protocol')
    source_cache = (ROOT / manifest['source_metrics']['path']).parent / 'cache'
    for item in manifest['files']:
        record, relative = item['ref'], Path(item['destination'])
        source = ROOT / record['path']
        if (relative.is_absolute() or '..' in relative.parts or relative.parts[0] != 'cache' or
                source.resolve() != (source_cache / Path(*relative.parts[1:])).resolve()):
            raise ValueError('reuse destination must preserve the source cache layout')
        if ref(source) != record:
            raise ValueError(f'reuse source checksum mismatch: {source.name}')
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            temporary = destination.with_suffix(destination.suffix + '.tmp')
            shutil.copyfile(source, temporary)
            temporary.replace(destination)
        if hash_file(destination) != record['sha256']:
            raise ValueError(f'reuse destination checksum mismatch: {destination.name}')
    print(f'Verified {len(manifest["files"])} immutable cache files from {manifest["source_run_id"]}; no weights reused', flush=True)


def verify_reference_reuse(output, model, config):
    """Reuse invariant zero/shared/source scores, with the source predictions retained."""
    sources = config['reference_reuse']
    for record in sources.values():
        if ref(ROOT/record['path']) != record:
            raise ValueError('reference reuse source checksum mismatch')
    previous = yaml.safe_load((ROOT/sources['config']['path']).read_text())
    metrics = json.loads((ROOT/sources['metrics']['path']).read_text())
    if metrics['status'] != 'completed' or not metrics['evaluation_completed'] or metrics['research'] != previous['research']:
        raise ValueError('reference reuse requires a completed bound source')
    for key in ('benchmark', 'data', 'fit_scope', 'seed', 'generation'):
        if previous[key] != config[key]:
            raise ValueError(f'reference reuse changes {key}')
    for key in ('single_source', 'panels', 'runtime'):
        if previous['evaluation'][key] != config['evaluation'][key]:
            raise ValueError(f'reference reuse changes evaluation {key}')
    with np.load(ROOT/sources['checkpoint']['path']) as source:
        if not all(np.array_equal(source[k], model[k]) for k in ('shared', 'targets', 'genes', 'trained_mask')):
            raise ValueError('shared response changed; invariant reference cannot be reused')
    manifest = json.loads((ROOT/sources['predictions']['path']).read_text())
    files = [r for r in manifest['files'] if Path(r['path']).stem in ('zero','shared','source')]
    if {Path(r['path']).stem for r in files} != {'zero','shared','source'}:
        raise ValueError('missing frozen reference predictions')
    for record in files:
        if ref(ROOT/record['path']) != record:
            raise ValueError('reference prediction checksum changed')
    frozen_scores = json.loads((ROOT/sources['score_manifest']['path']).read_text())['files']
    for record in frozen_scores:
        if ref(ROOT/record['path']) != record:
            raise ValueError('reference score checksum changed')
    evaluation, score_refs = {}, []
    for panel in config['evaluation']['panels']:
        evaluation[panel] = {}
        for arm in ('zero','shared','source'):
            result = metrics['evaluation'][panel][arm]
            if not np.isfinite(result['Overall']) or not all(np.isfinite(list(result['normalized'].values()))):
                raise ValueError('invalid reference scores')
            evaluation[panel][arm] = dict(result, reused_from=sources['metrics'])
            for name in ('raw.parquet','aggregate.csv','scores.csv','run_meta.json'):
                score_refs.append(ref(ROOT/result['result_directory']/name))
    if sorted(score_refs, key=lambda r:r['path']) != sorted(frozen_scores, key=lambda r:r['path']):
        raise ValueError('reference score manifest does not cover the requested scores')
    audit = {'sources': sources, 'reference_predictions': files, 'score_refs': score_refs,
             'shared_response_exact_match': True, 'scores_recomputed': False,
             'scope': 'unchanged data/split/generator/seed/scorer/reference; model predictions scored afresh'}
    write_json(output/'cache/reference-reuse.json', audit)
    print('Verified frozen zero/shared/source predictions and exact shared response; reusing their scores', flush=True)
    return evaluation
