"""Freeze challenge-aligned measurement and split contracts from read-only sources.

No model, historical output, or response-dependent selection enters this audit.
The same mapping and split functions are reusable by formal Experiment inputs.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, replace
from importlib import metadata
import json
from pathlib import Path
import hashlib

import h5py
import numpy as np
import pandas as pd
from scipy import sparse

from rna import RNAFile, hash_file, read_frame, value_hash

ROOT = Path(__file__).resolve().parents[2]
NTC = 'non-targeting'
SCORER_COMMIT = '5e64833518a6603a0301cbe28185d49c30f4a986'
PANELS = {
    'H1': ('ARC-H1', ['arc_vcc2025_h1/adata_Training.h5ad',
                     'arc_vcc2025_h1/adata_Validation.h5ad', 'arc_vcc2025_h1/adata_Test.h5ad']),
    'K562': ('REPLOGLE-2022', ['replogle2022/K562_gwps_raw_singlecell_01.h5ad']),
    'RPE1': ('REPLOGLE-2022', ['replogle2022/rpe1_raw_singlecell_01.h5ad']),
    'HepG2': ('NADIG-2025', ['nadig2025/GSE264667_hepg2_raw_singlecell_01.h5ad']),
    'Jurkat': ('NADIG-2025', ['nadig2025/GSE264667_jurkat_raw_singlecell_01.h5ad']),
}


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


class ChallengeIdentity:
    """Official labels and positions are immutable; HGNC only supplies bridges."""
    def __init__(self, official, hgnc):
        self.official = list(official)
        if len(self.official) != len(set(self.official)):
            raise ValueError('duplicate_official_axis')
        approved = hgnc.loc[hgnc.status.eq('Approved')].fillna('')
        self.approved = set(approved.symbol)
        self.aliases = defaultdict(set)
        for row in approved.to_dict('records'):
            for field in ('symbol', 'prev_symbol', 'alias_symbol', 'ensembl_gene_id'):
                for item in str(row.get(field, '')).split('|'):
                    if item:
                        self.aliases[item].add(row['symbol'])
        self.positions = {gene: i for i, gene in enumerate(self.official)}
        self.canonical_slots = defaultdict(set)
        for gene in self.official:
            for canonical in self.candidates(gene):
                self.canonical_slots[canonical].add(gene)

    def candidates(self, name):
        name = str(name)
        if name in self.approved:
            return {name}
        return self.aliases.get(name.split('.')[0] if name.startswith('ENSG') else name, set())

    def resolve(self, name, ensembl=''):
        name, ensembl = str(name), str(ensembl)
        symbols, ids = self.candidates(name), self.candidates(ensembl) if ensembl else set()
        if symbols and ids and not symbols.intersection(ids):
            return None, 'symbol_ensembl_conflict'
        if name in self.positions:
            # Do not collapse historical challenge slots (e.g. TIAF1/MYO18A).
            if len(ids) == 1 and next(iter(ids)) in self.positions and next(iter(ids)) != name:
                return None, 'official_slot_id_conflict'
            return name, 'literal_official' if ids else 'literal_official_id_unresolved'
        choices = symbols.intersection(ids) if symbols and ids else ids or symbols
        if len(choices) != 1:
            return None, 'ambiguous_identity' if choices else 'unresolved_identity'
        canonical = next(iter(choices))
        slots = self.canonical_slots.get(canonical, set())
        if len(slots) > 1:
            return None, 'ambiguous_official_slots'
        return (next(iter(slots)) if slots else canonical), 'unique_hgnc_bridge'

    def features(self, names, ids):
        rows = []
        for i, (name, ensembl) in enumerate(zip(names, ids, strict=True)):
            gene, reason = self.resolve(name, ensembl)
            rows.append({'source_position': i, 'source_symbol': str(name), 'source_ensembl': str(ensembl),
                         'mapped_gene': gene, 'official_position': self.positions.get(gene), 'reason': reason})
        result = pd.DataFrame(rows)
        duplicates = result.mapped_gene.notna() & result.mapped_gene.duplicated(keep=False)
        result.loc[duplicates, 'reason'] = 'duplicate_mapped_feature'
        result['measured'] = result.official_position.notna() & ~duplicates
        return result


def ranked_indices(keys, salt):
    return np.array(sorted(range(len(keys)), key=lambda i: (value_hash([salt, keys[i]]), keys[i])), dtype=int)


def split_ntc(frame, seed, input_cap):
    """Disjoint physical records, balanced within batch; cap only the input pool."""
    if frame.physical_id.duplicated().any():
        raise ValueError('duplicate_physical_NTC')
    result = pd.Series('score', index=frame.index, dtype=object)
    for batch, group in frame.groupby('batch', sort=True):
        order = ranked_indices(group.physical_id.tolist(), [seed, 'NTC', batch])
        result.loc[group.index[order[:len(order)//2]]] = 'input'
    pool = frame.loc[result.eq('input')]
    if len(pool) > input_cap:
        # Proportional deterministic quota per batch, largest-remainder allocation.
        sizes = pool.groupby('batch', sort=True).size()
        quota = sizes * input_cap / len(pool)
        counts = np.floor(quota).astype(int)
        for batch in (quota-counts).sort_values(ascending=False, kind='stable').index[:input_cap-counts.sum()]:
            counts[batch] += 1
        result.loc[pool.index] = 'unused'
        for batch, group in pool.groupby('batch', sort=True):
            order = ranked_indices(group.physical_id.tolist(), [seed, 'NTC-cap', batch])
            result.loc[group.index[order[:counts[batch]]]] = 'input'
    return result


def target_partition(target, seed):
    bucket = int(value_hash([seed, 'global-target', target])[:16], 16) % 5
    return 'test' if bucket == 0 else 'validation' if bucket == 1 else 'train'


def reference_indices(frame, seed, cap=400, diagnostic=False):
    """All rows are unique/valid before selection; never replicate scarce real cells."""
    order = ranked_indices(frame.physical_id.tolist(), [seed, 'reference'])
    size = min(cap, len(order)//2 if diagnostic else len(order))
    return frame.index[order[:size]]


def verify_input(path, root, records):
    before = path.stat()
    source = path.parent/'SOURCE.json'
    entry = next(x for x in json.loads(source.read_text())['files'] if x['name'] == path.name)
    digest = hash_file(path)
    if digest != entry['sha256']:
        raise ValueError(f'input_hash_mismatch:{path}')
    records.append({'path': str(path.relative_to(root)), 'sha256': digest, 'bytes': before.st_size,
                    'source_path': str(source.relative_to(root)), 'source_sha256': hash_file(source)})
    return before


def inspect_context(root, context, identity, config, inputs):
    study, files = PANELS[context]
    frames, maps, scans, duplicate_controls = [], [], [], {}
    duplicate_count = 0
    h1_axis = None
    for file_number, relative in enumerate(files):
        path = root/'data/raw'/relative
        print(f'Hash and full count scan: {context} {path.name}', flush=True)
        before = verify_input(path, root, inputs)
        with RNAFile(path) as source:
            obs, var = source.obs, source.var
            required = ['batch', 'guide_id', 'target_gene'] if context == 'H1' else ['gem_group', 'sgID_AB', 'gene', 'gene_id']
            if obs[required].isna().any().any() or obs.index.isna().any():
                raise ValueError(f'missing_physical_or_intervention_identity:{relative}')
            names = var.index.astype(str) if context == 'H1' else var.gene_name.astype(str)
            if context == 'H1':
                if h1_axis is None:
                    h1_axis = (list(names), var.gene_id.astype(str).tolist())
                if list(names) != h1_axis[0]:
                    raise ValueError('H1_native_gene_axis_mismatch')
                ids = var.gene_id.astype(str).tolist() if 'gene_id' in var else h1_axis[1]
                if list(ids) != h1_axis[1]:
                    raise ValueError('H1_native_gene_id_mismatch')
            else:
                ids = var.index.astype(str)
            mapping = identity.features(names, ids)
            mapping['ensembl_evidence_file'] = files[0] if context == 'H1' else relative
            mapping.insert(0, 'file', relative)
            maps.append(mapping)
            batch = obs['batch' if context == 'H1' else 'gem_group'].astype(str)
            raw_targets = obs['target_gene' if context == 'H1' else 'gene'].astype(str)
            raw_ids = pd.Series('', index=obs.index) if context == 'H1' else obs.gene_id.astype(str)
            # Resolve unique metadata pairs once, not millions of times.
            resolver = {(t, e): (NTC, 'control') if t == NTC else identity.resolve(t, e)
                        for t, e in set(zip(raw_targets, raw_ids))}
            targets = [resolver[(t, e)][0] for t, e in zip(raw_targets, raw_ids)]
            reasons = [resolver[(t, e)][1] for t, e in zip(raw_targets, raw_ids)]
            keys = [json.dumps([study, context, b, str(barcode)], separators=(',', ':'))
                    for b, barcode in zip(batch, obs.index)]
            if len(set(keys)) != len(keys):
                raise ValueError(f'duplicate_physical_records:{relative}')
            frame = pd.DataFrame({'physical_id': keys, 'source_row': np.arange(len(obs)),
                'file': relative, 'batch': batch.to_numpy(),
                'guide': obs['guide_id' if context == 'H1' else 'sgID_AB'].astype(str).to_numpy(),
                'source_target': raw_targets.to_numpy(), 'source_target_id': raw_ids.to_numpy(),
                'target': targets, 'target_mapping': reasons})
            selected = mapping.loc[mapping.measured].sort_values('official_position').source_position.to_numpy()
            if not len(selected):
                raise ValueError(f'no_official_measured_features:{relative}')
            native_total, mapped_total = np.zeros(len(obs)), np.zeros(len(obs))
            gene_detected = np.zeros(source.shape[1], dtype=np.int64)
            ntc = raw_targets.eq(NTC).to_numpy()
            row_hashes = {}
            axis_hash = value_hash(list(zip(names, ids)))
            for start in range(0, len(obs), config['chunk_rows']):
                stop = min(start+config['chunk_rows'], len(obs))
                if source.encoding == 'dense':
                    block = source.x[start:stop]
                    data = block
                elif source.encoding == 'csr_matrix':
                    ptr = source.indptr[start:stop+1]
                    lo, hi = int(ptr[0]), int(ptr[-1])
                    indices = source.x['indices'][lo:hi]
                    if np.any(indices < 0) or np.any(indices >= source.shape[1]):
                        raise ValueError('sparse_column_out_of_bounds')
                    stored = source.x['data'][lo:hi]
                    if not np.isfinite(stored).all() or (stored < 0).any() or not np.equal(stored, np.floor(stored)).all():
                        raise ValueError(f'invalid_stored_counts:{relative}:{start}')
                    block = sparse.csr_matrix((stored, indices, ptr-lo),
                                              shape=(stop-start, source.shape[1]))
                    block.sum_duplicates(); block.sort_indices(); block.eliminate_zeros()
                    data = block.data
                else:
                    raise ValueError('unsupported_matrix_layout')
                if not np.isfinite(data).all() or (data < 0).any() or not np.equal(data, np.floor(data)).all():
                    raise ValueError(f'invalid_counts:{relative}:{start}')
                native_total[start:stop] = np.asarray(block.sum(axis=1)).ravel()
                mapped_total[start:stop] = np.asarray(block[:, selected].sum(axis=1)).ravel()
                gene_detected += np.asarray((block > 0).sum(axis=0)).ravel().astype(np.int64)
                if context == 'H1':
                    for local in np.flatnonzero(ntc[start:stop]):
                        row = block[local].toarray().ravel() if sparse.issparse(block) else block[local]
                        digest = hashlib.sha256(axis_hash.encode()+np.asarray(row, dtype='<f8').tobytes()).hexdigest()
                        i = start+int(local)
                        row_hashes[keys[i]] = (digest, frame.guide.iloc[i], frame.target.iloc[i])
            frame['valid_counts'] = (native_total > 0) & (mapped_total > 0)
            if context == 'H1':
                if file_number == 0:
                    duplicate_controls = row_hashes
                elif row_hashes != duplicate_controls:
                    raise ValueError('H1_duplicate_NTC_expression_or_label_mismatch')
                else:
                    duplicate_count += len(row_hashes)
                    frame = frame.loc[~ntc].copy()
            scans.append({'file': relative, 'rows_scanned': len(obs), 'genes_scanned': source.shape[1],
                          'finite_nonnegative_integer': True,
                          'zero_native_libraries': int((native_total == 0).sum()),
                          'zero_official_measured_libraries': int((mapped_total == 0).sum()),
                          'all_zero_native_genes': int((gene_detected == 0).sum())})
            mapping['detected_cells'] = gene_detected
            frames.append(frame)
        after = path.stat()
        if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
            raise ValueError(f'input_changed_during_audit:{relative}')
    frame = pd.concat(frames, ignore_index=True)
    if frame.physical_id.duplicated().any():
        raise ValueError('cross_file_duplicate_after_NTC_dedup')
    measured_sets = [set(m.loc[m.measured, 'official_position'].astype(int)) for m in maps]
    if any(s != measured_sets[0] for s in measured_sets[1:]):
        raise ValueError('within_context_measurement_axis_mismatch')
    control = frame.loc[frame.target.eq(NTC) & frame.valid_counts]
    frame['ntc_pool'] = ''
    frame.loc[control.index, 'ntc_pool'] = split_ntc(control, config['split_seed'], config['input_ntc_cap'])
    pools = {}
    for pool in ('input', 'score', 'unused'):
        members = frame.loc[frame.ntc_pool.eq(pool)]
        pools[pool] = {'cells': len(members), 'identity_sha256': value_hash(sorted(members.physical_id)),
                       'by_batch': {str(k): int(v) for k, v in members.batch.value_counts().sort_index().items()}}
    matched_batches = set(pools['input']['by_batch']) & set(pools['score']['by_batch'])
    frame['eligible'] = frame.valid_counts & frame.target.notna() & frame.batch.isin(matched_batches)
    # Scoring and input pools must refer to exactly the retained batch population.
    if set(pools['input']['by_batch']) != set(pools['score']['by_batch']):
        raise ValueError(f'NTC_pool_batch_mismatch:{context}')
    return frame, pd.concat(maps, ignore_index=True), {
        'study': study, 'scans': scans, 'unique_physical_cells': len(frame),
        'duplicate_NTC_records_verified_and_removed': duplicate_count,
        'duplicate_NTC_content_sha256': value_hash(sorted(duplicate_controls.items())) if duplicate_controls else None,
        'official_measured_genes': len(measured_sets[0]),
        'official_gene_positions': sorted(measured_sets[0]), 'ntc_pools': pools,
        'independent_biological_replicates': None,
        'excluded_invalid_count_or_identity_or_control_support': int((~frame.eligible).sum()),
    }


def scorer_config(**runtime):
    """Restrict overrides to computational settings; scientific preset is immutable."""
    from cell_eval2 import EvalConfig
    permitted = {'device', 'num_threads', 'pert_chunk', 'outdir', 'cache_real', 'cache_pred'}
    if set(runtime)-permitted:
        raise ValueError('scientific_preset_override_forbidden')
    return replace(EvalConfig.from_preset('vcc2026'), pert_col='target_gene', **runtime)


def scorer_contract():
    import cell_eval2
    from cell_eval2.competition import competition_members
    package = Path(cell_eval2.__file__).parent
    direct = json.loads(metadata.distribution('cell-eval2').read_text('direct_url.json'))
    if direct.get('vcs_info', {}).get('commit_id') != SCORER_COMMIT:
        raise ValueError('cell_eval2_commit_mismatch')
    files = {str(p.relative_to(package)): hash_file(p) for p in sorted(package.rglob('*'))
             if p.is_file() and p.suffix in {'.py', '.yaml'}}
    return {'repository': 'https://github.com/ArcInstitute/cell-eval2', 'commit': SCORER_COMMIT,
            'version': metadata.version('cell-eval2'), 'preset': 'vcc2026',
            'effective_config': asdict(scorer_config()), 'package_files_sha256': files,
            'scored_metrics': list(competition_members()),
            'baseline': {'profile': 'cell_eval2.baseline.generic_response_profile', 'exclude_target_gene': True,
                         'emission': 'cell_eval2.baseline.build_baseline_prediction', 'emit': 'dispersed', 'seed': 0,
                         'scope': 'scorer-only oracle computed from the exact scoring reference; never training input'},
            'anchors': {'function': 'cell_eval2.real_bundle.build_real_bundle', 'base_seed': 0, 'n_splits': 5,
                        'supplied_de_real': False, 'status': 'recipe_frozen_values_require_reference_execution'},
            'score': {'raw': 'cell_eval2.compute_metrics', 'aggregate': 'cell_eval2.aggregate_metrics_wide',
                      'normalized': 'cell_eval2.score_metrics(real_bundle=..., user_meta=...)',
                      'failure_policy': 'Preserve official NaN/gates. Invalid bundle blocks normalized Overall; never drop a member or substitute anchors.'}}


def build_reference_bundle(real, directory, bundle_id, **runtime):
    """Execute the frozen official recipe; reference labels stay evaluator-only."""
    from cell_eval2.baseline import generic_response_profile, build_baseline_prediction
    from cell_eval2.real_bundle import build_real_bundle
    cfg = scorer_config(**runtime)
    profile = generic_response_profile(real, pert_col=cfg.pert_col, control=cfg.control,
                                       exclude_target_gene=True)
    baseline = build_baseline_prediction(profile, real, pert_col=cfg.pert_col,
                                         control=cfg.control, emit='dispersed', seed=0)
    return build_real_bundle(real, baseline, config=cfg, outdir=str(directory),
                             bundle_id=bundle_id, base_seed=0, n_splits=5)


def score_prediction(prediction, real, bundle, directory, **runtime):
    """Official raw/normalized results with strict reference and six-member gates."""
    from cell_eval2 import compute_metrics, aggregate_metrics_wide, score_metrics
    from cell_eval2.baseline import build_run_meta
    from cell_eval2.competition import competition_members
    from cell_eval2.run import metric_output_names
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    cfg = scorer_config(**runtime)
    if not prediction.var_names.equals(real.var_names):
        raise ValueError('prediction_reference_gene_axis_mismatch')
    raw = compute_metrics(prediction, real, config=cfg)
    aggregate = aggregate_metrics_wide(raw, metrics=metric_output_names(cfg))
    meta = build_run_meta(cfg, real, prediction)
    raw.write_parquet(directory/'raw.parquet')
    aggregate.write_csv(directory/'aggregate.csv')
    write_json(directory/'run_meta.json', meta)
    scored = score_metrics(aggregate, real_bundle=str(bundle), user_meta=meta)
    scored.write_csv(directory/'scores.csv')
    members = competition_members()
    scores = {m: scored.filter(scored['metric'] == m)['from_replicate'].item() for m in members}
    if not all(v is not None and np.isfinite(v) for v in scores.values()):
        raise ValueError('official_six_metric_score_is_not_finite')
    overall = scored.filter(scored['metric'] == 'avg_score')['from_replicate'].item()
    if not np.isclose(overall, np.mean(list(scores.values())), atol=1e-12):
        raise ValueError('official_six_metric_overall_mismatch')
    return {'normalized': scores, 'Overall': overall}


def build(root, output, config_path):
    config = json.loads(config_path.read_text())
    if any(output.is_relative_to(root/p) for p in ['data', 'models']):
        raise ValueError('audit_output_must_not_modify_shared_inputs')
    if any(not isinstance(config.get(k), int) or config[k] <= 0 for k in
           ['chunk_rows', 'input_ntc_cap', 'minimum_reference_cells']):
        raise ValueError('positive_integer_audit_parameters_required')
    output.mkdir(parents=True, exist_ok=False)
    inputs = []
    official_root = root/'data/raw/arc_vcc2026_controls'
    for name in ['gene_names.csv', 'pert_counts.csv', 'manifest.json']:
        verify_input(official_root/name, root, inputs)
    hgnc_path = root/'data/raw/networks/hgnc_complete_set.txt'
    verify_input(hgnc_path, root, inputs)
    genes = pd.read_csv(official_root/'gene_names.csv').gene_name.tolist()
    official_targets = pd.read_csv(official_root/'pert_counts.csv').target_gene.tolist()
    manifest = json.loads((official_root/'manifest.json').read_text())
    if config['panel_id'] != manifest['panel_id']:
        raise ValueError('official_panel_version_mismatch')
    if len(genes) != manifest['n_genes'] or len(official_targets) != manifest['n_constructs']:
        raise ValueError('official_manifest_axis_mismatch')
    identity = ChallengeIdentity(genes, pd.read_csv(hgnc_path, sep='\t', low_memory=False))
    summaries, tasks, mappings, target_mappings = {}, [], [], []
    for context in PANELS:
        frame, mapping, summary = inspect_context(root, context, identity, config, inputs)
        summaries[context] = summary
        mapping.insert(0, 'context', context)
        mappings.append(mapping)
        tm = frame.groupby(['source_target', 'source_target_id', 'target', 'target_mapping'], dropna=False).size().reset_index(name='cells')
        tm.insert(0, 'context', context); target_mappings.append(tm)
        for target, group in frame.loc[frame.target.notna() & ~frame.target.eq(NTC)].groupby('target', sort=True):
            eligible = group.loc[group.eligible]
            reference = reference_indices(eligible, config['split_seed'], manifest['cells_per_pert'])
            diagnostic = reference_indices(eligible, config['split_seed'], manifest['cells_per_pert'], diagnostic=True)
            tasks.append({'context': context, 'study': summary['study'], 'target': target,
                'official_target': target in official_targets, 'partition': target_partition(target, config['split_seed']),
                'source_cells': len(group), 'eligible_cells': len(eligible),
                'reference_cells': len(reference), 'reference_identity_sha256': value_hash(sorted(frame.loc[reference, 'physical_id'])),
                'S0_reference_cells': len(diagnostic), 'S0_reference_identity_sha256': value_hash(sorted(frame.loc[diagnostic, 'physical_id'])),
                'guide_pairs': eligible.guide.nunique(), 'batches': eligible.batch.nunique(),
                'structurally_eligible': len(reference) >= config['minimum_reference_cells'],
                'exclusion_reason': '' if len(reference) >= config['minimum_reference_cells'] else 'insufficient_unique_real_cells_for_split_half',
                'official_metric_gates': 'pending_actual_cell_eval2', 'biological_replicates': 'unknown'})
        print(f'{context}: {summary["official_measured_genes"]}/{len(genes)} measured genes; NTC input={summary["ntc_pools"]["input"]["cells"]}, score={summary["ntc_pools"]["score"]["cells"]}', flush=True)
    tasks = pd.DataFrame(tasks)
    mapping = pd.concat(mappings, ignore_index=True)
    mapping.to_csv(output/'gene_mapping.csv.gz', index=False, compression={'method': 'gzip', 'mtime': 0})
    pd.concat(target_mappings, ignore_index=True).to_csv(output/'target_mapping.csv', index=False)
    tasks.to_csv(output/'tasks.csv', index=False)
    axis = pd.DataFrame({'official_position': range(len(genes)), 'gene_name': genes})
    for context, summary in summaries.items():
        axis[context+'_measured'] = axis.official_position.isin(summary['official_gene_positions'])
    axis.to_csv(output/'official_gene_axis.csv', index=False)
    coverage = pd.DataFrame({'target_gene': official_targets})
    for context in summaries:
        counts = tasks.loc[tasks.context.eq(context)].set_index('target').eligible_cells
        coverage[context+'_eligible_cells'] = coverage.target_gene.map(counts).fillna(0).astype(int)
    coverage.to_csv(output/'official_target_coverage.csv', index=False)
    splits = build_splits(tasks, config)
    write_json(output/'splits.json', splits)
    write_json(output/'scorer.json', scorer_contract())
    write_json(output/'data_audit.json', {'inputs': inputs, 'contexts': summaries, 'config': config,
        'config_sha256': hash_file(config_path), 'official_manifest': manifest,
        'package_versions': {n: metadata.version(n) for n in ['numpy', 'scipy', 'pandas', 'h5py', 'cell-eval2']},
        'coverage_scope': 'full official axis with explicit masks; each local scorer panel is its measured subset in official order',
        'data_contract_complete': True, 'all_normalized_scores_available': False})
    write_json(output/'artifacts.json', {p.name: hash_file(p) for p in sorted(output.iterdir()) if p.is_file()})
    print('Completed immutable mapping/split audit; actual scorer gates and anchors remain runtime checks.', flush=True)


def build_splits(tasks, config):
    """Explicit training target sets prevent global holdout labels leaking across contexts."""
    eligible = tasks.loc[tasks.structurally_eligible]
    splits = []
    studies = {c: s for c, (s, _) in PANELS.items()}
    for scenario in ['S0', 'S1', 'S2', 'S3', 'S4']:
        holdouts = list(PANELS) if scenario in ['S2', 'S3'] else sorted(set(studies.values())) if scenario == 'S4' else ['all']
        for holdout in holdouts:
            test_contexts = [c for c in PANELS if c == holdout] if scenario in ['S2', 'S3'] else [c for c, s in studies.items() if s == holdout] if scenario == 'S4' else list(PANELS)
            train_contexts = list(PANELS) if scenario in ['S0', 'S1'] else [c for c in PANELS if c not in test_contexts]
            train = tasks.loc[tasks.context.isin(train_contexts) & tasks.eligible_cells.gt(0)]
            if scenario in ['S1', 'S3']:
                train = train.loc[train.partition.eq('train')]
            allowed_targets = sorted(set(train.target))
            test = eligible.loc[eligible.context.isin(test_contexts)].copy()
            if scenario in ['S1', 'S3']:
                test = test.loc[test.partition.eq('test')]
            if scenario == 'S2':
                test = test.loc[test.target.isin(allowed_targets)]
            if scenario == 'S0':
                test = test.loc[test.S0_reference_cells.ge(config['minimum_reference_cells'])]
            splits.append({'id': f'{scenario}-{holdout}', 'scenario': scenario,
                'training_contexts': train_contexts, 'evaluation_contexts': test_contexts,
                'training_targets': allowed_targets,
                'training_cells': 'eligible except held-out S0_reference identities' if scenario == 'S0' else 'eligible in training_contexts and training_targets',
                'validation_targets': sorted(set(tasks.loc[tasks.partition.eq('validation'), 'target'])) if scenario in ['S1', 'S3'] else [],
                'target_partition': 'global-train60-validation20-test20' if scenario in ['S1', 'S3'] else 'context-seen' if scenario == 'S2' else 'all-with-seen-unseen-strata',
                'evaluation_targets': {c: sorted(test.loc[test.context.eq(c), 'target']) for c in test_contexts},
                'official_overlap_targets': {c: sorted(test.loc[test.context.eq(c) & test.official_target, 'target']) for c in test_contexts},
                'evaluation_natural_unseen_targets': {c: sorted(test.loc[test.context.eq(c) & ~test.target.isin(allowed_targets), 'target']) for c in test_contexts},
                'gene_panel': 'official_gene_axis.csv:<context>_measured, in official order; no cross-context intersection reduction',
                'metric_gate_status': 'pending_cell_eval2_reference_and_anchors',
                'scope': 'pipeline diagnostic only' if scenario == 'S0' else 'local development evidence; not official leaderboard'})
    return splits


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    build(args.root.resolve(), args.output.resolve(), args.config.resolve())
