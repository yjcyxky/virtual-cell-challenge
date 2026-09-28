"""Training-only depth selection with exact target/guide/batch random controls.

Frozen input caches stay immutable. Selected pseudobulks are recomputed from raw
retained cells into cache/selected; input NTC and all evaluation populations stay
unchanged. Low library size is a proxy being tested, not a correctness label.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
import yaml

from data import ROOT, normalized, ref, write_json
from challenge import ChallengeIdentity, NTC
from rna import RNAFile, hash_file, quantiles, value_hash

STRATA = ['target', 'guide', 'batch']


def select_cells(frame, context, spec):
    """Return both selections; output depends only on training depth and identity."""
    frame = frame.copy()
    if frame.physical_id.duplicated().any() or not np.isfinite(frame.depth).all() or (frame.depth <= 0).any():
        raise ValueError('invalid selection identity/depth')
    frame['threshold'] = frame.groupby('batch').depth.transform(
        lambda x: x.quantile(spec['depth_quantile'], interpolation='linear'))
    frame['flagged'] = frame.depth.lt(frame.threshold)
    protected = frame.sort_values(['depth', 'physical_id'], ascending=[False, True]).drop_duplicates(STRATA).index
    frame['qc_keep'] = ~frame.flagged
    frame.loc[protected, 'qc_keep'] = True
    frame['remove_count'] = (~frame.qc_keep).groupby([frame[k] for k in STRATA]).transform('sum')
    frame['random_key'] = [hashlib.sha256(json.dumps([spec['selection_seed'], context, p],
                           separators=(',', ':')).encode()).hexdigest() for p in frame.physical_id]
    ordered = frame.sort_values(['random_key', 'physical_id'])
    rank = ordered.groupby(STRATA, sort=False).cumcount().reindex(frame.index)
    frame['random_keep'] = rank >= frame.remove_count
    counts = frame.groupby(STRATA, sort=True)[['qc_keep', 'random_keep']].sum()
    if not counts.qc_keep.equals(counts.random_keep) or (counts.qc_keep < 1).any():
        raise ValueError('selection lost exact stratum matching/coverage')
    return frame.drop(columns=['random_key', 'remove_count'])


def accumulate(sums, counts, block, groups, target_sum):
    """Sum per-cell normalized expression, never normalize a pooled count row."""
    if not len(groups):
        return
    present, inverse = np.unique(groups, return_inverse=True)
    selector = sparse.csr_matrix((np.ones(len(groups)), (inverse, np.arange(len(groups)))),
                                 shape=(len(present), len(groups)))
    sums[present] += (selector @ normalized(block, target_sum)).toarray()
    counts += np.bincount(groups, minlength=len(counts))


def verify_control(output, config):
    """Verify a historical comparison control; never relabel it as current shared."""
    spec = config['training_selection']
    references = spec['control_refs']
    for record in references.values():
        if ref(ROOT / record['path']) != record:
            raise ValueError('historical control checksum mismatch')
    previous = yaml.safe_load((ROOT / references['config']['path']).read_text())
    metrics = json.loads((ROOT / references['metrics']['path']).read_text())
    if (metrics['status'] != 'completed' or not metrics['evaluation_completed'] or
            metrics['research'] != previous['research'] or
            metrics['research']['node_id'] != 'masked-response-shared-s01'):
        raise ValueError('historical control is not completed and bound')
    validate_control_config(previous, config)
    manifest = json.loads((ROOT / references['predictions']['path']).read_text())
    for record in manifest['files']:
        if ref(ROOT / record['path']) != record:
            raise ValueError('historical control prediction changed')
    scores = json.loads((ROOT / references['scores']['path']).read_text())['files']
    for record in scores:
        if ref(ROOT / record['path']) != record:
            raise ValueError('historical control score changed')
    write_json(output / 'cache/control-provenance.json', {
        'control_run': 'masked-response-shared-s01', 'sources': references,
        'verified_predictions': manifest['files'], 'verified_scores': scores,
        'role': 'historical unfiltered control only; candidate shared is refit and scored afresh',
        'current_shared_equal_to_control': False, 'control_scores_merged_as_candidate_references': False})


def validate_control_config(previous, config):
    for key in ('benchmark', 'data', 'fit_scope', 'seed', 'generation', 'evaluation', 'model', 'representation', 'reuse'):
        if previous[key] != config[key]:
            raise ValueError(f'historical control mismatch: {key}')


def prepare_selected(output, config, axis, split):
    spec = config['training_selection']
    if (split['id'] != 'S2-H1' or set(split['training_contexts']) != {'K562', 'RPE1', 'HepG2', 'Jurkat'} or
            spec['kind'] not in ('low_depth', 'matched_random') or config['representation']['kind'] != 'shared'):
        raise ValueError('selection outside registered training scope')
    verify_control(output, config)
    directory = output / 'cache/selected'
    directory.mkdir(exist_ok=True)
    audit = json.loads((ROOT / config['benchmark']['data_audit']['path']).read_text())
    source_refs = {r['path']: r for r in audit['inputs']}
    identity = ChallengeIdentity(axis, pd.read_csv(ROOT / config['representation']['identities']['path'],
                                                   sep='\t', low_memory=False))
    records = {}
    for context in split['training_contexts']:
        done = directory / f'{context}-selection.json'
        if done.exists():
            record = json.loads(done.read_text())
            if record['selection_sha256'] != value_hash(spec):
                raise ValueError('selection resume configuration changed')
            for item in record['files']:
                if ref(ROOT / item['path']) != item:
                    raise ValueError('selected cache changed')
            records[context] = record
            continue
        base = dict(np.load(output / f'cache/{context}-statistics.npz'))
        frame = pd.read_parquet(output / f'cache/{context}-identities.parquet')
        frame = frame.loc[frame.eligible & frame.target.isin(split['training_targets']) & frame.target.ne(NTC)].copy()
        if frame.file.nunique() != 1 or set(frame.target) != set(base['labels'][1:]):
            raise ValueError('selection training target/source mismatch')
        if not np.array_equal(frame.target.value_counts().reindex(base['labels'][1:]), base['counts'][1:]):
            raise ValueError('selection base population counts changed')
        relative = frame.file.iloc[0]
        source_path = ROOT / 'data/raw' / relative
        source_ref = source_refs[str(source_path.relative_to(ROOT))]
        before = source_path.stat()
        print(f'Verifying raw selection source: {context}', flush=True)
        if hash_file(source_path) != source_ref['sha256']:
            raise ValueError('raw selection input checksum mismatch')
        with RNAFile(source_path) as source:
            frame['depth'] = source.obs.UMI_count.to_numpy()[frame.source_row.to_numpy()]
            selected = select_cells(frame, context, spec)
            keep_column = 'qc_keep' if spec['kind'] == 'low_depth' else 'random_keep'
            selected['retained'] = selected[keep_column]
            labels = base['labels']; label_index = {t: i for i, t in enumerate(labels)}
            mapping = identity.features(source.var.gene_name.astype(str), source.var.index.astype(str))
            measured = mapping.loc[mapping.measured].sort_values('official_position')
            positions = measured.official_position.to_numpy(dtype=int)
            if not np.array_equal(positions, base['positions']):
                raise ValueError('selected measured gene axis changed')
            columns = measured.source_position.to_numpy()
            row_groups = np.full(source.shape[0], -1, dtype=np.int64)
            retained = selected.loc[selected.retained]
            row_groups[retained.source_row] = retained.target.map(label_index).to_numpy()
            sums = np.zeros_like(base['mean'], dtype=np.float64)
            counts = np.zeros_like(base['counts'])
            print(f'Recomputing {context}: {len(retained)}/{len(frame)} training cells retained', flush=True)
            for start, block in source.blocks(config['data']['chunk_rows']):
                groups = row_groups[start:start + block.shape[0]]
                keep = groups >= 0
                if keep.any():
                    accumulate(sums, counts, block[keep][:, columns], groups[keep], config['data']['target_sum'])
            expected = retained.target.value_counts().reindex(labels[1:]).to_numpy()
            if not np.array_equal(counts[1:], expected) or (counts[1:] <= 0).any():
                raise ValueError('selected counts differ from cell selection')
            # Frozen input-NTC population and its exact float32 mean are invariant.
            counts[0] = base['counts'][0]
            mean = (sums / counts[:, None]).astype(np.float32)
            mean[0] = base['mean'][0]
        after = source_path.stat()
        if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
            raise ValueError('raw input changed while selecting')
        stats_path = directory / f'{context}-statistics.npz'
        with stats_path.with_suffix('.tmp').open('wb') as handle:
            np.savez_compressed(handle, labels=labels, positions=positions, mean=mean, counts=counts)
        stats_path.with_suffix('.tmp').replace(stats_path)
        selection_path = directory / f'{context}-cells.parquet'
        selected.to_parquet(selection_path, index=False)
        response_before = np.sqrt(np.mean((base['mean'][1:] - base['mean'][0])**2, axis=1))
        response_after = np.sqrt(np.mean((mean[1:] - mean[0])**2, axis=1))
        targets_path = directory / f'{context}-targets.csv'
        pd.DataFrame({'target': labels[1:], 'base_cells': base['counts'][1:], 'retained_cells': counts[1:],
                      'base_response_rms': response_before, 'selected_response_rms': response_after}).to_csv(targets_path, index=False)
        record = {'context': context, 'kind': spec['kind'], 'selection_sha256': value_hash(spec),
                  'source_ref': source_ref, 'base_statistics': ref(output / f'cache/{context}-statistics.npz'),
                  'training_cells_before': len(frame), 'training_cells_after': len(retained),
                  'targets': len(labels)-1, 'guide_pairs': frame.guide.nunique(), 'batches': frame.batch.nunique(),
                  'strata': len(frame.drop_duplicates(STRATA)), 'all_strata_preserved': True,
                  'qc_random_stratum_counts_equal': True, 'input_ntc_unchanged': True,
                  'depth_before': quantiles(frame.depth), 'depth_after': quantiles(retained.depth),
                  'selected_identity_sha256': value_hash(sorted(retained.physical_id)),
                  'files': [ref(stats_path), ref(selection_path), ref(targets_path)]}
        write_json(done, record)
        records[context] = record
        print(f'Selection complete: {context}, {len(retained)} cells, {len(labels)-1} targets', flush=True)
    write_json(output / 'cache/selection.json', records)
    return records
