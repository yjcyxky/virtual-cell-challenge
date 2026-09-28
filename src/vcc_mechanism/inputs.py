"""Reuse immutable populations, independently of the candidate's model/decoder."""
import gc
import json
import os
from pathlib import Path
import shutil

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
import yaml

from data import ROOT, NTC, ref, write_json, normalized
from challenge import ChallengeIdentity
from rna import RNAFile, hash_file


def verified(record):
    path = ROOT / record['path']
    if ref(path) != record:
        raise ValueError(f'Frozen input changed: {path}')
    return path


def matching_populations(previous, config):
    # Model/task/decoder may differ; physical populations, normalization scale,
    # masks, seed, target panel and scorer must match the historical control.
    for key in ('benchmark', 'data', 'fit_scope', 'seed'):
        if previous[key] != config[key]:
            raise ValueError(f'Population contract changed: {key}')
    before, after = dict(previous['evaluation']), dict(config['evaluation'])
    old_arms, new_arms = before.pop('arms'), after.pop('arms')
    if before != after or not set(new_arms) <= set(old_arms):
        raise ValueError('Evaluation contract changed')
    for key in ('cells_per_target', 'paired_rng', 'missing_input'):
        if previous['generation'][key] != config['generation'][key]:
            raise ValueError(f'Generation population changed: {key}')


def completed_source(metrics, previous, run_id):
    if (metrics['status'] != 'completed' or not metrics['evaluation_completed'] or
            metrics['research'] != previous['research'] or metrics['research']['node_id'] != run_id):
        raise ValueError('Source must be a completed bound run')


def prepare(output, config):
    manifest = json.loads(verified(config['reuse']).read_text())
    metrics = json.loads(verified(manifest['source_metrics']).read_text())
    previous = yaml.safe_load(verified(manifest['source_config']).read_text())
    completed_source(metrics, previous, manifest['source_run_id'])
    matching_populations(previous, config)
    source_cache = (ROOT / manifest['source_metrics']['path']).parent / 'cache'
    for item in manifest['files']:
        relative = Path(item['destination'])
        if relative.is_absolute() or '..' in relative.parts or relative.parts[0] != 'cache':
            raise ValueError('Invalid cache destination')
        source = verified(item['ref'])
        if source.resolve() != (source_cache / Path(*relative.parts[1:])).resolve():
            raise ValueError('Input reuse changed cache layout')
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            temporary = destination.with_suffix(destination.suffix + '.tmp')
            shutil.copyfile(source, temporary)
            temporary.replace(destination)
        if hash_file(destination) != item['ref']['sha256']:
            raise ValueError('Copied cache changed')
    sources = config['control_refs']
    for record in sources.values():
        verified(record)
    control = json.loads((ROOT / sources['metrics']['path']).read_text())
    control_config = yaml.safe_load((ROOT / sources['config']['path']).read_text())
    completed_source(control, control_config, 'masked-response-shared-s01')
    matching_populations(control_config, config)
    predictions = json.loads((ROOT / sources['predictions']['path']).read_text())['files']
    scores = json.loads((ROOT / sources['scores']['path']).read_text())['files']
    for record in predictions + scores:
        verified(record)
    write_json(output/'cache/control-provenance.json', {
        'sources': sources, 'verified_predictions': predictions, 'verified_scores': scores,
        'role': 'historical independent control; model and generator may change as registered',
        'candidate_scored_afresh': True, 'matched_populations_and_evaluation': True})
    splits = json.loads(verified(config['benchmark']['split_manifest']).read_text())
    split = next(s for s in splits if s['id'] == config['fit_scope']['outer_split'])
    if split['id'] != 'S2-H1':
        raise ValueError('Implementation only registered for S2-H1')
    axis = pd.read_csv(verified(config['benchmark']['gene_axis'])).gene_name.tolist()
    prepared = {c: json.loads((output/f'cache/{c}-prepared.json').read_text())
                for c in split['training_contexts'] + ['H1']}
    write_json(output/'cache/preparation.json', prepared)
    print(f'Verified {len(manifest["files"])} frozen input/bundle files and independent shared control', flush=True)
    return axis, split, prepared


def accumulate_moments(sums, squares, logs, inverse_library, counts, block, groups, target_sum):
    if not len(groups):
        return
    present, inverse = np.unique(groups, return_inverse=True)
    selector = sparse.csr_matrix((np.ones(len(groups)), (inverse, np.arange(len(groups)))),
                                shape=(len(present), len(groups)))
    raw = block.astype(np.float64)
    depth = np.asarray(raw.sum(1)).ravel()
    if (depth <= 0).any():
        raise ValueError('Empty measured library')
    cpm = (sparse.diags(target_sum/depth) @ raw).tocsr()
    sums[present] += (selector @ cpm).toarray()
    squares[present] += (selector @ cpm.power(2)).toarray()
    logs[present] += (selector @ normalized(block, target_sum)).toarray()
    inverse_library[present] += np.asarray(selector @ (1/depth)).ravel()
    counts += np.bincount(groups, minlength=len(counts))


def moments(output, axis, split, config):
    """Exact same selected cells; compute arithmetic CPM moments, never missing zeros."""
    directory = output/'cache/moments'
    directory.mkdir(exist_ok=True)
    audit = json.loads(verified(config['benchmark']['data_audit']).read_text())
    source_refs = {r['path']: r for r in audit['inputs']}
    identity = ChallengeIdentity(axis, pd.read_csv(verified(config['moments']['identities']), sep='\t', low_memory=False))
    for context in split['training_contexts'] + ['H1']:
        done = directory/f'{context}.json'
        if done.exists():
            record = json.loads(done.read_text())
            verified(record['moments_ref'])
            continue
        base = dict(np.load(output/f'cache/{context}-statistics.npz'))
        sums, squares, logs = [np.zeros_like(base['mean'], dtype=np.float64) for _ in range(3)]
        counts = np.zeros_like(base['counts'])
        inv = np.zeros(len(counts))
        if context == 'H1':
            ntc = ad.read_h5ad(output/'cache/H1-input.h5ad')
            for start in range(0, ntc.n_obs, config['data']['chunk_rows']):
                block = ntc.X[start:start+config['data']['chunk_rows']]
                accumulate_moments(sums, squares, logs, inv, counts, block, np.zeros(block.shape[0], dtype=int), config['data']['target_sum'])
            # H1 input file concatenation changes summation grouping only.
            np.testing.assert_allclose(logs/counts[:, None], base['mean'], rtol=1e-6, atol=1e-7)
            source_record = ref(output/'cache/H1-input.h5ad')
            del ntc
        else:
            frame = pd.read_parquet(output/f'cache/{context}-identities.parquet')
            frame = frame.loc[(frame.eligible & frame.target.isin(split['training_targets'])) | frame.ntc_pool.eq('input')]
            if frame.file.nunique() != 1:
                raise ValueError('Training context must have exactly one raw file')
            path = ROOT/'data/raw'/frame.file.iloc[0]
            source_record = source_refs[str(path.relative_to(ROOT))]
            before = path.stat()
            print(f'Hashing raw moments input: {context}', flush=True)
            if hash_file(path) != source_record['sha256']:
                raise ValueError('Raw moment source checksum mismatch')
            with RNAFile(path) as source:
                mapping = identity.features(source.var.gene_name.astype(str), source.var.index.astype(str))
                measured = mapping.loc[mapping.measured].sort_values('official_position')
                if not np.array_equal(measured.official_position.to_numpy(), base['positions']):
                    raise ValueError('Moment measured gene axis changed')
                row_groups = np.full(source.shape[0], -1, dtype=np.int64)
                labels = {t: i for i, t in enumerate(base['labels'])}
                row_groups[frame.source_row] = frame.target.map(labels).to_numpy()
                for start, block in source.blocks(config['data']['chunk_rows']):
                    groups = row_groups[start:start+block.shape[0]]
                    keep = groups >= 0
                    if keep.any():
                        accumulate_moments(sums, squares, logs, inv, counts,
                                           block[keep][:, measured.source_position.to_numpy()],
                                           groups[keep], config['data']['target_sum'])
            after = path.stat()
            if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
                raise ValueError('Raw moment input changed while reading')
            np.testing.assert_array_equal((logs/counts[:, None]).astype(np.float32), base['mean'])
        np.testing.assert_array_equal(counts, base['counts'])
        mean = sums/counts[:, None]
        variance = np.maximum(squares-counts[:, None]*mean**2, 0)/np.maximum(counts[:, None]-1, 1)
        path = directory/f'{context}.npz'
        with path.with_suffix('.tmp').open('wb') as handle:
            np.savez_compressed(handle, labels=base['labels'], positions=base['positions'], counts=counts,
                                mean=mean, variance=variance, inverse_library=inv/counts)
        path.with_suffix('.tmp').replace(path)
        write_json(done, {'context': context, 'source_ref': source_record, 'moments_ref': ref(path),
                          'base_statistics': ref(output/f'cache/{context}-statistics.npz'),
                          'cells': counts.tolist(), 'measured_genes': len(base['positions']),
                          'same_cells_and_log_means_verified': True,
                          'perturbed_labels_used': context != 'H1'})
        print(f'Moments complete: {context}, {len(counts)-1} training targets, {int(counts.sum())} cells', flush=True)
        del sums, squares, logs, mean, variance, base
        gc.collect()


def release_file_cache(output):
    """Advisory release of consumed immutable inputs; content and file state unchanged."""
    paths = list((output/'cache').rglob('*.h5ad')) + list((output/'cache').rglob('*.npz')) + list((output/'cache').rglob('*.npy')) + list((output/'predictions').glob('*.h5ad'))
    audit = json.loads((ROOT/'docs/research/challenge_2026/data_audit.json').read_text())
    paths += [ROOT/r['path'] for r in audit['inputs'] if r['path'].endswith('.h5ad')]
    for path in paths:
        with path.open('rb') as handle:
            os.posix_fadvise(handle.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
