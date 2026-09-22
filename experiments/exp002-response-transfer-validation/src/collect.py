"""Reconstruct frozen effects and collect disjoint cell-split sufficient statistics."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'scripts/dossier'))
import h5py
import numpy as np
import pandas as pd
from scipy import sparse
from heterogeneity_cells import CellContexts
from heterogeneity import safe_symbols, compare_vectors
from profile_responses import write_json
from rna import hash_file

SEEDS = [20260921, 20260922, 20260923]
VARIANTS = ['full'] + [f's{seed}h{half}' for seed in SEEDS for half in range(2)]
SOURCE = ROOT / 'data/assessments/cross-source-dossier-20260919'
FROZEN = ROOT / 'data/assessments/response-heterogeneity-dossier-20260919'


def split_cells(keys, strata, identity, seed):
    """Stable, disjoint, balanced split within each stratum; random odd remainder."""
    keys = np.asarray(keys, dtype=str)
    if len(np.unique(keys)) != len(keys):
        raise ValueError('duplicate split identities')
    result = np.full(len(keys), -1, dtype=np.int8)
    for name, indices in pd.Series(np.arange(len(keys))).groupby(np.asarray(strata), sort=True):
        indices = indices.to_numpy()
        indices = indices[np.argsort(keys[indices])]
        digest = hashlib.sha256(f'{identity}|{seed}|{name}'.encode()).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], 'little'))
        order = rng.permutation(indices)
        result[order] = (np.arange(len(order)) + rng.integers(2)) % 2
    assert np.all(result >= 0)
    return result


def moments(log, ids, membership):
    """Multiple overlapping estimands, accumulated in float64 with bounded memory."""
    n = membership.sum(axis=1).astype(int)
    sums = np.zeros((len(n), log.shape[1]), dtype=np.float64)
    squares = np.zeros_like(sums)
    for start in range(0, len(ids), 512):
        block = log[ids[start:start + 512]].astype(np.float64)
        weights = sparse.csr_matrix(membership[:, start:start + 512].astype(float))
        a = weights @ block
        b = weights @ (block.multiply(block) if sparse.issparse(block) else block * block)
        sums += a.toarray() if sparse.issparse(a) else a
        squares += b.toarray() if sparse.issparse(b) else b
    means = np.divide(sums, n[:, None], out=np.full_like(sums, np.nan), where=n[:, None] > 0)
    # Variance of sample mean, not variance of individual cells.
    vmean = np.divide(np.maximum(squares - sums * np.nan_to_num(means), 0),
                      (n * (n - 1))[:, None], out=np.full_like(sums, np.nan), where=n[:, None] > 1)
    return n, means, vmean


def geometry(target, baseline, safe):
    delta = target[safe] - baseline[safe]
    result = compare_vectors(target[safe], baseline[safe])
    sd = float(np.std(baseline[safe])) if safe.any() else None
    return {'NTC_target_correlation': result['correlation'], 'safe_genes': int(safe.sum()),
            'baseline_SD': sd, 'response_RMS': float(np.sqrt(np.mean(delta ** 2))) if len(delta) else None,
            'response_RMS_over_baseline_SD': float(np.sqrt(np.mean(delta ** 2)) / sd) if sd else None,
            **{f'fraction_abs_delta_gt_{t}': float(np.mean(np.abs(delta) > t)) if len(delta) else None
               for t in [.05, .1, .2, .5]}}


def collect_context(context, output):
    output.mkdir(parents=True, exist_ok=False)
    cells, log = context['cells'], context['log']
    tasks, labels = context['tasks'], context['task_labels']
    control = context['control']
    genes = context['genes']
    safe = np.asarray(safe_symbols(context['mapping']), dtype=object)
    if cells.source_batch.isna().any():
        raise ValueError('missing technical stratum')
    batches, names = pd.factorize(cells.source_batch, sort=True)
    ctrl_ids = np.flatnonzero(control)
    ctrl_n = np.bincount(batches[ctrl_ids], minlength=len(names))
    # Shared H1 controls have identical barcode identities across source copies.
    half = np.full((3, len(cells)), -1, dtype=np.int8)
    for si, seed in enumerate(SEEDS):
        half[si, ctrl_ids] = split_cells(cells.source_barcode.to_numpy()[ctrl_ids],
                                        cells.source_batch.to_numpy()[ctrl_ids], context['baseline_id'], seed)
    task_groups = {key: np.asarray(ids, dtype=int) for key, ids in
                   pd.Series(labels).groupby(pd.Series(labels), dropna=True, sort=False).groups.items()}
    for uid, ids in task_groups.items():
        for si, seed in enumerate(SEEDS):
            half[si, ids] = split_cells(cells.record_id.to_numpy()[ids],
                                       cells.source_batch.to_numpy()[ids], uid, seed)
    sidecar = cells[['record_id', 'input_sha256', 'row_index', 'source_barcode', 'source_batch']].copy()
    sidecar['task_uid'], sidecar['is_NTC'] = labels, control
    sidecar['baseline_id'] = context['baseline_id']
    sidecar['split_NTC_support_eligible'] = ctrl_n[batches] >= 4
    for i, seed in enumerate(SEEDS):
        sidecar[f'half_{seed}'] = half[i]
    sidecar.to_parquet(output / 'cell-splits.parquet', index=False, compression='zstd')
    context['mapping'].to_parquet(output / 'native-gene-mapping.parquet', index=False)
    tasks.reset_index(drop=True).to_parquet(output / 'tasks.parquet', index=False)
    # Baseline moments by technical stratum. NaN variance is retained for singleton NTC.
    cm = np.full((7, len(names), len(genes)), np.nan)
    cv = np.full_like(cm, np.nan)
    cn = np.zeros((7, len(names)), dtype=int)
    control_membership = np.vstack([np.ones(len(ctrl_ids), bool)] +
                                  [half[si, ctrl_ids] == h for si in range(3) for h in range(2)])
    _, pooled_mean, pooled_var = moments(log, ctrl_ids, control_membership)
    for bi in np.unique(batches[ctrl_ids]):
        ids = ctrl_ids[batches[ctrl_ids] == bi]
        membership = np.vstack([np.ones(len(ids), bool)] +
                              [half[si, ids] == h for si in range(3) for h in range(2)])
        cn[:, bi], cm[:, bi], cv[:, bi] = moments(log, ids, membership)
        if ctrl_n[bi] >= 4:
            assert (cn[1:, bi] >= 2).all()
    np.savez_compressed(output / 'control-moments.npz', mean=pooled_mean, variance_mean=pooled_var,
                        counts=control_membership.sum(axis=1), variants=np.array(VARIANTS),
                        stratum_names=np.asarray(names, str), stratum_counts=cn)
    rows, diagnostics, reproduction = [], [], []
    with h5py.File(output / 'moments.h5', 'w') as hf:
        hf.create_dataset('source_gene', data=np.asarray(genes, dtype=h5py.string_dtype()))
        hf.create_dataset('safe_symbol', data=np.asarray([x if pd.notna(x) else '' for x in safe], dtype=h5py.string_dtype()))
        hf.create_dataset('task_uid', data=np.asarray(tasks.task_uid, dtype=h5py.string_dtype()))
        for name in ['target', 'matched_control', 'target_variance_mean', 'matched_control_variance_mean']:
            hf.create_dataset(name, shape=(len(tasks), 7, len(genes)), dtype='float32',
                              chunks=(1, 1, min(len(genes), 8192)), compression='gzip', compression_opts=1,
                              fillvalue=np.nan)
        hf.create_dataset('target_excluded', shape=(len(tasks), len(genes)), dtype='bool',
                          chunks=(1, min(len(genes), 8192)), compression='gzip', compression_opts=1)
        for ti, task in enumerate(tasks.to_dict('records')):
            uid = task['task_uid']
            ids = task_groups.get(uid, np.array([], dtype=int))
            supported = ctrl_n[batches[ids]] >= 4
            original = ctrl_n[batches[ids]] > 0
            membership = np.vstack([original] + [supported & (half[si, ids] == h) for si in range(3) for h in range(2)])
            n, tm, tv = moments(log, ids, membership)
            matched_mean, matched_var = np.full_like(tm, np.nan), np.full_like(tv, np.nan)
            for v in range(7):
                if n[v]:
                    used = batches[ids[membership[v]]]
                    weight = np.bincount(used, minlength=len(names)) / n[v]
                    included = weight > 0
                    matched_mean[v] = weight[included] @ cm[v, included]
                    matched_var[v] = weight[included] ** 2 @ cv[v, included]
            source_target = np.zeros(len(genes), bool)
            error = None
            if task['effect_status'] == 'completed':
                path = ROOT / task['source_folder'] / task['source_gene_result_file']
                if hash_file(path) != task['source_gene_result_sha256']:
                    raise ValueError('frozen source gene result changed')
                old = pd.read_parquet(path, columns=['source_gene', 'is_target', 'effect_all_matched_cells',
                                                     'mean_logCP10K_target', 'mean_logCP10K_matched_control'])
                assert old.source_gene.astype(str).tolist() == genes
                source_target = old.is_target.to_numpy(dtype=bool)
                error = max(float(np.max(np.abs(a - b))) for a, b in
                            [(tm[0], old.mean_logCP10K_target.to_numpy()),
                             (matched_mean[0], old.mean_logCP10K_matched_control.to_numpy()),
                             (tm[0] - matched_mean[0], old.effect_all_matched_cells.to_numpy())])
                if not np.isfinite(error) or error > 2e-6:
                    raise ValueError(f'source mean reconstruction failed {uid} {error}')
            elif n[0]:
                raise ValueError('unexpected newly estimable source task')
            excluded = source_target | (safe == task['canonical_target'])
            downstream = pd.notna(safe) & ~excluded
            hf['target_excluded'][ti] = excluded
            for name, values in [('target', tm), ('matched_control', matched_mean),
                                 ('target_variance_mean', tv), ('matched_control_variance_mean', matched_var)]:
                hf[name][ti] = values
            reproduction.append({'task_uid': uid, 'source_status': task['effect_status'], 'maximum_absolute_error': error})
            base = {k: task[k] for k in ['task_uid', 'panel_id', 'canonical_target', 'source_task', 'effect_status']}
            base['context_id'] = context['context_id']
            for v, variant in enumerate(VARIANTS):
                eligible = n[v] >= (1 if v == 0 else 10)
                rows.append({**base, 'variant': variant, 'n_target': int(n[v]), 'observed_target_cells': len(ids),
                             'full_matched_target_cells': int(n[0]), 'split_supported_target_cells': int(supported.sum()),
                             'support_fraction': float(supported.sum() / n[0]) if n[0] else None,
                             'status': 'completed' if eligible else 'not_estimable',
                             'reason': None if eligible else ('no_matched_NTC' if not n[v] else 'fewer_than_10_half_target_cells')})
            if n[0]:
                for scale, baseline in [('source_matched', matched_mean[0]), ('control_only', pooled_mean[0])]:
                    diagnostics.append({**base, 'experiment': 'geometry', 'scale': scale, **geometry(tm[0], baseline, downstream)})
            for si, seed in enumerate(SEEDS):
                a, b = 1 + si * 2, 2 + si * 2
                for scale, baseline in [('source_matched', matched_mean), ('control_only', pooled_mean)]:
                    status = n[a] >= 10 and n[b] >= 10
                    d0, d1 = tm[a] - baseline[a], tm[b] - baseline[b]
                    record = {**base, 'experiment': 'split_reproducibility', 'scale': scale, 'seed': seed,
                              'status': 'completed' if status else 'not_estimable'}
                    if status:
                        record.update(compare_vectors(d0[downstream], d1[downstream]))
                        record['half_response_cross_product'] = float(np.mean(d0[downstream] * d1[downstream]))
                        # Cell-count weighted half means reproduce the supported target mean.
                        supported_delta = (n[a] * d0 + n[b] * d1) / (n[a] + n[b])
                        record['supported_vs_original_response_RMS_difference'] = float(np.sqrt(np.mean(
                            (supported_delta[downstream] - (tm[0] - baseline[0])[downstream]) ** 2)))
                    diagnostics.append(record)
            if (ti + 1) % 100 == 0:
                print(json.dumps({'context': context['panel_id'], 'tasks': ti + 1, 'total': len(tasks)}), flush=True)
    pd.DataFrame(rows).to_parquet(output / 'task-support.parquet', index=False, compression='zstd')
    pd.DataFrame(diagnostics).to_parquet(output / 'diagnostics.parquet', index=False, compression='zstd')
    pd.DataFrame(reproduction).to_parquet(output / 'source-reproduction.parquet', index=False)
    report = {k: context[k] for k in ['context_id', 'panel_id', 'source_context', 'source_metadata', 'baseline_id']}
    report.update(tasks=len(tasks), NTC_cells=int(control.sum()), source_mean_max_error=max(
        [r['maximum_absolute_error'] for r in reproduction if r['maximum_absolute_error'] is not None], default=None),
        status='completed')
    report['artifacts'] = {p.name: hash_file(p) for p in output.iterdir() if p.is_file()}
    write_json(output / 'report.json', report)
    return report


def run(output):
    output.mkdir(parents=True, exist_ok=True)
    identity = {'code_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                'source_report_sha256': hash_file(SOURCE / 'report.json'),
                'issue': 'https://github.com/yjcyxky/virtual-cell-challenge/issues/28',
                'seeds': SEEDS, 'variants': VARIANTS, 'minimum_half_targets': 10, 'minimum_stratum_NTC': 4,
                'source_code_sha256': hash_file(Path(__file__)),
                'uv_lock_sha256': hash_file(Path(__file__).resolve().parents[1] / 'uv.lock')}
    if (output / 'identity.json').exists():
        previous = json.loads((output / 'identity.json').read_text())
        for key in ['source_code_sha256', 'uv_lock_sha256', 'source_report_sha256']:
            assert previous[key] == identity[key], 'cannot resume with changed estimands/code/input'
        identity = previous
    else:
        write_json(output / 'identity.json', identity)
    start = time.monotonic()
    contexts = CellContexts(SOURCE)
    results = []
    for context in contexts.iter_contexts():
        folder = output / context['context_id']
        if (folder / 'report.json').exists():
            report = json.loads((folder / 'report.json').read_text())
            assert all(hash_file(folder / name) == digest for name, digest in report['artifacts'].items())
        else:
            report = collect_context(context, folder)
        results.append(report)
        print(json.dumps(report, default=str), flush=True)
    assert sum(r['tasks'] for r in results) == 37845
    write_json(output / 'consumed-inputs.json', contexts.verify_unchanged())
    write_json(output / 'report.json', {'status': 'completed', 'identity': identity, 'contexts': results,
                                      'tasks': 37845, 'seconds': time.monotonic() - start})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.output.resolve())
