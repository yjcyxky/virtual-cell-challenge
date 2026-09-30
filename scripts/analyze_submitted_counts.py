"""Read-only, bounded-memory distribution diagnostics for published VCC predictions.

This supplements saved official receipts; it never calls a scorer, emits cells,
or treats public NTC as hidden perturbed truth. Outputs live beside each run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy import sparse, stats

ROOT = Path(__file__).resolve().parents[1]
AUDIT = 'counts-audit-20260930'
CONTROL = ROOT / 'data/raw/arc_vcc2026_controls'
OUT = ROOT / 'experiments/init-linear/outputs/init-linear-s01/cache' / AUDIT


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while block := f.read(16 << 20):
            h.update(block)
    return h.hexdigest()


def ref(path):
    return {'path': str(Path(path).relative_to(ROOT)), 'sha256': digest(path)}


def clean(x):
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, np.ndarray)):
        return [clean(v) for v in x]
    if isinstance(x, (np.integer, np.bool_)):
        return x.item()
    if isinstance(x, (float, np.floating)):
        return float(x) if np.isfinite(x) else None
    return x


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2) + '\n')


def strings(node):
    if isinstance(node, h5py.Group):
        if 'values' in node:
            if node['mask'][:].any():
                raise ValueError('missing string value')
            return strings(node['values'])
        categories = strings(node['categories'])
        codes = node['codes'][:]
        if (codes < 0).any():
            raise ValueError('missing label')
        return categories[codes]
    return node.asstr()[:]


def block(handle, start, stop):
    x = handle['X']
    if not isinstance(x, h5py.Group) or x.attrs['encoding-type'] != 'csr_matrix':
        raise ValueError('Expected CSR counts')
    ptr = x['indptr'][start:stop+1]
    lo, hi = int(ptr[0]), int(ptr[-1])
    result = sparse.csr_matrix((x['data'][lo:hi].astype(np.float64),
                               x['indices'][lo:hi], ptr-ptr[0]),
                              shape=(stop-start, int(x.attrs['shape'][1])))
    result.sum_duplicates()
    result.sort_indices()
    result.eliminate_zeros()
    return result


def moments(x):
    n, g = x.shape
    mean = np.bincount(x.indices, weights=x.data, minlength=g) / n
    second = np.bincount(x.indices, weights=x.data*x.data, minlength=g) / n
    variance = np.maximum(second-mean*mean, 0) * n / (n-1)
    return mean, variance


def normalized(x):
    depth = np.asarray(x.sum(axis=1)).ravel()
    y = x.copy()
    y.data *= np.repeat(np.divide(1e4, depth, out=np.zeros_like(depth), where=depth>0), np.diff(x.indptr))
    return y


def quantiles(x, prefix):
    x = np.asarray(x)
    return {prefix + '_' + k: float(v) for k, v in zip(
        ['q01', 'q05', 'q50', 'q95', 'q99'], np.quantile(x, [.01, .05, .5, .95, .99]))}


def corr(a, b):
    a, b = np.asarray(a), np.asarray(b)
    keep = np.isfinite(a) & np.isfinite(b)
    a, b = a[keep], b[keep]
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float('nan')
    return float(np.corrcoef(a, b)[0, 1])


def row_hashes(x):
    result = []
    for i in range(x.shape[0]):
        lo, hi = x.indptr[i:i+2]
        h = hashlib.blake2b(digest_size=16)
        h.update(x.indices[lo:hi].astype('<i4').tobytes())
        h.update(x.data[lo:hi].astype('<f8').tobytes())
        result.append(h.digest())
    return result


def describe(x, reference=None, gene_positions=None, ntc_hashes=None):
    depth = np.asarray(x.sum(axis=1)).ravel()
    detected = np.diff(x.indptr)
    mean, var = moments(x)
    y = normalized(x)
    cpmean, cpvar = moments(y)
    y.data = np.log1p(y.data)
    logmean, logvar = moments(y)
    hashes = row_hashes(x)
    summary = {'n': x.shape[0], 'nnz': x.nnz, 'zero_fraction': 1-x.nnz/np.prod(x.shape),
               'negative_entries': int((x.data<0).sum()), 'nonfinite_entries': int((~np.isfinite(x.data)).sum()),
               'fractional_entries': int((x.data!=np.rint(x.data)).sum()),
               'empty_cells': int((depth==0).sum()), 'depth_above_1e6': int((depth>1e6).sum()),
               'depth_mean': depth.mean(), 'depth_min': depth.min(), 'depth_max': depth.max(),
               'depth_cv': depth.std(ddof=1)/depth.mean(), 'detected_mean': detected.mean(),
               'depth_detected_spearman': stats.spearmanr(depth, detected).statistic,
               'duplicate_fraction': 1-len(set(hashes))/len(hashes),
               **quantiles(depth, 'depth'), **quantiles(detected, 'detected')}
    expressed = mean >= .1
    summary['raw_fano_median_mean_ge_0p1'] = np.median(var[expressed]/mean[expressed])
    summary['raw_fano_below_one_fraction'] = np.mean(var[expressed]<mean[expressed])
    if reference is not None:
        valid = (reference['cpmean'] >= .1) & (reference['cpvar'] > 1e-8)
        lvalid = reference['logvar'] > 1e-8
        ratio = cpvar[valid]/reference['cpvar'][valid]
        summary.update({'cp_variance_ratio_median': np.median(ratio),
                        'cp_variance_ratio_below_0p1': np.mean(ratio<.1),
                        'cp_variance_ratio_above_10': np.mean(ratio>10),
                        'cp_variance_ratio_genes': int(valid.sum()),
                        'log_variance_ratio_median': np.median(logvar[lvalid]/reference['logvar'][lvalid]),
                        'log_response_rms': np.sqrt(np.mean((logmean-reference['logmean'])**2)),
                        'bulk_lfc_rms': np.sqrt(np.mean(np.log2((cpmean+.01)/(reference['cpmean']+.01))**2)),
                        'count_mean_pearson_ntc': corr(mean, reference['mean']),
                        'depth_wasserstein_over_ntc_mean': stats.wasserstein_distance(depth, reference['depth'])/np.mean(reference['depth']),
                        'detected_shift': detected.mean()-reference['detected'].mean(),
                        'ntc_zero_to_pred_detected_genes': int(((reference['mean']==0)&(mean>0)).sum()),
                        'ntc_cp_ge_0p1_to_all_zero_genes': int(((reference['cpmean']>=.1)&(mean==0)).sum()),
                        'exact_ntc_cell_fraction': np.mean([h in ntc_hashes for h in hashes])})
        z = y[:, gene_positions].toarray()
        cov = np.cov(z, rowvar=False)
        triangle = np.triu_indices(len(gene_positions), 1)
        summary['hvg128_covariance_pearson_ntc'] = corr(cov[triangle], reference['cov'][triangle])
        summary['hvg128_covariance_relative_error'] = np.linalg.norm(cov-reference['cov'])/np.linalg.norm(reference['cov'])
    return summary, dict(mean=mean, var=var, cpmean=cpmean, cpvar=cpvar,
                         logmean=logmean, logvar=logvar, depth=depth, detected=detected)


def geometry(delta):
    centroid = delta.mean(axis=0)
    energy = np.sum(delta*delta)
    residual = delta-centroid
    eig = np.maximum(np.linalg.eigvalsh(residual @ residual.T), 0)
    norm = np.linalg.norm(delta, axis=1)
    cos = (delta @ delta.T)/np.maximum(norm[:, None]*norm[None, :], 1e-30)
    return {'targets': len(delta), 'genes': delta.shape[1],
            'shared_centroid_energy_fraction': len(delta)*np.sum(centroid**2)/energy if energy else None,
            'centered_participation_rank': eig.sum()**2/np.sum(eig**2) if eig.sum()>0 else None,
            'centered_top_pc_energy_fraction': eig[-1]/eig.sum() if eig.sum()>0 else None,
            'pairwise_cosine_median': np.median(cos[np.triu_indices(len(delta), 1)]),
            'rms': np.sqrt(np.mean(delta**2)), 'centroid_rms': np.sqrt(np.mean(centroid**2)),
            'centered_rms': np.sqrt(np.mean(residual**2))}


def controls(genes, targets):
    reference, summaries, nulls = {}, [], []
    source = json.loads((CONTROL/'SOURCE.json').read_text())
    expected = {v['name']:v['sha256'] for v in source['files']}
    exclude = ~np.isin(genes, targets)
    for ci, c in enumerate('ABC'):
        p = CONTROL/f'context_{c}.h5ad'
        if digest(p) != expected[p.name]:
            raise ValueError('Changed official controls')
        with h5py.File(p) as h:
            if not np.array_equal(strings(h['var'][h['var'].attrs['_index']]), genes):
                raise ValueError('control gene axis mismatch')
            x = block(h, 0, int(h['X'].attrs['shape'][0]))
        summary, r = describe(x)
        hvg = np.argsort(np.where(exclude, r['logvar'], -1))[-128:]
        z = normalized(x[:, :])[:, hvg].toarray()
        r['cov'] = np.cov(np.log1p(z), rowvar=False)
        r['hvg'] = hvg
        r['hashes'] = set(row_hashes(x))
        rng = np.random.default_rng(20260930+ci)
        for i in range(30):
            ids = rng.choice(x.shape[0], 800, replace=False)
            a, am = describe(x[ids[:400]], r, hvg, r['hashes'])
            _, bm = describe(x[ids[400:]])
            a.update(context=c, repeat=i,
                     independent_split_log_rms=np.sqrt(np.mean((am['logmean']-bm['logmean'])**2)))
            nulls.append(a)
        summary['context'] = c
        summaries.append(summary)
        reference[c] = r
        print('NTC complete', c, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summaries).to_csv(OUT/'ntc-summary.csv', index=False)
    pd.DataFrame(nulls).to_csv(OUT/'ntc-resampling.csv', index=False)
    return reference


def analyze(path, genes, targets, references, selected=None):
    receipt = json.loads(path.read_text())
    if receipt.get('status') != 'published' or receipt.get('score_avg') is None:
        return None
    if selected and receipt['entry_id'] not in selected:
        return None
    directory = path.parent
    output = directory/AUDIT
    if (output/'summary.json').exists():
        return json.loads((output/'summary.json').read_text())
    print('VERIFY', receipt['model_name'], flush=True)
    generation = json.loads((directory/'generation.json').read_text())
    packing = json.loads((directory/'submission-file.json').read_text())
    prediction = ref(directory/'predictions.h5ad')
    packed = ref(directory/'predictions.vcc')
    if prediction['sha256'] != generation.get('sha256', generation.get('file_ref', {}).get('sha256')):
        raise ValueError('prediction does not match recorded export')
    if packed['sha256'] != packing.get('sha256', packing.get('file_ref', {}).get('sha256')):
        raise ValueError('submission archive does not match recorded export')
    rows, geometries, stored = [], [], {}
    seen = set(generation.get('seen_targets', []))
    unseen = set(generation.get('unseen_targets', []))
    feature_mask = ~np.isin(genes, targets)
    with h5py.File(directory/'predictions.h5ad') as h:
        axis = strings(h['var'][h['var'].attrs['_index']])
        context, target = strings(h['obs']['context']), strings(h['obs']['target_gene'])
        if not np.array_equal(axis, genes) or set(context) != set('ABC') or set(target) != set(targets):
            raise ValueError('official axis or labels mismatch')
        for c in 'ABC':
            cpmeans, logmeans = [], []
            r = references[c]
            for ti, t in enumerate(targets):
                ids = np.flatnonzero((context==c)&(target==t))
                if len(ids) != 400 or not np.all(np.diff(ids)==1):
                    raise ValueError('Expected 400 contiguous rows per submitted group')
                x = block(h, int(ids[0]), int(ids[-1])+1)
                s, m = describe(x, r, r['hvg'], r['hashes'])
                j = int(np.flatnonzero(genes==t)[0])
                s.update(context=c, target=t, model=receipt['model_name'], entry_id=receipt['entry_id'],
                         target_support='seen' if t in seen else 'unseen' if t in unseen else 'unknown',
                         target_ntc_cpm=r['cpmean'][j]*100,
                         target_pred_cpm=m['cpmean'][j]*100,
                         target_ratio=m['cpmean'][j]/r['cpmean'][j] if r['cpmean'][j]>0 else None,
                         target_lfc=np.log2((m['cpmean'][j]+.01)/(r['cpmean'][j]+.01)),
                         target_detection_pred=float(np.mean(x[:, j].toarray()>0)))
                rows.append(s)
                cpmeans.append(m['cpmean'].astype(np.float32))
                logmeans.append(m['logmean'].astype(np.float32))
                if (ti+1)%100==0:
                    print(receipt['model_name'], c, ti+1, flush=True)
            cpmeans, logmeans = np.array(cpmeans), np.array(logmeans)
            stored[c+'_cpmean'] = cpmeans
            stored[c+'_logmean'] = logmeans
            stored[c+'_ntc_cpmean'] = r['cpmean']
            stored[c+'_ntc_logmean'] = r['logmean']
            for support, mask in [('all', np.ones(len(targets), bool)), ('seen', np.isin(targets, list(seen))), ('unseen', np.isin(targets, list(unseen)))]:
                if mask.sum()<3:
                    continue
                d = logmeans[mask][:, feature_mask]-r['logmean'][feature_mask]
                geometries.append(dict(context=c, support=support, space='mean_log1p_CP10k', **geometry(d)))
                lfc = np.log2((cpmeans[mask][:, feature_mask]+.01)/(r['cpmean'][feature_mask]+.01))
                geometries.append(dict(context=c, support=support, space='bulk_log2FC_CP10k_pc0p01', **geometry(lfc)))
        stored_nnz = len(h['X']['data'])
    output.mkdir(exist_ok=True)
    table = pd.DataFrame(rows)
    table.to_csv(output/'per-target.csv', index=False)
    pd.DataFrame(geometries).to_csv(output/'response-geometry.csv', index=False)
    np.savez_compressed(output/'moments.npz', genes=genes.astype(str), targets=targets.astype(str), **stored)
    aggregate = []
    for c, group in table.groupby('context'):
        a = {'context': c, 'groups': len(group), 'cells': int(group.n.sum())}
        for col in table.select_dtypes(include='number').columns:
            a[col+'_median'] = group[col].median()
        valid = group.target_ntc_cpm >= 5
        a.update(target_kd_eligible=int(valid.sum()), target_ratio_median_eligible=group.loc[valid, 'target_ratio'].median(),
                 target_down_fraction_eligible=(group.loc[valid, 'target_ratio']<1).mean(),
                 target_half_fraction_eligible=(group.loc[valid, 'target_ratio']<=.5).mean(),
                 target_up_1p2_fraction_eligible=(group.loc[valid, 'target_ratio']>1.2).mean())
        aggregate.append(a)
    result = {'model': receipt['model_name'], 'entry_id': receipt['entry_id'], 'official': receipt,
              'prediction_ref': prediction, 'archive_ref': packed, 'receipt_ref': ref(path),
              'generation_ref': ref(directory/'generation.json'), 'prep_ref': ref(directory/'prep.json'),
              'diagnostic_code_ref': ref(Path(__file__)), 'python': platform.python_version(),
              'full_matrix_checked': True, 'rows': int(table.n.sum()), 'genes': len(genes),
              'stored_nnz': stored_nnz, 'stored_nnz_within_limit': stored_nnz<=4750000000,
              'invalid_entries': int(table[['negative_entries','fractional_entries','nonfinite_entries']].to_numpy().sum()),
              'empty_or_overlimit_cells': int(table[['empty_cells','depth_above_1e6']].to_numpy().sum()),
              'contexts': aggregate, 'geometry': geometries,
              'files': [ref(output/p) for p in ['per-target.csv','response-geometry.csv','moments.npz']],
              'limitations': 'NTC is not hidden perturbed truth. No new official scores. Historical heterogeneous packages are descriptive, not causal ablations. Rank is not noise-corrected; covariances use 128 NTC-selected genes.'}
    write(output/'summary.json', result)
    print('COMPLETE', receipt['model_name'], flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--entry', action='append')
    args = parser.parse_args()
    genes = pd.read_csv(CONTROL/'gene_names.csv').iloc[:, 0].astype(str).to_numpy()
    targets = pd.read_csv(CONTROL/'pert_counts.csv').iloc[:, 0].astype(str).to_numpy()
    references = controls(genes, targets)
    results = []
    for path in sorted((ROOT/'experiments').glob('*/outputs/**/official-status.json')):
        r = analyze(path, genes, targets, references, args.entry)
        if r is not None:
            results.append(r)
    write(OUT/'results.json', {'models': results, 'controls': [ref(OUT/p) for p in ['ntc-summary.csv','ntc-resampling.csv']],
                              'analysis_kind': 'retrospective_fixed_predictions', 'seed': 20260930})


if __name__ == '__main__':
    main()
