"""Full-axis diagnostics for a frozen, not-yet-submitted official prediction."""
from pathlib import Path
import json

import h5py
import numpy as np
import pandas as pd
from scipy import sparse
from analyze_submitted_counts import clean, describe, geometry, row_hashes

from .common import ref, verified, write_json, stable_seed
from .counts import profile, validate_counts
from .official import annotated, de_table, de_summary
from .population_emission import fit_population, draw_population
from .prediction_diagnostics import diagnose_panel
from .response_data import decode_composition


def null_diagnostics(ntc, directory, targets, config, positions):
    """Disjoint public NTC identity tasks; no hidden perturbed labels are used."""
    directory = Path(directory)
    marker = directory / 'null.json'
    if marker.exists():
        result = json.loads(marker.read_text())
        for item in result['files']:
            verified(item)
        return result
    x = sparse.csr_matrix(ntc.X)
    n = config['prediction_cells']
    seed = config['seed']
    context = str(ntc.obs.context.iloc[0])
    order = np.random.default_rng(stable_seed(seed, context, 'null-partition')).permutation(len(ntc))
    first, second = len(order) // 2, 3 * len(order) // 4
    source, reference, pool = x[order[:first]], x[order[first:second]], x[order[second:]]
    if min(source.shape[0], reference.shape[0], pool.shape[0]) < n:
        raise ValueError('Insufficient disjoint NTC support')
    baseline = profile(source)
    desired, _ = decode_composition(baseline, np.zeros(ntc.n_vars), own=-1, **config['decoder'])
    if not np.array_equal(desired, baseline):
        raise ValueError('Zero decoder changed the NTC baseline')
    fitted = fit_population(source, desired-baseline, **config['emitter'])
    _, moments = describe(reference.astype(np.float64))
    z = reference[:, positions].toarray().astype(float)
    z = np.log1p(z * (10_000/moments['depth'])[:, None])
    moments['cov'] = np.cov(z, rowvar=False)
    hashes = set(row_hashes(reference))
    blocks, labels, summaries = [], [], []
    for repeat in range(config['diagnostics']['null_repeats']):
        rng_seed = stable_seed(seed, context, 'official-null', repeat)
        generated = draw_population(source, fitted, rng_seed, n)
        original = source[np.random.default_rng(rng_seed).integers(0, source.shape[0], n)]
        if (generated != original).nnz:
            raise ValueError('Actual emitter does not preserve the zero-response identity')
        real_null = pool[np.random.default_rng(rng_seed).choice(pool.shape[0], n, replace=False)]
        for name, matrix in [('zero', generated), ('input', original), ('real_null', real_null)]:
            validate_counts(matrix)
            blocks.append(matrix); labels.extend([f'{name}_{repeat}'] * n)
            row, _ = describe(matrix.astype(float), moments, positions, hashes)
            summaries.append(clean(dict(row, kind=name, repeat=repeat)))
    matrix = sparse.vstack(blocks, format='csr')
    annotated(matrix, ntc.var_names, labels, context+'-null').write_h5ad(directory/'null-counts.h5ad', compression='gzip')
    table, backend = de_table(matrix, reference, ntc.var_names, labels, config['runtime']['num_threads'])
    table.write_parquet(directory/'null-de.parquet')
    de = de_summary(table)
    added = [de[f'zero_{r}']['significant_genes']-de[f'input_{r}']['significant_genes']
             for r in range(config['diagnostics']['null_repeats'])]
    if any(added):
        raise ValueError('Zero generator adds differential expression to matched resampling')
    pd.DataFrame(summaries).to_parquet(directory/'null-distributions.parquet', index=False)
    # Each comparison consists of two nonintersecting groups of the submitted n.
    sampling = []
    for repeat in range(config['diagnostics']['ntc_sampling_repeats']):
        ids = np.random.default_rng(stable_seed(seed, context, 'same-n', repeat)).choice(len(ntc), 2*n, replace=False)
        a, b = x[ids[:n]].astype(float), x[ids[n:]].astype(float)
        _, bm = describe(b)
        z = b[:, positions].toarray()
        bm['cov'] = np.cov(np.log1p(z * (10_000/bm['depth'])[:, None]), rowvar=False)
        row, am = describe(a, bm, positions, set(row_hashes(b)))
        row.update(repeat=repeat, independent_mean_log_rms=float(np.sqrt(np.mean((am['logmean']-bm['logmean'])**2))))
        sampling.append(clean(row))
    pd.DataFrame(sampling).to_parquet(directory/'same-n-ntc.parquet', index=False)
    result = {'status':'completed', 'zero_decoder_identity':True, 'zero_count_identity':True,
              'added_zero_de':added, 'de_summary':de, 'backend':backend,
              'partition':{'source':source.shape[0],
                           'reference':reference.shape[0], 'real_null_pool':pool.shape[0]},
              'scope':'Zero identity uses source half; DE reference and real-null pool are two disjoint quarters. Official prediction uses all public NTC. Same-n pairs never overlap within a pair.',
              'biological_replicates_added':0,
              'files':[ref(directory/name) for name in ('null-counts.h5ad','null-de.parquet','null-distributions.parquet','same-n-ntc.parquet')]}
    write_json(marker, result)
    return result


def diagnose_official(prediction, ntc, directory, targets, config, checkpoint_ref, coverage, global_supervision):
    directory = Path(directory)
    basic = diagnose_panel(prediction, ntc, directory, seed=config['seed'], checkpoint_ref=checkpoint_ref,
                           expected_targets=targets, expected_cells=config['prediction_cells'],
                           source_coverage=coverage, covariance_genes=config['diagnostics']['covariance_genes'])
    marker = directory/'completed.json'
    if marker.exists():
        result = json.loads(marker.read_text())
        for item in result['files'] + result['null']['files']:
            verified(item)
        return result
    null = null_diagnostics(ntc, directory, targets, config, basic['covariance_gene_positions'])
    with h5py.File(prediction) as h:
        detected = np.diff(h['X/indptr'][:])
    rows = pd.read_parquet(directory/'per-target.parquet')
    if rows.target.tolist() != sorted(targets):
        raise ValueError('Unexpected diagnostic target ordering')
    rows['detected_cv'] = [float(v.std(ddof=1)/v.mean())
                           for v in detected.reshape(len(targets), config['prediction_cells'])]
    rows[['target','detected_cv']].to_parquet(directory/'detected-cv.parquet', index=False)
    # The basic marker includes a hash of the table; keep the original intact.
    # Store supplements separately so its preexisting identity remains reusable.
    allowed = ~np.isin(ntc.var_names, targets)
    geometries = {}
    strata = []
    with h5py.File(directory/'gene-moments.h5') as src, h5py.File(directory/'derived-moments.h5','w') as out:
        src.copy('genes', out); src.copy('targets', out)
        mean, variance = src['mean'][:], src['var'][:]
        arrays = {
            'raw_fano':np.divide(variance, mean, out=np.full_like(mean,np.nan), where=mean>0),
            'mean_log_response':src['logmean'][:]-src['ntc_logmean'][:],
            'log2_mean_CP10k_ratio_pc0p01':np.log2((src['cpmean'][:]+.01)/(src['ntc_cpmean'][:]+.01)),
        }
        for name, value in arrays.items():
            out.create_dataset(name, data=value, compression='gzip', compression_opts=1)
            if name != 'raw_fano':
                geometries[name] = clean(geometry(value[:, allowed].astype(float)))
        for name, mask in [('source_supervised',np.asarray(global_supervision,dtype=bool)),
                           ('source_unobserved',~np.asarray(global_supervision,dtype=bool))]:
            mask = mask & allowed
            for i, target in enumerate(rows.target):
                values = arrays['mean_log_response'][i,mask]
                strata.append({'target':target,'stratum':name,'genes':int(mask.sum()),
                               'mean_log_response_rms':float(np.sqrt(np.mean(values**2))) if values.size else None})
        out.attrs['normalization'] = 'Per-cell full-axis CP10k; pseudo-count 0.01 CP10k = 1 CPM. Bulk log1p CP50K is separate.'
        out.attrs['undefined_fano'] = 'NaN where raw gene mean equals zero'
    pd.DataFrame(strata).to_parquet(directory/'readout-strata.parquet',index=False)
    result = {'status':'completed', 'hard_constraints_passed':True, 'checkpoint_ref':checkpoint_ref,
              'prediction_ref':basic['prediction_ref'], 'geometry':geometries, 'null':null,
              'supervision':{'seen_targets':int(rows.seen_target.sum()),
                             'unseen_targets':int((~rows.seen_target).sum()),
                             'missing_prior_template_targets':int(rows.missing_prior_template_fallback.sum()),
                             'zero_latent_targets':int(rows.zero_learned_response.sum())},
              'limitations':['Hidden perturbed truth unavailable; prediction-vs-NTC DE is not false DE.',
                             'Fixed own-target residual and inherited NTC morphology are not learned capabilities.'],
              'files':[ref(directory/name) for name in ('diagnostics.json','per-target.parquet','detected-cv.parquet','derived-moments.h5','readout-strata.parquet','null.json')]}
    write_json(marker, clean(result))
    return result
