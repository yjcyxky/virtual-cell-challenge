"""One streamed source scan; fold-neutral observations with explicit supervision masks.

All-context summaries are data artifacts, not a fitted representation. A downstream
predictor must select its training contexts and target partition before reading them.
"""
import gc
import json

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from rna import RNAFile

from .common import ROOT, NTC, ref, verified, write_json, stable_seed
from .counts import describe, thin_counts


def selected_panels(context, tasks, splits, per_panel, seed):
    available = tasks.loc[tasks.context.eq(context) & tasks.structurally_eligible]
    permitted = set(available.target)
    panels = {}
    for split in splits:
        if split['scenario'] not in ('S2', 'S3', 'S4') or context not in split['evaluation_contexts']:
            continue
        candidates = set(split['evaluation_targets'][context]) & permitted
        panels[split['id']] = sorted(candidates, key=lambda t: stable_seed(seed, context, split['id'], t))[:per_panel]
    return panels


def prepare_context(output, config, context):
    directory = output / 'cache' / context
    directory.mkdir(parents=True, exist_ok=True)
    manifest = directory / 'observations.json'
    if manifest.exists():
        record = json.loads(manifest.read_text())
        for item in record['files']:
            verified(item)
        return record
    audit = json.loads(verified(config['benchmark']['data_audit']).read_text())
    sources = {r['path']: r for r in audit['inputs']}
    frame = pd.read_parquet(verified(config['identity_refs'][context])).reset_index(drop=True)
    if frame.physical_id.duplicated().any():
        raise ValueError('Physical cell identities are not unique')
    for pool in ('input', 'score'):
        if int(frame.ntc_pool.eq(pool).sum()) != audit['contexts'][context]['ntc_pools'][pool]['cells']:
            raise ValueError('Frozen NTC pool changed')
    mapping = pd.read_csv(verified(config['mapping_ref']))
    mapping = mapping.loc[mapping.context.eq(context)]
    axis = pd.read_csv(verified(config['benchmark']['gene_axis'])).gene_name.to_numpy()
    positions = np.asarray(audit['contexts'][context]['official_gene_positions'])
    genes = axis[positions]
    gene_index = {g: i for i, g in enumerate(genes)}
    tasks = pd.read_csv(verified(config['tasks_ref']))
    splits = json.loads(verified(config['benchmark']['split_manifest']).read_text())
    panels = selected_panels(context, tasks, splits, config['data']['panel_targets'], config['seed'])
    selected = set().union(*map(set, panels.values()))
    labels = [NTC] + sorted(set(frame.loc[frame.eligible & frame.target.ne(NTC), 'target'].dropna()))
    label_index = {t: i for i, t in enumerate(labels)}
    shape = (len(labels), len(genes))
    sums, cpms, squares, logs = [np.zeros(shape, dtype=np.float64) for _ in range(4)]
    counts = np.zeros(len(labels), dtype=np.int64)
    batches = sorted(frame.batch.unique())
    batch_index = {b: i for i, b in enumerate(batches)}
    ntc_sums = np.zeros((len(batches), len(genes)))
    ntc_sq = np.zeros_like(ntc_sums)
    ntc_n = np.zeros(len(batches), dtype=np.int64)
    for column in ('native_library', 'measured_library', 'metadata_umi', 'on_target_native_cpm'):
        frame[column] = np.nan
    frame['on_target_measured'] = frame.target.isin(gene_index)
    frame['bank'] = ''
    for pool in ('input', 'score'):
        ids = frame.index[frame.ntc_pool.eq(pool)].tolist()
        ids.sort(key=lambda i: stable_seed(config['seed'], context, pool, frame.at[i, 'physical_id']))
        frame.loc[ids[:config['data']['ntc_bank_cells']], 'bank'] = pool
    for target in selected:
        ids = frame.index[frame.eligible & frame.target.eq(target)].tolist()
        ids.sort(key=lambda i: stable_seed(config['seed'], context, target, frame.at[i, 'physical_id']))
        frame.loc[ids[:config['data']['reference_cells']], 'bank'] = 'perturbation'
    blocks, indices, used_sources = [], [], []
    for relative, members in frame.groupby('file', sort=False):
        raw_ref = {k: sources['data/raw/' + relative][k] for k in ('path', 'sha256')}
        print(f'{context}: verify and scan {relative}', flush=True)
        path = verified(raw_ref)
        used_sources.append(raw_ref)
        before = path.stat()
        feature_map = mapping.loc[mapping.file.eq(relative) & mapping.measured].sort_values('official_position')
        if not np.array_equal(feature_map.official_position, positions):
            raise ValueError('Context measurement mask changed')
        columns = feature_map.source_position.to_numpy()
        with RNAFile(path) as source:
            row_to_frame = np.full(source.shape[0], -1, dtype=np.int64)
            row_to_frame[members.source_row] = members.index
            if 'UMI_count' in source.obs:
                frame.loc[members.index, 'metadata_umi'] = pd.to_numeric(
                    source.obs.UMI_count.iloc[members.source_row], errors='coerce').to_numpy()
            for start, raw in source.blocks(config['data']['chunk_rows']):
                mapping_rows = row_to_frame[start:start + raw.shape[0]]
                keep = mapping_rows >= 0
                if not keep.any():
                    continue
                ids = mapping_rows[keep]
                rows = frame.loc[ids]
                raw = raw[keep]
                native = np.asarray(raw.sum(1)).ravel()
                measured = raw[:, columns].tocsr()
                depth = np.asarray(measured.sum(1)).ravel()
                if np.any(native <= 0) or np.any(depth <= 0):
                    raise ValueError('Audited nonempty library changed')
                frame.loc[ids, 'native_library'] = native
                frame.loc[ids, 'measured_library'] = depth
                target_columns = np.array([gene_index.get(t, -1) for t in rows.target])
                known = target_columns >= 0
                own = np.asarray(measured[np.flatnonzero(known), target_columns[known]]).ravel()
                frame.loc[ids[known], 'on_target_native_cpm'] = own * 1e6 / native[known]
                fit = (rows.eligible & rows.target.ne(NTC)).to_numpy() | rows.ntc_pool.eq('input').to_numpy()
                groups = np.asarray([label_index.get(t, -1) for t in rows.target])[fit]
                if np.any(groups < 0):
                    raise ValueError('Invalid statistical task label')
                present, inverse = np.unique(groups, return_inverse=True)
                selector = sparse.csr_matrix((np.ones(len(groups)), (inverse, np.arange(len(groups)))),
                                             shape=(len(present), len(groups)))
                x = measured[fit]
                normalized = (sparse.diags(1e6 / depth[fit]) @ x).tocsr()
                logarithm = normalized.copy()
                logarithm.data = np.log1p(logarithm.data / 100)
                sums[present] += (selector @ x).toarray()
                cpms[present] += (selector @ normalized).toarray()
                squares[present] += (selector @ normalized.power(2)).toarray()
                logs[present] += (selector @ logarithm).toarray()
                counts += np.bincount(groups, minlength=len(labels))
                ntc = rows.ntc_pool.eq('input').to_numpy()
                if ntc.any():
                    b = np.asarray([batch_index[t] for t in rows.batch[ntc]])
                    bs = sparse.csr_matrix((np.ones(len(b)), (b, np.arange(len(b)))), shape=(len(batches), len(b)))
                    xn = sparse.diags(1e6 / native[ntc]) @ measured[ntc]
                    ntc_sums += (bs @ xn).toarray()
                    ntc_sq += (bs @ xn.power(2)).toarray()
                    ntc_n += np.bincount(b, minlength=len(batches))
                take = rows.bank.ne('').to_numpy()
                if take.any():
                    blocks.append(measured[take].astype(np.int32))
                    indices.extend(ids[take].tolist())
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError('Raw input changed while scanning')
    if frame.native_library.isna().any() or np.any(counts <= 0):
        raise ValueError('Incomplete observation scan')
    fraction = min(1., config['data']['native_median_target'] / float(
        frame.loc[frame.ntc_pool.eq('input'), 'native_library'].median()))
    bank_obs = frame.loc[indices].copy().set_index('physical_id')
    bank = sparse.vstack(blocks, format='csr')
    del blocks
    bank_path = directory / 'bank-native.h5ad'
    var = pd.DataFrame({'official_position': positions}, index=genes)
    ad.AnnData(bank, obs=bank_obs, var=var).write_h5ad(bank_path, compression='lzf')
    aligned = thin_counts(bank, fraction, stable_seed(config['seed'], context, 'thinning'))
    view_path = directory / 'bank-depth-view.h5ad'
    ad.AnnData(aligned, obs=bank_obs, var=var).write_h5ad(view_path, compression='lzf')
    statistics = directory / 'statistics.npz'
    bulk = np.log1p(50_000 * sums / sums.sum(1, keepdims=True))
    mean = cpms / counts[:, None]
    variance = np.maximum(squares - counts[:, None] * mean**2, 0) / np.maximum(counts[:, None] - 1, 1)
    np.savez_compressed(statistics, labels=np.asarray(labels), positions=positions, counts=counts,
        bulk=bulk.astype(np.float32), mean_cpm=mean.astype(np.float32), variance_cpm=variance.astype(np.float32),
        mean_log_cp10k=(logs / counts[:, None]).astype(np.float32))
    del sums, cpms, squares, logs, bulk, mean, variance
    # Dose estimates use native measured-library CPM and batch-matched input NTC.
    rows = frame.loc[frame.eligible & frame.target.ne(NTC)]
    dose = rows.groupby(['target', 'guide', 'batch'], dropna=False).agg(
        cells=('physical_id', 'size'), target_mean_cpm=('on_target_native_cpm', 'mean'),
        target_variance_cpm=('on_target_native_cpm', 'var')).reset_index()
    ci, status, effects, controls, control_counts = [], [], [], [], []
    for row in dose.itertuples():
        g, b = gene_index.get(row.target), batch_index[row.batch]
        cn = int(ntc_n[b]); control_counts.append(cn)
        mu = float(ntc_sums[b, g] / cn) if g is not None and cn else None
        controls.append(mu)
        reason = ('unmeasured_target' if g is None else 'insufficient_cells' if cn < 20 or row.cells < 10
                  else 'low_control_expression' if mu <= 5 else 'available')
        if reason != 'available':
            effects.append(None); ci.append(None); status.append(reason); continue
        effect = 1 - row.target_mean_cpm / mu
        var0 = max((ntc_sq[b, g] - cn * mu**2) / (cn - 1), 0)
        se = np.sqrt(max(row.target_variance_cpm, 0) / row.cells / mu**2 +
                     row.target_mean_cpm**2 * var0 / cn / mu**4)
        effects.append(float(effect)); ci.append(float(1.96 * se))
        status.append('uncertain' if 1.96 * se > config['data']['dose_ci_halfwidth'] else 'estimable')
    dose['control_cells'] = control_counts
    dose['control_mean_native_cpm'] = controls
    dose['knockdown_fraction_unclipped'] = effects
    dose['approximate_95ci_halfwidth'] = ci
    dose['reliability'] = status
    dose_path = directory / 'knockdown-by-guide-batch.parquet'
    dose.to_parquet(dose_path, index=False)
    cells_path = directory / 'cells.parquet'
    frame.to_parquet(cells_path, index=False)
    quantiles = [0, .05, .25, .5, .75, .95, 1]
    distributions = {}
    for name, subset in [('all', frame), ('input_ntc', frame.loc[frame.ntc_pool.eq('input')]),
                         ('perturbation', rows)]:
        distributions[name] = {'cells': len(subset), 'quantile_levels': quantiles}
        for field in ('native_library', 'measured_library', 'metadata_umi'):
            values = subset[field].dropna()
            distributions[name][field] = np.quantile(values, quantiles).tolist() if len(values) else None
        distributions[name]['official_fraction_of_native'] = np.quantile(
            subset.measured_library / subset.native_library, quantiles).tolist()
    record = {'context': context, 'study': audit['contexts'][context]['study'],
        'biological_replicates': None, 'target_tasks': len(labels) - 1,
        'measured_genes': len(genes), 'official_gene_count': len(axis),
        'gene_axis_scope': 'per-source measured official subset; missing genes never supervised as zero',
        'native_depth_thinning_probability': fraction,
        'depth_view_scope': 'uniform native-count thinning probability estimated only from input NTC; no upsampling; measured-axis median is not a full-transcriptome depth claim',
        'statistics_scope': 'unthinned source observations; restrict contexts/targets before any fitting; input NTC only in control statistics',
        'panels': panels, 'distributions': distributions,
        'dose_reliability': dose.reliability.value_counts().to_dict(),
        'dose_limitations': 'delta-method intervals describe cell sampling only, not biological replicates; no response-based filtering; assay/dose/context remain confounded',
        'bank_native': describe(bank), 'bank_depth_view': describe(aligned),
        'identity_ref': config['identity_refs'][context], 'source_refs': used_sources,
        'files': [ref(p) for p in [statistics, cells_path, dose_path, bank_path, view_path]]}
    write_json(manifest, record)
    del bank, aligned, frame, dose
    gc.collect()
    print(f'{context}: complete observation audit; thinning probability={fraction:.4f}', flush=True)
    return record
