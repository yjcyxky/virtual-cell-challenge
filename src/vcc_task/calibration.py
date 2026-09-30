"""Calibrate emission against independent controls and feasible oracle bulk effects."""
import json
import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from .common import NTC, ref, verified, write_json, stable_seed
from .counts import emit_counts, profile, describe
from .official import de_table, de_summary


def calibrate(output, config, context, observation):
    directory = output / 'predictions' / context
    directory.mkdir(parents=True, exist_ok=True)
    marker = directory / 'calibration.json'
    if marker.exists():
        record = json.loads(marker.read_text())
        for item in record['files']:
            verified(item)
        return record
    source = output / 'cache' / context / 'bank-depth-view.h5ad'
    bank = ad.read_h5ad(source)
    genes = bank.var_names.to_numpy()
    ntc = bank.X[bank.obs.bank.eq('input').to_numpy()]
    scoring = bank.X[bank.obs.bank.eq('score').to_numpy()]
    cfg = config['calibration']
    n = cfg['cells']
    if scoring.shape[0] < 2 * n:
        raise ValueError('Independent control calibration is underpowered for the frozen cell count')
    split = scoring.shape[0] // 2
    reference, null_pool = scoring[:split], scoring[split:]
    blocks, labels, records = [], [], []
    for i in range(cfg['null_repeats']):
        seed = stable_seed(config['seed'], context, 'null', i)
        rng = np.random.default_rng(seed)
        null = null_pool[rng.choice(null_pool.shape[0], n, replace=False)]
        base = ntc[np.random.default_rng(seed).integers(0, ntc.shape[0], n)]
        generated, details = emit_counts(ntc, np.zeros(len(genes)), seed, cells=n, **config['emitter'])
        exact = (base != generated).nnz == 0
        for name, matrix in [('real_null', null), ('input_resample', base), ('zero_emitter', generated)]:
            label = f'{name}_{i}'
            blocks.append(matrix); labels.extend([label] * n)
            records.append({'condition': label, 'kind': name, 'repeat': i,
                'bulk_rms_vs_control': float(np.sqrt(np.mean((profile(matrix) - profile(reference))**2))),
                'zero_identity': exact if name == 'zero_emitter' else None,
                'distribution': describe(matrix)})
    all_targets = sorted(set(bank.obs.loc[bank.obs.bank.eq('perturbation'), 'target']))
    targets = sorted(all_targets, key=lambda t: stable_seed(config['seed'], context, 'known', t))[:cfg['known_targets']]
    native_profile = profile(ntc)
    for target in targets:
        real = bank.X[bank.obs.target.eq(target).to_numpy()]
        desired = profile(real)
        for rule in ('independent', 'conservative'):
            spec = dict(config['emitter'], rounding=rule)
            seed = stable_seed(config['seed'], context, 'known', target)
            generated, details = emit_counts(ntc, desired - native_profile, seed, cells=n, **spec)
            label = f'{target}__{rule}'
            blocks.append(generated); labels.extend([label] * n)
            records.append({'condition': label, 'kind': 'known_' + rule, 'target': target,
                            'real_cells': real.shape[0], 'generation': details,
                            'distribution': describe(generated)})
        label = f'{target}__real'
        blocks.append(real); labels.extend([label] * real.shape[0])
    matrix = sparse.vstack(blocks, format='csr')
    table, backend = de_table(matrix, reference, genes, labels, config['runtime']['threads'])
    table_path = directory / 'official-de.parquet'
    table.write_parquet(table_path)
    summary = de_summary(table)
    for item in records:
        item['de'] = summary[item['condition']]
    frame = table.to_pandas()
    known = []
    for target in targets:
        truth = frame.loc[frame.target.eq(f'{target}__real') & frame.feature.ne(target)].set_index('feature')
        for rule in ('independent', 'conservative'):
            pred = frame.loc[frame.target.eq(f'{target}__{rule}') & frame.feature.ne(target)].set_index('feature')
            joined = truth[['p_adj', 'log2_fold_change']].join(pred[['p_adj', 'log2_fold_change']],
                                                            lsuffix='_real', rsuffix='_pred', how='inner')
            a, b = joined.p_adj_real.lt(.05), joined.p_adj_pred.lt(.05)
            union = int((a | b).sum())
            known.append({'target': target, 'rounding': rule,
                'real_significant': int(a.sum()), 'pred_significant': int(b.sum()),
                'overlap': int((a & b).sum()), 'set_jaccard_diagnostic': float((a & b).sum() / union) if union else 1.,
                'sign_correct_on_real_significant': float((np.sign(joined.loc[a, 'log2_fold_change_real']) ==
                    np.sign(joined.loc[a, 'log2_fold_change_pred'])).mean()) if a.any() else None})
    known_path = directory / 'known-effect-de.csv'
    pd.DataFrame(known).to_csv(known_path, index=False)
    prediction_path = directory / 'calibration-counts.h5ad'
    ad.AnnData(matrix, obs=pd.DataFrame({'target_gene': labels}, index=[f'cal-{i}' for i in range(len(labels))]),
               var=bank.var.copy()).write_h5ad(prediction_path, compression='lzf')
    null_excess = [summary[f'zero_emitter_{i}']['significant_genes'] -
                   summary[f'input_resample_{i}']['significant_genes'] for i in range(cfg['null_repeats'])]
    conservation = [r['generation']['row_totals_preserved'] for r in records if r['kind'] == 'known_conservative']
    errors = [r['generation']['bulk_realization_rms'] for r in records if r['kind'] == 'known_conservative']
    checks = {'zero_identity': all(r['zero_identity'] for r in records if r['kind'] == 'zero_emitter'),
              'no_additional_null_de': max(null_excess) <= cfg['max_added_null_de'],
              'integer_row_conservation': bool(conservation) and all(conservation),
              'known_bulk_reconstruction': bool(errors) and max(errors) <= cfg['max_known_bulk_rms']}
    result = {'context': context, 'kind': 'generator_diagnostic_not_predictive_model',
        'official_de_backend': backend, 'checks': checks, 'acceptance_fraction': sum(checks.values()) / len(checks),
        'records': records, 'known_effects': known, 'null_added_significant_genes': null_excess,
        'real_null_discoveries': [summary[f'real_null_{i}']['significant_genes'] for i in range(cfg['null_repeats'])],
        'input_resample_discoveries': [summary[f'input_resample_{i}']['significant_genes'] for i in range(cfg['null_repeats'])],
        'limitations': 'Oracle bulk effects only diagnose emission; DE overlap is descriptive and not an official score or a biological accuracy claim. Exact zero identity is necessary, not sufficient for nonzero effect distributions.',
        'source_ref': ref(source), 'files': [ref(p) for p in [table_path, known_path, prediction_path]]}
    write_json(marker, result)
    print(f'{context}: completed count/official-DE calibration {checks}', flush=True)
    return result
