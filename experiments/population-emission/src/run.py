"""Full fixed-response comparison of bag IPF and independent population draws."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts'), str(ROOT / 'scripts/dossier')]

import anndata as ad
import numpy as np
from scipy import sparse
from analyze_submitted_counts import describe, clean, row_hashes
from vcc_task.common import ref, verified, write_json, stable_seed
from vcc_task.counts import emit_counts, profile, validate_counts
from vcc_task.official import annotated, de_table, de_summary
from vcc_task.population_emission import fit_population, draw_population, state_record
from vcc_task.run_context import RegisteredRun
from research import bind


def context_batch(job, context):
    cfg = job.config
    directory = job.output / 'predictions' / context
    marker = directory / 'result.json'
    if marker.exists():
        record = json.loads(marker.read_text())
        for r in record['files']:
            verified(r)
        return record
    directory.mkdir(parents=True, exist_ok=True)
    bank = ad.read_h5ad(verified(cfg['bank_refs'][context]))
    old = json.loads(verified(cfg['calibration_refs'][context]).read_text())
    targets = sorted({r['target'] for r in old['known_effects']})
    ntc = bank.X[bank.obs.bank.eq('input').to_numpy()].tocsr()
    score = bank.X[bank.obs.bank.eq('score').to_numpy()].tocsr()
    reference, null_pool = score[:score.shape[0]//2], score[score.shape[0]//2:]
    _, ntc_stats = describe(ntc.astype(np.float64))
    ntc_hashes = set(row_hashes(ntc))
    allowed = ~np.isin(bank.var_names, targets)
    hvg = np.argsort(np.where(allowed, ntc_stats['logvar'], -1))[-cfg['diagnostics']['covariance_genes']:]
    z = ntc[:, hvg].toarray().astype(float)
    z = np.log1p(z * (10_000 / np.asarray(ntc.sum(1)).ravel())[:, None])
    ntc_stats['cov'] = np.cov(z, rowvar=False)
    records, summaries, blocks, labels, de_blocks, de_labels = [], [], [], [], [], []
    fitted_records = {}
    for target in targets:
        real = bank.X[bank.obs.target.eq(target).to_numpy()]
        desired = profile(real)
        delta = desired - profile(ntc)
        fitted = fit_population(ntc, delta, **cfg['population'])
        np.savez_compressed(directory / f'{target}-population.npz', **fitted)
        fitted_records[target] = state_record(fitted)
        for arm in cfg['arms']:
            cpmeans, cpvars, bulks, count_sums, libraries = [], [], [], [], []
            for repeat in range(cfg['repeats']):
                seed = stable_seed(cfg['seed'], context, target, repeat)
                if arm == 'population':
                    matrix = draw_population(ntc, fitted, seed, cfg['cells'])
                else:
                    matrix, _ = emit_counts(ntc, delta, seed, cells=cfg['cells'], **cfg['bag'])
                depth = validate_counts(matrix)
                rng = np.random.default_rng(seed)
                original_depth = np.asarray(ntc[rng.integers(0, ntc.shape[0], cfg['cells'])].sum(1)).ravel()
                if not np.array_equal(depth, original_depth):
                    raise ValueError('Generated row library changed')
                distribution, stats = describe(matrix.astype(np.float64), ntc_stats, hvg, ntc_hashes)
                label = f'{target}__{arm}_{repeat}'
                records.append(clean(dict(distribution, target=target, arm=arm, repeat=repeat, seed=seed,
                    response_bulk_rms=float(np.sqrt(np.mean((profile(matrix)-desired)**2))))))
                cpmeans.append(stats['cpmean']); cpvars.append(stats['cpvar'])
                bulks.append(profile(matrix)); count_sums.append(np.asarray(matrix.sum(0)).ravel())
                libraries.append(depth.sum())
                blocks.append(matrix); labels.extend([label] * matrix.shape[0])
                if repeat < cfg['de_repeats']:
                    de_blocks.append(matrix); de_labels.extend([label] * matrix.shape[0])
                    rng = np.random.default_rng(stable_seed(seed, 'nmatch'))
                    small = matrix[rng.choice(matrix.shape[0], real.shape[0], replace=False)]
                    de_blocks.append(small); de_labels.extend([label+'__nmatched'] * small.shape[0])
            empirical = np.var(cpmeans, axis=0, ddof=1)
            expected = np.mean(cpvars, axis=0) / cfg['cells']
            valid = allowed & (ntc_stats['cpmean'] >= .05) & (expected > 1e-10)
            ratio = float(empirical[valid].sum() / expected[valid].sum())
            pooled = np.sum(count_sums, axis=0)
            pooled_bulk = np.log1p(50_000 * pooled / sum(libraries))
            error = float(np.sqrt(np.mean((pooled_bulk-desired)**2)))
            low, high = cfg['acceptance']['variance_ratio']
            checks = {'row_conservation': True,
                      'mean_response': error <= cfg['acceptance']['pooled_bulk_rms'],
                      'sampling_variance': low <= ratio <= high}
            if arm == 'population':
                checks['population_expectation'] = fitted['population_bulk_rms'] <= cfg['acceptance']['population_bulk_rms']
            summaries.append({'context': context, 'target': target, 'arm': arm, 'real_cells': real.shape[0],
                'variance_ratio': ratio, 'variance_genes': int(valid.sum()), 'pooled_bulk_rms': error,
                'mean_bag_bulk_rms': float(np.mean([np.sqrt(np.mean((b-desired)**2)) for b in bulks])),
                'checks': checks})
        de_blocks.append(real); de_labels.extend([target+'__real'] * real.shape[0])
        print(context, target, summaries[-2:], flush=True)
    zero_checks = []
    for repeat in range(cfg['null_repeats']):
        seed = stable_seed(cfg['seed'], context, 'null', repeat)
        base = ntc[np.random.default_rng(seed).integers(0, ntc.shape[0], cfg['cells'])]
        zero = draw_population(ntc, fit_population(ntc, np.zeros(ntc.shape[1]), **cfg['population']), seed, cfg['cells'])
        zero_checks.append((base != zero).nnz == 0)
        real_null = null_pool[np.random.default_rng(seed).choice(null_pool.shape[0], cfg['cells'], replace=False)]
        for name, matrix in [('zero', zero), ('input', base), ('real_null', real_null)]:
            label = f'{name}_{repeat}'
            blocks.append(matrix); labels.extend([label]*matrix.shape[0])
            de_blocks.append(matrix); de_labels.extend([label]*matrix.shape[0])
    prediction = annotated(sparse.vstack(blocks, format='csr'), bank.var_names, labels, context)
    path = directory / 'counts.h5ad'
    prediction.write_h5ad(path, compression='gzip')
    diagnostics = directory / 'pre-score-diagnostics.json'
    write_json(diagnostics, {'status':'completed','hard_constraints_passed':True,'prediction_ref':ref(path),
        'records':records,'zero_identity':all(zero_checks),'fitted_populations':fitted_records})
    # The diagnostics marker is persisted before the official DE call.
    table, backend = de_table(sparse.vstack(de_blocks, format='csr'), reference, bank.var_names, de_labels,
                             cfg['runtime']['threads'])
    de_path = directory / 'official-de.parquet'; table.write_parquet(de_path)
    # Composite group IDs must not make the directly perturbed gene enter the
    # downstream diagnostic. Preserve the original official table unchanged.
    filtered = table.filter(table['feature'] != table['target'].str.split('__').list.first())
    de = de_summary(filtered)
    frame = filtered.to_pandas()
    by_group = {str(k): v.set_index('feature') for k, v in frame.groupby('target', sort=False)}
    overlaps = []
    for target in targets:
        truth = by_group[target+'__real']
        true_sig = set(truth.index[truth.p_adj.lt(.05)])
        for arm in cfg['arms']:
            for repeat in range(cfg['de_repeats']):
                for suffix in ('', '__nmatched'):
                    pred = by_group[f'{target}__{arm}_{repeat}'+suffix]
                    pred_sig = set(pred.index[pred.p_adj.lt(.05)])
                    union = true_sig | pred_sig
                    common = sorted(true_sig & set(pred.index))
                    agreement = (np.sign(truth.loc[common].log2_fold_change.to_numpy()) ==
                                 np.sign(pred.loc[common].log2_fold_change.to_numpy()))
                    overlaps.append({'target': target, 'arm': arm, 'repeat': repeat,
                        'nmatched': bool(suffix), 'real_significant': len(true_sig),
                        'pred_significant': len(pred_sig),
                        'descriptive_sig_jaccard': len(true_sig & pred_sig)/len(union) if union else None,
                        'direction_on_tested_real_sig': float(agreement.mean()) if len(common) else None,
                        'tested_real_sig': len(common)})
    zero_added = [de[f'zero_{i}']['significant_genes'] - de[f'input_{i}']['significant_genes']
                  for i in range(cfg['null_repeats'])]
    checks = {'zero_identity':all(zero_checks), 'no_added_null_de':max(zero_added)==0,
              'population_all_targets':all(all(s['checks'].values()) for s in summaries if s['arm']=='population')}
    result = {'context':context, 'checks':checks, 'summaries':summaries, 'official_de_backend':backend,
              'de_summary':de, 'known_de_diagnostics':overlaps, 'zero_added_de':zero_added,
              'files':[ref(path),ref(diagnostics),ref(de_path)]}
    write_json(marker, result)
    return result


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--config',type=Path,required=True)
    cfg = json.loads(parser.parse_args().config.read_text())
    research = bind(ROOT, cfg['run_id'], cfg, resume=os.environ.get('VCC_RESEARCH_RESUME') == '1')
    job = RegisteredRun(cfg, research)
    try:
        results = {c:context_batch(job,c) for c in cfg['contexts']}
        checks = [x for r in results.values() for x in r['checks'].values()]
        checkpoint = job.output/'checkpoints/population-specification.json'
        write_json(checkpoint, {'population':cfg['population'],'bag':cfg['bag'],'research':job.research})
        predictions = job.output/'predictions/manifest.json'
        write_json(predictions, {c:ref(job.output/'predictions'/c/'result.json') for c in results})
        job.complete({'acceptance':{'fraction':sum(checks)/len(checks)}, 'contexts':results,
                      'diagnostics_completed':True, 'submission_decision':'not_applicable_oracle_calibration'},
                     checkpoint,predictions,[job.output/'predictions'/c/'pre-score-diagnostics.json' for c in results])
    except BaseException:
        job.run.finish(exit_code=1); raise


if __name__ == '__main__':
    main()
