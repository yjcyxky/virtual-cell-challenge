"""Frozen, explicitly oracle controls for auditing the local evaluator.

This module never fits a predictive model. Reference-derived quantities remain
inside instrument diagnostics and cannot be used by a deployable predictor.
"""
import json

import anndata as ad
import numpy as np
from scipy import sparse

from .common import ROOT, NTC, ref, verified, write_json, stable_seed
from .counts import profile, emit_counts, describe
from .official import annotated, de_table
from .capability import ensure_bundle, evaluate_counts


def audit_cell_count_power(output, config, context):
    """Hold the generated population fixed; vary only sampled cell count.

    This is an evaluation diagnostic, never permission to change submission n.
    """
    directory = output / 'predictions' / 'cell-count-power' / context
    marker = directory / 'result.json'
    if marker.exists():
        result = json.loads(marker.read_text())
        for item in result['files']:
            verified(item)
        return result
    directory.mkdir(parents=True, exist_ok=True)
    source_ref = config['calibration_refs'][context]
    source = json.loads(verified(source_ref).read_text())
    counts_ref = next(r for r in source['files'] if r['path'].endswith('/calibration-counts.h5ad'))
    counts = ad.read_h5ad(verified(counts_ref))
    bank = ad.read_h5ad(verified(source['source_ref']))
    score = bank.X[bank.obs.bank.eq('score').to_numpy()]
    reference = score[:score.shape[0]//2]
    targets = sorted({r['target'] for r in source['known_effects']})
    blocks, labels, selections = [], [], []
    for target in targets:
        real_rows = np.flatnonzero(counts.obs.target_gene.eq(target + '__real').to_numpy())
        generated_rows = np.flatnonzero(counts.obs.target_gene.eq(target + '__conservative').to_numpy())
        if len(generated_rows) != config['prediction_cells'] or len(real_rows) > len(generated_rows):
            raise ValueError('Frozen source count policy changed')
        arms = [('real', real_rows), ('generated_400', generated_rows)]
        for repeat in range(config['power_diagnostic']['repeats']):
            seed = stable_seed(config['seed'], context, target, 'power', repeat)
            rows = np.random.default_rng(seed).choice(generated_rows, len(real_rows), replace=False)
            arms.append((f'generated_nmatched_{repeat}', rows))
        for arm, rows in arms:
            label = target + '__' + arm
            blocks.append(counts.X[rows]); labels.extend([label] * len(rows))
            selections.append({'condition': label, 'target': target, 'arm': arm,
                               'source_rows': rows.tolist(), 'cells': len(rows)})
    table, backend = de_table(sparse.vstack(blocks, format='csr'), reference,
                             counts.var_names.to_numpy(), labels, config['runtime']['num_threads'])
    table_path = directory / 'official-de.parquet'; table.write_parquet(table_path)
    frame = table.to_pandas()
    for item in selections:
        group = frame.loc[frame.target.eq(item['condition']) & frame.feature.ne(item['target'])]
        item['tested_genes'] = len(group)
        item['significant_genes'] = int(group.p_adj.lt(.05).sum())
    selection_path = directory / 'source-row-selections.json'
    write_json(selection_path, {'source_counts_ref': counts_ref, 'selections': selections})
    result = {'context': context, 'source_ref': source_ref, 'source_counts_ref': counts_ref,
        'reference_bank_ref': source['source_ref'], 'official_de_backend': backend,
        'arms': [{k:v for k,v in item.items() if k != 'source_rows'} for item in selections],
        'limitations': 'Fixed generated population, five without-replacement n-matched subsets. Diagnoses cell-count sensitivity conditional on this finite population; does not isolate all distribution error, create biological replicates, or alter the required 400 prediction cells.',
        'files': [ref(table_path), ref(selection_path)]}
    write_json(marker, result)
    print(f'{context}: completed fixed-population cell-count sensitivity diagnostic', flush=True)
    return result


def audit_panel(output, config, context, panel_id, observation_ref):
    directory = output / 'predictions' / panel_id / context
    marker = directory / 'panel-result.json'
    if marker.exists():
        result = json.loads(marker.read_text())
        for record in result['files']:
            verified(record)
        return result
    directory.mkdir(parents=True, exist_ok=True)
    observation = json.loads(verified(observation_ref).read_text())
    targets = observation['panels'][panel_id]
    if len(targets) < config['minimum_panel_targets']:
        raise ValueError('Panel below the registered target-count floor')
    bank_ref = next(r for r in observation['files'] if r['path'].endswith('/bank-depth-view.h5ad'))
    bank = ad.read_h5ad(verified(bank_ref))
    inp = bank.obs.bank.eq('input').to_numpy()
    score = bank.obs.bank.eq('score').to_numpy()
    pert = bank.obs.bank.eq('perturbation').to_numpy() & bank.obs.target.isin(targets).to_numpy()
    if set(bank.obs_names[inp]) & set(bank.obs_names[score | pert]):
        raise ValueError('Input cells overlap evaluation cells')
    ntc = bank.X[inp]
    selected = bank[score | pert].copy()
    real = annotated(selected.X, bank.var_names, selected.obs.target.astype(str).tolist(), 'real')
    real.obs['physical_id'] = selected.obs_names.to_numpy()
    real.var = bank.var.copy()
    if real.obs.physical_id.duplicated().any():
        raise ValueError('Reference has duplicate physical cells')
    real_path = directory / 'real.h5ad'
    real.write_h5ad(real_path, compression='lzf')
    real_ntc = real.X[real.obs.target_gene.eq(NTC).to_numpy()]
    ordered = sorted(targets, key=lambda t: stable_seed(config['seed'], context, panel_id, 'template', t))
    template_targets = ordered[:len(ordered)//2]
    diagnostic_targets = ordered[len(ordered)//2:]
    template = np.mean([profile(real.X[real.obs.target_gene.eq(t).to_numpy()]) - profile(real_ntc)
                        for t in template_targets], axis=0)
    scope = {'kind': 'evaluation_only_oracle_no_predictor_fit', 'fit_targets': template_targets,
             'diagnostic_targets': diagnostic_targets, 'partition': 'fixed_identity_hash_half',
             'reference_ref': ref(real_path)}
    np.save(directory / 'oracle-template.npy', template)
    bundle_status = ensure_bundle(real, directory / 'real-bundle',
                                 config['run_id'] + '-' + panel_id + '-' + context, config['runtime'])
    bundle = ROOT / bundle_status['bundle_directory']
    results, generations = {}, {}
    for condition in config['conditions']:
        print(f'{panel_id}/{context}: score {condition}', flush=True)
        prediction_path = directory / (condition + '.h5ad')
        generation_path = directory / (condition + '-generation.json')
        if prediction_path.exists() and generation_path.exists():
            generation = json.loads(generation_path.read_text())
            verified(generation['prediction_ref'])
            prediction = ad.read_h5ad(prediction_path)
        else:
            generation = {'kind': condition, 'oracle': condition in ('real_copy', 'target_derangement', 'oracle_template')}
            if condition == 'real_copy':
                prediction = real.copy()
            elif condition == 'target_derangement':
                prediction = real.copy()
                shuffled = sorted(targets, key=lambda t: stable_seed(config['seed'], context, panel_id, 'shuffle', t))
                mapping = dict(zip(shuffled, shuffled[1:] + shuffled[:1]))
                prediction.obs['target_gene'] = prediction.obs.target_gene.astype(str).map(
                    lambda t: mapping.get(t, t)).to_numpy()
                generation['source_to_destination'] = mapping
            elif condition in ('zero', 'oracle_template'):
                blocks, labels, diagnostics = [ntc], [NTC] * ntc.shape[0], []
                delta = np.zeros(len(bank.var_names)) if condition == 'zero' else (
                    profile(real_ntc) + template - profile(ntc))
                for target in targets:
                    seed = stable_seed(config['seed'], context, panel_id, target, 'generation')
                    matrix, details = emit_counts(ntc, delta, seed, cells=config['prediction_cells'], **config['emitter'])
                    blocks.append(matrix); labels.extend([target] * matrix.shape[0])
                    diagnostics.append({'target': target, 'generation_seed': seed,
                                        'generation': details, 'distribution': describe(matrix)})
                prediction = annotated(sparse.vstack(blocks, format='csr'), bank.var_names, labels, condition)
                generation['targets'] = diagnostics
            else:
                raise ValueError('Unregistered evaluator control')
            prediction.var = bank.var.copy()
            prediction.write_h5ad(prediction_path, compression='lzf')
            generation['prediction_ref'] = ref(prediction_path)
            write_json(generation_path, generation)
        generations[condition] = ref(generation_path)
        results[condition] = evaluate_counts(prediction, real, bundle, bundle_status,
            directory / condition, config['runtime'], targets, template, scope,
            config['pathway_ref'], diagnostic_targets=diagnostic_targets)
    oracle = results['real_copy']['response_diagnostics']
    shuffled = results['target_derangement']['response_diagnostics']
    checks = {
        'independent_input_reference': True,
        'oracle_bulk_identity': oracle['mean_response_rms_error'] <= config['acceptance']['oracle_rms'],
        'oracle_residual_identity': oracle['mean_residual_rms_error'] <= config['acceptance']['oracle_rms'],
        'label_derangement_detected': shuffled['mean_response_rms_error'] > config['acceptance']['derangement_rms'],
        'official_six_scores_defined': all(r['normalized'] is not None and
            len(r['normalized']) == 6 and all(v is not None for v in r['normalized'].values())
            and r['Overall'] is not None for r in results.values()),
    }
    pds = 'pds_cosine'
    if results['real_copy']['raw'].get(pds) is None or results['target_derangement']['raw'].get(pds) is None:
        checks['official_pds_detects_labels'] = False
    else:
        checks['official_pds_detects_labels'] = (results['real_copy']['raw'][pds] -
            results['target_derangement']['raw'][pds]) > config['acceptance']['derangement_pds_gap']
    rows = []
    for target in targets:
        real_matrix = real.X[real.obs.target_gene.eq(target).to_numpy()]
        rows.append({'target': target, **describe(real_matrix)})
    result = {'context': context, 'panel_id': panel_id, 'targets': targets, 'real_cell_counts':
        real.obs.target_gene.value_counts().to_dict(), 'source_ref': observation_ref, 'bank_ref': bank_ref,
        'measured_genes': real.n_vars, 'official_genes': observation['official_gene_count'],
        'reference_distributions': rows, 'template_scope': scope, 'bundle': bundle_status,
        'conditions': {c: {'raw': r['raw'], 'normalized': r['normalized'], 'Overall': r['Overall'],
                          'official_rejection': r['official_rejection'],
                          'diagnostics': r['response_diagnostics'], 'result_ref': ref(directory / c / 'result.json')}
                       for c, r in results.items()},
        'generation_refs': generations, 'checks': checks, 'acceptance_fraction': sum(checks.values())/len(checks),
        'limitations': 'Instrument controls only. Real-copy/deranged references keep original scarce cell counts; zero/template controls emit 400 cells. No predictor fitted, no model generalization claim. Oracle template is restricted to a fixed half-panel and diagnostics to the other half.',
        'files': [ref(p) for p in sorted(directory.rglob('*')) if p.is_file() and p != marker]}
    write_json(marker, result)
    print(f'{panel_id}/{context}: completed evaluator controls {checks}', flush=True)
    return result
