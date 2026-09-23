"""Paired comparison of completed trials without rewriting their historical metrics."""
import json
import re

import numpy as np
import pandas as pd
import torch
import yaml

from data import EXPERIMENT, BalancedSampler, data_spec_hash, hash_file, write_json


def comparison_sources(run_ids):
    records = []
    assert len(set(run_ids)) == len(run_ids), 'duplicate_comparison_run'
    for run_id in run_ids:
        assert re.fullmatch(r'[A-Za-z0-9_-]+', run_id), 'invalid_comparison_run_id'
        folder = EXPERIMENT / 'outputs' / run_id
        completion = json.loads((folder / 'complete.json').read_text())
        assert completion['status'] == 'completed', 'comparison_requires_completed_trial'
        config = yaml.safe_load((folder / 'config.yaml').read_text())
        records.append({'run_id': run_id, 'pipeline_commit': config['pipeline_commit'],
                        'config_sha256': hash_file(folder / 'config.yaml'),
                        'metrics_sha256': hash_file(folder / 'evaluation-metrics.parquet'),
                        'artifact': completion['artifact']})
    return records


def selected_exposure(data, folder, config, key):
    """Replay only sampling, including NTC RNG draws; prove agreement with saved state."""
    checkpoint_path = folder / 'checkpoints' / f'{key}-last.pt'
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    checkpoint_sha256 = hash_file(checkpoint_path)
    progress = checkpoint['progress']
    assert progress['complete'], 'exposure_requires_completed_fit'
    sampler = BalancedSampler(data, progress['training_contexts'], config['seed'] + sum(map(ord, key)),
                              config['unseen_target_percent'])
    selected = None
    for step in range(progress['next_step']):
        groups = sampler.sample(config['tasks_per_step'], config['cells_per_task'])
        context, _, batch, _ = groups[0]
        sampler.rng.choice(data.controls[(context, batch, 1)], config['cells_per_task'], replace=True)
        if step + 1 == progress['best_epoch'] * config['steps_per_epoch']:
            selected = {context: set(targets) for context, targets in sampler.seen.items()}
    assert selected is not None
    assert sampler.state_dict() == checkpoint['sampler'], 'sampler_replay_mismatch'
    rows = [{'fit': key, 'training_context': c, 'eligible_targets': len(sampler.targets[c]),
             'sampled_targets_at_selected_epoch': len(selected[c]), 'sampled_targets_at_last_epoch': len(sampler.seen[c]),
             'selected_epoch': progress['best_epoch'], 'last_epoch': len(progress['history']),
             'resume_count': progress['resume_count'], 'stop_reason': progress['stop_reason'],
             'replay_matches_checkpoint': True, 'checkpoint_sha256': checkpoint_sha256} for c in progress['training_contexts']]
    return set.union(*selected.values()), rows


def paired_difference(model, reference, repeats, seed):
    """Context-equal error reduction; seeds retained, targets clustered across contexts."""
    keys = ['seed', 'context', 'target']
    metric_names = ['response_mse', 'response_mae', 'energy_distance', 'original_weight_mse']
    assert set(map(tuple, model[keys].to_numpy())) == set(map(tuple, reference[keys].to_numpy())), 'unpaired_tasks'
    pair = model[keys + metric_names].merge(reference[keys + metric_names], on=keys,
                                           suffixes=('_model', '_reference'), validate='one_to_one')
    deltas = pair[keys].copy()
    for name in metric_names:
        deltas[name] = pair[name + '_reference'] - pair[name + '_model']
    per_seed = deltas.groupby(['seed', 'context'])[metric_names].mean().groupby('seed').mean()
    per_context = deltas.groupby(['seed', 'context'])[metric_names].mean().groupby('context').mean()
    # Seed averaging happens before resampling; intervals quantify target sampling
    # conditional on these seeds, not seed uncertainty or biological replication.
    target_means = deltas.groupby(['target', 'context']).response_mse.mean().unstack('context').to_numpy()
    rng = np.random.default_rng(seed)
    bootstrap = []
    for _ in range(repeats * 100):
        sampled = target_means[rng.integers(len(target_means), size=len(target_means))]
        if not np.isfinite(sampled).any(axis=0).all():
            continue
        bootstrap.append(float(np.nanmean(np.nanmean(sampled, axis=0))))
        if len(bootstrap) == repeats:
            break
    assert len(bootstrap) == repeats, 'insufficient_cluster_support_for_all_contexts'
    return {'positive_means_model_improves': True, 'paired_tasks_per_seed': deltas.groupby('seed').size().to_dict(),
            'mean_reference_minus_model': per_seed.mean().to_dict(),
            'seed_sample_sd': per_seed.std(ddof=1).to_dict() if len(per_seed) > 1 else None,
            'by_seed': per_seed.to_dict('index'), 'by_context': per_context.to_dict('index'),
            'mse_target_cluster_bootstrap_95': np.quantile(bootstrap, [.025, .975]).tolist(),
            'interval_scope': 'target clusters; fixed observed contexts and seeds; not biological replication'}


def compare_trials(data, config, output):
    records = config.get('comparison_sources')
    if not records:
        return None
    frames, exposure_rows, run_rows = [], [], []
    assert records == comparison_sources([record['run_id'] for record in records]), 'comparison_source_changed'
    folders = [EXPERIMENT / 'outputs' / record['run_id'] for record in records] + [output]
    protocol_fields = ['contexts', 'state_dimensions', 'state_components', 'hidden_dimensions', 'residual_dimensions',
                       'modules_per_family', 'minimum_module_genes', 'maximum_module_genes', 'network_minimum_score',
                       'module_off_support_weight', 'response_weight', 'learning_rate', 'weight_decay', 'epochs',
                       'steps_per_epoch', 'minimum_epochs', 'patience', 'kl_warmup_epochs', 'tasks_per_step',
                       'cells_per_task', 'validation_tasks', 'evaluation_cells', 'uv_lock_sha256',
                       'gene_axis_sha256', 'source_collection_sha256', 'prior_source_sha256']
    membership = np.load(output / 'cache' / 'priors' / 'modules.npz')['true']
    prior_targets = {data.genes[i] for i in np.flatnonzero(membership.sum(1))}
    primary = ['response_mse', 'response_mae', 'zero_mse', 'zero_mae', 'shared_mse', 'shared_mae',
               'ntc_sample_response_mse', 'ntc_sample_response_mae',
               'energy_distance', 'ntc_energy_distance', 'original_weight_mse', 'original_weight_zero_mse',
               'depth_mean_ratio', 'detection_mae', 'covariance_mse', 'state_composition_l1',
               'projected_quantile_mae', 'native_response_mse', 'native_zero_mse',
               'log_mean_raw_count_mse', 'log_mean_raw_count_zero_mse']
    baseline = None
    finite_columns = ['ntc_sample_response_mse', 'ntc_sample_response_mae']
    finite_baseline = pd.read_parquet(output / 'evaluation-metrics.parquet').set_index(['context', 'target'])[finite_columns].sort_index()
    for folder in folders:
        source = yaml.safe_load((folder / 'config.yaml').read_text())
        assert data_spec_hash(source) == data_spec_hash(config), 'comparison_data_mismatch'
        for field in protocol_fields:
            assert source[field] == config[field], 'comparison_protocol_mismatch:' + field
        frame = pd.read_parquet(folder / 'evaluation-metrics.parquet')
        assert source['run_id'] == folder.name
        assert set(frame.variant) == {source['variant']} and set(frame.seed) == {source['seed']}
        assert not frame.duplicated(['context', 'target']).any()
        assert set(frame.context) == set(config['contexts'])
        identity = frame.set_index(['context', 'target']).sort_index()[['zero_mse', 'zero_mae', 'shared_mse', 'ntc_energy_distance']]
        if baseline is None:
            baseline = identity
        else:
            pd.testing.assert_frame_equal(baseline, identity, check_exact=True)
        if set(finite_columns).issubset(frame.columns):
            pd.testing.assert_frame_equal(frame.set_index(['context', 'target'])[finite_columns].sort_index(), finite_baseline, check_exact=True)
        else:
            frame = frame.merge(finite_baseline.reset_index(), on=['context', 'target'], validate='one_to_one')
        frame['ntc_sample_baseline_source_run'] = output.name
        frame['run_id'] = folder.name
        frame['target_has_prior_membership'] = frame.target.isin(prior_targets)
        frame['actually_sampled_at_selected_checkpoint'] = False
        for context in config['contexts']:
            seen, coverage = selected_exposure(data, folder, source, 'holdout-' + context)
            frame.loc[frame.context.eq(context), 'actually_sampled_at_selected_checkpoint'] = frame.loc[frame.context.eq(context), 'target'].isin(seen)
            exposure_rows.extend([{'run_id': folder.name, 'variant': source['variant'], 'seed': source['seed'], **row} for row in coverage])
        _, coverage = selected_exposure(data, folder, source, 'final')
        exposure_rows.extend([{'run_id': folder.name, 'variant': source['variant'], 'seed': source['seed'], **row} for row in coverage])
        assert not (frame.reserved_target & frame.actually_sampled_at_selected_checkpoint).any(), 'reserved_target_leak'
        macro = frame.groupby('context')[primary].mean().mean().to_dict()
        run_rows.append({'run_id': folder.name, 'variant': source['variant'], 'seed': source['seed'],
                         'pipeline_commit': source['pipeline_commit'], 'tasks': len(frame), **macro})
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    observed = set(zip(combined.variant, combined.seed))
    expected = {(v, s) for v in ['true_prior', 'random_prior', 'no_prior'] for s in [17, 29, 43]}
    expected |= {(v, 17) for v in ['no_state', 'no_context', 'no_residual', 'no_module']}
    assert observed == expected and len(run_rows) == len(expected), 'incomplete_comparison_matrix'
    run_summary = pd.DataFrame(run_rows)
    run_summary.to_csv(output / 'comparison-runs.csv', index=False)
    pd.DataFrame(exposure_rows).to_csv(output / 'comparison-training-exposure.csv', index=False)
    combined.to_parquet(output / 'comparison-tasks.parquet', index=False)
    combined.groupby(['run_id', 'variant', 'seed', 'context'])[primary].mean().to_csv(output / 'comparison-contexts.csv')
    strata = ['run_id', 'variant', 'seed', 'context', 'reserved_target', 'target_seen_in_training',
              'actually_sampled_at_selected_checkpoint', 'target_has_prior_membership']
    grouped = combined.groupby(strata)
    grouped[primary].mean().join(grouped.size().rename('tasks')).to_csv(output / 'comparison-strata.csv')
    result = {'run_count': len(run_rows), 'task_instances': len(combined), 'source_run_ids': [f.name for f in folders],
              'original_target_seen_in_training_means': 'eligible training pool, not realized optimization exposure',
              'main_metrics': 'mean(log1p(CP10K(cell))) response; common genes excluding on-target; context equal',
              'comparisons': {}}
    true = combined[combined.variant.eq('true_prior')]
    for variant in ['random_prior', 'no_prior', 'no_state', 'no_context', 'no_residual', 'no_module']:
        reference = combined[combined.variant.eq(variant)]
        model = true[true.seed.isin(reference.seed.unique())]
        result['comparisons']['true_prior_vs_' + variant] = paired_difference(model, reference, config['bootstrap_repeats'], config['data_seed'])
    write_json(output / 'comparison.json', result)
    return result
