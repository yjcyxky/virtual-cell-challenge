"""Matched-population response and joint-distribution evaluation."""
import hashlib
import json

import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import cdist

from data import logcp, write_json, reserved


def task_seed(context, target):
    return int.from_bytes(hashlib.sha256(('evaluation-v1|' + context + '|' + target).encode()).digest()[:4], 'little')


def observed_task(data, view, context, target):
    groups = data.groups[(context, target)]
    weights = data.task_weights(context, target)
    original = data.task_weights(context, target, original=True)
    mask = data.masks[context]
    ids = data.tasks[(context, target)]
    observed = data.read(ids)
    local_index = {int(v): i for i, v in enumerate(ids)}
    lc = logcp(observed, data.common)
    cell_weights, original_weights = np.zeros(len(ids)), np.zeros(len(ids))
    sums = {k: np.zeros(len(data.genes), np.float64) for k in ['reference', 'reference_log', 'reference_log_common', 'original_reference_log_common']}
    group_metrics = []
    for pair, indices in groups.items():
        construct, batch = pair
        positions = np.asarray([local_index[int(v)] for v in indices])
        cell_weights[positions] = weights[pair] / len(positions)
        original_weights[positions] = original[pair] / len(positions)
        state = view.controls[(context, batch)]
        sums['reference'] += weights[pair] * state['reference']
        sums['reference_log'] += weights[pair] * state['reference_log']
        sums['reference_log_common'] += weights[pair] * state['reference_log_common']
        sums['original_reference_log_common'] += original[pair] * state['reference_log_common']
        group_metrics.append((pair, lc[positions].mean(0) - state['reference_log_common']))
    sums['log'] = np.average(logcp(observed, mask), axis=0, weights=cell_weights)
    sums['log_common'] = np.average(lc, axis=0, weights=cell_weights)
    sums['count'] = np.average(observed, axis=0, weights=cell_weights)
    sums['original_log_common'] = np.average(lc, axis=0, weights=original_weights)
    return sums, observed, cell_weights, group_metrics


def response_summary(data, view, context, target):
    """Common-gene baseline statistic; avoid constructing unused distribution diagnostics."""
    ids = data.tasks[(context, target)]
    common = data.common
    x = logcp(data.read(ids)[:, common])
    local_index = {int(v): i for i, v in enumerate(ids)}
    weights = data.task_weights(context, target)
    cell_weights = np.zeros(len(ids)); reference = np.zeros(int(common.sum()))
    for pair, members in data.groups[(context, target)].items():
        positions = np.asarray([local_index[int(v)] for v in members])
        cell_weights[positions] = weights[pair] / len(members)
        reference += weights[pair] * view.controls[(context, pair[1])]['reference_log_common'][common]
    return (np.average(x, axis=0, weights=cell_weights) - reference).astype(np.float32)


def shared_responses(data, view, contexts, config, directory):
    path = directory / 'shared-response.npz'
    if path.exists():
        saved = np.load(path)
        return {t: saved['responses'][i] for i, t in enumerate(saved['targets'])}, saved['global_response']
    totals, n = {}, {}
    global_by_context = []
    for context in contexts:
        rows = []
        for c, target in data.tasks:
            if c != context or reserved(target, config['unseen_target_percent']):
                continue
            response = response_summary(data, view, context, target)
            if target not in totals:
                totals[target] = np.zeros_like(response); n[target] = 0
            totals[target] += response; n[target] += 1; rows.append(response)
        global_by_context.append(np.mean(rows, axis=0))
        print(json.dumps({'stage': 'shared_baseline', 'training_context': context, 'targets': len(rows)}), flush=True)
    averaged = {t: v / n[t] for t, v in totals.items()}
    global_response = np.mean(global_by_context, axis=0).astype(np.float32)
    targets = sorted(averaged)
    np.savez_compressed(path, targets=targets, responses=np.stack([averaged[t] for t in targets]), global_response=global_response)
    return averaged, global_response


def allocate(weights, total):
    weights = np.asarray(weights, float); weights /= weights.sum()
    total = max(total, len(weights))
    numbers = np.ones(len(weights), int)
    extra = weights * (total - len(weights))
    numbers += np.floor(extra).astype(int)
    remaining = total - numbers.sum()
    if remaining:
        numbers[np.argsort(-(extra - np.floor(extra)))[:remaining]] += 1
    return numbers


def correlation(a, b):
    a, b = a - a.mean(), b - b.mean()
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / denominator) if denominator > 1e-10 else None


@torch.no_grad()
def evaluate_task(model, data, view, context, target, config, shared=None, save_cells=False):
    device = next(model.parameters()).device
    sums, observed, observed_weight, group_response = observed_task(data, view, context, target)
    groups = data.groups[(context, target)]
    weights = data.task_weights(context, target); original = data.task_weights(context, target, original=True)
    pairs = list(groups)
    number = allocate([weights[p] for p in pairs], config['evaluation_cells'])
    specifications = [(context, target, pair[1], int(n)) for pair, n in zip(pairs, number)]
    batch = view.inputs(specifications, device)
    # Common random streams make comparisons paired; evaluation never advances training RNG.
    with torch.random.fork_rng(devices=[device.index or 0] if device.type == 'cuda' else []):
        torch.manual_seed(task_seed(context, target))
        generated = model.generate(batch).cpu().numpy()
    assert np.isfinite(generated).all() and generated.min() >= 0 and np.array_equal(generated, np.floor(generated))
    assert generated.max(initial=0) <= np.iinfo(np.uint32).max, 'generated_counts_exceed_storage_type'
    generated_common = logcp(generated, data.common)
    generated_native = logcp(generated, data.masks[context])
    pred_log = np.zeros(len(data.genes)); pred_common = np.zeros(len(data.genes)); pred_count = np.zeros(len(data.genes)); pred_original = np.zeros(len(data.genes))
    predicted_weight = []
    ptr = 0
    for pair, count in zip(pairs, number):
        x = generated[ptr:ptr + count]
        lc = generated_common[ptr:ptr + count].mean(0)
        pred_log += weights[pair] * generated_native[ptr:ptr + count].mean(0)
        ptr += count
        pred_common += weights[pair] * lc
        pred_count += weights[pair] * x.mean(0)
        pred_original += original[pair] * lc
        predicted_weight.extend([weights[pair] / count] * count)
    common = data.common.copy(); common[data.gene_index[target]] = False
    native = data.masks[context].copy(); native[data.gene_index[target]] = False
    trained_native = native & model.trained_readouts.cpu().numpy()
    actual = sums['log_common'] - sums['reference_log_common']
    predicted = pred_common - sums['reference_log_common']
    residual = (predicted - actual)[common]
    actual_native = sums['log'] - sums['reference_log']
    pred_native = pred_log - sums['reference_log']
    pseudobulk_truth = logcp(sums['count'][None], data.common)[0] - logcp(sums['reference'][None], data.common)[0]
    pseudobulk_pred = logcp(pred_count[None], data.common)[0] - logcp(sums['reference'][None], data.common)[0]
    original_true = sums['original_log_common'] - sums['original_reference_log_common']
    original_pred = pred_original - sums['original_reference_log_common']
    common_indices = np.flatnonzero(data.common); common_target_mask = common[common_indices]
    share = np.zeros(data.common.sum()) if shared is None else shared
    rng = np.random.default_rng(task_seed(context, target))
    oi = rng.choice(len(observed), config['evaluation_cells'], p=observed_weight / observed_weight.sum())
    pi = rng.choice(len(generated), config['evaluation_cells'], p=np.asarray(predicted_weight) / sum(predicted_weight))
    a, b = view.project(observed[oi]), view.project(generated[pi])
    energy = float(2 * cdist(a, b).mean() - cdist(a, a).mean() - cdist(b, b).mean())
    covariance = float(np.mean((np.cov(a.T) - np.cov(b.T)) ** 2))
    # One finite NTC population with the same layer allocation as the model.
    # Preserve RNG draws and final population resampling for historical metrics.
    ntc_ids = np.concatenate([data.controls[(context, pair[1], 0)][rng.integers(len(data.controls[(context, pair[1], 0)]), size=n)]
                             for pair, n in zip(pairs, number)])
    ntc_generated = data.read(ntc_ids)
    ntc = ntc_generated[pi]
    ntc_sample_mean = np.average(logcp(ntc_generated, data.common), axis=0, weights=predicted_weight)
    ntc_sample_residual = (ntc_sample_mean - sums['log_common'])[common]
    ntc_z = view.project(ntc)
    ntc_energy = float(2 * cdist(a, ntc_z).mean() - cdist(a, a).mean() - cdist(ntc_z, ntc_z).mean())
    centers = view.controls[(context, pairs[0][1])]['centers']
    pa = np.bincount(cdist(a, centers).argmin(1), minlength=len(centers)) / len(a)
    pb = np.bincount(cdist(b, centers).argmin(1), minlength=len(centers)) / len(b)
    factor_variation = sum(weights[p] * np.mean((r[common] - actual[common]) ** 2) for p, r in group_response)
    metrics = {
        'context': context, 'target': target, 'reserved_target': reserved(target, config['unseen_target_percent']),
        'response_mse': float(np.mean(residual ** 2)), 'response_mae': float(np.mean(abs(residual))),
        'response_correlation': correlation(predicted[common], actual[common]),
        'zero_mse': float(np.mean(actual[common] ** 2)), 'zero_mae': float(np.mean(abs(actual[common]))),
        'ntc_sample_response_mse': float(np.mean(ntc_sample_residual ** 2)),
        'ntc_sample_response_mae': float(np.mean(abs(ntc_sample_residual))),
        'shared_mse': float(np.mean((share[common_target_mask] - actual[common]) ** 2)),
        'shared_mae': float(np.mean(abs(share[common_target_mask] - actual[common]))),
        'native_response_mse': float(np.mean((pred_native[native] - actual_native[native]) ** 2)),
        'native_response_mae': float(np.mean(abs(pred_native[native] - actual_native[native]))),
        'native_zero_mse': float(np.mean(actual_native[native] ** 2)),
        'native_trained_readout_mse': float(np.mean((pred_native[trained_native] - actual_native[trained_native]) ** 2)),
        'native_readouts_without_training_measurement': int((native & ~trained_native).sum()),
        'pseudobulk_response_mse': float(np.mean((pseudobulk_pred[common] - pseudobulk_truth[common]) ** 2)),
        'log_mean_raw_count_mse': float(np.mean((np.log1p(pred_count[native]) - np.log1p(sums['count'][native])) ** 2)),
        'log_mean_raw_count_zero_mse': float(np.mean((np.log1p(sums['reference'][native]) - np.log1p(sums['count'][native])) ** 2)),
        'observed_response_rms': float(np.sqrt(np.mean(actual[common] ** 2))),
        'predicted_response_rms': float(np.sqrt(np.mean(predicted[common] ** 2))),
        'direction_agreement_above_0_05': float(np.mean(np.sign(predicted[common & (abs(actual) > 0.05)]) == np.sign(actual[common & (abs(actual) > 0.05)]))) if np.any(common & (abs(actual) > 0.05)) else None,
        'original_weight_mse': float(np.mean((original_pred[common] - original_true[common]) ** 2)),
        'original_weight_zero_mse': float(np.mean(original_true[common] ** 2)),
        'factor_response_variance': float(factor_variation),
        'energy_distance': energy, 'ntc_energy_distance': ntc_energy, 'covariance_mse': covariance,
        'state_composition_l1': float(np.abs(pa - pb).sum()),
        'projected_quantile_mae': float(np.mean(abs(np.quantile(a, [0.1, 0.5, 0.9], axis=0) - np.quantile(b, [0.1, 0.5, 0.9], axis=0)))),
        'depth_mean_ratio': float(generated[pi].sum(1).mean() / max(observed[oi].sum(1).mean(), 1)),
        'depth_cv_predicted': float(generated[pi].sum(1).std() / max(generated[pi].sum(1).mean(), 1)),
        'depth_cv_observed': float(observed[oi].sum(1).std() / max(observed[oi].sum(1).mean(), 1)),
        'detection_mae': float(np.mean(abs((generated[pi] > 0).mean(0)[native] - (observed[oi] > 0).mean(0)[native]))),
        'variance_log_mse': float(np.mean((logcp(generated[pi]).var(0)[native] - logcp(observed[oi]).var(0)[native]) ** 2)),
        'on_target_readout_observed': bool(data.masks[context][data.gene_index[target]]),
        'on_target_mean_count_predicted': float(pred_count[data.gene_index[target]]) if data.masks[context][data.gene_index[target]] else None,
        'on_target_mean_count_observed': float(sums['count'][data.gene_index[target]]) if data.masks[context][data.gene_index[target]] else None,
        'technical_construct_layers': len(pairs), 'generated_cells': len(generated), 'cached_observed_cells': len(observed),
        'fraction_NTC_batches_with_target_support': len({p[1] for p in pairs}) / sum(c == context for c, b in view.controls),
    }
    prediction = {'target': target, 'common_response': predicted[data.common].astype(np.float32),
                  'native_mean_log': pred_log[data.masks[context]].astype(np.float32),
                  'native_mean_count': pred_count[data.masks[context]].astype(np.float32)}
    if save_cells:
        prediction['cells'] = generated.astype(np.uint32)
    return metrics, prediction


def evaluate_context(model, data, view, context, config, directory, shared, global_shared, validation=False):
    model.eval()
    targets = sorted((t for c, t in data.tasks if c == context and (not validation or not reserved(t, config['unseen_target_percent']))),
                     key=lambda t: task_seed(context, t))
    if validation:
        targets = targets[:config['validation_tasks']]
    rows, predictions = [], []
    for i, target in enumerate(targets):
        metrics, prediction = evaluate_task(model, data, view, context, target, config,
                                             shared.get(target, global_shared), save_cells=not validation and i < 2)
        rows.append(metrics)
        if not validation:
            predictions.append(prediction)
            if 'cells' in prediction:
                np.savez_compressed(directory / f'example-{i}.npz', target=target, context=context,
                                    genes=data.genes, counts=prediction.pop('cells'),
                                    available_readout_mask=data.masks[context],
                                    trained_readout_mask=model.trained_readouts.cpu().numpy())
        if not validation and (i + 1) % 100 == 0:
            print(json.dumps({'stage': 'outer_evaluation', 'context': context, 'targets_done': i + 1, 'total': len(targets)}), flush=True)
    frame = pd.DataFrame(rows)
    if not validation:
        frame.to_parquet(directory / 'metrics.parquet', index=False)
        np.savez_compressed(directory / 'predictions.npz', targets=targets,
                            common_genes=np.asarray(data.genes)[data.common], native_genes=np.asarray(data.genes)[data.masks[context]],
                            **{key: np.stack([p[key] for p in predictions]) for key in ['common_response', 'native_mean_log', 'native_mean_count']})
    return frame


def summarize(frames, config, output):
    metrics = pd.concat(frames, ignore_index=True)
    metrics.to_parquet(output / 'evaluation-metrics.parquet', index=False)
    numeric = metrics.select_dtypes(include=np.number).columns
    by_context = metrics.groupby('context')[numeric].mean()
    by_context.to_csv(output / 'evaluation-by-context.csv')
    metrics.groupby(['context', 'target_seen_in_training', 'reserved_target'])[numeric].mean().to_csv(output / 'evaluation-by-stratum.csv')
    macro = by_context.mean().to_dict()
    rng = np.random.default_rng(config['data_seed'])
    targets = sorted(metrics.target.unique())
    # Target-cluster resampling keeps repeated targets across backgrounds together.
    difference = metrics.assign(delta=metrics.zero_mse - metrics.response_mse).pivot(index='target', columns='context', values='delta').reindex(targets).to_numpy()
    boot = [float(np.nanmean(np.nanmean(difference[rng.integers(len(targets), size=len(targets))], axis=0)))
            for _ in range(config['bootstrap_repeats'])]
    result = {'macro': {k: float(v) for k, v in macro.items()}, 'tasks': len(metrics), 'contexts': len(by_context),
              'mse_improvement_vs_zero_fraction': float(1 - macro['response_mse'] / macro['zero_mse']),
              'mae_improvement_vs_zero_fraction': float(1 - macro['response_mae'] / macro['zero_mae']),
              'paired_zero_minus_model_mse_target_cluster_bootstrap_95': np.quantile(boot, [0.025, 0.975]).tolist(),
              'scope': 'retrospective development; task bootstrap and training seeds are not independent biological repeats'}
    write_json(output / 'metrics.json', result)
    return result
