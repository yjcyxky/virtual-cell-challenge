"""Nested whole-context validation of supervised gene-pair regressors."""
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import xgboost as xgb

from dataset import target_reserved
from features import PairFeatures
from profile_responses import write_json


class Supervision:
    def __init__(self, prepared, metadata):
        self.metadata = metadata
        self.genes = metadata['genes']
        self.contexts = metadata['contexts']
        self.values, self.target_indices = {}, {}
        for c in self.contexts:
            with np.load(prepared / 'contexts' / c / 'labels.npz') as z:
                self.values[c] = z['response']
                self.target_indices[c] = z['targets']
        self.rows = {c: {int(p): i for i, p in enumerate(self.target_indices[c])} for c in self.contexts}
        self.common = np.load(prepared / 'common-measured.npy')

    def rows_for(self, contexts, config, validation=False, release_reserved=False):
        records, y, weights = [], [], []
        cap = config['validation_genes_per_task'] if validation else config['genes_per_task']
        for c in contexts:
            target_records = []
            for p, row in self.rows[c].items():
                if not release_reserved and target_reserved(self.genes[p], config['unseen_target_percent']):
                    continue
                valid = np.flatnonzero(np.isfinite(self.values[c][row]))
                digest = hashlib.sha256(f"{config['seed']}|{c}|{self.genes[p]}".encode()).digest()
                rng = np.random.default_rng(int.from_bytes(digest[:8], 'little'))
                g = np.sort(rng.choice(valid, min(cap, len(valid)), replace=False))
                if len(g):
                    target_records.append((p, g, self.values[c][row, g]))
            for p, g, values in target_records:
                records.append((c, np.full(len(g), p, int), g))
                y.append(values)
                weights.append(np.full(len(g), 1. / (len(target_records) * len(g))))
        weight = np.concatenate(weights).astype(np.float32)
        weight *= len(weight) / weight.sum()
        return records, np.concatenate(y).astype(np.float32), weight


def feature_rows(features, records, references, model):
    # Build one background at a time; avoid retaining thousands of small arrays.
    matrices = []
    names = None
    contexts = list(dict.fromkeys(r[0] for r in records))
    for c in contexts:
        subset = [r for r in records if r[0] == c]
        p, g = np.concatenate([r[1] for r in subset]), np.concatenate([r[2] for r in subset])
        matrix, current_names = features.features(c, p, g, references, model)
        if names is not None:
            assert names == current_names
        names = current_names; matrices.append(matrix)
    return np.concatenate(matrices), names


def fit(features, supervision, contexts, validation, references, model, config, prefix, tracked,
        rounds=None, release_reserved=False):
    path = prefix.with_suffix('.ubj')
    record_path = prefix.with_suffix('.json')
    if record_path.exists():
        booster = xgb.Booster(); booster.load_model(path)
        return booster, json.loads(record_path.read_text())
    prefix.parent.mkdir(parents=True, exist_ok=True)
    records, labels, weights = supervision.rows_for(contexts, config, release_reserved=release_reserved)
    matrix, names = feature_rows(features, records, references, model)
    train = xgb.QuantileDMatrix(matrix, label=labels, weight=weights, feature_names=names,
                               max_bin=config['model_parameters']['max_bin'], nthread=config['model_parameters']['nthread'])
    del matrix
    evaluation = [(train, 'train')]
    if validation:
        records_v, labels_v, weights_v = supervision.rows_for(validation, config, validation=True)
        matrix_v, names_v = feature_rows(features, records_v, references, model)
        assert names_v == names
        valid = xgb.QuantileDMatrix(matrix_v, label=labels_v, weight=weights_v, feature_names=names,
                                   ref=train, max_bin=config['model_parameters']['max_bin'], nthread=config['model_parameters']['nthread'])
        evaluation.append((valid, 'validation'))
        del matrix_v
    parameters = {**config['model_parameters'], 'seed': config['model_seed']}
    history = {}
    start = time.monotonic()
    print(json.dumps({'stage': 'fit', 'model': model, 'checkpoint': prefix.name, 'contexts': contexts,
                      'validation': validation, 'rows': len(labels), 'features': len(names)}), flush=True)
    booster = xgb.train(parameters, train, num_boost_round=rounds or config['num_boost_round'],
                        evals=evaluation, evals_result=history,
                        early_stopping_rounds=config['early_stopping_rounds'] if validation else None,
                        verbose_eval=25)
    selected_rounds = booster.best_iteration + 1 if validation else rounds
    if validation:
        booster = booster[:selected_rounds]
    booster.save_model(path)
    gain = booster.get_score(importance_type='total_gain')
    record = {'model': model, 'training_contexts': contexts, 'validation_contexts': validation,
              'reference_contexts': references, 'rows': len(labels), 'features': names,
              'rounds': selected_rounds, 'history': history, 'seconds': time.monotonic() - start,
              'feature_total_gain': gain, 'reserved_targets_released': release_reserved,
              'row_weight': 'equal context, then equal target, then equal uniformly sampled readout',
              'early_stopping': 'weighted RMSE on a disjoint inner background; fixed uniform readout sample'}
    write_json(record_path, record)
    tracked.log({f'fit/{prefix.name}/rounds': selected_rounds, f'fit/{prefix.name}/seconds': record['seconds']})
    return booster, record


def predict(booster, features, context, targets, genes, references, model):
    result = np.empty((len(targets), len(genes)), np.float32)
    for start in range(0, len(targets), 8):
        selected = targets[start:start + 8]
        p, g = np.repeat(selected, len(genes)), np.tile(genes, len(selected))
        matrix, _ = features.features(context, p, g, references, model)
        result[start:start + len(selected)] = booster.inplace_predict(matrix).reshape(len(selected), len(genes))
    return result


def response_metrics(truth, prediction, common):
    result = {}
    for name, mask in [('measured', np.ones(len(truth), bool)), ('common', common)]:
        good = mask & np.isfinite(truth) & np.isfinite(prediction)
        a, b = truth[good].astype(float), prediction[good].astype(float)
        result[name + '_genes'] = len(a)
        if not len(a):
            continue
        result.update({name + '_MSE': float(np.mean((a - b) ** 2)), name + '_MAE': float(np.mean(np.abs(a - b))),
                       name + '_zero_MSE': float(np.mean(a ** 2)), name + '_zero_MAE': float(np.mean(np.abs(a))),
                       name + '_correlation': float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 0 and np.std(b) > 0 else None,
                       name + '_sign_agreement': float(np.mean(np.sign(a) == np.sign(b)))})
    return result


def shared_response(supervision, target, contexts, reserve_percent=0):
    values = [supervision.values[c][supervision.rows[c][target]] for c in contexts
              if target in supervision.rows[c] and not target_reserved(supervision.genes[target], reserve_percent)]
    if not values:
        return np.zeros(len(supervision.genes), np.float32), np.zeros(len(supervision.genes), int)
    values = np.asarray(values)
    count = np.isfinite(values).sum(0)
    mean = np.divide(np.nansum(values, axis=0), count, out=np.zeros(len(count), np.float32), where=count > 0)
    return mean, count


def fit_shared(supervision, contexts, config):
    grid = np.array([0, .25, .5, .75, 1.])
    curves = {1: [], 2: []}
    for held in contexts:
        other = [c for c in contexts if c != held]
        losses = {1: [], 2: []}
        for p, row in supervision.rows[held].items():
            if target_reserved(supervision.genes[p], config['unseen_target_percent']):
                continue
            pred, count = shared_response(supervision, p, other, config['unseen_target_percent'])
            truth = supervision.values[held][row]
            for kind, support in [(1, count == 1), (2, count >= 2)]:
                good = support & np.isfinite(truth)
                if good.any():
                    a, b = truth[good].astype(float), pred[good].astype(float)
                    losses[kind].append(np.mean(a * a) - 2 * grid * np.mean(a * b) + grid ** 2 * np.mean(b * b))
        for kind in curves:
            if losses[kind]:
                curves[kind].append(np.mean(losses[kind], axis=0))
    return {str(kind): float(grid[np.argmin(np.mean(values, axis=0))]) if values else 0.
            for kind, values in curves.items()}


def feature_sensitivity(booster, features, context, targets, references, model, config):
    rng = np.random.default_rng(config['seed'])
    p = rng.choice(targets, 2048)
    g = rng.integers(len(features.embedding), size=len(p))
    base, names = features.features(context, p, g, references, model)
    prediction = booster.inplace_predict(base)
    other, _ = features.features(references[0], p, g, references, model)
    context_rms = float(np.sqrt(np.mean((prediction - booster.inplace_predict(other)) ** 2)))
    relation_names = {'raw_correlation', 'adjusted_correlation', 'relation_split_gap', 'reference_correlation',
                      'reference_relation_support', 'relation_shift', 'relation_abs_shift', 'relation_sign_conformity',
                      'functional_context_interaction', 'physical_context_interaction', 'pathway_context_interaction'}
    columns = [j for j, name in enumerate(names) if name in relation_names]
    masked = base.copy(); masked[:, columns] = np.nan
    relation_rms = float(np.sqrt(np.mean((prediction - booster.inplace_predict(masked)) ** 2)))
    return {'context': context, 'model': model, 'pairs': len(p), 'context_swap_prediction_RMS': context_rms,
            'relation_mask_prediction_RMS': relation_rms,
            'interpretation': 'Functional input sensitivity, not causal mechanism validation'}


def evaluate(config, output, metadata, tracked):
    done = output / 'evaluation.json'
    if done.exists():
        return json.loads(done.read_text())
    prepared = output / 'prepared'
    supervision = Supervision(prepared, metadata)
    features = PairFeatures(prepared, metadata['contexts'] + metadata['official_contexts'])
    rows, rounds, sensitivities = [], [], []
    prediction_dir = output / 'predictions'; prediction_dir.mkdir(exist_ok=True)
    for hi, held in enumerate(metadata['contexts']):
        train_contexts = [c for c in metadata['contexts'] if c != held]
        inner_validation = [train_contexts[hi % len(train_contexts)]]
        inner_train = [c for c in train_contexts if c not in inner_validation]
        targets = supervision.target_indices[held]
        truth = supervision.values[held]
        shared_fit = fit_shared(supervision, train_contexts, config)
        for model in config['models']:
            inner, selected = fit(features, supervision, inner_train, inner_validation, inner_train, model, config,
                                  output / 'checkpoints' / f'{held}-{model}-inner', tracked)
            del inner
            booster, _ = fit(features, supervision, train_contexts, [], train_contexts, model, config,
                             output / 'checkpoints' / f'{held}-{model}-outer', tracked, rounds=selected['rounds'])
            rounds.append({'held': held, 'model': model, 'rounds': selected['rounds'], 'inner_validation': inner_validation})
            predictions = predict(booster, features, held, targets, np.arange(len(metadata['genes'])), train_contexts, model)
            np.savez_compressed(prediction_dir / f'{held}-{model}.npz', targets=targets, response=predictions)
            for i, p in enumerate(targets):
                sources = [c for c in train_contexts if p in supervision.rows[c]]
                reserved = target_reserved(metadata['genes'][p], config['unseen_target_percent'])
                rows.append({'context': held, 'target': metadata['genes'][p], 'model': model,
                             'reserved_target': reserved, 'training_target_backgrounds': 0 if reserved else len(sources),
                             **response_metrics(truth[i], predictions[i], supervision.common)})
            sensitivities.append(feature_sensitivity(booster, features, held, targets, train_contexts, model, config))
            del predictions, booster
        for i, p in enumerate(targets):
            prediction, count = shared_response(supervision, p, train_contexts, config['unseen_target_percent'])
            prediction *= np.where(count == 1, shared_fit['1'], shared_fit['2'])
            reserved = target_reserved(metadata['genes'][p], config['unseen_target_percent'])
            source_count = sum(p in supervision.rows[c] for c in train_contexts) if not reserved else 0
            for model, value in [('zero', np.zeros(len(prediction))), ('shared_shrunk', prediction)]:
                rows.append({'context': held, 'target': metadata['genes'][p], 'model': model, 'reserved_target': reserved,
                             'training_target_backgrounds': source_count, **response_metrics(truth[i], value, supervision.common)})
        pd.DataFrame(rows).to_parquet(output / 'evaluation-metrics.partial.parquet', index=False)
    frame = pd.DataFrame(rows)
    frame.to_parquet(output / 'evaluation-metrics.parquet', index=False)
    (output / 'evaluation-metrics.partial.parquet').unlink(missing_ok=True)
    by_context = frame.groupby(['model', 'context']).agg({k: 'mean' for k in frame if k.endswith(('_MSE', '_MAE', '_correlation', '_sign_agreement'))}).reset_index()
    by_context.to_parquet(output / 'evaluation-by-context.parquet', index=False)
    summary = by_context.groupby('model').mean(numeric_only=True).reset_index()
    summary['relative_common_MSE_improvement'] = 1 - summary.common_MSE / summary.common_zero_MSE
    summary['relative_common_MAE_improvement'] = 1 - summary.common_MAE / summary.common_zero_MAE
    summary.to_parquet(output / 'evaluation-summary.parquet', index=False)
    frame.groupby(['model', 'context', 'reserved_target', 'training_target_backgrounds']).mean(numeric_only=True).reset_index().to_parquet(output / 'evaluation-strata.parquet', index=False)
    write_json(output / 'input-sensitivity.json', sensitivities)
    result = {'status': 'completed', 'outer_contexts': metadata['contexts'], 'rounds': rounds,
              'summary': summary.to_dict('records'), 'paired_bootstrap': paired_comparisons(frame, config),
              'scope': 'Retrospective public-context evaluation with disjoint NTC pools; no official labels',
              'final_model_fixed_before_results': 'context_relation'}
    write_json(done, result)
    for row in result['summary']:
        tracked.summary['offline/' + row['model'] + '/relative_MSE_improvement'] = row['relative_common_MSE_improvement']
    return result


def paired_comparisons(frame, config):
    result = []
    rng = np.random.default_rng(config['seed'])
    for candidate, baseline in [('context_pair', 'static_pair'), ('context_relation', 'context_pair'),
                                 ('context_relation', 'shared_shrunk'), ('context_relation', 'zero')]:
        pair = frame.loc[frame.model.isin([candidate, baseline])].pivot(index=['context', 'target'], columns='model', values='common_MSE').dropna()
        delta = pair[candidate] - pair[baseline]
        # Target-cluster bootstrap retains a target's appearances in all contexts.
        target_ids = sorted(set(pair.index.get_level_values('target')))
        code = {p: i for i, p in enumerate(target_ids)}
        codes = np.array([code[p] for p in pair.index.get_level_values('target')])
        contexts = pair.index.get_level_values('context')
        stats = []
        for _ in range(config['bootstrap_repeats']):
            count = np.bincount(rng.integers(len(target_ids), size=len(target_ids)), minlength=len(target_ids))[codes]
            means = [np.average(delta.to_numpy()[contexts == c], weights=count[contexts == c])
                     for c in sorted(set(contexts)) if count[contexts == c].sum()]
            stats.append(np.mean(means))
        result.append({'candidate': candidate, 'baseline': baseline, 'MSE_difference': float(delta.groupby(level='context').mean().mean()),
                       'descriptive_target_cluster_95_interval': np.quantile(stats, [.025, .975]).tolist(),
                       'targets': len(target_ids), 'target_contexts': len(pair)})
    return result


def final_fit(config, output, metadata, evaluation, tracked):
    done = output / 'final-training.json'
    if done.exists():
        return json.loads(done.read_text())
    supervision = Supervision(output / 'prepared', metadata)
    features = PairFeatures(output / 'prepared', metadata['contexts'] + metadata['official_contexts'])
    rounds = max(1, int(np.median([r['rounds'] for r in evaluation['rounds'] if r['model'] == 'context_relation'])))
    booster, record = fit(features, supervision, metadata['contexts'], [], metadata['contexts'], 'context_relation', config,
                          output / 'checkpoints' / 'final-context_relation', tracked, rounds=rounds, release_reserved=True)
    gene_map = {g: i for i, g in enumerate(metadata['genes'])}
    targets = np.array([gene_map[p] for p in metadata['official_targets']])
    all_genes = np.arange(len(metadata['genes']))
    response = np.stack([predict(booster, features, c, targets, all_genes, metadata['contexts'], 'context_relation')
                         for c in metadata['official_contexts']])
    measured = np.any([np.isfinite(x).any(0) for x in supervision.values.values()], axis=0)
    support = np.broadcast_to(measured, response.shape).copy()
    response[~support] = 0
    np.savez_compressed(output / 'model.npz', genes=np.asarray(metadata['genes']), targets=np.asarray(metadata['official_targets']),
                        contexts=np.asarray(metadata['official_contexts']), response=response, response_support=support)
    record.update(status='completed', official_prediction_shape=list(response.shape),
                  genes_without_training_labels=int((~measured).sum()),
                  context_response_pair_RMS={a + '_' + b: float(np.sqrt(np.mean((response[i] - response[j]) ** 2)))
                                            for i, a in enumerate(metadata['official_contexts'])
                                            for j, b in enumerate(metadata['official_contexts']) if j > i})
    write_json(done, record)
    return record
