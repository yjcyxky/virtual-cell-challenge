"""Five complete LODO fits with coverage-clock optimization and official validation."""
import json
import os
import random
import time

import numpy as np
import pandas as pd
import torch

from data import BalancedSampler, reserved, write_json, hash_file
from evaluation import evaluate_context, shared_responses, summarize, task_seed
from model import ModuleCVAE
from objectives import TaskObjective, COMPONENTS, LOSS_TERMS
from priors import degree_matched_random, fold_evidence
from state import FoldView
from stopping import assess_stopping, validation_due
from official import OfficialValidation

def random_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_random(state):
    random.setstate(state['python']); np.random.set_state(state['numpy']); torch.set_rng_state(state['torch'])
    if state['cuda'] is not None:
        torch.cuda.set_rng_state_all(state['cuda'])


def save_model_state(path, state):
    """Commit checkpoint bytes before replacing the last valid model/state."""
    temporary = path.with_suffix('.tmp')
    with temporary.open('wb') as stream:
        torch.save(state, stream)
        stream.flush(); os.fsync(stream.fileno())
    temporary.replace(path)
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def save_checkpoint(path, model, optimizer, scheduler, sampler, progress):
    state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
             'sampler': sampler.state_dict(), 'random': random_state(), 'progress': progress}
    save_model_state(path, state)


def load_checkpoint(path, model, optimizer, scheduler, sampler, device):
    state = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
    scheduler.load_state_dict(state['scheduler']); sampler.load_state_dict(state['sampler'])
    # RNG tensors must stay on CPU even when other checkpoint tensors use CUDA.
    state['random']['torch'] = state['random']['torch'].cpu()
    if state['random']['cuda'] is not None:
        state['random']['cuda'] = [v.cpu() for v in state['random']['cuda']]
    restore_random(state['random'])
    return state['progress']


def construct_model(data, view, contexts, config, prior_directory, folder):
    archive = np.load(prior_directory / 'modules.npz')
    modules = json.loads((prior_directory / 'modules.json').read_text())
    families = [v['family'] for v in modules]
    if config['variant'] == 'random_prior':
        membership = archive['random']
        null = degree_matched_random(membership, families, config['data_seed'] + 1)
    else:
        membership, null = archive['true'], archive['random']
    strength_path = folder / 'prior-strength.npy'
    if strength_path.exists():
        strength = np.load(strength_path)
    else:
        strength = fold_evidence(data, contexts, membership, null, config, folder)
    if config['variant'] == 'no_prior':
        strength = np.ones_like(strength)
    seen = np.zeros(len(data.genes), bool)
    for c, target in data.tasks:
        if c in contexts and not reserved(target, config['unseen_target_percent']):
            seen[data.gene_index[target]] = True
    model = ModuleCVAE(len(data.genes), membership, strength, view.projection, view.origin,
                       data.common, seen, config)
    model.trained_readouts.copy_(torch.tensor(np.logical_or.reduce([data.masks[c] for c in contexts])))
    return model



def train_step(model, data, view, sampler, optimizer, scheduler, config, step, device, objective):
    groups = sampler.sample(config['tasks_per_step'], config['cells_per_task'])
    specifications = [(c, t, b, len(ids)) for c, t, b, ids in groups]
    selected = [ids for c, t, b, ids in groups]
    c, _, b, _ = groups[0]
    ntc_ids = sampler.rng.choice(data.controls[(c, b, 1)], config['cells_per_task'], replace=True)
    specifications.append((c, '__NTC__', b, len(ntc_ids))); selected.append(ntc_ids)
    x = torch.as_tensor(data.read(np.concatenate(selected)), device=device)
    inputs = view.inputs(specifications, device)
    beta = min(1.0, (step + 1) / config['kl_warmup_steps'])
    optimizer.zero_grad(set_to_none=True)
    elbo, logs = model(x, inputs, beta=beta)
    predictive = {k: v.mean() for k, v in objective.losses(model, [(c, t) for c, t, b, ids in groups], sampler.rng).items()}
    losses = {'elbo': elbo, **predictive}
    loss = objective.weighted(losses)
    if not torch.isfinite(loss):
        raise FloatingPointError('nonfinite_training_loss')
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10, error_if_nonfinite=True)
    optimizer.step(); scheduler.step()
    return {'loss': float(loss.detach()), **{k: float(v.detach()) for k, v in losses.items()},
            **{'weighted_' + k: float((objective.weights[k] * losses[k]).detach()) for k in LOSS_TERMS},
            'gradient_norm': float(norm),
            'kl_beta': beta, 'learning_rate': scheduler.get_last_lr()[0], **{k: float(v.detach()) for k, v in logs.items()}}


def monitor_plan(data, contexts, config, cache):
    """Fixed stratified MC panel: all targets, training-weighted layer draws per target."""
    path = cache / 'monitor-sampling.parquet'
    rows = []
    n = config['cells_per_task']
    draws = config['monitor_layers_per_target']
    assert isinstance(draws, int) and draws > 0
    coverage = {}
    for context in contexts:
        targets = sorted(t for c, t in data.tasks if c == context and not reserved(t, config['unseen_target_percent']))
        total_layers, sampled_layers = 0, set()
        for target in targets:
            weights = data.task_weights(context, target)
            keys = sorted(data.groups[(context, target)])
            probability = np.asarray([weights[key] for key in keys], dtype=float)
            assert np.all(probability > 0) and abs(probability.sum() - 1) < 1e-6
            probability /= probability.sum()
            layer_rng = np.random.default_rng(task_seed(context, f'loss-monitor|{target}|{config["data_seed"]}'))
            total_layers += len(keys)
            for draw, selected in enumerate(layer_rng.choice(len(keys), draws, replace=True, p=probability)):
                construct, batch = keys[selected]
                ids = data.groups[(context, target)][(construct, batch)]
                sampled_layers.add((target, construct, batch))
                seed = task_seed(context, f'loss-monitor-cells|{target}|{draw}|{config["data_seed"]}')
                rng = np.random.default_rng(seed)
                rows.append({'context': context, 'target': target, 'construct': construct, 'batch': batch,
                             'draw': draw, 'seed': seed, 'weight': 1 / (draws * len(targets)),
                             'cells': rng.choice(ids, n, replace=True).tolist(),
                             'controls': rng.choice(data.controls[(context, batch, 1)], n, replace=True).tolist()})
        coverage[context] = {'targets': len(targets), 'layer_draws': draws * len(targets),
                             'unique_sampled_layers': len(sampled_layers), 'eligible_layers': total_layers}
    frame = pd.DataFrame(rows)
    if path.exists():
        previous = pd.read_parquet(path)
        # Recompute identities, including the fixed observed-cell samples, on recovery.
        assert previous.drop(columns=['cells', 'controls']).equals(frame.drop(columns=['cells', 'controls']))
        assert all(np.array_equal(a, b) for field in ['cells', 'controls'] for a, b in zip(previous[field], frame[field]))
    else:
        frame.to_parquet(path, index=False)
    write_json(cache / 'monitor-scope.json', {'method': 'NB: fixed training-weighted layer samples; predictive: all tasks with fixed stratified batch draws',
               'layers_per_target': draws, 'cells_per_layer': n, 'contexts': coverage,
               'interpretation': 'Monte Carlo training-objective estimate; not exhaustive layer evaluation',
               'predictive_batch_draws_per_task': config['response_batch_draws'], 'sampling_sha256': hash_file(path)})
    return rows


@torch.no_grad()
def monitor_training(model, data, view, plan, contexts, config, step, objective):
    """Monitor NB per layer and predictive losses per complete training task."""
    result = {c: dict.fromkeys(['loss', 'nll', 'kl_per_gene', *COMPONENTS, 'response_mse', 'zero_response_mse', 'ntc_mse'], 0.) for c in contexts}
    device = next(model.parameters()).device
    beta = min(1., step / config['kl_warmup_steps'])
    pert_weight = config['tasks_per_step'] / (config['tasks_per_step'] + 1)
    def layer_losses(cells, controls, inputs, ctrl_inputs):
        pert_loss, pert_logs = model(cells, inputs, beta=beta)
        ctrl_loss, ctrl_logs = model(controls, ctrl_inputs, beta=beta)
        return {'loss': objective.weights['elbo'] * (pert_weight * pert_loss + (1 - pert_weight) * ctrl_loss),
                **{k: pert_weight * pert_logs[k] + (1 - pert_weight) * ctrl_logs[k] for k in ['nll', 'kl_per_gene']}}
    # Vectorize independent layer evaluations without pooling their observed means
    # or changing their sample sizes. Ordered 64-layer blocks fix latent RNG streams.
    batched_losses = torch.vmap(layer_losses, randomness='different')
    was_training = model.training
    try:
        model.eval()
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
            for start in range(0, len(plan), 64):
                rows = plan[start:start + 64]; size = len(rows); n = config['cells_per_task']
                torch.manual_seed(rows[0]['seed'])
                cells = torch.as_tensor(data.read(np.concatenate([r['cells'] for r in rows])), device=device).reshape(size, n, -1)
                controls = torch.as_tensor(data.read(np.concatenate([r['controls'] for r in rows])), device=device).reshape(size, n, -1)
                specs = [(r['context'], r['target'], r['batch'], n) for r in rows]
                ctrl_specs = [(c, '__NTC__', b, count) for c, t, b, count in specs]
                inputs = {k: v.reshape(size, n, *v.shape[1:]) for k, v in view.inputs(specs, device).items()}
                ctrl_inputs = {k: v.reshape(size, n, *v.shape[1:]) for k, v in view.inputs(ctrl_specs, device).items()}
                values = {k: v.cpu().numpy() for k, v in batched_losses(cells, controls, inputs, ctrl_inputs).items()}
                assert all(np.isfinite(v).all() for v in values.values()), 'nonfinite_monitor_loss'
                for j, row in enumerate(rows):
                    for k, value in values.items():
                        result[row['context']][k] += row['weight'] * float(value[j])
                if start // 10000 != (start + size) // 10000 or start + size == len(plan):
                    print(json.dumps({'stage': 'training_monitor', 'layers_done': start + size, 'layers': len(plan)}), flush=True)
            for context in contexts:
                keys = [key for key in objective.keys if key[0] == context]
                for start in range(0, len(keys), 16):
                    selected = keys[start:start+16]
                    seed = task_seed(context, f'predictive-monitor|{start}|{config["data_seed"]}')
                    torch.manual_seed(seed)
                    values = objective.losses(model, selected, np.random.default_rng(seed))
                    assert all(torch.isfinite(v).all() for v in values.values()), 'nonfinite_predictive_monitor'
                    for k, value in values.items():
                        result[context][k] += float(value.sum()) / len(keys)
                result[context]['loss'] += sum(objective.weights[k] * result[context][k] for k in COMPONENTS)
                result[context]['response_skill_vs_zero'] = 1 - result[context]['response_mse'] / max(result[context]['zero_response_mse'], 1e-12)
                print(json.dumps({'stage': 'predictive_monitor', 'context': context, 'tasks': len(keys)}), flush=True)
    finally:
        model.train(was_training)
    return result


def fit(data, contexts, validation_context, config, output, key, prior_directory, tracked):
    assert validation_context not in contexts
    cache = output / 'cache' / key; cache.mkdir(parents=True, exist_ok=True)
    checkpoints = output / 'checkpoints'; checkpoints.mkdir(exist_ok=True)
    last, best = checkpoints / f'{key}-last.pt', checkpoints / f'{key}-best.pt'
    device = torch.device('cuda')
    fit_seed = config['seed'] + sum(ord(c) for c in key)
    random.seed(fit_seed); np.random.seed(fit_seed); torch.manual_seed(fit_seed)
    view = FoldView(data, contexts, config, cache)
    model = construct_model(data, view, contexts, config, prior_directory, cache).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    # Identity schedule: LR remains fixed before/after every update and recovery.
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=1)
    sampler = BalancedSampler(data, contexts, fit_seed, config['unseen_target_percent'])
    objective = TaskObjective(data, view, contexts, config, cache)
    plan = monitor_plan(data, contexts, config, cache)
    evaluator = OfficialValidation(data, view, validation_context, config, cache, output / 'predictions' / key)
    write_json(cache / 'split.json', {'training_targets_by_context': sampler.targets,
               'validation_context': validation_context, 'validation_targets': evaluator.targets,
               'new_context_input': 'feature-half NTC only', 'model_variant': config['variant'],
               'monitor_plan_sha256': hash_file(cache / 'monitor-sampling.parquet'),
               'task_statistics_sha256': objective.signature,
               'global_target_reservation_percent': config['unseen_target_percent'],
               'external_prior_enters_generator': config['variant'] != 'no_prior'})
    progress = {'next_step': 0, 'history': [], 'best_score': None, 'best_cycle': None,
                'patience_reference': None, 'stale': 0, 'complete': False, 'resume_count': 0,
                'training_contexts': list(contexts), 'validation_context': validation_context,
                'cycle_log_sums': {}, 'cycle_log_count': 0}
    if last.exists():
        progress = load_checkpoint(last, model, optimizer, scheduler, sampler, device)
        assert progress['training_contexts'] == list(contexts) and progress['validation_context'] == validation_context
        progress['resume_count'] += 1
    if not progress['complete']:
        # Anchor preparation and all checks belong to this run, before optimization.
        # They cannot consume training RNG, including on a recovery after cache preparation.
        rng = random_state()
        try:
            objective.calibrate(model, contexts, cache, device)
            evaluator.prepare()
        finally:
            restore_random(rng)
        if 'loss_weights' in progress:
            assert progress['loss_weights'] == objective.weights, 'changed_frozen_loss_weights'
        progress['loss_weights'] = objective.weights
        tracked.log({f'loss_weights/{key}/{k}': v for k, v in objective.weights.items()})
        if not last.exists():
            save_checkpoint(last, model, optimizer, scheduler, sampler, progress)
        marker = output / 'cache' / 'optimization-started.json'
        if not marker.exists():
            write_json(marker, {'fit': key, 'pipeline_commit': config['pipeline_commit'], 'seed': config['seed']})
        initial_path = cache / 'initial-monitor.json'
        if not initial_path.exists():
            assert progress['next_step'] == 0, 'missing_initial_predictive_monitor'
            initial = monitor_training(model, data, view, plan, contexts, config, config['kl_warmup_steps'], objective)
            write_json(initial_path, initial)
            tracked.log({f'monitor/{key}/{c}/{k}': v for c, row in initial.items() for k, v in row.items()} | {f'train/{key}/cycle': 0})
        model.train(); rolling = []; started = time.monotonic()
        while not progress['complete']:
            step = progress['next_step']
            old_cycle = sampler.completed_cycles
            logs = train_step(model, data, view, sampler, optimizer, scheduler, config, step, device, objective)
            progress['next_step'] = step + 1; rolling.append(logs)
            for k, value in logs.items():
                progress['cycle_log_sums'][k] = progress['cycle_log_sums'].get(k, 0.) + value
            progress['cycle_log_count'] += 1
            if (step + 1) % 32 == 0:
                tracked.log({f'train/{key}/{k}': float(np.mean([r[k] for r in rolling])) for k in logs}
                            | {f'train/{key}/step': step + 1, f'train/{key}/coverage_time': sampler.coverage_time,
                               'resource/cuda_peak_allocated_bytes': torch.cuda.max_memory_allocated()})
                if (step + 1) % 512 == 0:
                    print(json.dumps({'stage': 'optimization', 'fit': key, 'step': step + 1,
                                      'coverage_time': sampler.coverage_time,
                                      'learning_rate': scheduler.get_last_lr()[0],
                                      'loss': float(np.mean([r['loss'] for r in rolling])),
                                      'response_loss': float(np.mean([r['response_loss'] for r in rolling])),
                                      'response_mse': float(np.mean([r['response_mse'] for r in rolling])),
                                      'zero_response_mse': float(np.mean([r['zero_response_mse'] for r in rolling]))}), flush=True)
                rolling = []
            if sampler.completed_cycles == old_cycle:
                if (step + 1) % config['checkpoint_every_steps'] == 0:
                    save_checkpoint(last, model, optimizer, scheduler, sampler, progress)
                continue
            cycle = sampler.completed_cycles
            rng = random_state()
            try:
                monitor_started = time.monotonic()
                training_losses = monitor_training(model, data, view, plan, contexts, config, step + 1, objective)
                monitor_seconds = time.monotonic() - monitor_started
                validation_started = time.monotonic()
                validation = evaluator.evaluate(model, cycle) if validation_due(cycle, config) else None
                validation_seconds = time.monotonic() - validation_started if validation is not None else 0.
            finally:
                restore_random(rng)
                model.train()
            score = None if validation is None else validation['mean_score']
            point = {'cycle': cycle, 'step': step + 1, 'learning_rate': scheduler.get_last_lr()[0],
                     'coverage': sampler.last_cycle_coverage, 'training_monitor': training_losses,
                     'validation_score': score, 'validation_score_sd': None if validation is None else validation['score_sd'],
                     'validation_seed_scores': None if validation is None else validation['seed_scores'],
                     'optimization_average': {k: v / progress['cycle_log_count'] for k, v in progress['cycle_log_sums'].items()}}
            if score is not None and (progress['best_score'] is None or score > progress['best_score']):
                evaluator.promote_best(cycle)
                save_model_state(best, {'model': model.state_dict(), 'config': config, 'training_contexts': contexts,
                                       'cycle': cycle, 'step': step + 1, 'validation_score': score,
                                       'sampled_training_targets': {c: sorted(v) for c, v in sampler.seen.items()}})
                progress['best_score'], progress['best_cycle'] = score, cycle
            progress['history'].append(point)
            decision = assess_stopping(progress['history'], progress['patience_reference'], progress['stale'], contexts, config)
            progress['patience_reference'], progress['stale'] = decision['reference'], decision['stale']
            point['stopping'] = decision
            progress['cycle_log_sums'], progress['cycle_log_count'] = {}, 0
            progress['coverage'] = {c: {'seen': len(sampler.seen[c]), 'eligible': len(sampler.targets[c])} for c in contexts}
            if decision['stop']:
                progress['complete'], progress['stop_reason'] = True, 'normal_sufficiency_early_stopping'
            # Publish after monitoring, any scheduled validation and selection agree.
            save_checkpoint(last, model, optimizer, scheduler, sampler, progress)
            write_json(cache / 'training.json', progress)
            write_json(cache / f'cycle-{cycle:04d}.json', point)
            timing = {'monitor_seconds': monitor_seconds, 'validation_seconds': validation_seconds}
            write_json(cache / f'cycle-{cycle:04d}-timing.json', timing)
            validation_logs = {} if validation is None else {
                f'validation/{key}/score': score, f'validation/{key}/score_sd': validation['score_sd']}
            tracked.log({**validation_logs, f'validation/{key}/performed': int(validation is not None),
                         f'train/{key}/cycle': cycle, f'train/{key}/stale': progress['stale'],
                         **{f'resource/{key}/{k}': v for k, v in timing.items()},
                         f'train/{key}/elapsed_seconds_since_resume': time.monotonic() - started,
                         **{f'monitor/{key}/{c}/{k}': v for c, row in training_losses.items() for k, v in row.items()}})
            print(json.dumps({'stage': 'cycle', 'fit': key, **point, **timing, 'stop': progress['complete']}), flush=True)
            if validation is not None:
                evaluator.clear_transient()
    model.load_state_dict(torch.load(best, map_location=device, weights_only=False)['model'])
    model.eval()
    evaluator.record_prediction_files()
    shared, global_shared = shared_responses(data, view, contexts, config, cache)
    return model, view, progress, shared, global_shared


def experiment(data, config, output, priors, tracked):
    assert len(config['contexts']) == 5 and config['unseen_target_percent'] == 0
    frames, folds = [], {}
    for held in config['contexts']:
        training_contexts = [c for c in config['contexts'] if c != held]
        key = 'holdout-' + held
        prediction_dir = output / 'predictions' / key; prediction_dir.mkdir(parents=True, exist_ok=True)
        if (prediction_dir / 'complete.json').exists():
            folds[held] = json.loads((prediction_dir / 'complete.json').read_text())
            frames.append(pd.read_parquet(prediction_dir / 'metrics.parquet'))
            continue
        print(json.dumps({'stage': 'fit_start', 'fit': key, 'training': training_contexts, 'validation': held}), flush=True)
        model, view, progress, shared, global_shared = fit(data, training_contexts, held, config, output, key, priors, tracked)
        frame = evaluate_context(model, data, view, held, config, prediction_dir, shared, global_shared)
        trained_targets = {t for c, t in data.tasks if c in training_contexts}
        frame['target_seen_in_training'] = frame.target.isin(trained_targets)
        frame['variant'], frame['seed'] = config['variant'], config['seed']
        frame.to_parquet(prediction_dir / 'metrics.parquet', index=False)
        frames.append(frame)
        selected = progress['history'][progress['best_cycle'] - 1]
        folds[held] = {'best_cycle': progress['best_cycle'], 'last_cycle': progress['history'][-1]['cycle'],
                      'best_score': progress['best_score'], 'best_seed_scores': selected['validation_seed_scores'],
                      'last_score': progress['history'][-1]['validation_score'],
                      'training_contexts': training_contexts, 'validation_context': held,
                      'stop_reason': progress['stop_reason'], 'coverage': progress['coverage'],
                      'best_before_stopping_window': progress['best_cycle'] < config['stopping_start_cycle']}
        write_json(prediction_dir / 'complete.json', folds[held])
        tracked.log({f'validation/{held}/selected_score': progress['best_score']})
        del model, view; torch.cuda.empty_cache()
    result = summarize(frames, config, output)
    result.update(status='trained_and_locally_evaluated', folds=folds, models_trained=5,
                  mean_validation_score=float(np.mean([f['best_score'] for f in folds.values()])),
                  official_submission=False, validation_used_for_selection=True,
                  historical_comparison='Different four/one split, full target exposure and official selection; not a paired reproduction of #32.')
    write_json(output / 'metrics.json', result)
    return result
