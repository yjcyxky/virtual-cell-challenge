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
from model import ModuleCVAE, masked_logcp
from priors import degree_matched_random, fold_evidence
from state import FoldView
from stopping import CoverageCosine, assess_stopping
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



def response_objective(model, inputs, observed_mean, reference):
    predicted = model.conditional_mean(inputs)
    ref = masked_logcp(reference, model.common)
    delta = (masked_logcp(predicted, model.common) - ref) - (masked_logcp(observed_mean, model.common) - ref)
    return (delta.square() * model.common).sum(-1).mean() / model.common.sum()


def train_step(model, data, view, sampler, optimizer, scheduler, config, step, device):
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
    loss, logs = model(x, inputs, beta=beta)
    t, n = config['tasks_per_step'], config['cells_per_task']
    task_inputs = {key: value[:t*n:n] for key, value in inputs.items()}
    observed_mean = x[:t*n].reshape(t, n, -1).mean(1)
    reference = torch.as_tensor(np.stack([view.controls[(c, b)]['reference'] for c, target, b, ids in groups]), device=device)
    response_loss = response_objective(model, task_inputs, observed_mean, reference)
    loss = loss + config['response_weight'] * response_loss
    if not torch.isfinite(loss):
        raise FloatingPointError('nonfinite_training_loss')
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10, error_if_nonfinite=True)
    optimizer.step(); scheduler.step(sampler.coverage_time)
    return {'loss': float(loss.detach()), 'response_loss': float(response_loss.detach()), 'gradient_norm': float(norm),
            'kl_beta': beta, 'learning_rate': scheduler.get_last_lr()[0], **{k: float(v.detach()) for k, v in logs.items()}}


def monitor_plan(data, contexts, config, cache):
    """Enumerate every target and construct/batch layer, using the training sampling law."""
    path = cache / 'monitor-sampling.parquet'
    rows = []
    n = config['cells_per_task']
    for context in contexts:
        targets = sorted(t for c, t in data.tasks if c == context and not reserved(t, config['unseen_target_percent']))
        for target in targets:
            weights = data.task_weights(context, target)
            for (construct, batch), ids in sorted(data.groups[(context, target)].items()):
                seed = task_seed(context, 'loss-monitor|' + target + '|' + str(construct) + '|' + str(batch))
                rng = np.random.default_rng(seed)
                rows.append({'context': context, 'target': target, 'construct': construct, 'batch': batch,
                             'seed': seed, 'weight': weights[(construct, batch)] / len(targets),
                             'cells': rng.choice(ids, n, replace=True).tolist(),
                             'controls': rng.choice(data.controls[(context, batch, 1)], n, replace=True).tolist()})
    frame = pd.DataFrame(rows)
    if path.exists():
        previous = pd.read_parquet(path)
        # Recompute identities, including the fixed observed-cell samples, on recovery.
        assert previous.drop(columns=['cells', 'controls']).equals(frame.drop(columns=['cells', 'controls']))
        assert all(np.array_equal(a, b) for field in ['cells', 'controls'] for a, b in zip(previous[field], frame[field]))
    else:
        frame.to_parquet(path, index=False)
    return rows


@torch.no_grad()
def monitor_training(model, data, view, plan, contexts, config, step):
    """Fixed MC evaluation of the SAME expectation, for all tasks/layers, no held labels.

    Enumerating layers integrates their exact training weights. Each layer uses the
    original cells-per-task sample size for the response loss. Evaluate perturbed and
    NTC terms separately to preserve the original 4:1 mixture even at a tail batch.
    """
    result = {c: dict.fromkeys(['loss', 'response_loss', 'nll', 'kl_per_gene'], 0.) for c in contexts}
    device = next(model.parameters()).device
    beta = min(1., step / config['kl_warmup_steps'])
    pert_weight = config['tasks_per_step'] / (config['tasks_per_step'] + 1)
    was_training = model.training
    try:
        model.eval()
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
            for i, row in enumerate(plan):
                torch.manual_seed(row['seed'])
                c, t, b = row['context'], row['target'], row['batch']
                cells = torch.as_tensor(data.read(row['cells']), device=device)
                controls = torch.as_tensor(data.read(row['controls']), device=device)
                inputs = view.inputs([(c, t, b, len(cells))], device)
                ctrl_inputs = view.inputs([(c, '__NTC__', b, len(controls))], device)
                pert_loss, pert_logs = model(cells, inputs, beta=beta)
                ctrl_loss, ctrl_logs = model(controls, ctrl_inputs, beta=beta)
                reference = torch.as_tensor(view.controls[(c, b)]['reference'][None], device=device)
                response = response_objective(model, {k: v[:1] for k, v in inputs.items()}, cells.mean(0, keepdim=True), reference)
                total = pert_weight * pert_loss + (1 - pert_weight) * ctrl_loss + config['response_weight'] * response
                values = {'loss': float(total), 'response_loss': float(response),
                          **{k: float(pert_weight * pert_logs[k] + (1 - pert_weight) * ctrl_logs[k]) for k in ['nll', 'kl_per_gene']}}
                assert all(np.isfinite(v) for v in values.values()), 'nonfinite_monitor_loss'
                for k, value in values.items():
                    result[c][k] += row['weight'] * value
                if (i + 1) % 1000 == 0:
                    print(json.dumps({'stage': 'training_monitor', 'layers_done': i + 1, 'layers': len(plan)}), flush=True)
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
    scheduler = CoverageCosine(optimizer, config['learning_rate'], config['minimum_learning_rate'], config['decay_cycles'])
    sampler = BalancedSampler(data, contexts, fit_seed, config['unseen_target_percent'])
    plan = monitor_plan(data, contexts, config, cache)
    evaluator = OfficialValidation(data, view, validation_context, config, cache, output / 'predictions' / key)
    write_json(cache / 'split.json', {'training_targets_by_context': sampler.targets,
               'validation_context': validation_context, 'validation_targets': evaluator.targets,
               'new_context_input': 'feature-half NTC only', 'model_variant': config['variant'],
               'monitor_plan_sha256': hash_file(cache / 'monitor-sampling.parquet'),
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
            evaluator.prepare()
        finally:
            restore_random(rng)
        if not last.exists():
            save_checkpoint(last, model, optimizer, scheduler, sampler, progress)
        marker = output / 'cache' / 'optimization-started.json'
        if not marker.exists():
            write_json(marker, {'fit': key, 'pipeline_commit': config['pipeline_commit'], 'seed': config['seed']})
        model.train(); rolling = []; started = time.monotonic()
        while not progress['complete']:
            step = progress['next_step']
            old_cycle = sampler.completed_cycles
            logs = train_step(model, data, view, sampler, optimizer, scheduler, config, step, device)
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
                                      'loss': float(np.mean([r['loss'] for r in rolling]))}), flush=True)
                rolling = []
            if sampler.completed_cycles == old_cycle:
                if (step + 1) % config['checkpoint_every_steps'] == 0:
                    save_checkpoint(last, model, optimizer, scheduler, sampler, progress)
                continue
            cycle = sampler.completed_cycles
            rng = random_state()
            try:
                training_losses = monitor_training(model, data, view, plan, contexts, config, step + 1)
                validation = evaluator.evaluate(model, cycle)
            finally:
                restore_random(rng)
                model.train()
            score = validation['mean_score']
            point = {'cycle': cycle, 'step': step + 1, 'learning_rate': scheduler.get_last_lr()[0],
                     'coverage': sampler.last_cycle_coverage, 'training_monitor': training_losses,
                     'validation_score': score, 'validation_score_sd': validation['score_sd'],
                     'validation_seed_scores': validation['seed_scores'],
                     'optimization_average': {k: v / progress['cycle_log_count'] for k, v in progress['cycle_log_sums'].items()}}
            if progress['best_score'] is None or score > progress['best_score']:
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
            # Publish only after loss/official evaluation, best prediction and stop state agree.
            save_checkpoint(last, model, optimizer, scheduler, sampler, progress)
            write_json(cache / 'training.json', progress)
            write_json(cache / f'cycle-{cycle:04d}.json', point)
            tracked.log({f'validation/{key}/score': score, f'validation/{key}/score_sd': validation['score_sd'],
                         f'train/{key}/cycle': cycle, f'train/{key}/stale': progress['stale'],
                         f'train/{key}/elapsed_seconds_since_resume': time.monotonic() - started,
                         **{f'monitor/{key}/{c}/{k}': v for c, row in training_losses.items() for k, v in row.items()}})
            print(json.dumps({'stage': 'cycle', 'fit': key, **point, 'stop': progress['complete']}), flush=True)
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
                      'best_before_decay_complete': progress['best_cycle'] < config['decay_cycles']}
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
