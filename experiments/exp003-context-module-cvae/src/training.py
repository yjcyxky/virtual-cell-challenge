"""Complete resumable training trials, with training-only priors and outer holdouts."""
import json
import random
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from data import EXPERIMENT, BalancedSampler, reserved, write_json, hash_file, data_spec_hash
from evaluation import evaluate_context, shared_responses, summarize, task_seed
from model import ModuleCVAE, masked_logcp
from priors import degree_matched_random, fold_evidence
from state import FoldView


def random_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_random(state):
    random.setstate(state['python']); np.random.set_state(state['numpy']); torch.set_rng_state(state['torch'])
    if state['cuda'] is not None:
        torch.cuda.set_rng_state_all(state['cuda'])


def save_checkpoint(path, model, optimizer, scheduler, sampler, progress):
    state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
             'sampler': sampler.state_dict(), 'random': random_state(), 'progress': progress}
    temporary = path.with_suffix('.tmp')
    torch.save(state, temporary); temporary.replace(path)


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


def reuse_preprocessing(config, cache, key, contexts):
    if not config.get('reuse_run'):
        return
    source_run = EXPERIMENT / 'outputs' / config['reuse_run']
    source_config = yaml.safe_load((source_run / 'config.yaml').read_text())
    assert source_config['pipeline_commit'] == config['pipeline_commit'], 'preprocessing_reuse_requires_same_code'
    assert source_config['uv_lock_sha256'] == config['uv_lock_sha256']
    assert data_spec_hash(source_config) == data_spec_hash(config)
    for field in ['state_dimensions', 'state_components', 'modules_per_family', 'minimum_module_genes', 'maximum_module_genes', 'network_minimum_score', 'validation_tasks']:
        assert source_config[field] == config[field]
    source = source_run / 'cache' / key
    split = json.loads((source / 'split.json').read_text())
    assert list(split['training_targets_by_context']) == list(contexts), 'preprocessing_fold_mismatch'
    files = ['state.npz', 'shared-response.npz']
    source_kind = 'random' if source_config['variant'] == 'random_prior' else 'true'
    kind = 'random' if config['variant'] == 'random_prior' else 'true'
    if kind == source_kind:
        files += ['prior-strength.npy', 'prior-evidence.parquet', 'evidence-scope.json']
    identities = []
    for name in files:
        original, destination = source / name, cache / name
        digest = hash_file(original)
        if destination.exists():
            assert hash_file(destination) == digest
        elif name == 'shared-response.npz':
            destination.symlink_to(original.resolve())
        else:
            shutil.copyfile(original, destination)
        identities.append({'file': name, 'source': str(original), 'sha256': digest})
    write_json(cache / 'reused-preprocessing.json', {'source_run': config['reuse_run'],
               'source_commit': source_config['pipeline_commit'], 'same_training_contexts': list(contexts),
               'source_artifact': json.loads((source_run / 'artifact.json').read_text()), 'files': identities,
               'model_optimizer_or_training_rng_reused': False})


def train_step(model, data, view, sampler, optimizer, scheduler, config, step, device):
    groups = sampler.sample(config['tasks_per_step'], config['cells_per_task'])
    specifications = [(c, t, b, len(ids)) for c, t, b, ids in groups]
    selected = [ids for c, t, b, ids in groups]
    # Independent reference NTC cells anchor the learned count generator at no intervention.
    c, _, b, _ = groups[0]
    ntc_ids = sampler.rng.choice(data.controls[(c, b, 1)], config['cells_per_task'], replace=True)
    specifications.append((c, '__NTC__', b, len(ntc_ids))); selected.append(ntc_ids)
    x = torch.as_tensor(data.read(np.concatenate(selected)), device=device)
    inputs = view.inputs(specifications, device)
    beta = min(1.0, (step + 1) / (config['kl_warmup_epochs'] * config['steps_per_epoch']))
    optimizer.zero_grad(set_to_none=True)
    loss, logs = model(x, inputs, beta=beta)
    t, n = config['tasks_per_step'], config['cells_per_task']
    task_inputs = {key: value[:t*n:n] for key, value in inputs.items()}
    predicted_mean = model.conditional_mean(task_inputs)
    observed_mean = x[:t*n].reshape(t, n, -1).mean(1)
    reference = torch.as_tensor(np.stack([view.controls[(c, b)]['reference'] for c, target, b, ids in groups]), device=device)
    ref = masked_logcp(reference, model.common)
    response_pred = masked_logcp(predicted_mean, model.common) - ref
    response_true = masked_logcp(observed_mean, model.common) - ref
    response_loss = ((response_pred - response_true).square() * model.common).sum(-1).mean() / model.common.sum()
    loss = loss + config['response_weight'] * response_loss
    if not torch.isfinite(loss):
        raise FloatingPointError('nonfinite_training_loss')
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10, error_if_nonfinite=True)
    optimizer.step(); scheduler.step()
    return {'loss': float(loss.detach()), 'response_loss': float(response_loss.detach()), 'gradient_norm': float(norm),
            'kl_beta': beta, 'learning_rate': scheduler.get_last_lr()[0], **{k: float(v.detach()) for k, v in logs.items()}}


def fit(data, contexts, validation_context, config, output, key, prior_directory, tracked, epochs=None):
    cache = output / 'cache' / key; cache.mkdir(parents=True, exist_ok=True)
    reuse_preprocessing(config, cache, key, contexts)
    checkpoints = output / 'checkpoints'; checkpoints.mkdir(exist_ok=True)
    last, best = checkpoints / f'{key}-last.pt', checkpoints / f'{key}-best.pt'
    device = torch.device('cuda')
    fit_seed = config['seed'] + sum(ord(c) for c in key)
    random.seed(fit_seed); np.random.seed(fit_seed); torch.manual_seed(fit_seed)
    view = FoldView(data, contexts, config, cache)
    model = construct_model(data, view, contexts, config, prior_directory, cache).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    maximum_epochs = config['epochs'] if epochs is None else epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=maximum_epochs * config['steps_per_epoch'], eta_min=config['learning_rate'] * 0.1)
    sampler = BalancedSampler(data, contexts, fit_seed, config['unseen_target_percent'])
    validation_targets = [] if validation_context is None else sorted(
        (t for c, t in data.tasks if c == validation_context and not reserved(t, config['unseen_target_percent'])),
        key=lambda t: task_seed(validation_context, t))[:config['validation_tasks']]
    write_json(cache / 'split.json', {'training_targets_by_context': sampler.targets,
               'inner_validation_context': validation_context, 'inner_validation_targets': validation_targets,
               'globally_reserved_targets': sorted({t for c, t in data.tasks if reserved(t, config['unseen_target_percent'])}),
               'new_context_input': 'feature-half NTC only', 'model_variant': config['variant'],
               'external_prior_enters_generator': config['variant'] != 'no_prior'})
    progress = {'next_step': 0, 'history': [], 'best_score': None, 'best_epoch': None, 'stale': 0,
                'complete': False, 'resume_count': 0, 'training_contexts': list(contexts), 'validation_context': validation_context,
                'maximum_epochs': maximum_epochs, 'epoch_log_sums': {}, 'epoch_log_count': 0}
    if last.exists():
        progress = load_checkpoint(last, model, optimizer, scheduler, sampler, device)
        assert progress['training_contexts'] == list(contexts) and progress['maximum_epochs'] == maximum_epochs
        progress['resume_count'] += 1
    shared, global_shared = shared_responses(data, view, contexts, config, cache)
    if not progress['complete']:
        model.train(); epoch_logs = []
        start_time = time.monotonic()
        marker = output / 'cache' / 'optimization-started.json'
        if not marker.exists():
            write_json(marker, {'fit': key, 'pipeline_commit': config['pipeline_commit'], 'seed': config['seed']})
        for step in range(progress['next_step'], maximum_epochs * config['steps_per_epoch']):
            logs = train_step(model, data, view, sampler, optimizer, scheduler, config, step, device)
            epoch_logs.append(logs); progress['next_step'] = step + 1
            for k, v in logs.items():
                progress['epoch_log_sums'][k] = progress['epoch_log_sums'].get(k, 0) + v
            progress['epoch_log_count'] += 1
            if (step + 1) % 32 == 0:
                mean_logs = {k: float(np.mean([r[k] for r in epoch_logs[-32:]])) for k in logs}
                tracked.log({f'train/{key}/{k}': v for k, v in mean_logs.items()} |
                            {f'train/{key}/step': step + 1, f'train/{key}/resume_count': progress['resume_count'],
                             'resource/cuda_peak_allocated_bytes': torch.cuda.max_memory_allocated()})
            if (step + 1) % config['checkpoint_every_steps'] == 0:
                save_checkpoint(last, model, optimizer, scheduler, sampler, progress)
            if (step + 1) % config['steps_per_epoch']:
                continue
            epoch = (step + 1) // config['steps_per_epoch']
            entry = {'epoch': epoch, 'step': step + 1, **{k: v / progress['epoch_log_count'] for k, v in progress['epoch_log_sums'].items()}}
            if validation_context is not None:
                rng = random_state()
                try:
                    val = evaluate_context(model, data, view, validation_context, config, cache, shared, global_shared, validation=True)
                finally:
                    restore_random(rng)
                score = float(val.pseudobulk_response_mse.mean())
                entry.update(validation_pseudobulk_mse=score, validation_response_mse=float(val.response_mse.mean()),
                             validation_energy=float(val.energy_distance.mean()), validation_ntc_energy=float(val.ntc_energy_distance.mean()))
                if progress['best_score'] is None or score < progress['best_score'] - 1e-6:
                    progress['best_score'], progress['best_epoch'], progress['stale'] = score, epoch, 0
                    torch.save({'model': model.state_dict(), 'config': config, 'training_contexts': contexts, 'epoch': epoch}, best)
                else:
                    progress['stale'] += 1
            else:
                progress['best_epoch'] = epoch
                torch.save({'model': model.state_dict(), 'config': config, 'training_contexts': contexts, 'epoch': epoch}, best)
            progress['history'].append(entry)
            progress['epoch_log_sums'], progress['epoch_log_count'] = {}, 0
            progress['coverage'] = {c: {'seen': len(sampler.seen[c]), 'eligible': len(sampler.targets[c])} for c in contexts}
            should_stop = epoch == maximum_epochs or (validation_context is not None and epoch >= config['minimum_epochs'] and progress['stale'] >= config['patience'])
            if should_stop:
                progress['complete'] = True
                progress['stop_reason'] = 'maximum_epochs' if epoch == maximum_epochs else 'normal_early_stopping'
            save_checkpoint(last, model, optimizer, scheduler, sampler, progress)
            write_json(cache / 'training.json', progress)
            tracked.log({f'validation/{key}/{k}': v for k, v in entry.items()} |
                        {f'train/{key}/elapsed_seconds_since_resume': time.monotonic() - start_time})
            print(json.dumps({'stage': 'epoch', 'fit': key, **entry, 'coverage': progress['coverage'], 'stop': should_stop}), flush=True)
            epoch_logs = []; model.train()
            if should_stop:
                break
    model.load_state_dict(torch.load(best, map_location=device, weights_only=False)['model'])
    model.eval()
    return model, view, progress, shared, global_shared


def experiment(data, config, output, priors, tracked):
    frames, best_epochs = [], []
    for index, held in enumerate(config['contexts']):
        inner = config['contexts'][(index + 1) % len(config['contexts'])]
        training = [c for c in config['contexts'] if c not in (held, inner)]
        key = 'holdout-' + held
        prediction_dir = output / 'predictions' / key; prediction_dir.mkdir(parents=True, exist_ok=True)
        if (prediction_dir / 'complete.json').exists():
            metadata = json.loads((prediction_dir / 'complete.json').read_text())
            frames.append(pd.read_parquet(prediction_dir / 'metrics.parquet')); best_epochs.append(metadata['best_epoch'])
            continue
        print(json.dumps({'stage': 'fit_start', 'fit': key, 'training': training, 'inner_validation': inner, 'outer_test': held}), flush=True)
        model, view, progress, shared, global_shared = fit(data, training, inner, config, output, key, priors, tracked)
        frame = evaluate_context(model, data, view, held, config, prediction_dir, shared, global_shared)
        trained_targets = {t for c, t in data.tasks if c in training and not reserved(t, config['unseen_target_percent'])}
        frame['target_seen_in_training'] = frame.target.isin(trained_targets)
        frame['variant'], frame['seed'] = config['variant'], config['seed']
        frame.to_parquet(prediction_dir / 'metrics.parquet', index=False)
        frames.append(frame); best_epochs.append(progress['best_epoch'])
        write_json(prediction_dir / 'complete.json', {'best_epoch': progress['best_epoch'], 'training_contexts': training,
                   'inner_validation': inner, 'outer_context': held, 'stop_reason': progress['stop_reason']})
        tracked.log({f'test/{held}/{k}': float(v) for k, v in frame.select_dtypes(include=np.number).mean().items()})
        del model, view; torch.cuda.empty_cache()
    result = summarize(frames, config, output)
    epochs = max(1, int(np.median(best_epochs)))
    model, view, progress, _, _ = fit(data, config['contexts'], None, config, output, 'final', priors, tracked, epochs=epochs)
    result['final_training'] = {'epochs': epochs, 'stop_reason': progress['stop_reason'], 'coverage': progress['coverage'],
                                'reserved_targets_remain_unseen': True}
    result['status'] = 'trained_and_locally_evaluated'
    write_json(output / 'metrics.json', result)
    tracked.log({f'result/{k}': v for k, v in result.items() if isinstance(v, (int, float))})
    return result
