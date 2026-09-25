"""Training-only task statistics and differentiable prior-predictive supervision."""
import json

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from data import hash_file, logcp, reserved, write_json
from evaluation import task_seed


COMPONENTS = ('response_loss', 'ntc_loss', 'depth_loss')
LOSS_TERMS = ('elbo', *COMPONENTS)


def predictive_losses(target, control, target_second, control_second, observed, mask, depth_scale):
    """Contrast two predictions; anchor controls; calibrate raw library moments.

    All observations/scales are frozen. NB ELBO remains responsible for the full
    conditional count likelihood; these moments alone do not identify a distribution.
    """
    from model import masked_logcp
    pt, pc = masked_logcp(target, mask), masked_logcp(control, mask)
    delta = pt - pc
    residual = (delta - observed['delta']) / observed['response_scale']
    weights = observed['response_weight'] * mask
    response = (F.huber_loss(residual, torch.zeros_like(residual), reduction='none') * weights).sum(-1) / weights.sum(-1)
    error = (pc - observed['ntc']) / observed['ntc_scale']
    ntc = (F.huber_loss(error, torch.zeros_like(error), reduction='none') * mask).sum(-1) / mask.sum(-1)
    # Means are on the native measured axis; CP normalization cannot supervise depth.
    depths = torch.stack([observed['predicted_target_depth'], observed['predicted_ntc_depth']], -1)
    seconds = torch.stack([target_second, control_second], -1)
    stds = (seconds - depths.square()).clamp_min(1e-6).sqrt()
    moments = torch.cat([depths.log1p(), stds.log1p()], -1)
    depth = F.huber_loss((moments - observed['depth']) / depth_scale, torch.zeros_like(moments), reduction='none').mean(-1)
    return {'response_loss': response, 'ntc_loss': ntc, 'depth_loss': depth,
            'response_mse': ((delta - observed['delta']).square() * mask).sum(-1) / mask.sum(-1),
            'zero_response_mse': (observed['delta'].square() * mask).sum(-1) / mask.sum(-1),
            'ntc_mse': ((pc - observed['ntc']).square() * mask).sum(-1) / mask.sum(-1)}


class TaskObjective:
    """Exact observed task mixtures; stratified Monte Carlo predictive mixtures.

    The empirical measure gives each unique cell mass pi_layer / n_layer. Prediction
    draws use the same batch marginal, shared across target/NTC. Monte Carlo is exact
    in expectation before the nonlinear transforms, not an unbiased loss estimator.
    """
    fields = ('delta', 'ntc', 'response_scale', 'ntc_scale', 'response_weight')

    def __init__(self, data, view, contexts, config, cache):
        self.data, self.view, self.config = data, view, config
        self.keys = sorted((c, t) for c, t in data.tasks if c in contexts and not reserved(t, config['unseen_target_percent']))
        self.index = {key: i for i, key in enumerate(self.keys)}
        self.axis = np.flatnonzero(data.common)
        self.mixtures = {}
        for key in self.keys:
            mass = {}
            for (_, batch), weight in data.task_weights(*key).items():
                mass[batch] = mass.get(batch, 0.) + weight
            batches = sorted(mass)
            weights = np.array([mass[b] for b in batches], dtype=np.float64)
            weights /= weights.sum()
            self.mixtures[key] = batches, weights
        self.directory = cache / 'task-statistics'; self.directory.mkdir(exist_ok=True)
        marker = self.directory / 'complete.json'
        identity = {'version': 1, 'keys': self.keys, 'axis': self.axis.tolist(),
                    'scale_floor': config['response_scale_floor'], 'signal_weight_max': config['response_signal_weight_max'],
                    'source': hash_file(data.directory / 'complete.json') if hasattr(data, 'directory') else 'test-fixture'}
        # JSON normalization makes tuple/list identities equivalent on recovery.
        identity = json.loads(json.dumps(identity))
        paths = {k: self.directory / f'{k}.npy' for k in [*self.fields, 'depth']}
        if marker.exists():
            recorded = json.loads(marker.read_text())
            assert recorded['identity'] == identity, 'task_statistics_identity_mismatch'
            assert all(hash_file(paths[k]) == digest for k, digest in recorded['sha256'].items())
        else:
            arrays = {k: np.lib.format.open_memmap(path, mode='w+', dtype=np.float32,
                      shape=(len(self.keys), 4 if k == 'depth' else len(self.axis))) for k, path in paths.items()}
            self._prepare(arrays)
            for array in arrays.values():
                array.flush()
            del arrays
            write_json(marker, {'identity': identity, 'sha256': {k: hash_file(p) for k, p in paths.items()},
                       'statistics': 'all unique training task cells; construct/batch-balanced mass; reference-half NTC',
                       'scale': 'delta-method weighted-cell uncertainty proxy; includes heterogeneity, not calibrated biological replicate SE',
                       'prediction': 'stratified batch Monte Carlo; shared target/NTC Gaussian noise; nonlinear finite-sample bias remains',
                       'held_perturbation_labels_used': False})
        self.arrays = {k: np.load(path, mmap_mode='r') for k, path in paths.items()}
        self.signature = hash_file(marker)
        self.weights = None

    def _prepare(self, arrays):
        controls = {}
        rows = []
        floor = self.config['response_scale_floor']
        for i, key in enumerate(self.keys):
            c, target = key
            groups, mass = self.data.groups[key], self.data.task_weights(*key)
            ids = np.concatenate(list(groups.values()))
            assert len(ids) == len(np.unique(ids)), 'duplicate_observed_task_cells'
            weight = np.concatenate([np.full(len(ids), mass[layer] / len(ids)) for layer, ids in groups.items()])
            weight /= weight.sum()
            x = self.data.read(ids)
            depths = x.sum(1, dtype=np.float64)
            x = x[:, self.axis].astype(np.float64)
            mean = weight @ x
            library = x.sum(1)
            ratio = mean / max(mean.sum(), 1.)
            derivative = 10000 / np.maximum(mean.sum() + 10000 * mean, 1.)
            # Weighted empirical influence variance: each independent row occurs once.
            influence = x - library[:, None] * ratio
            variance = (weight**2) @ (influence**2)
            variance *= derivative**2 / max(1 - np.sum(weight**2), 1e-6)
            batches, probability = self.mixtures[key]
            for batch in batches:
                if (c, batch) in controls:
                    continue
                ref = self.data.control(c, batch, 1).astype(np.float64)
                total = ref.sum(1)
                y = ref[:, self.axis]; lib = y.sum(1)
                controls[c, batch] = dict(mean=y.mean(0), var=y.var(0, ddof=1),
                    cov=((y-y.mean(0)) * (lib-lib.mean())[:, None]).sum(0) / (len(y)-1),
                    libvar=lib.var(ddof=1), n=len(y), depth=total.mean(), second=np.mean(total**2))
            reference = sum(p * controls[c, b]['mean'] for b, p in zip(batches, probability))
            ref_ratio = reference / max(reference.sum(), 1.)
            ref_variance = sum(p*p / controls[c, b]['n'] *
                (controls[c, b]['var'] - 2 * ref_ratio * controls[c, b]['cov'] + ref_ratio**2 * controls[c, b]['libvar'])
                for b, p in zip(batches, probability))
            ref_variance = np.maximum(ref_variance, 0) * (10000 / np.maximum(reference.sum() + 10000 * reference, 1.))**2
            delta = logcp(mean) - logcp(reference)
            arrays['delta'][i], arrays['ntc'][i] = delta, logcp(reference)
            arrays['response_scale'][i] = np.sqrt(variance + ref_variance + floor**2)
            arrays['ntc_scale'][i] = np.sqrt(ref_variance + floor**2)
            arrays['response_weight'][i] = 1 + (self.config['response_signal_weight_max'] - 1) * delta**2 / (delta**2 + variance + ref_variance + floor**2)
            depth = float(weight @ depths)
            depth0 = sum(p * controls[c, b]['depth'] for b, p in zip(batches, probability))
            var = max(float(weight @ (depths-depth)**2), 0)
            var0 = max(sum(p * controls[c, b]['second'] for b, p in zip(batches, probability)) - depth0**2, 0)
            arrays['depth'][i] = np.log1p([depth, depth0, np.sqrt(var), np.sqrt(var0)])
            rows.append({'context': c, 'target': target, 'independent_cells': len(ids), 'effective_cells': 1 / np.sum(weight**2),
                         'layers': len(groups), 'batches': len(batches), 'target_depth': depth, 'ntc_depth': depth0})
            if (i+1) % 1000 == 0 or i+1 == len(self.keys):
                print(json.dumps({'stage': 'task_statistics', 'tasks_done': i+1, 'tasks': len(self.keys)}), flush=True)
        pd.DataFrame(rows).to_parquet(self.directory / 'support.parquet', index=False)

    def batch_specs(self, keys, rng):
        draws = self.config['response_batch_draws']
        specs = []
        for c, target in keys:
            batches, probability = self.mixtures[c, target]
            # Randomized stratification has exactly the correct marginal expected mass.
            positions = (np.arange(draws) + rng.random(draws)) / draws
            indices = np.searchsorted(np.cumsum(probability), positions).clip(max=len(batches)-1)
            specs.extend((c, target, batches[j], 1) for j in indices)
        return specs

    def losses(self, model, keys, rng):
        device = next(model.parameters()).device
        specs = self.batch_specs(keys, rng)
        inputs = self.view.inputs(specs, device)
        ntc_inputs = dict(inputs, target=torch.full_like(inputs['target'], model.genes))
        epsilon = torch.randn(len(specs), self.config['state_components'], self.config['response_latent_samples'],
                              model.d, device=device)
        target, target_second = model.conditional_moments(inputs, epsilon)
        control, control_second = model.conditional_moments(ntc_inputs, epsilon)
        n, draws = len(keys), self.config['response_batch_draws']
        target = target.reshape(n, draws, -1).mean(1)
        control = control.reshape(n, draws, -1).mean(1)
        target_second = target_second.reshape(n, draws).mean(1)
        control_second = control_second.reshape(n, draws).mean(1)
        indices = [self.index[key] for key in keys]
        observed = {k: torch.as_tensor(np.asarray(v[indices]), device=device) for k, v in self.arrays.items()}
        observed['predicted_target_depth'] = target.sum(-1)
        observed['predicted_ntc_depth'] = control.sum(-1)
        axis = torch.as_tensor(self.axis, device=device)
        target, control = target.index_select(1, axis), control.index_select(1, axis)
        return predictive_losses(target, control, target_second, control_second, observed,
                                 torch.ones_like(target), self.config['depth_log_scale'])

    def weighted(self, losses):
        assert self.weights is not None
        return sum(self.weights[k] * losses[k] for k in LOSS_TERMS)

    def calibrate(self, model, contexts, cache, device):
        """Freeze coefficients from training-only gradients; never update parameters."""
        path = cache / 'loss-calibration.json'
        if path.exists():
            saved = json.loads(path.read_text())
            assert saved['statistics_sha256'] == self.signature
            self.weights = saved['weights']
            return
        parameters = [p for name, p in model.named_parameters() if p.requires_grad and not name.startswith(('posterior.', 'encoder.', 'mask_encoder.'))]
        norms = {k: [] for k in LOSS_TERMS}
        module_norms = []
        for context in contexts:
            keys = [key for key in self.keys if key[0] == context]
            rng = np.random.default_rng(task_seed(context, 'loss-calibration'))
            for _ in range(self.config['loss_calibration_batches_per_context']):
                selected = [keys[j] for j in rng.choice(len(keys), self.config['tasks_per_step'], replace=len(keys) < self.config['tasks_per_step'])]
                specs = self.batch_specs(selected, rng)
                ids = []
                for c, t, b, _ in specs:
                    groups = self.data.groups[c, t]; mass = self.data.task_weights(c, t)
                    layers = [layer for layer in groups if layer[1] == b]
                    p = np.array([mass[layer] for layer in layers]); p /= p.sum()
                    layer = layers[rng.choice(len(layers), p=p)]
                    ids.append(rng.choice(groups[layer]))
                x = torch.as_tensor(self.data.read(ids), device=device)
                inputs = self.view.inputs(specs, device)
                perturbed_elbo, _ = model(x, inputs, beta=1.)
                control_ids = [rng.choice(self.data.controls[c, b, 1]) for c, t, b, _ in specs]
                control_x = torch.as_tensor(self.data.read(control_ids), device=device)
                control_inputs = dict(inputs, target=torch.full_like(inputs['target'], model.genes))
                control_elbo, _ = model(control_x, control_inputs, beta=1.)
                fraction = self.config['tasks_per_step'] / (self.config['tasks_per_step'] + 1)
                elbo = fraction * perturbed_elbo + (1-fraction) * control_elbo
                losses = {'elbo': elbo, **{k: v.mean() for k, v in self.losses(model, selected, rng).items() if k in COMPONENTS}}
                row = {'context': context}
                for k, loss in losses.items():
                    gradients = torch.autograd.grad(loss, parameters, allow_unused=True, retain_graph=True)
                    norm = float(torch.sqrt(sum(g.detach().square().sum() for g in gradients if g is not None)))
                    assert np.isfinite(norm), 'nonfinite_calibration_gradient'
                    norms[k].append(norm); row[k] = norm
                module_norms.append(row)
        median = {k: float(np.median(v)) for k, v in norms.items()}
        assert all(v > 1e-10 for v in median.values()), 'disconnected_predictive_objective'
        # Delta is the primary objective with coefficient exactly one. Calibrate
        # auxiliary contributions against its gradients, not against NB magnitude.
        self.weights = {'response_loss': 1., **{
            k: float(np.clip(ratio * median['response_loss'] / median[k],
                            self.config['loss_weight_min'], self.config['loss_weight_max']))
            for k, ratio in self.config['auxiliary_gradient_ratios'].items()}}
        write_json(path, {'weights': self.weights, 'median_gradient_norms': median, 'gradient_samples': module_norms,
                         'initial_relative_gradient_norms': {k: self.weights[k] * median[k] / median['response_loss'] for k in LOSS_TERMS},
                         'statistics_sha256': self.signature, 'optimizer_steps': 0,
                         'method': 'primary delta coefficient 1; auxiliary initial shared-generator gradient norm ratios; frozen',
                         'warning': 'heuristic initialization, not a validation-selected optimum'})
        print(json.dumps({'stage': 'loss_calibration', 'weights': self.weights, 'gradient_norms': median}), flush=True)
