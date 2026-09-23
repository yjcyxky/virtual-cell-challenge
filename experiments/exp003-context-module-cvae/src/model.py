"""Conditional latent count model with adaptive signed module loadings."""
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def mlp(inputs, hidden, outputs, zero=False):
    net = nn.Sequential(nn.Linear(inputs, hidden), nn.SiLU(), nn.Linear(hidden, outputs))
    if zero:
        nn.init.zeros_(net[-1].weight); nn.init.zeros_(net[-1].bias)
    return net


def log_nb(x, mu, theta):
    mu = mu.clamp(1e-5, 1e6); theta = theta.clamp(0.05, 1e4)
    return (torch.lgamma(x + theta) - torch.lgamma(theta) - torch.lgamma(x + 1)
            + theta * (torch.log(theta) - torch.log(theta + mu))
            + x * (torch.log(mu) - torch.log(theta + mu)))


def masked_logcp(x, mask):
    x = x * mask
    return torch.log1p(10000 * x / x.sum(-1, keepdim=True).clamp_min(1))


class ModuleCVAE(nn.Module):
    def __init__(self, genes, membership, strength, projection, origin, common, seen_targets, config):
        super().__init__()
        self.config, self.variant, self.genes = config, config['variant'], genes
        g, m = membership.shape; d = config['state_dimensions']; h = config['hidden_dimensions']; r = config['residual_dimensions']
        self.d, self.m = d, m
        c = 2 * d + 5
        mask = np.asarray(membership, np.float32)
        if self.variant == 'no_prior':
            anchor = np.ones_like(mask)
            target_prior = np.zeros_like(mask)
        else:
            anchor = config['module_off_support_weight'] + mask * np.asarray(strength)[None]
            target_prior = mask / np.maximum(mask.sum(1, keepdims=True), 1)
        # All controls have identical dense parameters; the prior changes their
        # soft connection scale, never removes the residual or its capacity.
        reference_scale = np.ones_like(mask) if self.variant == 'no_prior' else config['module_off_support_weight'] + mask
        anchor /= np.maximum(np.sqrt(np.mean(reference_scale ** 2, axis=1, keepdims=True)), 1e-5)
        self.register_buffer('anchor', torch.tensor(anchor, dtype=torch.float32))
        self.register_buffer('target_prior', torch.tensor(np.vstack([target_prior, np.zeros(m)]), dtype=torch.float32))
        self.register_buffer('projection', torch.tensor(projection, dtype=torch.float32))
        self.register_buffer('origin', torch.tensor(origin, dtype=torch.float32))
        self.register_buffer('common', torch.tensor(common, dtype=torch.float32))
        self.register_buffer('seen', torch.tensor(np.r_[seen_targets, False], dtype=torch.float32))
        self.register_buffer('trained_readouts', torch.ones(g, dtype=torch.bool))
        self.target_embedding = nn.Embedding(g + 1, 24)
        nn.init.normal_(self.target_embedding.weight, std=0.1)
        self.target_encoder = mlp(m + 24, h, 32)
        self.prior = mlp(c + 32 + d, h, 2 * d + 1, zero=True)
        self.posterior = mlp(2 * d + c + 32 + 2, h, 2 * d, zero=True)
        self.encoder = nn.Linear(g, d, bias=False)
        self.mask_encoder = nn.Linear(g, d, bias=False)
        nn.init.normal_(self.encoder.weight, std=0.01 / math.sqrt(g))
        nn.init.normal_(self.mask_encoder.weight, std=0.01 / math.sqrt(g))
        self.baseline_modules = mlp(d + c, h, m)
        self.response_modules = mlp(d + c + 32, h, m, zero=True)
        self.module_interaction = nn.Parameter(torch.zeros(m, m))
        self.loadings = nn.Parameter(torch.randn(g, m) * (0.05 / math.sqrt(m)))
        self.gene_gate = nn.Parameter(torch.randn(g, r) * 0.1)
        self.state_gate = mlp(c + d, h, r + m, zero=True)
        self.residual_loadings = nn.Parameter(torch.randn(g, r) * (0.03 / math.sqrt(r)))
        self.baseline_residual = mlp(d + c, h, r)
        self.response_residual = mlp(d + c + 32, h, r, zero=True)
        self.dispersion = mlp(c + 32, h, r, zero=True)
        self.dispersion_loadings = nn.Parameter(torch.randn(g, r) * 0.05)
        self.on_target = mlp(c + 32, h, 1, zero=True)

    def condition(self, batch):
        index = batch['target'].clamp_max(self.genes - 1)
        target_mean = batch['base'].gather(1, index[:, None])
        target_mean = torch.log1p(target_mean) * (batch['target'] < self.genes)[:, None]
        target_measured = batch['mask'].gather(1, index[:, None]) * (batch['target'] < self.genes)[:, None]
        context = torch.cat([batch['condition'], target_mean, target_measured], -1)
        embedding = self.target_embedding(batch['target']) * self.seen[batch['target'], None]
        target = self.target_encoder(torch.cat([self.target_prior[batch['target']], embedding], -1))
        active = (batch['target'] < self.genes).float()[:, None]
        return context, target * active, active

    def response_condition(self, context):
        return torch.zeros_like(context) if self.variant == 'no_context' else context

    def prior_parameters(self, batch):
        context, target, active = self.condition(batch)
        centers = batch['centers']; n, k, d = centers.shape
        c = self.response_condition(context)
        conditioning_centers = centers * 0 if self.variant == 'no_context' else centers
        inputs = torch.cat([c[:, None].expand(-1, k, -1), target[:, None].expand(-1, k, -1), conditioning_centers], -1)
        shift, logscale, logits = torch.split(self.prior(inputs), [d, d, 1], -1)
        if self.variant in ['no_state', 'no_context', 'no_module']:
            shift, logscale, logits = shift * 0, logscale * 0, logits * 0
        # The NTC prior remains an empirical state distribution; only perturbation
        # shifts/reweights it. No observed perturbed cell is supplied at generation.
        mean = centers + 2 * torch.tanh(shift) * active[:, None]
        std = batch['scales'] * torch.exp(torch.tanh(logscale) * active[:, None])
        weights = F.softmax(torch.log(batch['probability'].clamp_min(1e-5)) + logits[..., 0] * active, -1)
        return mean, std.clamp_min(0.05), weights

    def sample_prior(self, batch):
        mean, std, weights = self.prior_parameters(batch)
        component = torch.multinomial(weights, 1).squeeze(1)
        rows = torch.arange(len(component), device=component.device)
        return mean[rows, component] + std[rows, component] * torch.randn_like(mean[:, 0])

    def conditional_mean(self, batch, samples_per_component=4):
        """Differentiable mixture expectation, without interpolating categorical states."""
        mean, std, weights = self.prior_parameters(batch)
        n, k, d = mean.shape
        z = mean[:, :, None] + std[:, :, None] * torch.randn(n, k, samples_per_component, d, device=mean.device)
        repeated = {key: value.repeat_interleave(k * samples_per_component, dim=0) for key, value in batch.items()}
        mu, _, _ = self.decode(z.reshape(-1, d), repeated)
        return (mu.reshape(n, k, samples_per_component, -1).mean(2) * weights[..., None]).sum(1)

    def decode(self, z, batch):
        context, target, active = self.condition(batch)
        response_context = self.response_condition(context)
        response_z = torch.zeros_like(z) if self.variant in ['no_state', 'no_context'] else z
        baseline_context = torch.cat([context[:, :-2], torch.zeros_like(context[:, -2:])], -1)
        base_features = torch.cat([z, baseline_context], -1)
        response_features = torch.cat([response_z, response_context, target], -1)
        baseline = self.baseline_modules(base_features)
        response = self.response_modules(response_features) * active
        if self.variant == 'no_module':
            response = response * 0
        interaction = torch.tanh(self.module_interaction) / math.sqrt(self.m)
        def module_decode(values, c, state):
            gate = self.state_gate(torch.cat([c, state], -1))
            gene_factors, module_factors = gate.split([self.config['residual_dimensions'], self.m], -1)
            gene_gate = 1 + 0.5 * torch.tanh(gene_factors @ self.gene_gate.T)
            module_gate = 1 + 0.5 * torch.tanh(module_factors)
            values = values + torch.tanh(values @ interaction)
            return ((values * module_gate) @ (self.loadings * self.anchor).T) * gene_gate
        module_base = module_decode(baseline, baseline_context, z)
        module_response = module_decode(response, response_context, response_z)
        residual_base = self.baseline_residual(base_features) @ self.residual_loadings.T
        residual_response = (self.response_residual(response_features) * active) @ self.residual_loadings.T
        if self.variant == 'no_residual':
            residual_base, residual_response = residual_base * 0, residual_response * 0
        on_target = self.on_target(torch.cat([response_context, target], -1)) * active
        target_delta = torch.zeros_like(batch['base']).scatter(1, batch['target'].clamp_max(self.genes - 1)[:, None], on_target)
        # Separate bounded baseline/response terms prevent ablated state effects
        # leaking back through a shared nonlinear transform of their sum.
        response_raw = module_response + residual_response + target_delta
        log_ratio = 2 * torch.tanh((module_base + residual_base) / 2) + 4 * torch.tanh(response_raw / 4)
        # An additive induction route permits expression absent from feature NTC;
        # it shares learned module/residual effects and adds no control-only capacity.
        birth = (F.softplus(response_raw - 4) - F.softplus(torch.full_like(response_raw, -4))).clamp_min(0) * active
        mu = ((batch['base'] + 0.02) * torch.exp(log_ratio) + birth).clamp(1e-5, 1e6)
        dispersion = self.dispersion(torch.cat([context, target], -1)) @ self.dispersion_loadings.T
        theta = batch['theta'] * torch.exp(2 * torch.tanh(dispersion / 2))
        return mu, theta.clamp(0.05, 1000), {'module_rms': module_response.square().mean().sqrt(),
                                         'residual_rms': residual_response.square().mean().sqrt(), 'induction_mean': birth.mean()}

    def forward(self, x, batch, beta=1.0):
        context, target, active = self.condition(batch)
        normalized = masked_logcp(x, batch['mask'])
        projected = masked_logcp(x, self.common) @ self.projection - self.origin
        features = self.encoder(normalized * batch['mask']) + self.mask_encoder(batch['mask'] - 1)
        depth = torch.log1p((x * batch['mask']).sum(1, keepdim=True)) / 10
        coverage = batch['mask'].mean(1, keepdim=True)
        correction, logstd = self.posterior(torch.cat([projected, features, context, target, depth, coverage], -1)).chunk(2, -1)
        qmean = projected + torch.tanh(correction)
        qstd = torch.exp(logstd.clamp(-3, 2))
        z = qmean + qstd * torch.randn_like(qmean)
        mu, theta, diagnostics = self.decode(z, batch)
        likelihood = -(log_nb(x, mu, theta) * batch['mask']).sum(-1) / batch['mask'].sum(-1).clamp_min(1)
        means, stds, weights = self.prior_parameters(batch)
        logp_component = (-0.5 * ((z[:, None] - means) / stds).square() - stds.log() - 0.5 * math.log(2 * math.pi)).sum(-1)
        logp = torch.logsumexp(logp_component + weights.log(), -1)
        logq = (-0.5 * ((z - qmean) / qstd).square() - qstd.log() - 0.5 * math.log(2 * math.pi)).sum(-1)
        kl = (logq - logp) / batch['mask'].sum(-1).clamp_min(1)
        return (likelihood + beta * kl).mean(), {'nll': likelihood.mean(), 'kl_per_gene': kl.mean(), **diagnostics}

    @torch.no_grad()
    def generate(self, batch):
        z = self.sample_prior(batch)
        mu, theta, _ = self.decode(z, batch)
        # Gamma-Poisson is an exact NB draw. Masked genes remain explicitly unobserved.
        rate = torch.distributions.Gamma(theta, theta / mu.clamp_min(1e-5)).sample()
        return torch.poisson(rate) * batch['mask']
