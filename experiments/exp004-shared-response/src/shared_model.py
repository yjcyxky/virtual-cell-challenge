"""Shared biological gene features and nonseparable conditional NB readouts."""
import hashlib
import gzip
import json

import numpy as np
from scipy import sparse
import torch
from torch import nn
from torch.nn import functional as F

from data import ROOT, hash_file, write_json
from model import ModuleCVAE, mlp
from training import construct_model as reference_model


def gene_features(genes, priors, config, output):
    """Public annotations only; no expression labels or gene-ID trainable rows."""
    path = output / 'gene-features.npz'
    marker = output / 'gene-features.json'
    source = {name: hash_file(priors / name) for name in ['embedding.npy', 'physical.npz', 'functional.npz', 'pathways.npz', 'modules.npz']}
    source['gene_order'] = hashlib.sha256('\n'.join(genes).encode()).hexdigest()
    source['goa_human.gaf.gz'] = hash_file(ROOT / 'data/raw/networks/goa_human.gaf.gz')
    source['prior-strength.npy'] = hash_file(output / 'prior-strength.npy')
    identity = dict(version=1, source=source, seed=config['data_seed'], prior_mode=config['prior_mode'])
    if marker.exists():
        record = json.loads(marker.read_text())
        assert record['identity'] == identity and record['sha256'] == hash_file(path)
        return np.load(path)['features']
    embedding = np.load(priors / 'embedding.npy')
    graph = sparse.load_npz(priors / 'physical.npz')
    degree = np.asarray(graph.sum(1)).ravel()
    physical = (graph @ embedding) / np.maximum(degree[:, None], 1)
    pathways = sparse.load_npz(priors / 'pathways.npz')
    rng = np.random.default_rng(config['data_seed'])
    projection = rng.choice([-1., 1.], size=(pathways.shape[1], 32)).astype(np.float32)
    reactome = (pathways @ projection) / np.sqrt(np.maximum(np.asarray(pathways.sum(1)), 1))
    # Stable signed hashing of complete GO memberships; NOT annotations excluded.
    go = np.zeros((len(genes), 32), np.float32)
    index = {g: i for i, g in enumerate(genes)}
    terms = set()
    with gzip.open(ROOT / 'data/raw/networks/goa_human.gaf.gz', 'rt') as stream:
        for line in stream:
            if line.startswith('!'): continue
            fields = line.rstrip('\n').split('\t')
            if len(fields) < 15 or 'NOT' in fields[3].split('|') or fields[2] not in index: continue
            pair = (index[fields[2]], fields[4])
            if pair in terms: continue
            terms.add(pair)
            digest = hashlib.sha256(fields[4].encode()).digest()
            go[pair[0]] += np.frombuffer(digest, dtype=np.uint8).astype(np.float32) % 2 * 2 - 1
    go_counts = np.bincount([i for i, term in terms], minlength=len(genes))
    go /= np.sqrt(np.maximum(go_counts[:, None], 1))
    membership = np.load(priors / 'modules.npz')['true']
    features = np.concatenate([embedding, physical, reactome, go, membership], axis=1).astype(np.float32)
    features /= np.maximum(np.sqrt(np.mean(features**2, axis=0, keepdims=True)), 1e-6)
    features = np.clip(features, -5, 5)
    # Training NTC evidence downweights unsupported module annotations after scaling.
    features[:, -membership.shape[1]:] *= np.load(output / 'prior-strength.npy')[None]
    permutation = np.arange(len(genes))
    if config['prior_mode'] == 'random':
        # Preserve annotation-availability and degree-bin composition, not exact edges.
        functional = sparse.load_npz(priors / 'functional.npz')
        degree_bin = np.floor(np.log2(1 + np.diff(functional.indptr))).astype(int)
        bins = np.stack([degree_bin, (go_counts > 0), np.asarray(pathways.sum(1)).ravel() > 0, np.any(features, axis=1)], axis=1)
        _, labels = np.unique(bins, axis=0, return_inverse=True)
        for label in np.unique(labels):
            ids = np.flatnonzero(labels == label); permutation[ids] = rng.permutation(ids)
        features = features[permutation]
    elif config['prior_mode'] == 'none':
        features = np.zeros_like(features)
    assert np.isfinite(features).all()
    np.savez_compressed(path, features=features, permutation=permutation)
    write_json(marker, {'identity': identity, 'sha256': hash_file(path), 'feature_dimensions': features.shape[1],
                       'genes_with_nonzero_features': int(np.any(features, axis=1).sum()),
                       'features': 'functional spectral basis, physical diffusion, complete Reactome projection, complete GO signed hashing, module membership',
                       'random_control': 'row permutation within functional-degree/annotation-availability strata; not exact edge-degree rewiring',
                       'held_perturbation_expression_used': False, 'raw_gene_symbols_are_not_trainable_features': True})
    return features


class SharedResponseCVAE(ModuleCVAE):
    def __init__(self, *args, features, **kwargs):
        super().__init__(*args, **kwargs)
        width = self.config['gene_feature_dimensions']; rank = self.config['relation_rank']
        self.register_buffer('gene_features', torch.as_tensor(features, dtype=torch.float32))
        self.feature_encoder = mlp(features.shape[1], width, width)
        self.shared_loadings = nn.Linear(width, self.m, bias=False)
        self.shared_residual = nn.Linear(width, self.config['residual_dimensions'], bias=False)
        self.shared_dispersion = nn.Linear(width, self.config['residual_dimensions'], bias=False)
        self.shared_gene_gate = nn.Linear(width, self.config['residual_dimensions'], bias=False)
        self.shared_target = nn.Linear(width, 24, bias=False) if self.config['shared_target'] else None
        self.relation_gene = nn.Linear(width, rank, bias=False)
        self.relation_module = nn.Parameter(torch.randn(self.m, rank) * (0.1 / np.sqrt(self.m)))
        self.relation_context = mlp(3*self.d + 5, self.config['hidden_dimensions'], rank, zero=True)
        # These free gene-specific generator rows would invalidate readout extrapolation.
        for name in ['loadings', 'residual_loadings', 'dispersion_loadings', 'gene_gate']:
            delattr(self, name)
        for head in [self.shared_loadings, self.shared_residual, self.shared_dispersion, self.shared_gene_gate, self.relation_gene]:
            nn.init.normal_(head.weight, std=0.05 / np.sqrt(width))

    def condition(self, batch):
        index = batch['target'].clamp_max(self.genes-1)
        active = (batch['target'] < self.genes).float()[:, None]
        mean = torch.log1p(batch['base'].gather(1, index[:, None])) * active
        measured = batch['mask'].gather(1, index[:, None]) * active
        context = torch.cat([batch['condition'], mean, measured], -1)
        embedding = self.target_embedding(batch['target']) * self.seen[batch['target'], None]
        if self.shared_target is not None:
            embedding = embedding + self.shared_target(self.feature_encoder(self.gene_features[index])) * active
        target = self.target_encoder(torch.cat([self.target_prior[batch['target']], embedding], -1))
        return context, target * active, active

    @staticmethod
    def relation_decode(values, base, left, right, coefficients):
        return values @ base.T + ((values @ right) * coefficients) @ left.T

    def decode(self, z, batch):
        context, target, active = self.condition(batch)
        response_context = self.response_condition(context)
        baseline_context = torch.cat([context[:, :-2], torch.zeros_like(context[:, -2:])], -1)
        response_z = torch.zeros_like(z) if self.variant in ['no_state', 'no_context'] else z
        base_features = torch.cat([z, baseline_context], -1)
        response_features = torch.cat([response_z, response_context, target], -1)
        baseline = self.baseline_modules(base_features)
        response = self.response_modules(response_features) * active
        encoded = self.feature_encoder(self.gene_features)
        loadings = self.shared_loadings(encoded)
        residual = self.shared_residual(encoded)
        dispersion = self.shared_dispersion(encoded)
        gene_gate = self.shared_gene_gate(encoded)
        relation_left = self.relation_gene(encoded)
        interaction = torch.tanh(self.module_interaction) / np.sqrt(self.m)
        def module_decode(values, c, state):
            values = values + torch.tanh(values @ interaction)
            # Keep the old separable adaptation in A; B adds a controlled new operator.
            factors, modules = self.state_gate(torch.cat([c, state], -1)).split([self.config['residual_dimensions'], self.m], -1)
            mode = self.config['relation_mode']
            if mode == 'off':
                coefficients = torch.zeros_like(state[:, :1]).expand(-1, self.config['relation_rank'])
            else:
                inputs = torch.cat([torch.cat([c[:, :-2], torch.zeros_like(c[:, -2:])], -1), state], -1)
                if mode == 'constant': inputs = torch.zeros_like(inputs)
                coefficients = self.config['relation_scale'] * torch.tanh(self.relation_context(inputs))
            output = self.relation_decode(values * (1+0.5*torch.tanh(modules)), loadings,
                                          relation_left, self.relation_module, coefficients)
            return output * (1+0.5*torch.tanh(factors @ gene_gate.T))
        module_base = module_decode(baseline, baseline_context, z)
        module_response = module_decode(response, response_context, response_z)
        residual_base = self.baseline_residual(base_features) @ residual.T
        residual_response = (self.response_residual(response_features) * active) @ residual.T
        on_target = self.on_target(torch.cat([response_context, target], -1)) * active
        target_delta = torch.zeros_like(batch['base']).scatter(1, batch['target'].clamp_max(self.genes-1)[:, None], on_target)
        response_raw = module_response + residual_response + target_delta
        log_ratio = 2*torch.tanh((module_base+residual_base)/2) + 4*torch.tanh(response_raw/4)
        birth = (F.softplus(response_raw-4)-F.softplus(torch.full_like(response_raw,-4))).clamp_min(0)*active
        mu = ((batch['base']+.02)*torch.exp(log_ratio)+birth).clamp(1e-5,1e6)
        adjust = self.dispersion(torch.cat([response_context,target],-1)) @ dispersion.T
        theta = batch['theta'] * torch.exp(2*torch.tanh(adjust/2))
        return mu, theta.clamp(.05,1000), {'module_rms':module_response.square().mean().sqrt(),
            'residual_rms':residual_response.square().mean().sqrt(),'induction_mean':birth.mean()}


def construct(data, view, contexts, config, prior_directory, folder):
    assert config['architecture'] in ['reference', 'shared']
    assert config['prior_mode'] in ['true', 'random', 'none']
    assert config['relation_mode'] in ['off', 'constant', 'conditional']
    assert config['response_auxiliary'] in ['none', 'amplitude', 'specificity']
    if config['architecture'] == 'reference':
        return reference_model(data, view, contexts, config, prior_directory, folder)
    # Reuse the audited source/strength construction, then instantiate shared parameters.
    reference = reference_model(data, view, contexts, config, prior_directory, folder)
    membership = np.load(prior_directory/'modules.npz')['true']
    strength = np.load(folder/'prior-strength.npy')
    features = gene_features(data.genes, prior_directory, config, folder)
    if config['prior_mode'] != 'true':
        membership = np.zeros_like(membership) if config['prior_mode']=='none' else np.load(prior_directory/'modules.npz')['random']
    model = SharedResponseCVAE(len(data.genes), membership, strength, view.projection, view.origin,
        data.common, reference.seen[:-1].cpu().numpy(), config, features=features)
    model.trained_readouts.copy_(reference.trained_readouts)
    return model
