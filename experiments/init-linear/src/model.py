"""Masked ridge with target indicators and a linear NTC-mean feature map.

Eliminating the diagonal target block makes the exact ridge solution small.
SVD is only an exact reparameterization of the four training NTC vectors.
"""
import json
import numpy as np
from scipy import sparse


def solve_ridge(target, features, response, weights, n_targets, alpha):
    incidence = sparse.csr_matrix((weights, (target, np.arange(len(target)))), shape=(n_targets, len(target)))
    mass = np.bincount(target, weights=weights, minlength=n_targets)
    denominator = mass + alpha
    cross = incidence @ features
    right = incidence @ response
    schur = features.T @ (features * weights[:, None]) + alpha * np.eye(features.shape[1])
    schur -= cross.T @ (cross / denominator[:, None])
    beta = np.linalg.solve(schur, features.T @ (response * weights[:, None]) - cross.T @ (right / denominator[:, None]))
    effect = (right - cross @ beta) / denominator[:, None]
    shared = np.divide(right, mass[:, None], out=np.zeros_like(right), where=mass[:, None] > 0)
    return effect, beta, shared


def context_features(statistics, training_contexts, output_genes):
    common = sorted(set.intersection(*(set(statistics[c]['positions']) for c in training_contexts)))
    means = np.array([statistics[c]['mean'][0, np.searchsorted(statistics[c]['positions'], common)] for c in training_contexts])
    center = means.mean(axis=0)
    scale = max(len(common), 1) ** .5
    # Keep all singular vectors: no rank search and no response-derived representation.
    _, _, basis = np.linalg.svd((means-center)/scale, full_matrices=False)
    return np.asarray(common), center, basis, scale


def task_features(stats, labels, genes, common, center, basis, scale):
    positions = stats['positions']
    ntc = np.zeros(len(genes), dtype=np.float64)
    measured = np.zeros(len(genes), dtype=bool)
    ntc[positions], measured[positions] = stats['mean'][0], True
    # Missing NTC context features are mean-imputed and explicitly diagnosed.
    context = np.where(measured[common], ntc[common], center)
    embedding = ((context-center)/scale) @ basis.T
    x = np.empty((len(labels), len(embedding)+3), dtype=np.float64)
    x[:, 0], x[:, 1:1+len(embedding)] = 1, embedding
    index = {g: i for i, g in enumerate(genes)}
    for i, target in enumerate(labels):
        p = index.get(str(target))
        x[i, -2] = ntc[p] / 5 if p is not None and measured[p] else 0
        x[i, -1] = float(p is not None and measured[p])
    return x


def fit(directory, genes, split, config):
    contexts = split['training_contexts']
    stats = {c: dict(np.load(directory/f'{c}-statistics.npz')) for c in contexts}
    targets = np.asarray(sorted(set(t for s in stats.values() for t in s['labels'][1:])))
    target_map = {t: i for i, t in enumerate(targets)}
    common, center, basis, scale = context_features(stats, contexts, genes)
    feature_args = (genes, common, center, basis, scale)
    features = {c: task_features(stats[c], stats[c]['labels'][1:], *feature_args) for c in contexts}
    dimensions = next(iter(features.values())).shape[1]
    effects = np.zeros((len(targets), len(genes)), np.float32)
    coefficients = np.zeros((dimensions, len(genes)), np.float32)
    shared = np.zeros_like(effects)
    masks = np.zeros((len(contexts), len(genes)), dtype=bool)
    for ci, c in enumerate(contexts):
        masks[ci, stats[c]['positions']] = True
    patterns = np.sum(masks * (2**np.arange(len(contexts)))[:, None], axis=0)
    loss, mass = 0., 0.
    for pattern in sorted(set(patterns)-{0}):
        available = [c for i, c in enumerate(contexts) if pattern & (1 << i)]
        columns = np.flatnonzero(patterns == pattern)
        target = np.concatenate([[target_map[t] for t in stats[c]['labels'][1:]] for c in available])
        x = np.concatenate([features[c] for c in available])
        weights = np.concatenate([stats[c]['counts'][1:] / config['model']['weight_cell_unit'] for c in available])
        for start in range(0, len(columns), config['model']['gene_chunk']):
            col = columns[start:start+config['model']['gene_chunk']]
            response = np.concatenate([stats[c]['mean'][1:, np.searchsorted(stats[c]['positions'], col)] -
                                       stats[c]['mean'][0, np.searchsorted(stats[c]['positions'], col)] for c in available]).astype(np.float64)
            a, beta, pooled = solve_ridge(target, x, response, weights, len(targets), config['model']['alpha'])
            effects[:, col], coefficients[:, col], shared[:, col] = a, beta, pooled
            residual = a[target] + x @ beta - response
            loss += float(np.sum(weights[:, None] * residual**2))
            mass += weights.sum()*len(col)
        print(f'Ridge solved measurement pattern {pattern}: {len(columns)} genes, {len(target)} tasks', flush=True)
    return dict(targets=targets, effects=effects, beta=coefficients, shared=shared,
                common=common, center=center, basis=basis, scale=np.asarray(scale),
                trained_mask=masks.any(axis=0), genes=np.asarray(genes),
                training_mse=np.asarray(loss/mass), training_tasks=np.asarray(sum(len(s['labels'])-1 for s in stats.values())))


def predict(model, stats, targets, arm, source=None):
    genes = model['genes']
    result = np.zeros((len(targets), len(genes)), np.float32)
    index = {str(t): i for i, t in enumerate(model['targets'])}
    if arm == 'zero':
        return result
    if arm == 'source':
        labels = {str(t): i for i, t in enumerate(source['labels'])}
        for i, t in enumerate(targets):
            if t in labels:
                result[i, source['positions']] = source['mean'][labels[t]] - source['mean'][0]
        return result
    if arm == 'linear':
        x = task_features(stats, targets, genes, model['common'], model['center'], model['basis'], float(model['scale']))
        result[:] = x @ model['beta']
    for i, t in enumerate(targets):
        if t in index:
            result[i] += model['effects' if arm == 'linear' else 'shared'][index[t]]
    result[:, ~model['trained_mask']] = 0
    return result
