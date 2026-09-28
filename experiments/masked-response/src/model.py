"""Shared observed responses with masked, fixed-basis completion of missing entries."""
import runtime  # Configure immutable data/representation providers first.
import numpy as np
from scipy import sparse
from data import write_json
from representation import pathway_incidence, randomize_incidence, randomized_pca


def shared_responses(statistics, contexts, genes, weight_cell_unit, gene_chunk):
    """Recompute the original cell-weighted shared estimator and its support mask."""
    targets = np.asarray(sorted({t for c in contexts for t in statistics[c]['labels'][1:]}))
    lookup = {t: i for i, t in enumerate(targets)}
    masks = np.zeros((len(contexts), len(genes)), dtype=bool)
    for i, c in enumerate(contexts):
        masks[i, statistics[c]['positions']] = True
    patterns = np.sum(masks * (2**np.arange(len(contexts)))[:, None], axis=0)
    shared = np.zeros((len(targets), len(genes)), dtype=np.float32)
    observed = np.zeros(shared.shape, dtype=bool)
    loss = total_mass = 0.
    for pattern in sorted(set(patterns) - {0}):
        available = [c for i, c in enumerate(contexts) if pattern & (1 << i)]
        columns = np.flatnonzero(patterns == pattern)
        target = np.concatenate([[lookup[t] for t in statistics[c]['labels'][1:]] for c in available])
        weights = np.concatenate([statistics[c]['counts'][1:] / weight_cell_unit for c in available])
        incidence = sparse.csr_matrix((weights, (target, np.arange(len(target)))),
                                      shape=(len(targets), len(target)))
        mass = np.bincount(target, weights=weights, minlength=len(targets))
        observed[:, columns] = (mass > 0)[:, None]
        for start in range(0, len(columns), gene_chunk):
            col = columns[start:start+gene_chunk]
            response = np.concatenate([
                statistics[c]['mean'][1:, np.searchsorted(statistics[c]['positions'], col)] -
                statistics[c]['mean'][0, np.searchsorted(statistics[c]['positions'], col)]
                for c in available]).astype(np.float64)
            right = incidence @ response
            pooled = np.divide(right, mass[:, None], out=np.zeros_like(right), where=mass[:, None] > 0)
            shared[:, col] = pooled
            loss += float(np.sum(weights[:, None] * (pooled[target] - response)**2))
            total_mass += float(mass.sum() * len(col))
        print(f'Shared response pattern {pattern}: {len(columns)} genes', flush=True)
    return dict(targets=targets, genes=np.asarray(genes), shared=shared, observed=observed,
                trained_mask=masks.any(axis=0), training_mse=np.asarray(loss/total_mass),
                training_tasks=np.asarray(sum(len(statistics[c]['labels'])-1 for c in contexts)))


def response_basis(shared, observed, incidence, specification):
    """Fit rank-r bases on the same output genes; never use missing entries as labels.

    PCA here is explicitly uncentered response SVD. Only completely observed
    training-target rows enter its fit. Zero is a meaningful response origin.
    """
    kind, rank = specification['kind'], specification['dimensions']
    complete = observed.all(axis=1)
    audit = {'complete_training_targets': int(complete.sum()), 'kind': kind}
    if kind == 'shared':
        return np.empty((0, shared.shape[1])), audit
    if kind == 'pca':
        matrix = shared[complete].astype(np.float64)
        basis, singular = randomized_pca(matrix, rank, specification['fit_seed'],
                                         specification['pca_oversample'], specification['pca_power_iterations'])
        audit['captured_response_energy'] = float(np.sum(singular**2)/np.sum(matrix**2))
    elif kind in ('program', 'random_program'):
        if kind == 'random_program':
            incidence, audit['randomization'] = randomize_incidence(
                incidence, specification['fit_seed'], specification['swaps_per_edge'])
        degree = np.asarray(incidence.sum(axis=1)).ravel()
        size = np.asarray(incidence.sum(axis=0)).ravel()
        normalized = sparse.diags(1/np.sqrt(degree)) @ incidence @ sparse.diags(1/np.sqrt(size))
        rng = np.random.default_rng(specification['fit_seed'])
        sketch = normalized @ rng.normal(size=(incidence.shape[1], rank))
        basis = np.linalg.qr(sketch, mode='reduced')[0].T
        singular = np.linalg.svd(sketch, compute_uv=False)
    else:
        raise ValueError('unknown response representation')
    if basis.shape != (rank, shared.shape[1]) or np.linalg.matrix_rank(basis) != rank:
        raise ValueError('invalid response basis rank')
    np.testing.assert_allclose(basis @ basis.T, np.eye(rank), atol=1e-8)
    if np.any(np.linalg.norm(basis, axis=0) <= 1e-12):
        raise ValueError('basis silently changes matched gene coverage')
    audit['singular_values'] = singular.tolist()
    return basis, audit


def complete_responses(shared, observed, basis, alpha):
    """Masked ridge projection; observed values are clamped exactly after reconstruction."""
    result = shared.copy()
    if not len(basis):
        return result
    # Training coverage has few patterns; solve all targets sharing a mask together.
    patterns, group = np.unique(observed, axis=0, return_inverse=True)
    for i, mask in enumerate(patterns):
        rows = np.flatnonzero(group == i)
        if mask.all() or not mask.any():
            continue
        design = basis[:, mask].T
        gram = design.T @ design + alpha*np.eye(len(basis))
        right = shared[np.ix_(rows, np.flatnonzero(mask))].astype(np.float64) @ design
        latent = np.linalg.solve(gram, right.T).T
        result[np.ix_(rows, np.flatnonzero(~mask))] = latent @ basis[:, ~mask]
    if not np.isfinite(result).all() or not np.array_equal(result[observed], shared[observed]):
        raise ValueError('invalid completion or changed observed responses')
    return result


def fit(directory, genes, split, config):
    contexts = split['training_contexts']
    statistics_directory = directory/'selected' if 'training_selection' in config else directory
    statistics = {c: dict(np.load(statistics_directory/f'{c}-statistics.npz')) for c in contexts}
    model = shared_responses(statistics, contexts, genes, config['model']['weight_cell_unit'],
                             config['model']['gene_chunk'])
    spec = config['representation']
    positions, modules, incidence = pathway_incidence(genes, np.flatnonzero(model['trained_mask']), spec)
    shared, observed = model['shared'][:, positions], model['observed'][:, positions]
    basis, audit = response_basis(shared, observed, incidence, spec)
    completed = complete_responses(shared, observed, basis, config['model']['completion_alpha'])
    model['effects'] = model['shared'].copy()
    model['effects'][:, positions] = completed
    model.update(basis=basis, completion_positions=positions)
    audit.update(training_contexts=contexts, training_response_used=spec['kind'] == 'pca',
                 heldout_ntc_or_response_used=False, source_refs=[spec['pathways'], spec['identities']],
                 specification=spec, covered_genes=len(positions), module_ids=modules, edges=incidence.nnz,
                 training_union_genes=int(model['trained_mask'].sum()), input_positions=positions.tolist(),
                 missing_eligible_entries=int((~observed).sum()),
                 changed_entries=int(np.count_nonzero(completed-shared)),
                 observed_coordinates_preserved=bool(np.array_equal(model['effects'][model['observed']],
                                                                     model['shared'][model['observed']])),
                 completion_rms=float(np.sqrt(np.mean((completed[~observed]-shared[~observed])**2))),
                 complete_target_ids=model['targets'][observed.all(axis=1)].tolist())
    write_json(directory/'representation.json', audit)
    print(f'Response {spec["kind"]}: {len(positions)} genes, {len(basis)} dimensions, '
          f'{audit["complete_training_targets"]} complete target rows', flush=True)
    return model


def predict(model, stats, targets, arm, source=None):
    """Same evaluator interface; no background conditioning or held-out fitting."""
    if arm != 'linear':
        raise ValueError('registered candidate is exposed as the evaluator linear arm only')
    index = {str(t): i for i, t in enumerate(model['targets'])}
    result = np.zeros((len(targets), len(model['genes'])), dtype=np.float32)
    for i, target in enumerate(targets):
        if target in index:
            result[i] = model['effects'][index[target]]
    return result
