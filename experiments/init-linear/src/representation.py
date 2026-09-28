"""Frozen NTC-only projections with matched pathway coverage and feature scale."""
import zipfile
import numpy as np
import pandas as pd
import anndata as ad
from scipy import sparse
from data import ROOT, NTC, ChallengeIdentity, normalized, ref, write_json


def pathway_incidence(genes, common, specification):
    for key in ('pathways', 'identities'):
        source = specification[key]
        if ref(ROOT/source['path']) != source:
            raise ValueError(f'changed representation source: {key}')
    identity = ChallengeIdentity(genes, pd.read_csv(ROOT/specification['identities']['path'],
                                                   sep='\t', low_memory=False))
    common = set(common)
    modules = {}
    with zipfile.ZipFile(ROOT/specification['pathways']['path']) as archive:
        for line in archive.read('ReactomePathways.gmt').decode().splitlines():
            _, module, *members = line.split('\t')
            positions = set()
            for member in members:
                gene, _ = identity.resolve(member)
                position = identity.positions.get(gene)
                if position in common:
                    positions.add(position)
            if module.startswith('R-HSA-') and specification['min_module_size'] <= len(positions) <= specification['max_module_size']:
                modules[module] = tuple(sorted(positions))
    unique = {}
    for module in sorted(modules):
        unique.setdefault(modules[module], module)
    positions = np.array(sorted(set().union(*map(set, unique))), dtype=int)
    lookup = {p: i for i, p in enumerate(positions)}
    columns, rows, names = [], [], []
    for members, module in sorted(unique.items(), key=lambda item: item[1]):
        column = len(names)
        names.append(module)
        rows.extend(lookup[p] for p in members)
        columns.extend([column]*len(members))
    matrix = sparse.csr_matrix((np.ones(len(rows)), (rows, columns)), shape=(len(positions), len(names)))
    if not len(names) or (np.asarray(matrix.sum(axis=1)).ravel() == 0).any():
        raise ValueError('empty pathway coverage')
    return positions, names, matrix


def randomize_incidence(matrix, seed, swaps_per_edge):
    """Double-edge swaps preserve each gene degree AND each module size."""
    coo = matrix.tocoo()
    rows, cols = coo.row.copy(), coo.col.copy()
    edges = set(zip(rows.tolist(), cols.tolist()))
    original = edges.copy()
    rng = np.random.default_rng(seed)
    wanted = swaps_per_edge*len(rows)
    successes = attempts = 0
    while successes < wanted:
        if attempts >= 100*wanted:
            raise ValueError('degree-preserving randomization failed to mix')
        i, j = rng.integers(len(rows), size=2)
        a, b, c, d = int(rows[i]), int(cols[i]), int(rows[j]), int(cols[j])
        attempts += 1
        if a == c or b == d or (a, d) in edges or (c, b) in edges:
            continue
        edges.remove((a, b)); edges.remove((c, d))
        edges.add((a, d)); edges.add((c, b))
        cols[i], cols[j] = d, b
        successes += 1
    result = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=matrix.shape)
    if not (np.array_equal(matrix.sum(axis=0), result.sum(axis=0)) and
            np.array_equal(matrix.sum(axis=1), result.sum(axis=1)) and result.data.max() == 1):
        raise ValueError('randomization changed degrees or module sizes')
    return result, {'successful_swaps': successes, 'attempts': attempts,
                    'edge_overlap_fraction': len(original & edges)/len(original),
                    'gene_degrees_and_module_sizes_preserved': True}


def randomized_pca(matrix, dimensions, seed, oversample, power_iterations):
    """Fixed-budget randomized SVD; matrix is centered, weighted training NTC."""
    width = min(dimensions+oversample, *matrix.shape)
    if dimensions >= min(matrix.shape):
        raise ValueError('PCA dimension exceeds training NTC matrix rank bound')
    rng = np.random.default_rng(seed)
    q = np.linalg.qr(matrix @ rng.normal(size=(matrix.shape[1], width)), mode='reduced')[0]
    for _ in range(power_iterations):
        right = np.linalg.qr(matrix.T @ q, mode='reduced')[0]
        q = np.linalg.qr(matrix @ right, mode='reduced')[0]
    _, singular, basis = np.linalg.svd(q.T @ matrix, full_matrices=False)
    return basis[:dimensions], singular[:dimensions]


def ntc_projection(statistics, contexts, genes, directory, config):
    spec = config['representation']
    common = sorted(set.intersection(*(set(statistics[c]['positions']) for c in contexts)))
    positions, names, incidence = pathway_incidence(genes, common, spec)
    means = np.array([statistics[c]['mean'][0, np.searchsorted(statistics[c]['positions'], positions)] for c in contexts], dtype=float)
    center = means.mean(axis=0)
    audit = {'kind': spec['kind'], 'input_positions': positions.tolist(),
             'training_common_genes': len(common), 'covered_genes': len(positions),
             'module_ids': names, 'edges': incidence.nnz, 'training_contexts': contexts,
             'training_response_used': False, 'heldout_ntc_used_for_fitting': False,
             'source_refs': [spec['pathways'], spec['identities']], 'specification': spec}
    dimensions = spec['dimensions']
    if spec['kind'] == 'raw':
        _, singular, basis = np.linalg.svd(means-center, full_matrices=False)
        basis = basis[singular > singular[0]*1e-10]
    elif spec['kind'] == 'pca':
        counts = [int(statistics[c]['counts'][0]) for c in contexts]
        matrix = np.empty((sum(counts), len(positions)), dtype=float)
        offset = 0
        for context, count in zip(contexts, counts, strict=True):
            ntc = ad.read_h5ad(directory/f'{context}-input.h5ad')
            if (ntc.n_obs != count or not ntc.obs.target_gene.eq(NTC).all() or
                    ntc.var_names.tolist() != [genes[p] for p in statistics[context]['positions']]):
                raise ValueError('PCA input differs from allowed training NTC')
            columns = np.searchsorted(statistics[context]['positions'], positions)
            for start in range(0, count, config['data']['chunk_rows']):
                stop = min(start+config['data']['chunk_rows'], count)
                # Normalize over the context's original measured axis, before projection.
                matrix[offset+start:offset+stop] = (normalized(ntc.X[start:stop], config['data']['target_sum'])[:, columns].toarray()-center)/np.sqrt(len(contexts)*count)
            offset += count
            del ntc
        basis, singular = randomized_pca(matrix, dimensions, spec['fit_seed'], spec['pca_oversample'], spec['pca_power_iterations'])
        audit.update(training_ntc_cells=sum(counts), context_ntc_counts=dict(zip(contexts, counts)),
                     captured_weighted_energy=float(np.sum(singular**2)/np.sum(matrix**2)))
        del matrix
    elif spec['kind'] in ('program', 'random_program'):
        if spec['kind'] == 'random_program':
            incidence, audit['randomization'] = randomize_incidence(incidence, spec['fit_seed'], spec['swaps_per_edge'])
        degree = np.asarray(incidence.sum(axis=1)).ravel()
        size = np.asarray(incidence.sum(axis=0)).ravel()
        normalized_graph = sparse.diags(1/np.sqrt(degree)) @ incidence @ sparse.diags(1/np.sqrt(size))
        # A fixed sketch keeps every connected component represented; a truncated
        # graph SVD could silently omit small components and change gene coverage.
        rng = np.random.default_rng(spec['fit_seed'])
        sketch = normalized_graph @ rng.normal(size=(incidence.shape[1], dimensions))
        basis = np.linalg.qr(sketch, mode='reduced')[0].T
        singular = np.linalg.svd(sketch, compute_uv=False)
    else:
        raise ValueError('unregistered NTC representation')
    # Common scalar calibration: equal context-feature trace, without per-axis whitening.
    embedding = (means-center) @ basis.T
    scale = float(np.sqrt(np.mean(np.sum(embedding**2, axis=1))))
    if scale <= 1e-12 or not np.isfinite(scale):
        raise ValueError('degenerate training NTC projection')
    np.testing.assert_allclose(basis @ basis.T, np.eye(len(basis)), rtol=1e-8, atol=1e-8)
    nonzero_genes = int(np.sum(np.linalg.norm(basis, axis=0) > 1e-12))
    if spec['kind'] in ('program', 'random_program') and nonzero_genes != len(positions):
        raise ValueError('projection silently removed covered input genes')
    audit.update(dimensions=int(len(basis)), training_context_rank=int(np.linalg.matrix_rank(embedding, tol=1e-8)),
                 nonzero_projection_genes=nonzero_genes,
                 singular_values=singular.tolist(), feature_scale=scale,
                 mean_training_context_squared_norm=float(np.mean(np.sum((embedding/scale)**2, axis=1))))
    write_json(directory/'representation.json', audit)
    print(f'NTC representation {spec["kind"]}: {len(positions)} genes, {len(basis)} dimensions, context rank {audit["training_context_rank"]}', flush=True)
    return positions, center, basis, scale
