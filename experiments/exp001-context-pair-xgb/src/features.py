"""External priors and NTC-only, context-dependent gene-pair features.

Correlation and conformity are observational proxies, never mechanistic labels.
No function in this module accepts perturbation response labels.
"""
import gzip
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import eigsh


def canonical_map(hgnc, genes):
    """Only unambiguous approved symbols/aliases mapping onto the output axis."""
    approved = set(hgnc.symbol.dropna())
    aliases = {}
    for row in hgnc.to_dict('records'):
        for field in ['symbol', 'alias_symbol', 'prev_symbol']:
            if pd.notna(row.get(field)):
                for alias in str(row[field]).split('|'):
                    aliases.setdefault(alias, set()).add(row['symbol'])
    canonical = {g: (g if g in approved else next(iter(aliases[g]))
                     if g in aliases and len(aliases[g]) == 1 else None) for g in genes}
    reverse = {}
    for i, g in enumerate(genes):
        if canonical[g] is not None:
            reverse.setdefault(canonical[g], []).append(i)
    axis = {g: ids[0] for g, ids in reverse.items() if len(ids) == 1}
    result = {}
    for alias, values in aliases.items():
        value = alias if alias in approved else next(iter(values)) if len(values) == 1 else None
        if value in axis:
            result[alias] = axis[value]
    return result


def build_priors(directory, genes, config, output):
    hgnc = pd.read_csv(directory / 'hgnc_complete_set.txt', sep='\t', low_memory=False)
    aliases = canonical_map(hgnc, genes)
    info = pd.read_csv(directory / '9606.protein.info.v12.0.txt.gz', sep='\t')
    protein = {str(r['#string_protein_id']): aliases[r['preferred_name']]
               for r in info.to_dict('records') if r['preferred_name'] in aliases}
    n = len(genes)
    graphs = {}
    for kind, name in [('functional', '9606.protein.links.v12.0.txt.gz'),
                       ('physical', '9606.protein.physical.links.v12.0.txt.gz')]:
        edges = {}
        with gzip.open(directory / name, 'rt') as stream:
            next(stream)
            for line in stream:
                a, b, score = line.split()
                score = int(score)
                if score < config['network_minimum_score'] or a not in protein or b not in protein:
                    continue
                i, j = protein[a], protein[b]
                if i != j:
                    key = (min(i, j), max(i, j))
                    edges[key] = max(edges.get(key, 0), score / 1000)
        rr, cc, vv = [], [], []
        for (i, j), value in sorted(edges.items()):
            rr.extend([i, j]); cc.extend([j, i]); vv.extend([value, value])
        graphs[kind] = sparse.csr_matrix((np.asarray(vv, np.float32), (rr, cc)), shape=(n, n))
        sparse.save_npz(output / (kind + '.npz'), graphs[kind])
    pathways = []
    with zipfile.ZipFile(directory / 'ReactomePathways.gmt.zip') as zf:
        with zf.open('ReactomePathways.gmt') as stream:
            for line in stream:
                fields = line.decode().strip().split('\t')
                if fields[1].startswith('R-HSA-'):
                    members = sorted({aliases[g] for g in fields[2:] if g in aliases})
                    if len(members) >= 2:
                        pathways.append((fields[1], members))
    rr, cc = [], []
    for j, (_, members) in enumerate(pathways):
        rr.extend(members); cc.extend([j] * len(members))
    membership = sparse.csr_matrix((np.ones(len(rr), np.float32), (rr, cc)), shape=(n, len(pathways)))
    sparse.save_npz(output / 'pathways.npz', membership)
    graph = graphs['functional']
    degree = np.asarray(graph.sum(1)).ravel()
    inv = np.divide(1., np.sqrt(degree), out=np.zeros_like(degree), where=degree > 0)
    normalized = sparse.diags(inv) @ graph @ sparse.diags(inv)
    rng = np.random.default_rng(config['seed'])
    values, vectors = eigsh(normalized, k=config['graph_dimensions'], which='LA',
                            v0=rng.normal(size=n), tol=1e-6)
    order = np.argsort(values)[::-1]
    embedding = vectors[:, order].astype(np.float32)
    # Fix the otherwise arbitrary eigenvector signs and scale to RMS one.
    for j in range(embedding.shape[1]):
        embedding[:, j] *= np.sign(embedding[np.argmax(np.abs(embedding[:, j])), j])
    embedding /= np.maximum(np.sqrt(np.mean(embedding ** 2, axis=0)), 1e-8)
    embedding[degree == 0] = 0
    np.save(output / 'embedding.npy', embedding)
    (output / 'prior-summary.json').write_text(json.dumps({
        'functional_edges': graph.nnz // 2, 'physical_edges': graphs['physical'].nnz // 2,
        'pathways': len(pathways), 'genes_with_functional_edges': int((degree > 0).sum()),
        'graph_dimensions': embedding.shape[1], 'eigenvalues': values[order].tolist(),
        'semantics': 'Unsigned external associations; no context-specific activity or regulatory sign is asserted.'
    }, indent=2) + '\n')


def residualize(log, batches, log_depth):
    """Remove technical-stratum means and within-stratum log library depth."""
    result = np.asarray(log, np.float32).copy()
    depth = np.asarray(log_depth, np.float64).copy()
    _, group = np.unique(batches, return_inverse=True)
    for b in np.unique(group):
        ids = group == b
        result[ids] -= result[ids].mean(0)
        depth[ids] -= depth[ids].mean()
    if depth @ depth > 1e-10:
        coefficient = depth @ result / (depth @ depth)
        result -= (depth[:, None] * coefficient).astype(np.float32)
    return result, len(result) - len(np.unique(group)) - 1


def unit_profiles(log, valid):
    centered = np.asarray(log, np.float32) - np.asarray(log, np.float32).mean(0)
    norm = np.sqrt(np.sum(centered.astype(np.float64) ** 2, axis=0))
    good = np.asarray(valid) & (norm > 1e-6)
    centered /= np.where(good, norm, 1.)
    centered[:, ~good] = 0
    return centered, good


def relation_matrices(log, batches, log_depth, measured, config, output):
    """Exact sample correlations; separate split discrepancy quantifies instability."""
    n, g = log.shape
    detected = (log > 0).sum(0)
    valid = measured & (detected >= config['minimum_relation_detections'])
    raw, raw_ok = unit_profiles(log, valid)
    adjusted, dof = residualize(log, batches, log_depth)
    adj, adj_ok = unit_profiles(adjusted, valid & (dof >= config['minimum_relation_degrees_freedom']))
    halves = []
    # Stable stratified assignment, independent of the reference-control pool.
    split = np.zeros(n, np.int8)
    rng = np.random.default_rng(config['seed'])
    for b in np.unique(batches):
        rows = rng.permutation(np.flatnonzero(batches == b))
        split[rows] = np.arange(len(rows)) % 2
    for h in [0, 1]:
        ids = split == h
        residual, half_dof = residualize(log[ids], batches[ids], log_depth[ids])
        halves.append(unit_profiles(residual, measured & ((log[ids] > 0).sum(0) >= config['minimum_relation_detections'])
                                    & (half_dof >= config['minimum_relation_degrees_freedom'])))
    matrices = {name: np.lib.format.open_memmap(output / (name + '.npy'), mode='w+', dtype=np.float32, shape=(g, g))
                for name in ['raw_correlation', 'correlation', 'split_gap']}
    for start in range(0, g, 256):
        end = min(start + 256, g)
        for name, array, good in [('raw_correlation', raw, raw_ok), ('correlation', adj, adj_ok)]:
            block = np.clip(array[:, start:end].T @ array, -1., 1.)
            block[~(good[start:end, None] & good[None, :])] = np.nan
            matrices[name][start:end] = block
        a, good_a = halves[0]; b, good_b = halves[1]
        stable = good_a & good_b
        difference = np.abs(np.clip(a[:, start:end].T @ a, -1, 1) - np.clip(b[:, start:end].T @ b, -1, 1))
        difference[~(stable[start:end, None] & stable[None, :])] = np.nan
        matrices['split_gap'][start:end] = difference
    for matrix in matrices.values():
        matrix.flush()
    return {'sampled_controls': n, 'technical_residual_degrees_freedom': dof,
            'raw_supported_genes': int(raw_ok.sum()), 'adjusted_supported_genes': int(adj_ok.sum()),
            'split_supported_genes': int((halves[0][1] & halves[1][1]).sum()),
            'interpretation': 'NTC association/conformity proxy, not biochemical constraint ground truth'}


class PairFeatures:
    def __init__(self, directory, contexts):
        self.directory = Path(directory)
        priors = self.directory / 'priors'
        self.embedding = np.load(priors / 'embedding.npy')
        self.functional = sparse.load_npz(priors / 'functional.npz')
        self.physical = sparse.load_npz(priors / 'physical.npz')
        self.pathways = sparse.load_npz(priors / 'pathways.npz')
        self.pathway_size = np.asarray(self.pathways.sum(1)).ravel()
        # Matrix product is count of shared pathways, not a causal edge.
        self.shared_pathways = (self.pathways @ self.pathways.T).tocsr()
        self.degree = np.asarray(self.functional.sum(1)).ravel()
        self.contexts = {}
        for name in contexts:
            folder = self.directory / 'contexts' / name
            state = np.load(folder / 'state.npz')
            self.contexts[name] = {k: state[k] for k in state.files}
            self.contexts[name].update({k: np.load(folder / (k + '.npy'), mmap_mode='r')
                                       for k in ['raw_correlation', 'correlation', 'split_gap']})

    def features(self, context, p, g, reference_contexts, model):
        p, g = np.asarray(p, int), np.asarray(g, int)
        ep, eg = self.embedding[p], self.embedding[g]
        functional = np.asarray(self.functional[p, g]).ravel()
        physical = np.asarray(self.physical[p, g]).ravel()
        shared = np.asarray(self.shared_pathways[p, g]).ravel()
        union = self.pathway_size[p] + self.pathway_size[g] - shared
        jaccard = np.divide(shared, union, out=np.zeros_like(shared), where=union > 0)
        values = [ep, eg, functional[:, None], physical[:, None], jaccard[:, None],
                  np.log1p(self.degree[p])[:, None], np.log1p(self.degree[g])[:, None],
                  (self.pathway_size[p] > 0)[:, None], (self.pathway_size[g] > 0)[:, None]]
        names = ([f'target_graph_{j}' for j in range(ep.shape[1])] + [f'readout_graph_{j}' for j in range(eg.shape[1])]
                 + ['functional_prior', 'physical_prior', 'pathway_jaccard', 'target_degree', 'readout_degree',
                    'target_pathway_known', 'readout_pathway_known'])
        if model == 'static_pair':
            return np.column_stack(values).astype(np.float32), names
        c = self.contexts[context]
        for key in ['mean', 'std', 'detection', 'neighbor_mean', 'neighbor_detection', 'measured']:
            values.extend([c[key][p, None], c[key][g, None]])
            names.extend(['target_' + key, 'readout_' + key])
        values.append(np.broadcast_to(c['global_state'], (len(p), len(c['global_state']))))
        names.extend([f'context_program_{j}' for j in range(len(c['global_state']))])
        if model == 'context_pair':
            return np.column_stack(values).astype(np.float32), names
        if model != 'context_relation':
            raise ValueError('unknown model ' + model)
        rho = c['correlation'][p, g]
        references = [self.contexts[b]['correlation'][p, g] for b in reference_contexts if b != context]
        if references:
            references = np.asarray(references)
            count = np.isfinite(references).sum(0)
            mean = np.divide(np.nansum(references, axis=0), count, out=np.full(len(p), np.nan), where=count > 0)
        else:
            count, mean = np.zeros(len(p)), np.full(len(p), np.nan)
        relation_values = [c['raw_correlation'][p, g], rho, c['split_gap'][p, g], mean, count,
                           rho - mean, np.abs(rho - mean), rho * mean, functional * rho,
                           physical * rho, jaccard * rho]
        relation_names = ['raw_correlation', 'adjusted_correlation', 'relation_split_gap', 'reference_correlation',
                          'reference_relation_support', 'relation_shift', 'relation_abs_shift', 'relation_sign_conformity',
                          'functional_context_interaction', 'physical_context_interaction', 'pathway_context_interaction']
        values.extend(np.asarray(x)[:, None] for x in relation_values)
        names.extend(relation_names)
        return np.column_stack(values).astype(np.float32), names
