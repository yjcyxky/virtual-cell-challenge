"""Semantic module anchors, degree-matched nulls and training-only evidence."""
from collections import defaultdict
import gzip
import json
import sys
import zipfile

import numpy as np
import pandas as pd
from scipy import sparse

from data import ROOT, hash_file, write_json, logcp, reserved

sys.path.append(str(ROOT / 'experiments/exp001-context-pair-xgb/src'))
from features import canonical_map, build_priors


def select_modules(candidates, count, minimum, maximum):
    candidates = [(name, set(members)) for name, members in candidates if minimum <= len(members) <= maximum]
    selected, covered = [], set()
    while candidates and len(selected) < count:
        index = max(range(len(candidates)), key=lambda i: (len(candidates[i][1] - covered) / np.sqrt(len(candidates[i][1])), candidates[i][0]))
        name, members = candidates.pop(index)
        selected.append((name, sorted(members))); covered.update(members)
    if len(selected) != count:
        raise ValueError('insufficient_prior_modules')
    return selected


def degree_matched_random(membership, families, seed):
    """Bipartite double-edge swaps preserve per-family gene degree and module size."""
    rng = np.random.default_rng(seed)
    result = membership.copy().astype(bool)
    for family in sorted(set(families)):
        columns = np.flatnonzero(np.asarray(families) == family)
        rows, local = np.nonzero(result[:, columns]); cols = columns[local]
        for _ in range(20 * len(rows)):
            a, b = rng.integers(len(rows), size=2)
            ga, gb, ma, mb = rows[a], rows[b], cols[a], cols[b]
            if ga != gb and ma != mb and not result[ga, mb] and not result[gb, ma]:
                result[ga, ma] = result[gb, mb] = False
                result[ga, mb] = result[gb, ma] = True
                cols[a], cols[b] = mb, ma
        assert np.array_equal(result[:, columns].sum(1), membership[:, columns].sum(1))
    assert np.array_equal(result.sum(0), membership.sum(0))
    return result.astype(np.float32)


def representation_config(config):
    """Keep the historical binary representation as the default."""
    method = config.get('prior_representation', 'binary')
    if method == 'binary':
        return {'method': method}
    if method != 'restart_diffusion':
        raise ValueError('unknown_prior_representation')
    # A randomized seed set on the real graph still contains biological knowledge.
    # Do not silently label that construction a matched random-prior control.
    if config['variant'] != 'true_prior':
        raise ValueError('diffusion_requires_true_prior')
    alpha = config['prior_diffusion_propagation']
    steps = config['prior_diffusion_steps']
    minimum = config['prior_diffusion_minimum']
    if not 0 <= alpha < 1 or not isinstance(steps, int) or steps < 1 or not 0 <= minimum < 1:
        raise ValueError('invalid_prior_diffusion_configuration')
    return {'method': method, 'propagation': alpha, 'steps': steps, 'minimum': minimum}


def annotation_incidence(candidates, genes):
    """All mapped positive terms with at least two genes; no response filtering."""
    rows, columns = [], []
    terms = 0
    for _, members in sorted(candidates):
        if len(members) >= 2:
            members = sorted(set(members))
            rows.extend(members); columns.extend([terms] * len(members))
            terms += 1
    return sparse.csr_matrix((np.ones(len(rows), np.float32), (rows, columns)), shape=(genes, terms))


def diffuse_membership(membership, relation, settings, annotation=False):
    """Restart diffusion to named landmarks, preserving their exact membership.

    For annotations, each gene chooses one of its terms uniformly and then a
    different member uniformly. This avoids dense gene-by-gene matrices and
    prevents a large term contributing quadratically more edge mass.
    """
    seeds = np.asarray(membership, np.float32)
    relation = sparse.csr_matrix(relation, dtype=np.float32)
    if relation.shape[0] != len(seeds) or not np.isfinite(relation.data).all() or np.any(relation.data < 0):
        raise ValueError('invalid_prior_relation')
    if annotation:
        sizes = np.asarray(relation.sum(0)).ravel()
        if np.any(sizes < 2) or np.any(relation.data != 1):
            raise ValueError('annotation_requires_binary_terms_with_two_genes')
        inverse_size = 1 / (sizes - 1)
        diagonal = np.asarray(relation @ inverse_size).ravel()
        degree = np.asarray(relation.sum(1)).ravel()
        def neighbors(values):
            return relation @ ((relation.T @ values) * inverse_size[:, None]) - diagonal[:, None] * values
    else:
        if relation.shape[1] != len(seeds) or np.any(relation.diagonal() != 0):
            raise ValueError('network_requires_square_without_self_edges')
        degree = np.asarray(relation.sum(1)).ravel()
        def neighbors(values):
            return relation @ values
    inverse_degree = np.divide(1., degree, out=np.zeros_like(degree), where=degree > 0)
    values = seeds.copy()
    alpha = settings['propagation']
    for _ in range(settings['steps']):
        transported = np.maximum(neighbors(values), 0) * inverse_degree[:, None]
        values = (1 - alpha) * seeds + alpha * transported
    # Original curated membership remains 1; inferred proximity is a soft feature,
    # never promoted to a new curated member or evidence of causal direction.
    values = np.maximum(seeds, np.clip(values, 0, 1))
    values[values < settings['minimum']] = 0
    if not np.isfinite(values).all():
        raise ValueError('nonfinite_prior_representation')
    return values.astype(np.float32)


def diffusion_representation(membership, families, candidates, directory, settings):
    result = membership.copy()
    for family in ['functional', 'physical', 'reactome', 'go']:
        columns = np.flatnonzero(np.asarray(families) == family)
        relation = (sparse.load_npz(directory / f'{family}.npz') if family in ['functional', 'physical']
                    else annotation_incidence(candidates[family], len(membership)))
        result[:, columns] = diffuse_membership(membership[:, columns], relation, settings,
                                                annotation=family in ['reactome', 'go'])
    return result


def prepare_priors(config, genes, output):
    settings = representation_config(config)
    directory = output / 'cache' / 'priors'; directory.mkdir(parents=True, exist_ok=True)
    done = directory / 'complete.json'
    if done.exists():
        report = json.loads(done.read_text())
        assert report.get('representation', {'method': 'binary'}) == settings, 'prior_representation_changed'
        for name, digest in report['artifacts'].items():
            assert hash_file(directory / name) == digest
        return directory
    network = ROOT / 'data/raw/networks'
    source = json.loads((network / 'SOURCE.json').read_text())
    for item in source['files']:
        assert hash_file(network / item['name']) == item['sha256']
    # Reuse the registered graph/mapping implementation, retaining its graph basis
    # as a diagnostic of network coverage rather than a second target embedding.
    graph_config = dict(config, seed=config['data_seed'], graph_dimensions=config['state_dimensions'])
    build_priors(network, genes, graph_config, directory)
    aliases = canonical_map(pd.read_csv(network / 'hgnc_complete_set.txt', sep='\t', low_memory=False), genes)
    candidates = defaultdict(list)
    for family in ['functional', 'physical']:
        graph = sparse.load_npz(directory / f'{family}.npz')
        for i in range(len(genes)):
            start, end = graph.indptr[i:i+2]
            neighbor = graph.indices[start:end]
            if len(neighbor) >= config['minimum_module_genes'] - 1:
                strength = graph.data[start:end]
                order = np.lexsort((neighbor, -strength))[:config['maximum_module_genes'] - 1]
                candidates[family].append((f'{family}:{genes[i]}', {i, *neighbor[order].tolist()}))
    with zipfile.ZipFile(network / 'ReactomePathways.gmt.zip') as z:
        with z.open('ReactomePathways.gmt') as fh:
            for line in fh:
                name, ident, *members = line.decode().strip().split('\t')
                if ident.startswith('R-HSA-'):
                    candidates['reactome'].append((ident + ':' + name, {aliases[g] for g in members if g in aliases}))
    go = defaultdict(set); evidence = defaultdict(lambda: defaultdict(int)); ignored_not = 0
    with gzip.open(network / 'goa_human.gaf.gz', 'rt') as fh:
        for line in fh:
            if line.startswith('!'):
                continue
            fields = line.rstrip('\n').split('\t')
            if len(fields) < 15:
                raise ValueError('malformed_GAF')
            if 'NOT' in fields[3].split('|'):
                ignored_not += 1; continue
            if fields[2] in aliases:
                go[fields[4]].add(aliases[fields[2]])
                evidence[fields[4]][fields[6]] += 1
    candidates['go'] = sorted(go.items())
    modules, families = [], []
    for family in ['functional', 'physical', 'reactome', 'go']:
        selected = select_modules(candidates[family], config['modules_per_family'],
                                  config['minimum_module_genes'], config['maximum_module_genes'])
        modules.extend(selected); families.extend([family] * len(selected))
    membership = np.zeros((len(genes), len(modules)), np.float32)
    rows = []
    for j, ((name, members), family) in enumerate(zip(modules, families)):
        membership[members, j] = 1
        rows.append({'module': j, 'name': name, 'family': family, 'size': len(members),
                     'members': [genes[i] for i in members], 'go_evidence_counts': dict(evidence.get(name, {}))})
    random = degree_matched_random(membership, families, config['data_seed'])
    arrays = {'true': membership, 'random': random}
    if settings['method'] == 'restart_diffusion':
        arrays['representation'] = diffusion_representation(membership, families, candidates, directory, settings)
    np.savez_compressed(directory / 'modules.npz', **arrays)
    write_json(directory / 'modules.json', rows)
    artifacts = {str(p.relative_to(directory)): hash_file(p) for p in directory.iterdir() if p.is_file()}
    represented = arrays.get('representation', membership)
    write_json(done, {'source': source, 'artifacts': artifacts, 'ignored_NOT_annotations': ignored_not,
                     'representation': settings,
                     'representation_genes_covered': int((represented.sum(1) > 0).sum()),
                     'representation_by_family': {family: {
                         'binary_genes': int((membership[:, np.asarray(families) == family].sum(1) > 0).sum()),
                         'represented_genes': int((represented[:, np.asarray(families) == family].sum(1) > 0).sum())}
                         for family in sorted(set(families))},
                     'genes_covered': int((membership.sum(1) > 0).sum()), 'modules': len(modules),
                     'random_membership_overlap': float((membership * random).sum() / membership.sum()),
                     'module_semantics': 'unsigned membership, learned signed effects; not directed causal edges'})
    return directory


def coherence(x, membership, mask, weights=None, strata=None):
    """Mean off-diagonal correlation per module, computed without dense gene covariance."""
    x = logcp(x, mask)
    weights = np.ones(len(x)) / len(x) if weights is None else np.asarray(weights) / np.sum(weights)
    if strata is not None:
        strata = np.asarray(strata)
        x = x.astype(np.float64)
        for layer in np.unique(strata):
            keep = strata == layer
            x[keep] -= np.average(x[keep], axis=0, weights=weights[keep])
    mean = np.average(x, axis=0, weights=weights)
    std = np.sqrt(np.average((x - mean) ** 2, axis=0, weights=weights))
    variable = mask & (std > 0.05)
    x = (x - mean) / np.maximum(std, 0.05)
    m = sparse.csc_matrix(membership * variable[:, None])
    n = np.asarray(m.sum(0)).ravel()
    sums = x @ m
    total_correlation = np.average(sums ** 2, axis=0, weights=weights) - n
    return np.divide(total_correlation, n * (n - 1), out=np.full(len(n), np.nan), where=n >= 3)


def fold_evidence(data, contexts, true, random, config, directory):
    """No held perturbation labels are read. Evidence changes strength, not truth labels."""
    rows, support = [], []
    for context in contexts:
        batches = sorted(b for c, b, h in data.controls if c == context and h == 0)
        # Equal cells per eligible technical layer removes NTC layer-size dominance.
        pooled = np.concatenate([data.control(context, b, 0)[:32] for b in batches])
        pool_weights = np.concatenate([np.full(min(32, len(data.controls[(context, b, 0)])),
                                                 1 / min(32, len(data.controls[(context, b, 0)]))) for b in batches])
        pool_strata = np.concatenate([[b] * min(32, len(data.controls[(context, b, 0)])) for b in batches])
        pooled_tc = coherence(pooled, true, data.masks[context], pool_weights)
        tc = coherence(pooled, true, data.masks[context], pool_weights, pool_strata)
        rc = coherence(pooled, random, data.masks[context], pool_weights, pool_strata)
        support.append(np.nan_to_num(tc - rc))
        for j in range(true.shape[1]):
            rows.append({'context': context, 'module': j, 'kind': 'NTC', 'candidate_coherence': float(tc[j]),
                         'matched_null_coherence': float(rc[j]), 'pooled_coherence_before_batch_centering': float(pooled_tc[j]),
                         'residual_rows_after_batch_centering': len(pooled) - len(batches),
                         'sample_support_sufficient': len(pooled) - len(batches) >= 30, 'target': None})
        tasks = sorted((t for c, t in data.tasks if c == context and not reserved(t, config['unseen_target_percent'])),
                       key=lambda t: value_key(t))[:config['prior_diagnostic_tasks']]
        for target in tasks:
            groups = data.groups[(context, target)]; weights = data.task_weights(context, target)
            observed, matched, observed_w, matched_w, observed_strata, matched_strata = [], [], [], [], [], []
            for pair, ids in groups.items():
                x = data.read(ids); ntc = data.control(context, pair[1], 0)
                observed.append(x); matched.append(ntc)
                observed_w.extend([weights[pair] / len(x)] * len(x)); matched_w.extend([weights[pair] / len(ntc)] * len(ntc))
                observed_strata.extend([pair[1]] * len(x)); matched_strata.extend([pair[1]] * len(ntc))
            pc = coherence(np.concatenate(observed), true, data.masks[context], observed_w, observed_strata)
            pr = coherence(np.concatenate(observed), random, data.masks[context], observed_w, observed_strata)
            matched_tc = coherence(np.concatenate(matched), true, data.masks[context], matched_w, matched_strata)
            residual_rows = len(observed_w) - len(set(observed_strata))
            for j in range(true.shape[1]):
                rows.append({'context': context, 'module': j, 'kind': 'perturbation', 'target': target,
                             'candidate_coherence': float(pc[j]), 'matched_null_coherence': float(pr[j]),
                             'coherence_change_vs_NTC': float(pc[j] - matched_tc[j]),
                             'residual_rows_after_batch_centering': residual_rows,
                             'sample_support_sufficient': residual_rows >= 30})
    pd.DataFrame(rows).to_parquet(directory / 'prior-evidence.parquet', index=False)
    # A conservative floor explicitly retains priors unobservable in RNA.
    strength = 0.25 + 0.75 * np.clip(np.mean(support, axis=0) / 0.15, 0, 1)
    np.save(directory / 'prior-strength.npy', strength.astype(np.float32))
    write_json(directory / 'evidence-scope.json', {'training_contexts': list(contexts),
               'uses_held_perturbations': False, 'candidate_prior': 'degree_matched_random' if config['variant'] == 'random_prior' else 'real_membership_diagnostic',
               'strength_method': '0.25 + 0.75 clip(mean train NTC within-batch candidate-minus-null coherence / 0.15)',
               'perturbation_diagnostic': 'changes in off-diagonal standardized covariance; endpoint association, not causal validity',
               'global_target_reservation_percent': config['unseen_target_percent'], 'minimum_strength': float(strength.min())})
    return strength.astype(np.float32)


def value_key(text):
    import hashlib
    return hashlib.sha256(('module-evidence-v1|' + text).encode()).hexdigest()
