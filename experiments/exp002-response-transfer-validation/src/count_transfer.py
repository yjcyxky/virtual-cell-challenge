"""Masked native-axis sufficient statistics and context-held-out response transfer."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT / 'experiments/exp003-context-module-cvae/src'))
from data import CountData, hash_file, write_json, verify_cache
from estimators import ntc_distances, response_mean


def bulk(counts, target_sum=50000.):
    counts = np.asarray(counts, dtype=np.float64)
    return np.log1p(target_sum * counts / np.maximum(counts.sum(-1, keepdims=True), 1e-12))


def stable_seed(seed, *parts):
    value = '|'.join(map(str, (seed, *parts)))
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:8], 'little')


def count_moments(data, ids):
    sums = np.zeros(len(data.genes), np.float64)
    squares, detected = sums.copy(), sums.copy()
    depths = []
    for start in range(0, len(ids), 256):
        x = data.read(ids[start:start + 256]).astype(np.float64)
        sums += x.sum(0); squares += np.square(x).sum(0); detected += (x > 0).sum(0)
        depths.extend(x.sum(1).tolist())
    mean = sums / len(ids)
    variance = np.maximum(squares / len(ids) - mean ** 2, 0.)
    return mean, variance, detected / len(ids), np.asarray(depths)


class Statistics:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.meta = json.loads((self.directory / 'complete.json').read_text())
        self.genes, self.targets, self.contexts = (self.meta[k] for k in ['genes', 'targets', 'contexts'])
        self.target_index = {t: i for i, t in enumerate(self.targets)}
        self.gene_index = {g: i for i, g in enumerate(self.genes)}
        for name in ['response', 'target_bulk', 'depth_response', 'available', 'masks', 'feature_mean',
                     'feature_variance', 'feature_detection', 'reference_mean', 'generic']:
            setattr(self, name, np.load(self.directory / (name + '.npy'), mmap_mode='r'))
        self.common = self.masks.all(0)
        self.features = bulk(self.feature_mean[:, self.common])


def save_response_library(stats, path):
    """Persist inference aggregates independently of the preprocessing cache."""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    marker = path.with_suffix('.json')
    if marker.exists():
        assert json.loads(marker.read_text())['sha256'] == hash_file(path)
        return
    arrays = {'genes': np.asarray(stats.genes), 'contexts': np.asarray(stats.contexts), 'targets': np.asarray(stats.targets),
              'feature_mean': stats.feature_mean, 'masks': stats.masks, 'generic': stats.generic}
    for ci, context in enumerate(stats.contexts):
        ids = np.flatnonzero(stats.available[ci])
        arrays[f'{context}_targets'] = ids
        arrays[f'{context}_response'] = stats.response[ci, ids]
        arrays[f'{context}_depth'] = stats.depth_response[ci, ids]
    temporary = path.with_name('response-library.partial.npz')
    np.savez_compressed(temporary, **arrays); temporary.replace(path)
    write_json(marker, {'sha256': hash_file(path), 'stats_sha256': hash_file(stats.directory / 'complete.json'),
                       'contents': 'fitted response library and NTC aggregates; no raw cells',
                       'fold_scope': 'each checkpoint training_contexts is an enforced whitelist'})


class ResponseLibrary:
    """Load a checkpoint's donor labels without access to cell or statistics caches."""
    def __init__(self, path, fitted):
        path = Path(path)
        metadata = json.loads(path.with_suffix('.json').read_text())
        assert metadata['sha256'] == hash_file(path), 'response_library_changed'
        assert metadata['stats_sha256'] == fitted['stats_sha256'], 'response_library_checkpoint_mismatch'
        with np.load(path, allow_pickle=False) as archive:
            self.genes, self.targets, self.contexts = (archive[k].tolist() for k in ['genes', 'targets', 'contexts'])
            self.target_index = {t: i for i, t in enumerate(self.targets)}
            self.gene_index = {g: i for i, g in enumerate(self.genes)}
            self.feature_mean, self.masks, self.generic = (archive[k] for k in ['feature_mean', 'masks', 'generic'])
            self.available = np.zeros((len(self.contexts), len(self.targets)), bool)
            self.depth_response = np.full(self.available.shape, np.nan)
            self.tables, self.positions = {}, {}
            for context in fitted['training_contexts']:
                ci = self.contexts.index(context); ids = archive[f'{context}_targets']
                self.available[ci, ids] = True
                self.depth_response[ci, ids] = archive[f'{context}_depth']
                self.tables[ci] = archive[f'{context}_response']
                positions = np.full(len(self.targets), -1, int); positions[ids] = np.arange(len(ids))
                self.positions[ci] = positions
        self.common = self.masks.all(0); self.features = bulk(self.feature_mean[:, self.common])
        self.response = self

    def __getitem__(self, key):
        context, targets = key
        positions = self.positions[context][targets]
        values = np.full((len(positions), len(self.genes)), np.nan, np.float32)
        valid = positions >= 0
        values[valid] = self.tables[context][positions[valid]]
        return values


def prepare_statistics(data, config, directory):
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    marker = directory / 'complete.json'
    source_digest = hash_file(data.directory / 'complete.json')
    if marker.exists():
        saved = json.loads(marker.read_text())
        assert saved['source_sha256'] == source_digest
        for name, digest in saved['files'].items():
            assert hash_file(directory / name) == digest, 'statistics_changed:' + name
        return Statistics(directory)
    contexts = config['contexts']; genes = data.genes
    targets = sorted({t for c, t in data.tasks if c in contexts})
    ti = {t: i for i, t in enumerate(targets)}
    c, p, g = len(contexts), len(targets), len(genes)
    shape = (c, p, g)
    response = np.lib.format.open_memmap(directory / 'response.npy', mode='w+', dtype=np.float32, shape=shape)
    truth = np.lib.format.open_memmap(directory / 'target_bulk.npy', mode='w+', dtype=np.float32, shape=shape)
    response[:] = np.nan; truth[:] = np.nan
    available = np.zeros((c, p), bool); depth_response = np.full((c, p), np.nan, np.float32)
    masks = np.stack([data.masks[x] for x in contexts])
    controls = {k: np.zeros((c, g), np.float64) for k in ['feature_mean', 'feature_variance', 'feature_detection', 'reference_mean']}
    generic = np.full((c, g), np.nan, np.float32)
    support, nulls = [], []
    panels = {}
    for ci, context in enumerate(contexts):
        feature = np.concatenate([ids for (cc, b, h), ids in data.controls.items() if cc == context and h == 0])
        reference = np.concatenate([ids for (cc, b, h), ids in data.controls.items() if cc == context and h == 1])
        assert not np.intersect1d(feature, reference).size
        fm, fv, fd, fdepth = count_moments(data, feature)
        rm, rv, rd, rdepth = count_moments(data, reference)
        for key, value in [('feature_mean', fm), ('feature_variance', fv), ('feature_detection', fd), ('reference_mean', rm)]:
            controls[key][ci] = value
        mask = masks[ci]; ref_bulk = bulk(rm, config['bulk_target_sum'])
        nulls.append({'context': context, 'feature_cells': len(feature), 'reference_cells': len(reference),
                      'bulk_mse': float(np.mean((bulk(fm)[mask] - ref_bulk[mask]) ** 2)),
                      'detection_mae': float(np.mean(np.abs(fd[mask] - rd[mask]))),
                      'variance_log_mse': float(np.mean((np.log1p(fv[mask]) - np.log1p(rv[mask])) ** 2)),
                      'feature_depth_mean': float(fdepth.mean()), 'reference_depth_mean': float(rdepth.mean()),
                      'feature_depth_cv': float(fdepth.std() / fdepth.mean()),
                      'reference_depth_cv': float(rdepth.std() / rdepth.mean())})
        context_targets = sorted(t for cc, t in data.tasks if cc == context)
        panels[context] = sorted(context_targets, key=lambda t: (stable_seed(config['seed'], context, t), t))[:config['official_panel_targets']]
        for number, target in enumerate(context_targets):
            pi = ti[target]; ids = data.tasks[context, target]
            mu, _, _, depths = count_moments(data, ids)
            value = bulk(mu, config['bulk_target_sum'])
            truth[ci, pi, mask] = value[mask]
            response[ci, pi, mask] = (value - ref_bulk)[mask]
            available[ci, pi] = True
            depth_response[ci, pi] = np.log(depths.mean() / rdepth.mean())
            support.append({'context': context, 'target': target, 'independent_cells': len(ids),
                            'measured_readouts': int(mask.sum()), 'target_depth': float(depths.mean()),
                            'target_depth_cv': float(depths.std() / depths.mean())})
            if (number + 1) % 500 == 0:
                print(json.dumps({'stage': 'native_statistics', 'context': context, 'done': number + 1, 'total': len(context_targets)}), flush=True)
        ids = np.flatnonzero(available[ci])
        # Summation avoids a large fancy-index copy of the K562 response table.
        total = np.zeros(g, np.float64)
        for start in range(0, len(ids), 64):
            total += np.nansum(response[ci, ids[start:start + 64]], axis=0, dtype=np.float64)
        generic[ci, mask] = total[mask] / len(ids)
        print(json.dumps({'stage': 'native_statistics_complete', 'context': context, 'tasks': len(ids), 'readouts': int(mask.sum())}), flush=True)
    response.flush(); truth.flush(); del response, truth
    arrays = dict(controls, available=available, masks=masks, depth_response=depth_response, generic=generic)
    for name, value in arrays.items():
        np.save(directory / (name + '.npy'), value)
    pd.DataFrame(support).to_parquet(directory / 'support.parquet', index=False)
    write_json(directory / 'ntc-diagnostics.json', nulls)
    write_json(directory / 'panels.json', {'seed': config['seed'], 'selection': 'identity hash only; no expression or model values', 'contexts': panels})
    files = {p.name: hash_file(p) for p in sorted(directory.iterdir()) if p.is_file() and p.name != 'complete.json'}
    write_json(marker, {'source_sha256': source_digest, 'source': str(data.directory), 'genes': genes, 'targets': targets,
                       'contexts': contexts, 'files': files, 'estimand': 'all cached independent cells equally weighted; disjoint reference NTC',
                       'bulk_target_sum': config['bulk_target_sum'], 'missing_measurement': 'NaN, never zero labels'})
    return Statistics(directory)


def donor_predictions(stats, training, targets, query_mean, temperature):
    """Read labels exclusively from the explicitly supplied training contexts."""
    indices = [stats.contexts.index(c) for c in training]
    positions = np.array([stats.target_index.get(t, -1) for t in targets])
    known = positions >= 0
    response = np.full((len(indices), len(targets), len(stats.genes)), np.nan, np.float32)
    available = np.zeros((len(indices), len(targets)), bool)
    depths = np.full((len(indices), len(targets)), np.nan)
    for j, ci in enumerate(indices):
        response[j, known] = stats.response[ci, positions[known]]
        available[j, known] = stats.available[ci, positions[known]]
        depths[j, known] = stats.depth_response[ci, positions[known]]
    count = np.isfinite(response).sum(0)
    feature = bulk(np.asarray(query_mean)[stats.common])
    distance = ntc_distances(np.vstack([feature, stats.features[indices]]))[0, 1:]
    shared = response_mean(response, available)
    kernel = response_mean(response, available, distance, temperature)
    generic = stats.generic[indices, None, :]
    ga = np.ones((len(indices), 1), bool)
    fallback = response_mean(generic, ga)[0]
    weighted_fallback = response_mean(generic, ga, distance, temperature)[0]
    shared = np.where(count > 0, shared, fallback)
    kernel = np.where(count > 0, kernel, weighted_fallback)
    supported = np.isfinite(shared)
    shared, kernel = np.nan_to_num(shared), np.nan_to_num(kernel)
    dn = np.isfinite(depths).sum(0)
    dsum = np.nansum(depths, axis=0)
    generic_depth = np.mean([np.nanmean(stats.depth_response[ci]) for ci in indices])
    depth = np.divide(dsum, dn, out=np.full(len(targets), generic_depth), where=dn > 0)
    return shared, kernel - shared, count, supported, depth


def quadratic_fit(a, b, y2, ridge, bound):
    penalty = np.diag([0., ridge * max(float(np.trace(a)), 1e-12)])
    matrix = a + penalty
    # The convex two-variable box problem has an interior stationary point or an
    # edge optimum. Enumerate those exactly; avoid line-search failure near zero loss.
    candidates = [np.array([x, y], float) for x in [0., bound] for y in [0., bound]]
    interior = np.linalg.lstsq(matrix, b, rcond=1e-12)[0]
    if np.all(interior >= 0) and np.all(interior <= bound):
        candidates.append(interior)
    for axis in range(2):
        other = 1 - axis
        for fixed in [0., bound]:
            point = np.zeros(2); point[axis] = fixed
            point[other] = np.clip((b[other] - matrix[other, axis] * fixed) / matrix[other, other], 0, bound) if matrix[other, other] > 0 else 0
            candidates.append(point)
    value = lambda w: float(w @ matrix @ w - 2 * b @ w)
    selected = min(candidates, key=lambda w: (value(w), w[1], w[0]))
    gradient = 2 * (matrix @ selected - b)
    projected = gradient.copy()
    projected[(selected == 0) & (gradient >= 0)] = 0
    projected[(selected == bound) & (gradient <= 0)] = 0
    residual = float(np.max(np.abs(projected)))
    if not np.isfinite(selected).all() or residual > 1e-8 * max(float(np.max(np.abs(b))), 1e-12):
        raise RuntimeError('quadratic_optimality_check_failed')
    return {'coefficients': selected.tolist(), 'penalized_loss': value(selected) + float(y2),
            'unpenalized_loss': float(selected @ a @ selected - 2 * b @ selected + y2),
            'optimizer_status': 'exact_box_quadratic_optimum', 'projected_gradient_residual': residual,
            'candidates_checked': len(candidates)}


def fit_transfer(stats, training, config):
    """Nested held-context calibration; no external held labels or stopping feedback."""
    records = []
    for temperature in config['temperatures']:
        a = np.zeros((2, 2)); b = np.zeros(2); y2 = 0.
        fallback_x2 = fallback_xy = fallback_y2 = depth_x2 = depth_xy = 0.
        tasks = 0; fold_records = []
        for held in training:
            ci = stats.contexts.index(held)
            sources = [c for c in training if c != held]
            ids = np.flatnonzero(stats.available[ci])
            fold_records.append({'held_context': held, 'training_contexts': sources, 'tasks': len(ids)})
            for start in range(0, len(ids), config['calibration_chunk_targets']):
                chunk = ids[start:start + config['calibration_chunk_targets']]
                targets = [stats.targets[i] for i in chunk]
                shared, residual, count, supported, depth = donor_predictions(stats, sources, targets, stats.feature_mean[ci], temperature)
                truth = np.asarray(stats.response[ci, chunk], float)
                valid = np.isfinite(truth) & supported
                valid[np.arange(len(chunk)), [stats.gene_index[t] for t in targets]] = False
                denominator = np.maximum(valid.sum(1), 1)
                weight = valid / denominator[:, None] / len(ids) / len(training)
                measured = count > 0
                y = np.nan_to_num(truth)
                x = [shared, residual]
                for j in range(2):
                    b[j] += np.sum(weight * measured * x[j] * y)
                    for k in range(2):
                        a[j, k] += np.sum(weight * measured * x[j] * x[k])
                y2 += np.sum(weight * measured * y * y)
                fw = weight * ~measured
                fallback_x2 += np.sum(fw * shared * shared)
                fallback_xy += np.sum(fw * shared * y)
                fallback_y2 += np.sum(fw * y * y)
                dy = stats.depth_response[ci, chunk]
                depth_x2 += np.sum(depth ** 2) / len(ids) / len(training)
                depth_xy += np.sum(depth * dy) / len(ids) / len(training)
                tasks += len(chunk)
        fit = quadratic_fit(a, b, y2, config['residual_ridge'], config['maximum_response_coefficient'])
        generic = float(np.clip(fallback_xy / max(fallback_x2, 1e-20), 0, 1))
        shared_coefficient = float(np.clip(b[0] / max(a[0, 0], 1e-20), 0, config['maximum_response_coefficient']))
        fit.update(temperature=float(temperature), fallback_coefficient=generic, shared_coefficient=shared_coefficient,
                   fallback_loss=float(fallback_y2 - 2 * generic * fallback_xy + generic ** 2 * fallback_x2),
                   depth_coefficient=float(np.clip(depth_xy / max(depth_x2, 1e-20), 0, 1)),
                   sufficient_statistics={'xx': a.tolist(), 'xy': b.tolist(), 'yy': float(y2),
                                          'fallback_x2': float(fallback_x2), 'fallback_xy': float(fallback_xy),
                                          'depth_x2': float(depth_x2), 'depth_xy': float(depth_xy)},
                   inner_folds=fold_records, tasks=tasks)
        records.append(fit)
        print(json.dumps({'stage': 'calibration', 'training_contexts': training, 'temperature': temperature,
                          'coefficients': fit['coefficients'], 'loss': fit['penalized_loss'], 'tasks': tasks}), flush=True)
    best = min(records, key=lambda r: (r['penalized_loss'] + r['fallback_loss'], -r['temperature']))
    return {'training_contexts': list(training), 'selected': best, 'candidates': records,
            'stats_sha256': hash_file(stats.directory / 'complete.json'),
            'training_complete': True, 'optimizer': 'converged bounded quadratic calibration; all admitted training tasks',
            'external_held_response_used': False}


def predict_transfer(stats, fitted, targets, query_mean, method='conditional'):
    selected = fitted['selected']
    shared, residual, count, supported, depth = donor_predictions(stats, fitted['training_contexts'], targets,
                                                                 query_mean, selected['temperature'])
    if method == 'zero':
        prediction = shared * 0; depth *= 0
    elif method == 'shared':
        prediction = shared
    elif method == 'shared_calibrated':
        prediction = np.where(count > 0, selected['shared_coefficient'] * shared, selected['fallback_coefficient'] * shared)
        depth *= selected['depth_coefficient']
    elif method == 'conditional':
        alpha, beta = selected['coefficients']
        prediction = np.where(count > 0, alpha * shared + beta * residual, selected['fallback_coefficient'] * shared)
        depth *= selected['depth_coefficient']
    else:
        raise ValueError('unknown_response_method:' + method)
    prediction[~supported] = 0
    return prediction, depth, count, supported


def evaluate_means(stats, fitted, held, config, directory):
    """Full held task set, including unseen targets and unsupported readouts."""
    from count_generation import desired_mean
    ci = stats.contexts.index(held); ids = np.flatnonzero(stats.available[ci]); rows = []
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(ids), config['calibration_chunk_targets']):
        chunk = ids[start:start + config['calibration_chunk_targets']]
        targets = [stats.targets[i] for i in chunk]
        for method in ['zero', 'shared', 'shared_calibrated', 'conditional']:
            predicted, depth, donors, support = predict_transfer(stats, fitted, targets, stats.feature_mean[ci], method)
            observed = stats.response[ci, chunk]
            for i, target in enumerate(targets):
                axis = stats.masks[ci]
                mean, _ = desired_mean(stats.feature_mean[ci, axis], predicted[i, axis], depth[i], config)
                actual_response = np.zeros(len(stats.genes))
                actual_response[axis] = bulk(mean) - bulk(stats.reference_mean[ci, axis])
                base = axis.copy(); base[stats.gene_index[target]] = False
                for stratum, mask in [('all', base), ('target_readout_supported', base & (donors[i] > 0)),
                                      ('target_readout_unsupported', base & (donors[i] == 0)),
                                      ('no_training_measurement', base & ~support[i])]:
                    n = int(mask.sum())
                    if not n:
                        continue
                    y, p = observed[i, mask].astype(float), actual_response[mask]
                    ac, bc = y - y.mean(), p - p.mean(); denominator = np.linalg.norm(ac) * np.linalg.norm(bc)
                    signal = np.abs(y) > .05
                    rows.append({'context': held, 'target': target, 'method': method, 'stratum': stratum, 'genes': n,
                                 'target_seen': bool((donors[i] > 0).any()),
                                 'mse': float(np.mean((y - p) ** 2)), 'mae': float(np.mean(np.abs(y - p))),
                                 'zero_mse': float(np.mean(y ** 2)), 'correlation': float(ac @ bc / denominator) if denominator > 0 else None,
                                 'direction_accuracy': float(np.mean(np.sign(y[signal]) == np.sign(p[signal]))) if signal.any() else None,
                                 'observed_rms': float(np.sqrt(np.mean(y ** 2))), 'predicted_rms': float(np.sqrt(np.mean(p ** 2))),
                                 'depth_log_prediction': float(depth[i])})
    frame = pd.DataFrame(rows)
    frame.to_parquet(directory / 'mean-metrics.parquet', index=False)
    return frame
