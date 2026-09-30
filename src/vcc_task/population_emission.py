"""Calibrate an empirical NTC population before drawing independent cells.

The sparse plus rank-one form avoids constructing a dense population during IPF.
No target reference is used except when explicitly supplied by calibration code.
"""
import numpy as np
from scipy import sparse

from .counts import conservative_round, profile, validate_counts


def fit_population(ntc, delta_bulk, *, iterations=128, smoothing=1e-6, tolerance=1e-7):
    ntc = sparse.csr_matrix(ntc, dtype=np.float64)
    depth = validate_counts(ntc)
    delta = np.asarray(delta_bulk, dtype=np.float64)
    if delta.shape != (ntc.shape[1],) or not np.isfinite(delta).all():
        raise ValueError('Invalid measured-axis response')
    base = profile(ntc)
    requested = base + delta
    composition = np.expm1(np.clip(requested, 0, 30))
    if composition.sum() <= 0:
        raise ValueError('Empty requested composition')
    composition /= composition.sum()
    projected = np.log1p(50_000 * composition)
    zero = bool(np.count_nonzero(delta) == 0)
    weights = np.ones(ntc.shape[1], dtype=np.float64)
    current = np.asarray(ntc.sum(0)).ravel() / depth.sum()
    step = -1
    if not zero:
        desired = composition * depth.sum()
        for step in range(iterations):
            denominator = ntc @ weights + smoothing * depth * (composition @ weights)
            row_scale = depth / denominator
            mass = weights * (ntc.T @ row_scale + smoothing * composition * (depth @ row_scale))
            weights *= np.divide(desired, mass, out=np.zeros_like(desired), where=mass > 0)
            denominator = ntc @ weights + smoothing * depth * (composition @ weights)
            row_scale = depth / denominator
            current = weights * (ntc.T @ row_scale + smoothing * composition * (depth @ row_scale)) / depth.sum()
            if np.abs(current - composition).sum() <= tolerance:
                break
    return {'weights': weights, 'composition': composition, 'smoothing': smoothing,
            'zero': zero, 'iterations': step + 1,
            'population_l1': float(np.abs(current - composition).sum()),
            'population_bulk_rms': float(np.sqrt(np.mean((np.log1p(50_000 * current) - projected)**2))),
            'projection_rms': float(np.sqrt(np.mean((projected - requested)**2))),
            'projected_bulk': projected}


def draw_population(ntc, fitted, seed, cells=400):
    ntc = sparse.csr_matrix(ntc)
    rng = np.random.default_rng(seed)
    rows = rng.integers(0, ntc.shape[0], cells)
    raw = ntc[rows].toarray().astype(np.float64)
    depth = raw.sum(1).astype(np.int64)
    if fitted['zero']:
        return sparse.csr_matrix(raw.astype(np.int32))
    expected = raw + fitted['smoothing'] * depth[:, None] * fitted['composition']
    expected *= fitted['weights']
    expected *= (depth / expected.sum(1))[:, None]
    counts = conservative_round(expected, depth, rng)
    validate_counts(counts)
    return sparse.csr_matrix(counts)


def state_record(fitted):
    return {k: v for k, v in fitted.items() if k not in ('weights', 'composition', 'projected_bulk')}
