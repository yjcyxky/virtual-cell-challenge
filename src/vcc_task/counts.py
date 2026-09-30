"""Explicit measured-axis counts and empirical NTC-anchored emission.

None of these descriptive statistics replaces an official metric.
"""
import numpy as np
from scipy import sparse


def validate_counts(matrix, maximum=1_000_000):
    values = matrix.data if sparse.issparse(matrix) else np.asarray(matrix)
    if not np.isfinite(values).all() or np.any(values < 0) or np.any(values != np.floor(values)):
        raise ValueError('Expected finite nonnegative integer counts')
    depth = np.asarray(matrix.sum(axis=1)).ravel()
    if np.any(depth <= 0) or np.any(depth > maximum):
        raise ValueError('Empty or out-of-contract library')
    return depth


def profile(matrix, total=50_000):
    summed = np.asarray(matrix.sum(axis=0)).ravel().astype(np.float64)
    if summed.sum() <= 0:
        raise ValueError('Empty population')
    return np.log1p(total * summed / summed.sum())


def proportions(matrix):
    depth = validate_counts(matrix)
    return sparse.diags(1 / depth) @ sparse.csr_matrix(matrix)


def describe(matrix):
    depth = validate_counts(matrix)
    matrix = sparse.csr_matrix(matrix)
    detected = np.asarray((matrix > 0).sum(1)).ravel()
    mean = np.asarray(matrix.mean(0)).ravel()
    variance = np.maximum(np.asarray(matrix.power(2).mean(0)).ravel() - mean**2, 0)
    levels = [0, .05, .25, .5, .75, .95, 1]
    return {'cells': matrix.shape[0], 'measured_genes': matrix.shape[1],
            'quantile_levels': levels, 'library_quantiles': np.quantile(depth, levels).tolist(),
            'detected_quantiles': np.quantile(detected, levels).tolist(),
            'zero_fraction': float(1 - detected.mean() / matrix.shape[1]),
            'mean_gene_variance': float(variance.mean()),
            'library_detected_correlation': float(np.corrcoef(depth, detected)[0, 1])
            if np.std(depth) > 0 and np.std(detected) > 0 else None}


def thin_counts(matrix, fraction, seed):
    if not 0 < fraction <= 1:
        raise ValueError('Thinning cannot create observations')
    result = sparse.csr_matrix(matrix, dtype=np.int64).copy()
    if fraction < 1:
        result.data = np.random.default_rng(seed).binomial(result.data, fraction)
        result.eliminate_zeros()
    validate_counts(result)
    return result


def conservative_round(expected, library, rng):
    """Randomized systematic rounding: unbiased marginals, exact integer row sums.

    A new column permutation per cell avoids a fixed gene-order correlation.
    This is discretization, not an independent biological noise model.
    """
    expected = np.asarray(expected, dtype=np.float64)
    library = np.asarray(library, dtype=np.int64)
    if np.any(expected < 0) or not np.isfinite(expected).all():
        raise ValueError('Invalid expected counts')
    if not np.allclose(expected.sum(1), library, rtol=0, atol=1e-6):
        raise ValueError('Expected counts must have the declared row totals')
    result = np.floor(expected).astype(np.int32)
    for i, total in enumerate(library):
        remainder = int(total - result[i].sum())
        if remainder == 0:
            continue
        fractions = expected[i] - result[i]
        order = rng.permutation(len(fractions))
        cumulative = np.cumsum(fractions[order])
        # Floating error in IPF's row scaling must not lose the last count.
        cumulative[-1] = remainder
        indices = np.searchsorted(cumulative, rng.random() + np.arange(remainder), side='right')
        if len(np.unique(indices)) != remainder:
            raise ValueError('Rounding fractional mass is inconsistent')
        result[i, order[indices]] += 1
    if not np.array_equal(result.sum(1), library):
        raise ValueError('Rounding failed count conservation')
    return result


def emit_counts(ntc, delta_bulk, seed, cells=400, iterations=64,
                smoothing=1e-6, tolerance=1e-7, rounding='conservative'):
    """Real NTC states, requested bulk log-CP50K effect, explicit count emission.

    Nonzero effects are projected onto the composition simplex, then fitted by
    row/column scaling. Zero effect is exact resampling, with no added noise.
    """
    rng = np.random.default_rng(seed)
    ntc = sparse.csr_matrix(ntc)
    validate_counts(ntc)
    delta = np.asarray(delta_bulk, dtype=np.float64)
    if delta.shape != (ntc.shape[1],) or not np.isfinite(delta).all():
        raise ValueError('Response/gene-axis mismatch')
    rows = rng.integers(0, ntc.shape[0], cells)
    raw = ntc[rows].toarray().astype(np.float64)
    depth = raw.sum(1).astype(np.int64)
    baseline = profile(ntc)
    requested = baseline + delta
    if np.count_nonzero(delta) == 0:
        return sparse.csr_matrix(raw.astype(np.int32)), {
            'zero_identity': True, 'row_totals_preserved': True,
            'composition_projection_rms': 0., 'ipf_l1': 0., 'iterations': 0,
            'bulk_realization_rms': float(np.sqrt(np.mean((profile(raw) - baseline)**2)))}
    composition = np.expm1(np.clip(requested, 0, 30))
    if composition.sum() <= 0:
        raise ValueError('Requested response has no positive mass')
    composition /= composition.sum()
    projected = np.log1p(50_000 * composition)
    expected = raw + smoothing * depth[:, None] * composition
    desired = depth.sum() * composition
    for step in range(iterations):
        current = expected.sum(0)
        expected *= np.divide(desired, current, out=np.zeros_like(desired), where=current > 0)
        row_mass = expected.sum(1)
        if np.any(row_mass <= 0):
            raise ValueError('No feasible mass for an empirical cell state')
        expected *= (depth / row_mass)[:, None]
        error = float(np.abs(expected.sum(0) / depth.sum() - composition).sum())
        if error <= tolerance:
            break
    if rounding == 'conservative':
        counts = conservative_round(expected, depth, rng)
    elif rounding == 'independent':
        counts = np.floor(expected + rng.random(expected.shape)).astype(np.int32)
    else:
        raise ValueError('Unregistered rounding rule')
    validate_counts(counts)
    return sparse.csr_matrix(counts), {
        'zero_identity': False, 'row_totals_preserved': bool(np.array_equal(counts.sum(1), depth)),
        'composition_projection_rms': float(np.sqrt(np.mean((projected - requested)**2))),
        'ipf_l1': error, 'iterations': step + 1,
        'bulk_realization_rms': float(np.sqrt(np.mean((profile(counts) - projected)**2)))}
