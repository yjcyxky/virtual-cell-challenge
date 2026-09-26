"""Separate a declared response mean from count-valued distribution generation."""
import numpy as np


def desired_mean(control_mean, response, depth_log_change, config):
    control_mean, response = np.asarray(control_mean, float), np.asarray(response, float)
    if not np.isfinite(control_mean).all() or not np.isfinite(response).all() or (control_mean < 0).any():
        raise ValueError('invalid_mean_or_response')
    depth = control_mean.sum()
    if depth <= 0:
        raise ValueError('empty_control_profile')
    baseline = np.log1p(config['bulk_target_sum'] * control_mean / depth)
    proposed = baseline + response
    if proposed.max(initial=0) > 50:
        raise ValueError('response_exceeds_supported_mean_scale')
    mass = np.expm1(np.maximum(proposed, 0))
    if mass.sum() <= 0:
        raise ValueError('response_removes_all_expression')
    bounded_depth = float(np.clip(depth_log_change, -config['maximum_depth_log_change'], config['maximum_depth_log_change']))
    mean = mass / mass.sum() * depth * np.exp(bounded_depth)
    return mean, {'negative_log_profile_coordinates': int((proposed < 0).sum()),
                  'depth_log_change': bounded_depth, 'depth_log_clamped': bounded_depth != depth_log_change}


def negative_binomial(mean, control_mean, control_variance, cells, rng):
    theta = np.clip(control_mean ** 2 / np.maximum(control_variance - control_mean, 1e-6), .05, 1e6)
    probability = theta / (theta + mean)
    return rng.negative_binomial(theta, probability, size=(cells, len(mean)))


def template_counts(templates, mean, rng, iterations, tolerance, identity=False):
    """Fit prescribed row/column margins, then unbiased integer rounding.

    The fitted rows preserve the sampled library-size CV; column masses match the
    prescribed population mean. Zero intervention is explicitly the empirical NTC.
    """
    templates = np.asarray(templates)
    if identity:
        if not np.array_equal(templates, np.floor(templates)) or (templates < 0).any():
            raise ValueError('templates_must_be_counts')
        return templates.astype(np.uint32), {'iterations': 0, 'row_relative_error': 0., 'identity': True}
    depths = templates.sum(1, dtype=np.float64)
    if (depths <= 0).any():
        raise ValueError('empty_template_cell')
    target_rows = depths / depths.mean() * mean.sum()
    target_columns = mean * len(templates)
    # A declared small positive support permits induced genes absent from all templates.
    matrix = templates.astype(np.float64) + 1e-4 * mean[None, :]
    matrix[:, mean == 0] = 0
    error = float('inf')
    for step in range(iterations):
        matrix *= (target_rows / matrix.sum(1))[:, None]
        column_sums = matrix.sum(0)
        matrix *= np.divide(target_columns, column_sums, out=np.zeros_like(mean), where=column_sums > 0)[None, :]
        error = float(np.max(np.abs(matrix.sum(1) - target_rows) / target_rows))
        if error <= tolerance:
            break
    if not np.isfinite(matrix).all() or error > tolerance:
        raise RuntimeError(f'count_margin_fit_not_converged: {error}, iterations={iterations}')
    floor = np.floor(matrix)
    counts = floor + (rng.random(matrix.shape) < matrix - floor)
    return counts, {'iterations': step + 1, 'row_relative_error': error, 'identity': False,
                    'support_pseudomass_fraction': 1e-4}


def generate_counts(control_mean, control_variance, templates, response, depth_log_change, method, seed, config):
    rng = np.random.default_rng(seed)
    mean, diagnostics = desired_mean(control_mean, response, depth_log_change, config)
    if method == 'template':
        counts, details = template_counts(templates, mean, rng, config['ipf_iterations'], config['ipf_tolerance'],
                                          identity=bool(np.all(response == 0) and depth_log_change == 0))
        diagnostics.update(details)
    elif method == 'nb':
        counts = negative_binomial(mean, control_mean, control_variance, len(templates), rng)
    else:
        raise ValueError('unknown_count_generator:' + method)
    depth = counts.sum(1, dtype=np.float64)
    if (counts < 0).any() or not np.isfinite(counts).all() or counts.max(initial=0) > np.iinfo(np.uint32).max:
        raise ValueError('invalid_generated_count_values')
    if (depth <= 0).any() or depth.max(initial=0) > config['max_counts_per_cell']:
        raise ValueError('generated_counts_violate_official_limit')
    diagnostics.update(mean_depth=float(depth.mean()), depth_cv=float(depth.std() / depth.mean()),
                       mean_profile_relative_l1=float(np.abs(counts.mean(0) - mean).sum() / mean.sum()),
                       expected_depth=float(mean.sum()), nonzero_fraction=float(np.count_nonzero(counts) / counts.size))
    return counts.astype(np.uint32), diagnostics
