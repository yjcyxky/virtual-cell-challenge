"""Train-only response estimators and uncertainty-aware operational classifications."""
import numpy as np
from scipy.stats import norm

TEMPERATURES = [0., .01, .1, 1., np.inf]
LAMBDAS = [0., .25, .5, .75, 1.]


def relation(mean, variance, tolerance):
    """0 unknown, 1 equivalent, 2 different; simultaneous within one gene/set."""
    mean, variance = np.asarray(mean, float), np.asarray(variance, float)
    count, genes = mean.shape
    state = np.zeros(genes, np.int8)
    spread = np.full(genes, np.nan)
    if count < 2:
        return state, spread
    pairs = count * (count - 1) // 2
    z90, z95 = norm.ppf(1 - .1 / (2 * pairs)), norm.ppf(1 - .05 / (2 * pairs))
    equivalent = np.ones(genes, bool)
    different = np.zeros(genes, bool)
    valid = np.isfinite(mean).all(0) & np.isfinite(variance).all(0) & (variance >= 0).all(0)
    spread = np.max(mean, axis=0) - np.min(mean, axis=0)
    for a in range(count):
        for b in range(a):
            distance = np.abs(mean[a] - mean[b])
            se = np.sqrt(variance[a] + variance[b])
            equivalent &= distance + z90 * se < tolerance
            different |= distance - z95 * se > tolerance
    state[valid & equivalent] = 1
    state[valid & different] = 2
    return state, spread


def classify(baseline, baseline_variance, response, response_variance, tolerance):
    b, bspread = relation(baseline, baseline_variance, tolerance)
    r, rspread = relation(response, response_variance, tolerance)
    n, genes = response.shape
    activity = np.zeros(genes, np.int8)  # 1 reliable near-zero in all, 2 active in at least one
    conserved_nonzero = np.zeros(genes, bool)
    if n:
        valid = np.isfinite(response).all(0) & np.isfinite(response_variance).all(0)
        z90, z95 = norm.ppf(1 - .1 / (2 * n)), norm.ppf(1 - .05 / (2 * n))
        se = np.sqrt(response_variance)
        nearzero = (np.abs(response) + z90 * se < tolerance / 2).all(0)
        active = (np.abs(response) - z95 * se > tolerance / 2).any(0)
        activity[valid & nearzero] = 1
        activity[valid & active] = 2
        same_direction = (response - z95 * se > 0).all(0) | (response + z95 * se < 0).all(0)
        conserved_nonzero = valid & (r == 1) & same_direction & ~nearzero
    quadrant = np.zeros(genes, np.int8)
    known = (b > 0) & (r > 0)
    quadrant[known] = (b[known] - 1) * 2 + r[known]
    return {'baseline_state': b, 'response_state': r, 'quadrant': quadrant, 'activity': activity,
            'conserved_nonzero': conserved_nonzero, 'baseline_spread': bspread, 'response_spread': rspread}


def ntc_distances(baseline):
    x = baseline - baseline.mean(axis=1, keepdims=True)
    normed = np.linalg.norm(x, axis=1, keepdims=True)
    x = np.divide(x, normed, out=np.zeros_like(x), where=normed > 0)
    return np.clip(1 - x @ x.T, 0, 2)


def response_mean(response, available, distance=None, temperature=np.inf):
    """Context-balanced mean; missing contexts/genes never become zero labels."""
    count, targets, genes = response.shape
    weights = available.astype(float)
    if distance is not None and not np.isinf(temperature):
        distance = np.asarray(distance)
        if temperature == 0:
            masked = np.where(available, distance[:, None], np.inf)
            weights = available * (masked == masked.min(0))
        else:
            weights *= np.exp(-(distance - distance.min())[:, None] / temperature)
    finite = np.isfinite(response) & available[:, :, None]
    denominator = (weights[:, :, None] * finite).sum(0)
    numerator = (np.where(finite, response, 0) * weights[:, :, None]).sum(0)
    return np.divide(numerator, denominator, out=np.full((targets, genes), np.nan), where=denominator > 0)


def choose_hyperparameters(response, available, baseline):
    """Only inner training contexts are visible; rows balance target/context equally."""
    errors = {t: [] for t in TEMPERATURES}
    shrink_errors = {lam: [] for lam in LAMBDAS}
    distance = ntc_distances(baseline)
    for held in range(len(response)):
        train = np.arange(len(response)) != held
        eligible = available[held] & (available[train].sum(0) >= 1)
        if not eligible.any():
            continue
        truth = response[held, eligible]
        shared = response_mean(response[train], available[train])[eligible]
        for lam in LAMBDAS:
            shrink_errors[lam].extend(np.nanmean((truth - lam * shared) ** 2, axis=1).tolist())
        kernel_eligible = available[held] & (available[train].sum(0) >= 2)
        for temp in TEMPERATURES:
            if kernel_eligible.any():
                prediction = response_mean(response[train], available[train], distance[held, train], temp)[kernel_eligible]
                errors[temp].extend(np.nanmean((response[held, kernel_eligible] - prediction) ** 2, axis=1).tolist())
    if not shrink_errors[0.]:
        # No test-dependent fallback. Fixed prior shared amplitude, fixed preregistered kernel.
        return 1., .1, {'status': 'not_identifiable', 'reason': 'fewer_than_3_usable_training_contexts',
                        'lambda_fallback': 1., 'temperature_fallback': .1}
    scores = {lam: float(np.mean(values)) for lam, values in shrink_errors.items()}
    kernel = {temp: float(np.mean(values)) for temp, values in errors.items() if values}
    best_lambda = min(LAMBDAS, key=lambda lam: (scores[lam], lam))
    best_temperature = min(TEMPERATURES, key=lambda temp: (kernel[temp], -temp)) if kernel else .1
    return best_lambda, best_temperature, {'status': 'trained_inner_context_CV', 'lambda_MSE': scores,
                                          'temperature_MSE': {str(k): v for k, v in kernel.items()},
                                          'inner_target_contexts': len(shrink_errors[0.]),
                                          'kernel_status': 'trained' if kernel else 'fixed_insufficient_training_contexts'}


def score(truth, prediction):
    valid = np.isfinite(truth) & np.isfinite(prediction)
    if not valid.any():
        return {'genes': 0, 'MSE': None, 'MAE': None, 'correlation': None}
    a, b = truth[valid], prediction[valid]
    mse = float(np.mean((a - b) ** 2))
    zero = float(np.mean(a ** 2))
    ac, bc = a - a.mean(), b - b.mean()
    denominator = np.linalg.norm(ac) * np.linalg.norm(bc)
    return {'genes': int(valid.sum()), 'MSE': mse, 'MAE': float(np.mean(np.abs(a - b))),
            'zero_MSE': zero, 'relative_MSE_improvement': 1 - mse / zero if zero > 0 else None,
            'correlation': float(ac @ bc / denominator) if denominator > 0 else None,
            'sign_agreement': float(np.mean(np.sign(a) == np.sign(b))),
            'prediction_RMS': float(np.sqrt(np.mean(b ** 2))), 'truth_RMS': float(np.sqrt(zero))}
