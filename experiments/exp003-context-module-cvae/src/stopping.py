"""Coverage-based validation and stopping, independent of learning-rate policy."""
import numpy as np


def validation_due(cycle, config):
    """Sparse early selection; every cycle in the stopping window is evaluated."""
    return cycle >= config['stopping_start_cycle'] or (cycle - 1) % config['validation_interval_cycles'] == 0


def assess_stopping(history, reference, stale, contexts, config):
    """Pure transition; checkpoint selection deliberately does not use min_delta.

    The assessment at the configured coverage boundary starts patience. Three
    subsequent intervals (four loss assessments) must lie after that boundary.
    The full sliding loss window must pass independently of validation patience.
    """
    point = history[-1]
    cycle, score = point['cycle'], point['validation_score']
    if score is None:
        return {'reference': reference, 'stale': stale, 'stable': False, 'ranges': {}, 'stop': False}
    assert np.isfinite(score)
    window = config['patience_cycles']
    ready = (cycle >= config['stopping_start_cycle'] and point['step'] >= config['kl_warmup_steps'])
    if not ready:
        return {'reference': None, 'stale': 0, 'stable': False, 'ranges': {}, 'stop': False}
    if reference is None:
        reference, stale = score, 0
    elif score - reference >= config['validation_min_delta']:
        reference, stale = score, 0
    else:
        stale += 1
    tail = history[-(window + 1):]
    eligible = (len(tail) == window + 1 and all(p['validation_score'] is not None for p in tail)
                and tail[0]['cycle'] >= config['stopping_start_cycle']
                and tail[0]['step'] >= config['kl_warmup_steps'])
    ranges = {}
    if eligible:
        assert [p['cycle'] for p in tail] == list(range(cycle - window, cycle + 1))
        for context in contexts:
            for metric in ['loss', 'response_loss']:
                values = np.asarray([p['training_monitor'][context][metric] for p in tail], dtype=float)
                assert np.isfinite(values).all()
                ranges[context + '/' + metric] = float(np.ptp(values) / max(abs(values.mean()), config['stability_epsilon']))
    stable = bool(eligible and all(v <= config['loss_relative_range'] for v in ranges.values()))
    # A sliding four-point test already requires stability over ALL three intervals.
    # Do not reset stale merely because an earlier, overlapping window was unstable:
    # that would silently require five stable intervals instead of the agreed three.
    return {'reference': reference, 'stale': stale, 'stable': stable, 'ranges': ranges,
            'stop': bool(stable and stale >= window)}
