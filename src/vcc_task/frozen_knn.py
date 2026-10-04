"""Inference from a completed DepMap kNN checkpoint, without rebuilding sources."""
import json
import numpy as np

from .response_data import center_numpy


def predict_knn(stored, targets, neighbors):
    if stored['mode'] != 'knn' or not stored['model_state']['complete']:
        raise ValueError('A completed kNN checkpoint is required')
    if stored['response_gauge'] != 'zero_mean_over_source_observed_readouts_excluding_own':
        raise ValueError('Unknown frozen response gauge')
    axis = np.asarray(stored['axis']).astype(str)
    labels = stored['shared_labels']
    features = stored['inference_descriptors']
    lookup = {str(t): i for i, t in enumerate(features['symbols'])}
    label_lookup = {str(t): i for i, t in enumerate(labels)}
    mass = np.unpackbits(stored['shared_support_bits'], count=stored['shared_support_size'])
    mass = mass.reshape(len(labels), len(axis)) > 0
    observed = stored['observed_readouts']
    shared, template = stored['shared_values'], stored['source_template']
    donors = [t for t in labels if t in lookup]
    if len(donors) < neighbors or neighbors < 1:
        raise ValueError('Invalid frozen neighbor count')
    donor_indices = np.asarray([label_lookup[t] for t in donors])
    donor_features = features['raw'][[lookup[t] for t in donors]]
    values = np.zeros((len(targets), len(axis)), dtype=np.float32)
    selected_donors = {}
    for i, target in enumerate(targets):
        if target not in lookup:
            values[i] = template
            selected_donors[target] = []
            continue
        cosine = donor_features @ features['raw'][lookup[target]]
        selected = np.argsort(-cosine, kind='stable')[:neighbors]
        rows = donor_indices[selected]
        support = mass[rows]
        values[i] = np.divide((shared[rows] * support).sum(0), support.sum(0),
                              out=template.copy(), where=support.sum(0) > 0)
        selected_donors[target] = [str(donors[j]) for j in selected]
    gauge_lookup = {str(axis[p]): i for i, p in enumerate(np.flatnonzero(observed))}
    own = np.asarray([gauge_lookup.get(str(t), -1) for t in targets])
    values[:, observed] = center_numpy(values[:, observed], own)
    values[:, ~observed] = 0
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite frozen response')
    coverage = {}
    for i, target in enumerate(targets):
        direct = mass[label_lookup[target]] if target in label_lookup else np.zeros(len(axis), bool)
        donor_rows = [label_lookup[t] for t in selected_donors[target]]
        neighbor_support = mass[donor_rows].any(0) if donor_rows else np.zeros(len(axis), bool)
        coverage[target] = {
            'seen_target': target in label_lookup, 'prior_available': target in lookup,
            'direct_source_contexts': stored['source_contexts'].get(target, []),
            'direct_supervised_readouts': int(direct.sum()),
            'globally_supervised_readouts': int(observed.sum()),
            'measured_untrained_readouts': int((~observed).sum()),
            'neighbor_targets': selected_donors[target],
            'neighbor_source_contexts_json': json.dumps({t:stored['source_contexts'].get(t,[]) for t in selected_donors[target]},sort_keys=True),
            'neighbor_supervised_readouts': int(neighbor_support.sum()),
            'template_fallback_readouts': int((observed & ~neighbor_support).sum()),
            'zero_learned_response': bool(np.count_nonzero(values[i]) == 0),
            'functional_extrapolation': target not in label_lookup and target in lookup,
            'missing_prior_template_fallback': target not in lookup,
            'fixed_on_target_rule': stored['decoder']['own_residual'],
        }
    return values, coverage
