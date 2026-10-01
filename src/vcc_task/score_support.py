"""Describe support in official tidy results without changing aggregation rules."""
import polars as pl
from cell_eval2.catalog import CATALOG


def metric_support(raw, names):
    if set(raw.columns) != {'perturbation', 'metric', 'value'}:
        raise ValueError('Unexpected official tidy result schema')
    def observed(name):
        part = raw.filter(pl.col('metric') == name)
        finite = part.filter(pl.col('value').is_finite().fill_null(False))
        return {'raw_rows': part.height,
                'finite_raw_targets': finite['perturbation'].n_unique(),
                'nonfinite_raw_rows': part.height - finite.height}, set(finite['perturbation'])
    support = {}
    for name in names:
        derived = CATALOG[name].derived
        if derived is None:
            support[name], _ = observed(name)
            support[name]['kind'] = 'per_target_raw'
        else:
            numerator, a = observed(derived.numerator)
            denominator, b = observed(derived.denominator)
            support[name] = {'kind': 'aggregate_ratio_of_sums', 'finite_raw_targets': None,
                'numerator': {'metric': derived.numerator, **numerator},
                'denominator': {'metric': derived.denominator, **denominator},
                'finite_component_target_intersection': len(a & b),
                'note': 'No per-target ratio exists; signed finite denominator terms remain included. Official aggregation and rejection rules are unchanged.'}
    return support
