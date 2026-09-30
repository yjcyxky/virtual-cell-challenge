"""Official scores plus perturbation-specific and pathway recovery diagnostics.

No diagnostic defined here replaces the frozen competition metrics. Template
provenance is required because an evaluator-only oracle is not a model input.
"""
import gc
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import polars as pl
from cell_eval2 import compute_metrics, aggregate_metrics_wide, score_metrics
from cell_eval2.baseline import build_run_meta
from cell_eval2.competition import competition_members
from cell_eval2.run import metric_output_names
from challenge import scorer_config, build_reference_bundle

from .common import ROOT, NTC, ref, verified, write_json
from .counts import profile, proportions, validate_counts


def finite_number(value):
    return float(value) if value is not None and np.isfinite(value) else None


def reference_unavailable(error):
    """Only explicit reference/gate failures; API and numeric bugs remain fatal."""
    return any(message in str(error).lower() for message in (
        'baseline leg is degenerate', 'degenerate baseline for metric',
        'no usable replicate scale', 'ratio of sums is undefined or sign-flipped',
        'ratio of sums has nothing to sum'))


def aggregate_with_unavailable(raw, names):
    """Retain EVERY metric, using official aggregation and explicit nulls.

    A refused whole-panel aggregate is never passed to normalized scoring. The
    remaining raw columns are diagnostic reports, not a substitute competition.
    """
    try:
        return aggregate_metrics_wide(raw, metrics=names), {}
    except ValueError as exc:
        if not reference_unavailable(exc):
            raise
    columns, errors, result = {}, {}, None
    for name in names:
        try:
            part = aggregate_metrics_wide(raw, metrics=[name]).select('statistic', name)
        except ValueError as exc:
            if not reference_unavailable(exc):
                raise
            errors[name] = str(exc)
        else:
            if result is None:
                result = part.select('statistic')
            columns[name] = part
    if result is None:
        raise ValueError('No official statistic rows available for a raw diagnostic report')
    for name in sorted(names):
        if name in errors:
            result = result.with_columns(pl.lit(None, dtype=pl.Float64).alias(name))
        else:
            result = result.join(columns[name], on='statistic', how='left', maintain_order='left')
    return result, errors


def pearson(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(x) < 3 or not np.isfinite(x).all() or not np.isfinite(y).all():
        return None
    a, b = x - x.mean(), y - y.mean()
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / denominator) if denominator > 1e-12 else None


def pathway_vectors(pathway_ref, genes, minimum=5):
    """Frozen local Reactome gene sets; aliases stay on the frozen symbol axis."""
    path = verified(pathway_ref)
    with ZipFile(path) as archive:
        names = [n for n in archive.namelist() if n.endswith('.gmt')]
        if len(names) != 1:
            raise ValueError('Ambiguous pathway archive')
        text = archive.read(names[0]).decode()
    lookup = {str(g): i for i, g in enumerate(genes)}
    vectors, records = [], []
    for line in text.splitlines():
        parts = line.split('\t')
        if len(parts) < 3:
            continue
        members = set(parts[2:])
        positions = sorted({lookup[g] for g in members if g in lookup})
        if len(positions) < minimum:
            continue
        vector = np.zeros(len(genes))
        vector[positions] = 1 / np.sqrt(len(positions))
        vectors.append(vector)
        records.append({'pathway': parts[0], 'id': parts[1], 'measured_members': len(positions),
                        'listed_members': len(members), 'coverage': len(positions) / len(members)})
    if not vectors:
        raise ValueError('No pathway meets the frozen measurement coverage')
    return np.stack(vectors), records


def response_diagnostics(prediction, reference, targets, template, template_scope, pathway_ref, directory,
                         excluded_targets=None):
    if not prediction.var_names.equals(reference.var_names):
        raise ValueError('Response diagnostics require identical measured axes')
    genes = reference.var_names.to_numpy()
    ctrl = reference.X[reference.obs.target_gene.eq(NTC).to_numpy()]
    origin = profile(ctrl)
    real = np.stack([profile(reference.X[reference.obs.target_gene.eq(t).to_numpy()]) - origin for t in targets])
    pred = np.stack([profile(prediction.X[prediction.obs.target_gene.eq(t).to_numpy()]) - origin for t in targets])
    # One fixed feature space for template directions and all target comparisons.
    allowed = ~np.isin(genes, targets if excluded_targets is None else excluded_targets)
    if template.shape != (len(genes),) or not np.isfinite(template).all():
        raise ValueError('Template/gene-axis mismatch')
    direction = np.asarray(template, dtype=float).copy()
    direction[~allowed] = 0
    norm = np.linalg.norm(direction)
    if norm > 1e-12:
        direction /= norm
    else:
        direction[:] = 0
    real[:, ~allowed] = 0; pred[:, ~allowed] = 0
    real_residual = real - np.outer(real @ direction, direction)
    pred_residual = pred - np.outer(pred @ direction, direction)
    table = []
    for i, target in enumerate(targets):
        a, b = real[i, allowed], pred[i, allowed]
        ar, br = real_residual[i, allowed], pred_residual[i, allowed]
        real_counts = reference.X[reference.obs.target_gene.eq(target).to_numpy()]
        pred_counts = prediction.X[prediction.obs.target_gene.eq(target).to_numpy()]
        real_cpm = np.asarray(proportions(real_counts).mean(0)).ravel() * 1e6
        pred_cpm = np.asarray(proportions(pred_counts).mean(0)).ravel() * 1e6
        table.append({'target': target, 'response_pearson': pearson(a, b),
            'residual_pearson': pearson(ar, br), 'response_rms_error': float(np.sqrt(np.mean((a-b)**2))),
            'residual_rms_error': float(np.sqrt(np.mean((ar-br)**2))),
            'real_response_rms': float(np.sqrt(np.mean(a**2))),
            'pred_response_rms': float(np.sqrt(np.mean(b**2))),
            'real_residual_rms': float(np.sqrt(np.mean(ar**2))),
            'pred_residual_rms': float(np.sqrt(np.mean(br**2))),
            'cell_mean_cpm_rms_error': float(np.sqrt(np.mean((real_cpm[allowed]-pred_cpm[allowed])**2))),
            'real_template_coefficient': float(real[i] @ direction),
            'pred_template_coefficient': float(pred[i] @ direction)})
    pd.DataFrame(table).to_csv(directory / 'response-diagnostics.csv', index=False)
    vectors, records = pathway_vectors(pathway_ref, genes[allowed])
    r, p = real[:, allowed] @ vectors.T, pred[:, allowed] @ vectors.T
    rr, pr = real_residual[:, allowed] @ vectors.T, pred_residual[:, allowed] @ vectors.T
    for i, record in enumerate(records):
        record.update(response_pearson=pearson(r[:, i], p[:, i]),
                      residual_pearson=pearson(rr[:, i], pr[:, i]),
                      response_rmse=float(np.sqrt(np.mean((r[:, i]-p[:, i])**2))),
                      residual_rmse=float(np.sqrt(np.mean((rr[:, i]-pr[:, i])**2))),
                      real_residual_rms=float(np.sqrt(np.mean(rr[:, i]**2))),
                      pred_residual_rms=float(np.sqrt(np.mean(pr[:, i]**2))), targets=len(targets))
    pd.DataFrame(records).to_csv(directory / 'pathway-recovery.csv', index=False)
    total_energy = float(np.sum(pred**2))
    finite = [r['residual_pearson'] for r in table if r['residual_pearson'] is not None]
    return {'template_scope': template_scope, 'pathway_ref': pathway_ref,
        'target_genes_excluded': 'all panel targets from diagnostic feature space',
        'template_direction_defined': bool(norm > 1e-12),
        'pred_template_energy_fraction': float(np.sum((pred @ direction)**2) / total_energy) if total_energy else None,
        'mean_residual_pearson': float(np.mean(finite)) if finite else None,
        'mean_response_rms_error': float(np.mean([r['response_rms_error'] for r in table])),
        'mean_residual_rms_error': float(np.mean([r['residual_rms_error'] for r in table])),
        'mean_pred_residual_rms': float(np.mean([r['pred_residual_rms'] for r in table])),
        'valid_residual_targets': len(finite), 'pathways': len(records),
        'files': [ref(directory / 'response-diagnostics.csv'), ref(directory / 'pathway-recovery.csv')]}


def de_call_diagnostics(directory):
    real = pl.read_parquet(directory / 'de_real.parquet').to_pandas()
    pred = pl.read_parquet(directory / 'de_pred.parquet').to_pandas()
    rows = []
    for target, a in real.groupby('target', sort=True):
        a = a.loc[a.feature.ne(target)].set_index('feature')
        b = pred.loc[pred.target.eq(target) & pred.feature.ne(target)].set_index('feature')
        joined = a[['p_adj', 'log2_fold_change']].join(b[['p_adj', 'log2_fold_change']],
                                                     how='outer', lsuffix='_real', rsuffix='_pred')
        r, p = joined.p_adj_real.lt(.05), joined.p_adj_pred.lt(.05)
        adjudicable = joined.log2_fold_change_real.notna() & joined.log2_fold_change_real.ne(0)
        correct = np.sign(joined.log2_fold_change_real) == np.sign(joined.log2_fold_change_pred)
        rows.append({'target': target, 'real_significant': int(r.sum()), 'pred_significant': int(p.sum()),
            'overlap': int((r & p).sum()), 'false_positive_calls': int((p & ~r).sum()),
            'missed_calls': int((r & ~p).sum()), 'pred_adjudicable': int((p & adjudicable).sum()),
            'pred_correct_direction': int((p & adjudicable & correct).sum())})
    path = directory / 'de-call-diagnostics.csv'
    pd.DataFrame(rows).to_csv(path, index=False)
    return ref(path)


def ensure_bundle(real, directory, bundle_id, runtime):
    directory = Path(directory)
    marker = directory.parent / (directory.name + '-result.json')
    if marker.exists():
        result = json.loads(marker.read_text())
        for record in result.get('files', []):
            verified(record)
        return result
    original = directory
    attempt = 0
    while directory.exists() and any(directory.iterdir()):
        attempt += 1
        directory = original.with_name(original.name + f'-attempt-{attempt}')
    try:
        build_reference_bundle(real, directory, bundle_id, **runtime)
    except ValueError as exc:
        if not reference_unavailable(exc):
            raise
        result = {'available': False, 'official_rejection': str(exc), 'files': []}
    else:
        result = {'available': True, 'files': [ref(p) for p in sorted(directory.rglob('*')) if p.is_file()]}
    result['bundle_directory'] = str(directory.relative_to(ROOT))
    result['preserved_incomplete_attempts'] = attempt
    write_json(marker, result)
    return result


def evaluate_counts(prediction, real, bundle, bundle_status, directory, runtime,
                    targets, template, template_scope, pathway_ref, diagnostic_targets=None):
    directory = Path(directory)
    marker = directory / 'result.json'
    if marker.exists():
        result = json.loads(marker.read_text())
        for item in result['files']:
            verified(item)
        return result
    directory.mkdir(parents=True, exist_ok=True)
    validate_counts(prediction.X); validate_counts(real.X)
    if set(prediction.obs.target_gene) != set(real.obs.target_gene):
        raise ValueError('Prediction/reference perturbation panel mismatch')
    if not prediction.var_names.equals(real.var_names):
        raise ValueError('Prediction/reference measured gene axis mismatch')
    cfg = scorer_config(outdir=str(directory), **runtime)
    raw = compute_metrics(prediction, real, config=cfg, write_de=True)
    raw.write_parquet(directory / 'raw.parquet')
    aggregate, aggregate_errors = aggregate_with_unavailable(raw, metric_output_names(cfg))
    meta = build_run_meta(cfg, real, prediction)
    aggregate.write_csv(directory / 'aggregate.csv')
    write_json(directory / 'run_meta.json', meta)
    raw_means = aggregate.filter(pl.col('statistic') == 'mean').to_dicts()[0]
    result = {'raw': {k: finite_number(v) for k, v in raw_means.items() if k != 'statistic'},
              'raw_aggregate': json.loads(aggregate.to_pandas().to_json(orient='records')),
              'normalized': None, 'Overall': None,
              'raw_aggregation_rejections': aggregate_errors,
              'official_rejection': bundle_status.get('official_rejection')}
    if aggregate_errors:
        result['official_rejection'] = 'Official raw aggregation unavailable: ' + '; '.join(aggregate_errors.values())
    if bundle_status['available'] and not aggregate_errors:
        try:
            scored = score_metrics(aggregate, real_bundle=str(bundle), user_meta=meta)
        except ValueError as exc:
            if not reference_unavailable(exc):
                raise
            result['official_rejection'] = str(exc)
        else:
            scored.write_csv(directory / 'scores.csv')
            result['normalized'] = {m: finite_number(scored.filter(pl.col('metric') == m)['from_replicate'].item())
                                    for m in competition_members()}
            result['Overall'] = finite_number(scored.filter(pl.col('metric') == 'avg_score')['from_replicate'].item())
            if all(v is not None for v in result['normalized'].values()):
                if result['Overall'] is None or not np.isclose(result['Overall'], np.mean(list(result['normalized'].values()))):
                    raise ValueError('Official six-member Overall mismatch')
    result['de_diagnostics_ref'] = de_call_diagnostics(directory)
    result['response_diagnostics'] = response_diagnostics(prediction, real,
        targets if diagnostic_targets is None else diagnostic_targets, template,
        template_scope, pathway_ref, directory, excluded_targets=targets)
    result['files'] = [ref(p) for p in sorted(directory.iterdir()) if p.is_file() and p.name != marker.name]
    write_json(marker, result)
    gc.collect()
    return result
