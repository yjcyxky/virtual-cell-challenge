"""Fit the CRISPRi shared-response baseline and emit count-valued cell profiles.

The learned object is a response mean plus two donor-support calibrations.
Count emission is explicitly an approximation, with a separate on-target prior.
"""
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from evaluate import load_group, responses
from estimators import response_mean, score
from heterogeneity import safe_symbols
from profile_responses import write_json
from rna import hash_file, mapping_audit
from summarize import balanced_metrics

ROOT = Path(__file__).resolve().parents[3]


def squared_components(truth, prediction):
    valid = np.isfinite(truth) & np.isfinite(prediction)
    n = valid.sum(axis=1)
    a, b = np.where(valid, truth, 0).astype(float), np.where(valid, prediction, 0).astype(float)
    return [np.divide(x.sum(axis=1), n, out=np.full(len(n), np.nan), where=n > 0)
            for x in [a * a, a * b, b * b]]


def calibrate(response, available, grid):
    """Balance donor cases within target, then targets within held background.

Single-donor deployment is calibrated with ordered pairs of distinct backgrounds.
Multi-donor deployment uses all remaining available backgrounds, with >=2 donors.
Neither function arguments nor loss selection include an external held background.
"""
    curves = {'single': [], 'multiple': []}
    counts = {'single': 0, 'multiple': 0}
    grid = np.asarray(grid, float)
    for held in range(len(response)):
        other = np.arange(len(response)) != held
        sums = np.zeros((len(available[held]), len(grid)))
        cases = np.zeros(len(available[held]), int)
        for donor in np.flatnonzero(other):
            ids = np.flatnonzero(available[held] & available[donor])
            if not len(ids):
                continue
            a, b, c = squared_components(response[held, ids], response[donor, ids])
            losses = a[:, None] - 2 * b[:, None] * grid + c[:, None] * grid ** 2
            good = np.isfinite(losses).all(1)
            sums[ids[good]] += losses[good]
            cases[ids[good]] += 1
        good = cases > 0
        if good.any():
            curves['single'].append((sums[good] / cases[good, None]).mean(0))
            counts['single'] += int(good.sum())
        ids = np.flatnonzero(available[held] & (available[other].sum(0) >= 2))
        if len(ids):
            shared = response_mean(response[other], available[other])[ids]
            a, b, c = squared_components(response[held, ids], shared)
            losses = a[:, None] - 2 * b[:, None] * grid + c[:, None] * grid ** 2
            good = np.isfinite(losses).all(1)
            if good.any():
                curves['multiple'].append(losses[good].mean(0))
                counts['multiple'] += int(good.sum())
    result = {}
    for name in curves:
        values = np.mean(curves[name], axis=0) if curves[name] else None
        result[name] = {'lambda': float(grid[np.argmin(values)]) if values is not None else 0.,
                        'grid': grid.tolist(), 'MSE': values.tolist() if values is not None else [],
                        'held_backgrounds': len(curves[name]), 'target_backgrounds': counts[name],
                        'status': 'trained' if values is not None else 'zero_response_no_calibration_support'}
    return result


def shrink(shared, donor_count, fitted):
    coefficient = np.where(donor_count == 1, fitted['single']['lambda'], fitted['multiple']['lambda'])
    if coefficient.ndim < shared.ndim:
        coefficient = coefficient[..., None]
    return np.where(np.isfinite(shared), shared * coefficient, 0.)


def outer_prediction(response, available, held, grid):
    train = np.arange(len(response)) != held
    fitted = calibrate(response[train], available[train], grid)
    shared = response_mean(response[train], available[train])
    donors = available[train].sum(0)
    return fitted, shared, shrink(shared, donors, fitted), donors


def validate_sources(collection, panels):
    report = json.loads((collection / 'report.json').read_text())
    assert report['status'] == 'completed'
    selected = [r for r in report['contexts'] if r['panel_id'] in panels]
    assert {r['panel_id'] for r in selected} == set(panels)
    identities = []
    for item in selected:
        meta = item['source_metadata']
        assert meta['intervention_mechanism'] == 'CRISPRi' and meta['additional_experimental_treatment'] is False
        for name in ['moments.h5', 'control-moments.npz', 'tasks.parquet', 'task-support.parquet']:
            path = collection / item['context_id'] / name
            digest = hash_file(path)
            if digest != item['artifacts'][name]:
                raise ValueError('frozen_training_input_changed: ' + str(path))
            identities.append({'path': str(path.relative_to(ROOT)), 'sha256': digest})
    return selected, identities


def deployable_responses(collection, reports, targets, genes, output):
    """Union of measured readouts; missing donors never contribute zero labels."""
    hgnc_path = ROOT / 'data/raw/networks/hgnc_complete_set.txt'
    mapping = mapping_audit(genes, pd.read_csv(hgnc_path, sep='\t', low_memory=False), genes)
    mapping.to_parquet(output / 'official-gene-mapping.parquet', index=False)
    official_map = {g: i for i, g in enumerate(safe_symbols(mapping)) if pd.notna(g)}
    target_map = {g: i for i, g in enumerate(targets)}
    lines = sorted({r['source_metadata']['cell_line'] for r in reports})
    values = np.zeros((len(lines), len(targets), len(genes)), np.float64)
    panels_per_gene = np.zeros(values.shape, np.uint8)
    target_presence = np.zeros(values.shape[:2], bool)
    for report in reports:
        folder = collection / report['context_id']
        tasks = pd.read_parquet(folder / 'tasks.parquet')
        support = pd.read_parquet(folder / 'task-support.parquet')
        supported = set(support.loc[support.variant.eq('full') & support.status.eq('completed'), 'task_uid'])
        control = np.load(folder / 'control-moments.npz')['mean'][0]
        li = lines.index(report['source_metadata']['cell_line'])
        with h5py.File(folder / 'moments.h5') as hf:
            symbols = hf['safe_symbol'].asstr()[:]
            src = np.array([i for i, symbol in enumerate(symbols) if symbol in official_map], int)
            dest = np.array([official_map[symbols[i]] for i in src], int)
            for target, group in tasks.groupby('canonical_target'):
                if target not in target_map:
                    continue
                rows = [i for i in group.index if tasks.iloc[i].task_uid in supported]
                if not rows:
                    continue
                pi = target_map[target]
                target_presence[li, pi] = True
                response = np.mean([hf['target'][i, 0, :][src] - control[src] for i in rows], axis=0)
                finite = np.isfinite(response)
                values[li, pi, dest[finite]] += response[finite]
                panels_per_gene[li, pi, dest[finite]] += 1
    values = np.divide(values, panels_per_gene, out=np.full_like(values, np.nan), where=panels_per_gene > 0)
    donors = np.isfinite(values).sum(0).astype(np.uint8)
    shared = np.divide(np.nansum(values, axis=0), donors, out=np.full(donors.shape, np.nan), where=donors > 0)
    ledger = pd.DataFrame({'target_gene': targets, 'backgrounds': target_presence.sum(0),
                           'donor_backgrounds': [','.join(np.array(lines)[target_presence[:, i]]) for i in range(len(targets))],
                           'genes_no_donor': (donors == 0).sum(1), 'genes_one_donor': (donors == 1).sum(1),
                           'genes_multiple_donors': (donors >= 2).sum(1)})
    ledger.to_parquet(output / 'training-support.parquet', index=False)
    return shared, donors, ledger, {'hgnc_sha256': hash_file(hgnc_path), 'training_backgrounds': lines}


def train(config, output, genes, targets):
    collection = ROOT / config['collection']
    reports, identities = validate_sources(collection, config['panels'])
    write_json(output / 'training-inputs.json', identities)
    print(json.dumps({'stage': 'training', 'verified_panels': len(reports)}), flush=True)
    data = load_group(collection, reports, supplemental=True, genetic=True)
    _, _, delta, _ = responses(data, 'control_only')
    response = delta[:, :, 0].astype(float)
    response[:, data['excluded']] = np.nan
    available = data['available'][:, :, 0]
    rows, folds = [], []
    for held, line in enumerate(data['lines']):
        fitted, shared, prediction, donors = outer_prediction(response, available, held, config['lambda_grid'])
        folds.append({'held_background': line, 'training_backgrounds': [x for x in data['lines'] if x != line],
                      'fitted': fitted, 'held_response_used_for_fitting': False})
        for pi in np.flatnonzero(available[held] & (donors > 0)):
            for name, value in [('zero', np.zeros(len(data['genes']))), ('shared', shared[pi]), ('shared_shrunk', prediction[pi])]:
                metric = score(response[held, pi], value)
                rows.append({'model': name, 'canonical_target': data['targets'][pi], 'balance_group': line,
                             'donor_backgrounds': int(donors[pi]), 'zero_MAE': float(np.nanmean(np.abs(response[held, pi]))), **metric})
        print(json.dumps({'stage': 'outer_evaluation', 'held': line, 'calibration': fitted}), flush=True)
    frame = pd.DataFrame(rows)
    frame.to_parquet(output / 'offline-prediction-metrics.parquet', index=False)
    summary = [{'model': name, **balanced_metrics(part)} for name, part in frame.groupby('model')]
    fitted = calibrate(response, available, config['lambda_grid'])
    write_json(output / 'calibration.json', {'final': fitted, 'outer_folds': folds,
                                            'genes': data['genes'], 'targets': data['targets']})
    shared, donors, ledger, meta = deployable_responses(collection, reports, targets, genes, output)
    response = shrink(shared, donors, fitted).astype(np.float32)
    np.savez_compressed(output / 'model.npz', genes=np.asarray(genes), targets=np.asarray(targets),
                        response=response, donor_counts=donors)
    result = {'status': 'completed', 'model': 'support_calibrated_shared_CRISPRi_response', 'fitted': fitted,
              'offline_summary': summary, 'training_genes': len(data['genes']), 'training_targets': len(data['targets']),
              'official_targets': len(targets), 'official_genes': len(genes), 'source_run': '20260922-c',
              'official_target_background_coverage': ledger.backgrounds.value_counts().sort_index().to_dict(),
              'model_sha256': hash_file(output / 'model.npz'), **meta}
    write_json(output / 'training.json', result)
    return result


def balanced_templates(guides, n, seed):
    names = sorted(set(guides))
    rng = np.random.default_rng(seed)
    extra = set(rng.choice(len(names), n % len(names), replace=False))
    rows = []
    for j, name in enumerate(names):
        ids = np.flatnonzero(np.asarray(guides) == name)
        number = n // len(names) + int(j in extra)
        if number > len(ids):
            raise ValueError('insufficient_control_templates')
        rows.extend(rng.choice(ids, number, replace=False).tolist())
    return np.asarray(rows, int)


def gene_factors(control_log, response, donors, target_index, config):
    raw_desired = control_log + response
    desired = np.maximum(raw_desired, 0)
    denominator = np.expm1(control_log)
    factor = np.divide(np.expm1(desired), denominator, out=np.ones_like(desired), where=denominator > 1e-12)
    factor[donors == 0] = 1.
    diagnostics = {'clipped_negative_log_means': int((raw_desired < 0).sum()),
                   'zero_control_unsupported_ratio': int(((denominator <= 1e-12) & (response != 0)).sum()),
                   'capped_gene_factors': int((factor > config['maximum_gene_factor']).sum()),
                   'zero_donor_genes': int((donors == 0).sum())}
    factor = np.clip(factor, 0, config['maximum_gene_factor'])
    factor[target_index] = config['on_target_remaining_fraction']
    return factor, diagnostics


def emit_counts(templates, factors, seed):
    """Preserve each template's library size in expectation, stochastic rounding."""
    depth = templates.sum(axis=1, dtype=np.float64)
    weighted = templates.astype(np.float64) * factors
    total = weighted.sum(axis=1)
    if (total <= 0).any() or not np.isfinite(weighted).all():
        raise ValueError('invalid_emission_mass')
    weighted *= (depth / total)[:, None]
    base = np.floor(weighted)
    rng = np.random.default_rng(seed)
    counts = (base + (rng.random(base.shape) < weighted - base)).astype(np.int32)
    return counts
