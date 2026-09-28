"""Moment-fitted compositional and Gamma-Poisson response packages."""
import gc
import json
import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from data import ROOT, ref, write_json
from research import digest
from vcc_mechanism.inputs import moments


def dispersion(stats, spec):
    mean = stats['mean']
    noise = mean*spec['target_sum']*stats['inverse_library'][:, None]
    result = np.divide(np.maximum(stats['variance']-noise, 0), mean**2,
                       out=np.zeros_like(mean), where=mean > 0)
    result = np.clip(result, 0, spec['dispersion_max'])
    if len(mean) > 1:
        weight = np.where(stats['counts'][1:] > 1, stats['counts'][1:]/(stats['counts'][1:]+spec['dispersion_prior_cells']), 0)
        result[1:] = weight[:, None]*result[1:] + (1-weight[:, None])*result[0]
    return result


def fit(output, genes, split, config, run):
    moments(output, genes, split, config)
    spec = config['population']
    checkpoint = output/'checkpoints/population.npz'
    if checkpoint.exists():
        model = dict(np.load(checkpoint))
        if str(model['config_sha256']) != digest(config) or str(model['training_state']) != 'analytic_fit_complete':
            raise ValueError('Population checkpoint mismatch')
        return checkpoint, model, json.loads((output/'cache/fit.json').read_text())
    # Use actual supervised labels; split may also contain targets with no cells.
    labels = sorted({str(t) for c in split['training_contexts'] for t in np.load(output/f'cache/moments/{c}.npz')['labels'][1:]})
    lookup = {t: i for i, t in enumerate(labels)}
    shape = (len(labels), len(genes))
    values, mass = np.zeros(shape), np.zeros(shape)
    dispersion_values = np.zeros(shape) if spec['kind'] == 'negative_binomial' else None
    tasks = cells = 0
    for c in split['training_contexts']:
        s = dict(np.load(output/f'cache/moments/{c}.npz'))
        ix = np.ix_([lookup[str(t)] for t in s['labels'][1:]], s['positions'])
        weights = s['counts'][1:, None]/spec['weight_cell_unit']
        if spec['kind'] == 'composition_lfc':
            response = np.log((s['mean'][1:]+spec['pseudocount'])/(s['mean'][0]+spec['pseudocount']))
        else:
            response = s['mean'][1:]-s['mean'][0]
            phi = dispersion(s, spec)
            ratios = np.log((phi[1:]+spec['dispersion_floor'])/(phi[0]+spec['dispersion_floor']))
            dispersion_values[ix] += weights*ratios
        values[ix] += weights*response; mass[ix] += weights
        tasks += len(s['labels'])-1; cells += int(s['counts'][1:].sum())
        del s, response
    model = dict(targets=np.asarray(labels), genes=np.asarray(genes),
                 response=np.divide(values, mass, out=np.zeros_like(values), where=mass > 0).astype(np.float32),
                 support=mass > 0, specification=np.asarray(json.dumps(spec)),
                 config_sha256=np.asarray(digest(config)), training_state=np.asarray('analytic_fit_complete'))
    if dispersion_values is not None:
        model['dispersion_response'] = np.divide(dispersion_values, mass, out=np.zeros_like(values), where=mass > 0).astype(np.float32)
    with checkpoint.with_suffix('.tmp').open('wb') as handle:
        np.savez_compressed(handle, **model)
    checkpoint.with_suffix('.tmp').replace(checkpoint)
    audit = {'kind': spec['kind'], 'training_tasks': tasks, 'training_cells': cells,
             'training_targets': len(labels), 'source_measured_union_genes': int((mass > 0).any(axis=0).sum()),
             'fitting': 'full_masked_cell_weighted_moments', 'heldout_H1_response_used': False,
             'count_likelihood_optimized': False}
    write_json(output/'cache/fit.json', audit)
    run.log({'train/tasks': tasks, 'train/cells': cells, 'train/complete': 1})
    print(f'Complete {spec["kind"]} fit: {tasks} tasks, {cells} cells, {len(labels)} targets', flush=True)
    del values, mass, dispersion_values
    gc.collect()
    return checkpoint, model, audit


def predict(model, stats, targets, arm='linear', source=None):
    """Arithmetic CPM profiles and dispersions on target measured genes."""
    if arm != 'linear':
        raise ValueError('Only registered candidate supported')
    spec = json.loads(str(model['specification']))
    lookup = {str(t): i for i, t in enumerate(model['targets'])}
    positions = stats['positions']
    baseline = stats['mean'][0]
    phi0 = dispersion(stats, spec)[0] if spec['kind'] == 'negative_binomial' else None
    profiles, phis = [], []
    for target in targets:
        response = model['response'][lookup[target], positions] if target in lookup else np.zeros(len(positions))
        if spec['kind'] == 'composition_lfc':
            mean = np.maximum((baseline+spec['pseudocount'])*np.exp(response)-spec['pseudocount'], 0)
        else:
            mean = np.maximum(baseline+response, 0)
            ratio = model['dispersion_response'][lookup[target], positions] if target in lookup else np.zeros(len(positions))
            phis.append(np.clip((phi0+spec['dispersion_floor'])*np.exp(ratio)-spec['dispersion_floor'], 0, spec['dispersion_max']))
        if not np.isfinite(mean).all() or mean.sum() <= 0:
            raise ValueError('Invalid population profile')
        profiles.append(mean/mean.sum())
    return np.asarray(profiles), np.asarray(phis) if phis else None


def composition_counts(raw, profile, rng, spec):
    library = raw.sum(1)
    q = (1-spec['template_smoothing'])*raw/library[:, None] + spec['template_smoothing']*profile
    desired = len(raw)*profile
    for _ in range(spec['ipf_iterations']):
        q *= np.divide(desired, q.sum(0), out=np.zeros_like(desired), where=q.sum(0) > 0)
        q /= q.sum(1, keepdims=True)
    error = float(np.abs(q.mean(0)-profile).sum())
    expected = library[:, None]*q
    return np.floor(expected+rng.random(expected.shape)).astype(np.float32), error


def negative_binomial_counts(raw, profile, phi, rng, spec):
    expected = raw.sum(1)[:, None]*profile
    active = phi > spec['poisson_threshold']
    rate = expected.copy()
    rate[:, active] = rng.gamma(1/phi[active], expected[:, active]*phi[active])
    return rng.poisson(rate).astype(np.float32), 0.


def draw(ntc, profile, phi, seed, config):
    rng = np.random.default_rng(seed)
    rows = rng.integers(0, ntc.shape[0], config['generation']['cells_per_target'])
    raw = ntc[rows].toarray().astype(np.float64)
    if config['population']['kind'] == 'composition_lfc':
        counts, error = composition_counts(raw, profile, rng, config['generation'])
    else:
        counts, error = negative_binomial_counts(raw, profile, phi, rng, config['generation'])
    if not np.isfinite(counts).all() or (counts < 0).any() or (counts.sum(1) <= 0).any() or (counts.sum(1) > 1e6).any():
        raise ValueError('Generated cells violate official count bounds')
    return sparse.csr_matrix(counts), error


def generate(output, model, genes, split, config):
    directory = output/'predictions'; directory.mkdir(exist_ok=True)
    manifest = directory/'manifest.json'
    if manifest.exists():
        result = json.loads(manifest.read_text())
        if any(ref(ROOT/r['path']) != r for r in result['files']):
            raise ValueError('Prediction changed on resume')
        return result
    stats = dict(np.load(output/'cache/moments/H1.npz'))
    ntc = ad.read_h5ad(output/'cache/H1-input.h5ad')
    targets = split['evaluation_targets']['H1']; positions = stats['positions']
    profiles, phis = predict(model, stats, targets)
    blocks, diagnostics = [], []
    for i, target in enumerate(targets):
        local, error = draw(ntc.X, profiles[i], phis[i] if phis is not None else None, config['seed']+i, config)
        depth = np.asarray(local.sum(1)).ravel()
        realized = np.asarray((sparse.diags(1/depth) @ local).mean(0)).ravel()
        diagnostics.append({'target': target, 'expected_profile_l1': error,
                            'realized_profile_l1': float(np.abs(realized-profiles[i]).sum()),
                            'mean_library': float(depth.mean()), 'mean_detected': float(np.diff(local.indptr).mean())})
        blocks.append(sparse.csr_matrix((local.data, positions[local.indices], local.indptr), shape=(local.shape[0], len(genes))))
        if (i+1) % 40 == 0:
            print(f'Generated {i+1}/{len(targets)} populations', flush=True)
    labels = np.repeat(targets, config['generation']['cells_per_target'])
    obs = pd.DataFrame({'target_gene': labels}, index=[f'linear-{i}' for i in range(len(labels))])
    prediction = ad.AnnData(sparse.vstack(blocks, format='csr'), obs=obs, var=pd.DataFrame(index=genes))
    path = directory/'linear.h5ad'; prediction.write_h5ad(path, compression='lzf')
    pd.DataFrame(diagnostics).to_csv(directory/'generation-diagnostics.csv', index=False)
    result = {'files': [ref(path), ref(directory/'generation-diagnostics.csv')], 'official_gene_axis': config['benchmark']['gene_axis'],
              'targets': targets, 'cells_per_target': config['generation']['cells_per_target'],
              'missing_input_completion': 'zero baseline and zero response; not observed ground truth',
              'unmeasured_input_positions': np.flatnonzero(~np.isin(np.arange(len(genes)), positions)).tolist()}
    write_json(manifest, result)
    del blocks, prediction, ntc
    gc.collect()
    return result
