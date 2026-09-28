"""One fixed count generator and the registered official six-metric recipe."""
import gc
import json
import numpy as np
import pandas as pd
import anndata as ad
from scipy import sparse
from data import ROOT, NTC, write_json, ref
from challenge import build_reference_bundle, score_prediction, scorer_contract
from model import predict


def generate_counts(ntc, delta, seed, n_cells, target_sum):
    rng = np.random.default_rng(seed)
    rows = rng.integers(0, ntc.shape[0], n_cells)
    raw = ntc[rows].toarray().astype(np.float64)
    library = raw.sum(axis=1, keepdims=True)
    shifted = np.expm1(np.maximum(np.log1p(raw*target_sum/library) + delta, 0))
    total = shifted.sum(axis=1, keepdims=True)
    if not np.isfinite(shifted).all() or (total <= 0).any():
        raise ValueError('invalid generated expression')
    expected = shifted * library / total
    counts = np.floor(expected + rng.random(expected.shape)).astype(np.float32)
    if (counts.sum(axis=1) <= 0).any() or (counts.sum(axis=1) > 1e6).any():
        raise ValueError('generated library violates official count bounds')
    return sparse.csr_matrix(counts)


def generate(output, model, genes, split, config):
    directory = output / 'predictions'
    directory.mkdir(exist_ok=True)
    manifest = directory / 'manifest.json'
    if manifest.exists():
        result = json.loads(manifest.read_text())
        if any(ref(ROOT/r['path']) != r for r in result['files']):
            raise ValueError('prediction checksum mismatch on resume')
        return result
    stats = dict(np.load(output/'cache/H1-statistics.npz'))
    source = dict(np.load(output/'cache/K562-statistics.npz'))
    ntc = ad.read_h5ad(output/'cache/H1-input.h5ad')
    positions = stats['positions']
    targets = split['evaluation_targets']['H1']
    files = []
    for arm in config['evaluation']['arms']:
        delta = predict(model, stats, targets, arm, source)
        blocks = []
        for i, target in enumerate(targets):
            # Identical random stream per target across all arms.
            local = generate_counts(ntc.X, delta[i, positions], config['seed']+i,
                                    config['generation']['cells_per_target'], config['data']['target_sum'])
            expanded = sparse.csr_matrix((local.data, positions[local.indices], local.indptr),
                                         shape=(local.shape[0], len(genes)))
            blocks.append(expanded)
        labels = np.repeat(targets, config['generation']['cells_per_target'])
        obs = pd.DataFrame({'target_gene': labels}, index=[f'{arm}-{i}' for i in range(len(labels))])
        prediction = ad.AnnData(sparse.vstack(blocks, format='csr'), obs=obs, var=pd.DataFrame(index=genes))
        path = directory/f'{arm}.h5ad'
        prediction.write_h5ad(path, compression='lzf')
        files.append(ref(path))
        print(f'Generated {arm}: {prediction.shape}, full official axis; missing input genes remain explicit zero-response completion', flush=True)
        del blocks, prediction
        gc.collect()
    result = {'files': files, 'official_gene_axis': config['benchmark']['gene_axis'],
              'targets': targets, 'cells_per_target': config['generation']['cells_per_target'],
              'unmeasured_input_positions': np.flatnonzero(~np.isin(np.arange(len(genes)), positions)).tolist(),
              'missing_input_completion': 'zero baseline and zero response; not observed ground truth'}
    write_json(manifest, result)
    return result


def evaluate(output, split, config, run):
    # Verify the actual installed official implementation, including all source hashes.
    expected = json.loads((ROOT/config['benchmark']['scorer']['path']).read_text())
    actual = scorer_contract()
    if actual != expected:
        raise ValueError('installed cell-eval2 differs from the frozen scorer recipe')
    real_all = ad.read_h5ad(output/'cache/H1-reference.h5ad')
    measured = np.load(output/'cache/H1-statistics.npz')['positions']
    results = {}
    runtime = config['evaluation']['runtime']
    for panel, targets in [('all', split['evaluation_targets']['H1']),
                           ('official_overlap', split['official_overlap_targets']['H1'])]:
        real = real_all[real_all.obs.target_gene.isin([NTC]+targets)].copy()
        bundle = output/'cache'/f'bundle-{panel}'
        if not (bundle/'manifest.json').exists():
            print(f'Building actual official five-split anchors: {panel}, {real.shape}, {len(targets)} targets', flush=True)
            build_reference_bundle(real, bundle, f'{config["run_id"]}-{panel}', **runtime)
            gc.collect()
        results[panel] = {}
        for arm in config['evaluation']['arms']:
            directory = output/'cache'/f'score-{panel}-{arm}'
            done = directory/'result.json'
            if done.exists():
                result = json.loads(done.read_text())
            else:
                # Incomplete evaluation has no scientific output; keep each attempt for diagnosis.
                attempt = directory
                suffix = 0
                while attempt.exists():
                    suffix += 1
                    attempt = directory.with_name(directory.name+f'-attempt{suffix}')
                prediction = ad.read_h5ad(output/'predictions'/f'{arm}.h5ad')
                prediction = prediction[prediction.obs.target_gene.isin(targets), measured].copy()
                print(f'Official evaluation: {panel}/{arm}', flush=True)
                result = score_prediction(prediction, real, bundle, attempt, **runtime)
                aggregate = pd.read_csv(attempt/'aggregate.csv')
                # Preserve raw means/counts exactly as emitted, alongside normalized scores.
                result['raw_aggregate'] = json.loads(aggregate.to_json(orient='records'))
                result['result_directory'] = str(attempt.relative_to(ROOT))
                directory.mkdir(exist_ok=True)
                write_json(done, result)
                del prediction
                gc.collect()
            results[panel][arm] = result
            run.log({f'evaluation/{panel}/{arm}/Overall': result['Overall'],
                     **{f'evaluation/{panel}/{arm}/{m}': v for m, v in result['normalized'].items()}})
        del real
        gc.collect()
    return results
