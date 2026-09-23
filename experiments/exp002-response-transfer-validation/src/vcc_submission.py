"""Reproducible Issue #30 training, count emission, official prep and submission."""
import importlib.metadata
import json
from pathlib import Path
import re
import subprocess
import sys
import time

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from scipy import sparse

from profile_responses import write_json
from rna import hash_file
from vcc_baseline import train, balanced_templates, gene_factors, emit_counts

EXPERIMENT = Path(__file__).resolve().parents[1]
ROOT = EXPERIMENT.parents[1]
TEMPLATE = EXPERIMENT / 'configs/vcc2026-shared.json'
VCC = str(Path(sys.executable).parent / 'vcc')


def official_inputs(config):
    directory = ROOT / config['official']
    lock = json.loads((ROOT / 'data/registry.lock.json').read_text())
    source = next(s for s in lock['sources'] if s['id'] == 'arc_vcc2026_controls')
    if json.loads((directory / 'SOURCE.json').read_text()) != source['provenance']:
        raise ValueError('official_source_identity_changed')
    identities = []
    for item in source['files']:
        path = directory / item['name']
        digest = hash_file(path)
        if digest != item['checksum'].removeprefix('sha256:') or path.stat().st_size != item['bytes']:
            raise ValueError('official_snapshot_changed: ' + item['name'])
        identities.append({'file': item['name'], 'sha256': digest, 'bytes': path.stat().st_size})
    manifest = json.loads((directory / 'manifest.json').read_text())
    genes = pd.read_csv(directory / 'gene_names.csv').gene_name.astype(str).tolist()
    targets = pd.read_csv(directory / 'pert_counts.csv').target_gene.astype(str).tolist()
    assert manifest['partition'] == 'val' and manifest['contexts'] == ['A', 'B', 'C']
    assert len(genes) == len(set(genes)) == manifest['n_genes'] == 18533
    assert len(targets) == len(set(targets)) == manifest['n_constructs'] == 300
    assert set(targets) <= set(genes) and manifest['cells_per_pert'] == config['cells_per_perturbation']
    return directory, manifest, genes, targets, identities


def control_summary(adata):
    size = np.asarray(adata.X.sum(axis=1)).ravel().astype(float)
    if (size <= 0).any():
        raise ValueError('empty_official_control')
    total = np.zeros(adata.n_vars, float)
    for start in range(0, adata.n_obs, 1024):
        x = adata.X[start:start + 1024].astype(np.float64)
        x.data *= np.repeat(10000 / size[start:start + 1024], np.diff(x.indptr))
        np.log1p(x.data, out=x.data)
        total += np.asarray(x.sum(axis=0)).ravel()
    return total / adata.n_obs


def append_sparse(group, block, start_row):
    block = sparse.csr_matrix(block)
    block.eliminate_zeros()
    block.sort_indices()
    old = len(group['data'])
    for name, value in [('data', block.data), ('indices', block.indices)]:
        group[name].resize(old + block.nnz, axis=0)
        group[name][old:] = value
    group['indptr'][start_row + 1:start_row + block.shape[0] + 1] = block.indptr.astype(np.int64)[1:] + old
    return block.nnz


def context_prediction(model, context):
    """Select explicitly named context predictions or the shared baseline."""
    response = model['response']
    support = model['response_support'] if 'response_support' in model else model['donor_counts']
    if response.ndim == 3:
        contexts = model['contexts'].tolist()
        if len(contexts) != len(set(contexts)) or context not in contexts:
            raise ValueError('invalid_prediction_context_identity')
        ci = contexts.index(context)
        response, support = response[ci], support[ci]
    if response.ndim != 2 or support.shape != response.shape:
        raise ValueError('invalid_prediction_response_support_shape')
    return response, support


def generate(config, output, official, manifest, genes, targets):
    completed = output / 'generation.json'
    final = output / 'predictions.h5ad'
    if completed.exists():
        record = json.loads(completed.read_text())
        assert hash_file(final) == record['sha256']
        return record
    model = np.load(output / 'model.npz')
    assert model['genes'].tolist() == genes and model['targets'].tolist() == targets
    contexts = manifest['contexts']
    per_target = config['cells_per_perturbation']
    rows = len(contexts) * len(targets) * per_target
    obs = pd.DataFrame({'context': np.repeat(contexts, len(targets) * per_target),
                        'target_gene': np.tile(np.repeat(targets, per_target), len(contexts))},
                       index=[f'pred_{i:06d}' for i in range(rows)])
    for name in obs:
        obs[name] = pd.Categorical(obs[name])
    temporary = output / 'predictions.partial.h5ad'
    if temporary.exists():
        temporary.unlink()  # same-run deterministic generation restart only
    ad.AnnData(X=sparse.csr_matrix((rows, len(genes)), dtype=np.int32), obs=obs,
               var=pd.DataFrame(index=pd.Index(genes, name='gene_name'))).write_h5ad(temporary)
    diagnostics, template_rows = [], {}
    pointer = 0
    with h5py.File(temporary, 'r+') as hf:
        del hf['X']
        x = hf.create_group('X')
        x.attrs.update({'encoding-type': 'csr_matrix', 'encoding-version': '0.1.0', 'shape': [rows, len(genes)]})
        for name in ['data', 'indices']:
            x.create_dataset(name, (0,), maxshape=(None,), chunks=(1048576,), dtype='int32', compression='gzip', compression_opts=1)
        x.create_dataset('indptr', (rows + 1,), dtype='int64')
        for ci, context in enumerate(contexts):
            response, response_support = context_prediction(model, context)
            controls = ad.read_h5ad(official / f'context_{context}.h5ad')
            assert controls.var_names.tolist() == genes
            assert set(controls.obs.context.astype(str)) == {context}
            assert set(controls.obs.target_gene.astype(str)) == {'non-targeting'}
            assert controls.n_obs == manifest['per_context'][context]['control_cells']
            sizes = controls.obs.ntc_id.value_counts()
            assert len(sizes) == 46 and (sizes == 400).all()
            mean_log = control_summary(controls)
            selected = balanced_templates(controls.obs.ntc_id.astype(str).to_numpy(), per_target, config['seed'] + ci)
            template_rows[context] = selected.tolist()
            templates = controls.X[selected].toarray().astype(np.int32)
            del controls
            template_depth = templates.sum(1, dtype=float)
            template_cp = templates / template_depth[:, None] * 10000
            for pi, target in enumerate(targets):
                target_index = genes.index(target)
                factors, diagnostic = gene_factors(mean_log, response[pi], response_support[pi], target_index, config)
                emitted = emit_counts(templates, factors, config['seed'] + 1000 + ci * len(targets) + pi)
                depth = emitted.sum(axis=1, dtype=np.int64)
                if (emitted < 0).any() or (depth <= 0).any() or depth.max() > config['max_counts_per_cell']:
                    raise ValueError('invalid_emitted_counts')
                nnz = append_sparse(x, emitted, pointer)
                pointer += per_target
                if len(x['data']) > config['max_nnz']:
                    raise ValueError('prediction_density_exceeds_official_cap')
                cp = emitted / depth[:, None] * 10000
                actual_mean_log = np.log1p(cp).mean(0)
                intended = np.maximum(mean_log + response[pi], 0)
                off_target = np.arange(len(genes)) != target_index
                diagnostic.update(context=context, target_gene=target, cells=per_target, nnz=nnz,
                    maximum_cell_counts=int(depth.max()), minimum_cell_counts=int(depth.min()),
                    mean_library_size_absolute_deviation=float(np.abs(depth - template_depth).mean()),
                    off_target_mean_log_mapping_RMSE=float(np.sqrt(np.mean((actual_mean_log[off_target] - intended[off_target]) ** 2))),
                    on_target_CP10K_ratio_to_templates=float(cp[:, target_index].mean() / template_cp[:, target_index].mean())
                        if template_cp[:, target_index].mean() > 0 else None)
                diagnostics.append(diagnostic)
                if (pi + 1) % 25 == 0:
                    print(json.dumps({'stage': 'generate', 'context': context, 'targets_completed': pi + 1,
                                      'cells_written': pointer, 'nnz': len(x['data'])}), flush=True)
            hf.flush()
        assert pointer == rows
    temporary.replace(final)
    pd.DataFrame(diagnostics).to_parquet(output / 'emission-diagnostics.parquet', index=False)
    write_json(output / 'template-rows-private.json', template_rows)
    result = {'status': 'completed', 'file': final.name, 'sha256': hash_file(final), 'bytes': final.stat().st_size,
              'cells': rows, 'genes': len(genes), 'target_contexts': len(diagnostics),
              'nnz': sum(r['nnz'] for r in diagnostics), 'seed': config['seed'],
              'distribution_method': 'NTC templates, gene scaling, library preservation, stochastic rounding',
              'on_target_remaining_fraction_is_model_prior': config['on_target_remaining_fraction'],
              'official_control_files_publicly_redistributed': False}
    write_json(completed, result)
    return result


def audit_prediction(path, genes, targets, contexts, per_target, max_nnz, max_counts):
    """Independent full-file count/identity audit before the official validator."""
    n = len(targets) * len(contexts) * per_target
    data = ad.read_h5ad(path, backed='r')
    try:
        assert data.shape == (n, len(genes)) and data.var_names.tolist() == genes
        assert data.obs_names.is_unique and set(data.obs.context.astype(str)) == set(contexts)
        sizes = data.obs.groupby(['context', 'target_gene'], observed=True).size()
        assert set(sizes.index) == {(c, t) for c in contexts for t in targets}
        assert (sizes == per_target).all()
    finally:
        data.file.close()
    maximum, minimum, nnz = 0, max_counts, 0
    with h5py.File(path) as hf:
        x = hf['X']
        ptr = x['indptr'][:]
        assert len(ptr) == n + 1 and ptr[0] == 0 and np.all(np.diff(ptr) > 0)
        assert ptr[-1] == len(x['data']) == len(x['indices']) <= max_nnz
        for start in range(0, n, 2048):
            end = min(n, start + 2048)
            a, b = int(ptr[start]), int(ptr[end])
            values, indices = x['data'][a:b], x['indices'][a:b]
            assert np.isfinite(values).all() and (values > 0).all() and np.equal(values, np.floor(values)).all()
            assert (indices >= 0).all() and (indices < len(genes)).all()
            totals = np.add.reduceat(values.astype(np.int64), ptr[start:end] - a)
            assert (totals > 0).all() and totals.max() <= max_counts
            minimum, maximum = min(minimum, int(totals.min())), max(maximum, int(totals.max()))
            nnz += b - a
    return {'status': 'completed', 'cells': n, 'genes': len(genes), 'target_contexts': len(sizes), 'nnz': nnz,
            'all_values_finite_nonnegative_integers': True, 'gene_order_matches': True,
            'exact_context_target_cell_counts': True, 'minimum_cell_counts': minimum, 'maximum_cell_counts': maximum}


def cli_json(arguments):
    result = subprocess.run([VCC, *map(str, arguments), '--json'], text=True, capture_output=True)
    try:
        value = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError('official_CLI_returned_non_JSON: ' + str(result.returncode)) from error
    if result.returncode:
        raise RuntimeError('official_CLI_failed: ' + json.dumps(value))
    return value


def package(config, output, official):
    report = output / 'prep.json'
    final = output / 'predictions.vcc'
    if not report.exists():
        result = cli_json(['prep', output / 'predictions.h5ad', '-g', official / 'gene_names.csv',
                           '--perts', official / 'pert_counts.csv', '-o', final])
        write_json(report, result)
    digest = hash_file(final)
    identity_path = output / 'submission-file.json'
    identity = {'file': final.name, 'bytes': final.stat().st_size, 'sha256': digest,
                'cli_version': importlib.metadata.version('vcc-cli')}
    if identity_path.exists():
        assert json.loads(identity_path.read_text()) == identity
    write_json(identity_path, identity)
    return identity


def submit_and_score(config, output, tracked):
    entry_path = output / 'submission-entry.json'
    entry = json.loads(entry_path.read_text()) if entry_path.exists() else None
    if entry is None:
        # Recover only a pending upload of this exact run's file; never a teammate's.
        from vcc import auth
        candidates = [value for value in auth.list_pending_uploads('default').values()
                      if Path(value.get('local_path', '')).resolve() == (output / 'predictions.vcc').resolve()]
        if len(candidates) > 1:
            raise ValueError('ambiguous_pending_submission')
        if candidates:
            entry = {'entry_id': candidates[0]['entry_id']}
            write_json(entry_path, entry)
    status = cli_json(['status', entry['entry_id']]) if entry else None
    if status is None or status.get('status') == 'uploading':
        command = [VCC, 'submit', str(output / 'predictions.vcc'), '-m', config['model_name'],
                   '-d', config.get('submission_description',
                                    'Pure CRISPRi shared-response baseline, donor-support calibrated shrinkage; Issue #30.')]
        if entry:
            command.extend(['--resume', entry['entry_id']])
        process = subprocess.Popen(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1)
        with (output / 'submission.log').open('a') as log:
            for line in process.stdout:
                # Never retain query-bearing capability URLs in experiment logs.
                line = re.sub(r'https://\S*\?\S+', '[redacted-capability-url]', line)
                log.write(line); log.flush()
                print(line.rstrip(), flush=True)
                found = re.search(r'^\s*entry:\s+(\S+)', line)
                if found:
                    entry = {'entry_id': found.group(1), 'model_name': config['model_name'],
                             'submission_sha256': json.loads((output / 'submission-file.json').read_text())['sha256']}
                    write_json(entry_path, entry)
        if process.wait() != 0:
            raise RuntimeError('submission_failed; inspect submission.log and resume the same entry')
    if entry is None:
        raise RuntimeError('official_submission_returned_no_entry_id')
    while True:
        status = cli_json(['status', entry['entry_id']])
        write_json(output / 'official-status.json', status)
        print(json.dumps({'stage': 'official_scoring', 'entry_id': entry['entry_id'], 'status': status.get('status')}), flush=True)
        tracked.summary['official/entry_id'] = entry['entry_id']
        tracked.summary['official/status'] = status.get('status')
        if status.get('status') == 'published':
            keys = ['score_pds', 'score_mse', 'score_nmae', 'score_fid', 'score_reach', 'score_jac']
            scores = np.array([status[k] for k in keys], dtype=float)
            assert np.isfinite(scores).all() and np.isclose(scores.mean(), status['score_avg'], atol=1e-6)
            assert status['partition'] == 'val' and status['panel_id'] == 'vcc2026-val-1' and status['anchor_version']
            summary = {k: status.get(k) for k in keys + ['score_avg', 'rank', 'partition', 'panel_id', 'anchor_version']}
            summary.update(entry_id=entry['entry_id'], model_name=config['model_name'], status='published')
            write_json(output / 'official-summary.json', summary)
            for key, value in summary.items():
                if value is not None:
                    tracked.summary['official/' + key] = value
            return status
        if status.get('is_terminal'):
            raise RuntimeError('official_scoring_terminal_without_published_score')
        time.sleep(30)


def run(run_id, submit=False):
    from run import tracking
    if not re.fullmatch(r'[A-Za-z0-9_-]+', run_id):
        raise ValueError('invalid_run_id')
    output = EXPERIMENT / 'outputs' / run_id
    output.mkdir(parents=True, exist_ok=True)
    template = json.loads(TEMPLATE.read_text())
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    subprocess.run(['git', 'diff', '--quiet', 'HEAD', '--', str(EXPERIMENT)], cwd=ROOT, check=True)
    config = {**template, 'experiment_id': EXPERIMENT.name, 'run_id': run_id, 'pipeline_commit': commit,
              'uv_lock_sha256': hash_file(EXPERIMENT / 'uv.lock'), 'model_name': template['model_name_prefix'] + '-' + run_id,
              'configuration_sha256': hash_file(TEMPLATE), 'run_type': 'trained_baseline_official_validation_submission',
              'collection_report_sha256': hash_file(ROOT / template['collection'] / 'report.json'),
              'registry_sha256': hash_file(ROOT / 'data/registry.lock.json'),
              'hgnc_sha256': hash_file(ROOT / 'data/raw/networks/hgnc_complete_set.txt'),
              'versions': {p: importlib.metadata.version(p) for p in ['numpy', 'scipy', 'pandas', 'h5py', 'anndata', 'vcc-cli', 'wandb']}}
    config_path = output / 'config.yaml'
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError('run_identity_changed; changed code/data/model requires a new run')
    write_json(config_path, config)
    tracked, tracking_info = tracking(output, config)
    write_json(output / 'tracking.json', tracking_info)
    try:
        with (output / 'tests.log').open('w') as log:
            subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', str(EXPERIMENT / 'tests'), '-v'],
                           cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        official, manifest, genes, targets, identities = official_inputs(config)
        write_json(output / 'official-inputs.json', {'manifest': manifest, 'identities': identities})
        if (output / 'training.json').exists():
            trained = json.loads((output / 'training.json').read_text())
            assert hash_file(output / 'model.npz') == trained['model_sha256']
        else:
            trained = train(config, output, genes, targets)
        for row in trained['offline_summary']:
            tracked.summary['offline/' + row['model'] + '/relative_MSE_improvement'] = row['relative_MSE_improvement']
        tracked.summary['training/lambda_single'] = trained['fitted']['single']['lambda']
        tracked.summary['training/lambda_multiple'] = trained['fitted']['multiple']['lambda']
        import wandb
        if not (output / 'model-artifact.json').exists():
            artifact = wandb.Artifact(EXPERIMENT.name + '-' + run_id, type='model')
            for name in ['model.npz', 'config.yaml', 'training.json', 'calibration.json', 'training-support.parquet', 'training-inputs.json', 'offline-prediction-metrics.parquet']:
                artifact.add_file(str(output / name), name=name)
            logged = tracked.log_artifact(artifact)
            if tracking_info['mode'] == 'online':
                logged.wait()
                write_json(output / 'model-artifact.json', {'name': logged.name, 'version': logged.version, 'digest': logged.digest})
        generated = generate(config, output, official, manifest, genes, targets)
        if not (output / 'prediction-audit.json').exists():
            audit = audit_prediction(output / 'predictions.h5ad', genes, targets, manifest['contexts'],
                                     config['cells_per_perturbation'], config['max_nnz'], config['max_counts_per_cell'])
            write_json(output / 'prediction-audit.json', audit)
        identity = package(config, output, official)
        tracked.summary['prediction/cells'] = generated['cells']
        tracked.summary['prediction/nnz'] = generated['nnz']
        result = {'status': 'prepared', 'issue': config['issue'], 'run_id': run_id, 'model_name': config['model_name'],
                  'pipeline_commit': commit, 'training': trained, 'generation': generated, 'submission_file': identity}
        write_json(output / 'report.json', result)
        if submit:
            result['official'] = submit_and_score(config, output, tracked)
            result['status'] = 'completed'
            write_json(output / 'report.json', result)
            artifact = wandb.Artifact(EXPERIMENT.name + '-' + run_id + '-official-score', type='evaluation')
            for name in ['official-summary.json', 'submission-entry.json', 'submission-file.json', 'prediction-audit.json',
                         'generation.json', 'emission-diagnostics.parquet', 'prep.json']:
                artifact.add_file(str(output / name), name=name)
            logged = tracked.log_artifact(artifact)
            if tracking_info['mode'] == 'online':
                logged.wait()
                write_json(output / 'score-artifact.json', {'name': logged.name, 'version': logged.version, 'digest': logged.digest})
        tracked.finish()
        print(json.dumps({'status': result['status'], 'output': str(output)}), flush=True)
    except Exception as error:
        write_json(output / 'pipeline-failure.json', {'status': 'failed', 'error': repr(error), 'pipeline_commit': commit})
        tracked.finish(exit_code=1)
        raise
