"""One resumable native-count transfer run, from inputs through paired evaluation."""
import fcntl
import importlib.metadata
import json
from pathlib import Path
import re
import subprocess
import sys
import traceback

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import torch
import yaml
from scipy import sparse

from count_transfer import (ROOT, CountData, verify_cache, hash_file, write_json,
                            prepare_statistics, fit_transfer, evaluate_means, predict_transfer, stable_seed,
                            save_response_library, ResponseLibrary)
from count_generation import generate_counts
from count_evaluation import PanelEvaluation, ARMS
from vcc_baseline import balanced_templates
from vcc_submission import official_inputs, append_sparse, audit_prediction, package, submit_and_score

EXPERIMENT = Path(__file__).resolve().parents[1]


class Tee:
    def __init__(self, stream, log):
        self.stream, self.log = stream, log
    def write(self, value):
        self.stream.write(value); self.log.write(value); self.log.flush()
        return len(value)
    def flush(self):
        self.stream.flush(); self.log.flush()


def source_identity(template):
    files = list((EXPERIMENT / 'src').glob('*.py')) + list((EXPERIMENT / 'tests').glob('*.py'))
    files += list((ROOT / 'experiments/exp003-context-module-cvae/src').glob('*.py'))
    files += [ROOT / 'experiments/exp001-context-pair-xgb/src/features.py']
    files += list((ROOT / 'scripts/dossier').glob('*.py'))
    files += [EXPERIMENT / 'pyproject.toml', EXPERIMENT / 'uv.lock', EXPERIMENT / 'reproduce.sh']
    files += [Path(template).resolve()]
    paths = [str(p.relative_to(ROOT)) for p in files]
    subprocess.run(['git', 'diff', '--quiet', 'HEAD', '--', *paths], cwd=ROOT, check=True)
    untracked = subprocess.check_output(['git', 'ls-files', '--others', '--exclude-standard', '--', *paths], cwd=ROOT, text=True)
    if untracked.strip():
        raise ValueError('formal_training_requires_committed_source')
    return {str(p.relative_to(ROOT)): hash_file(p) for p in files}


def config_identity(template, run_id):
    config = json.loads(Path(template).read_text())
    if config['protocol'] != 'native-count-transfer-v1' or config['bulk_target_sum'] != 50000:
        raise ValueError('unsupported_training_protocol')
    source = ROOT / config['cache_source']
    producer_path = source.parent.parent / 'config.yaml'
    producer = yaml.safe_load(producer_path.read_text())
    source_meta = json.loads((source / 'complete.json').read_text())
    registry = json.loads((ROOT / 'data/registry.lock.json').read_text())
    official_source = next(s for s in registry['sources'] if s['id'] == 'arc_vcc2026_controls')
    config.update(experiment_id=EXPERIMENT.name, run_id=run_id, run_type='calibrated_counts',
                  model_name=config['model_name_prefix'] + '-' + run_id,
                  submission_description='Native-count shared perturbation response with held-context calibrated residuals and NTC count-distribution calibration.',
                  source_files=source_identity(template), configuration_sha256=hash_file(Path(template)),
                  uv_lock_sha256=hash_file(EXPERIMENT / 'uv.lock'), python=sys.version,
                  python_executable=str(Path(sys.executable).resolve()),
                  data_reference={'metadata_sha256': hash_file(source / 'complete.json'),
                                  'producer_config_sha256': hash_file(producer_path),
                                  'producer_commit': producer['pipeline_commit'],
                                  'producer_data_spec_hash': source_meta['data_spec_hash']},
                  official_data_reference=official_source,
                  cuda_runtime=torch.version.cuda,
                  versions={p: importlib.metadata.version(p) for p in ['numpy', 'scipy', 'pandas', 'anndata', 'cell-eval2', 'gpudge', 'torch', 'wandb', 'vcc-cli']})
    assert config['versions']['cell-eval2'] == config['official_version']
    return config


def load_or_fit(stats, training, config, output, name):
    path = output / 'checkpoints' / (name + '.json')
    if path.exists():
        fitted = json.loads(path.read_text())
        assert fitted['training_contexts'] == training and fitted['training_complete']
        assert fitted['stats_sha256'] == hash_file(stats.directory / 'complete.json')
        return fitted
    fitted = fit_transfer(stats, training, config)
    write_json(path, fitted)
    return fitted


def export_official(stats, fitted, selection, config, output):
    directory = output / 'predictions/official'; directory.mkdir(parents=True, exist_ok=True)
    marker = directory / 'generation.json'
    if marker.exists():
        result = json.loads(marker.read_text())
        assert result['sha256'] == hash_file(directory / 'predictions.h5ad')
        return directory
    official, manifest, genes, targets, identities = official_inputs(config)
    assert genes == stats.genes
    write_json(directory / 'inputs.json', {'manifest': manifest, 'identities': identities})
    response_method, generator = ARMS[selection]
    n = config['cells_per_perturbation']; contexts = manifest['contexts']; total = len(contexts) * len(targets) * n
    obs = pd.DataFrame({'context': np.repeat(contexts, len(targets) * n),
                        'target_gene': np.tile(np.repeat(targets, n), len(contexts))}, index=[f'pred-{i}' for i in range(total)])
    for column in obs:
        obs[column] = pd.Categorical(obs[column])
    temporary = directory / 'predictions.partial.h5ad'
    ad.AnnData(sparse.csr_matrix((total, len(genes)), dtype=np.uint32), obs=obs,
               var=pd.DataFrame(index=pd.Index(genes, name='gene_name'))).write_h5ad(temporary)
    pointer = 0; diagnostics = []; support_rows = []
    with h5py.File(temporary, 'r+') as handle:
        del handle['X']; group = handle.create_group('X')
        group.attrs.update({'encoding-type': 'csr_matrix', 'encoding-version': '0.1.0', 'shape': [total, len(genes)]})
        for name in ['data', 'indices']:
            group.create_dataset(name, (0,), maxshape=(None,), chunks=(1048576,), dtype='int32', compression='gzip', compression_opts=1)
        group.create_dataset('indptr', (total + 1,), dtype='int64')
        for context in contexts:
            controls = ad.read_h5ad(official / f'context_{context}.h5ad')
            assert controls.var_names.tolist() == genes
            mu = np.zeros(len(genes)); second = mu.copy()
            for start in range(0, controls.n_obs, 256):
                values = controls.X[start:start + 256].toarray().astype(float)
                mu += values.sum(0); second += (values ** 2).sum(0)
            mu /= controls.n_obs; variance = np.maximum(second / controls.n_obs - mu ** 2, 0)
            response, depth, counts, supported = predict_transfer(stats, fitted, targets, mu, response_method)
            for pi, target in enumerate(targets):
                ids = balanced_templates(controls.obs.ntc_id.astype(str).to_numpy(), n, stable_seed(config['seed'], context, target, 'templates'))
                templates = controls.X[ids].toarray()
                generated, diagnostic = generate_counts(mu, variance, templates, response[pi], float(depth[pi]), generator,
                                                          stable_seed(config['seed'], context, target, 'official'), config)
                append_sparse(group, generated, pointer); pointer += n
                if len(group['data']) > config['max_nnz']:
                    raise ValueError('official_sparse_limit_exceeded')
                diagnostics.append({'context': context, 'target': target, **diagnostic})
                support_rows.append({'context': context, 'target': target, 'target_seen': bool((counts[pi] > 0).any()),
                                     'readouts_without_target_donor': int((counts[pi] == 0).sum()),
                                     'readouts_without_training_measurement': int((~supported[pi]).sum())})
                if (pi + 1) % 50 == 0:
                    print(json.dumps({'stage': 'official_generation', 'context': context, 'targets': pi + 1}), flush=True)
            del controls
    assert pointer == total
    final = directory / 'predictions.h5ad'; temporary.replace(final)
    pd.DataFrame(diagnostics).to_parquet(directory / 'generation-diagnostics.parquet', index=False)
    pd.DataFrame(support_rows).to_parquet(directory / 'training-support.parquet', index=False)
    audit = audit_prediction(final, genes, targets, contexts, n, config['max_nnz'], config['max_counts_per_cell'])
    write_json(directory / 'prediction-audit.json', audit)
    identity = package(config, directory, official)
    write_json(marker, {'sha256': hash_file(final), 'cells': total, 'genes': len(genes), 'selected_arm': selection,
                       'checkpoint': 'checkpoints/final.json', 'submission_file': identity, 'training_contexts': fitted['training_contexts']})
    return directory


def publish_artifact(tracked, config, output, online):
    import wandb
    marker = output / 'artifact.json'
    metrics_digest = hash_file(output / 'metrics.json')
    if marker.exists():
        saved = json.loads(marker.read_text())
        if saved['status'] == 'verified_online' and saved['metrics_sha256'] == metrics_digest:
            return saved
    artifact = wandb.Artifact(EXPERIMENT.name + '-' + output.name, type='model', metadata={'training_commit': config['pipeline_commit'], 'raw_cells_uploaded': False})
    paths = [output / 'config.yaml', output / 'metrics.json', output / 'cache/statistics/complete.json', output / 'cache/statistics/panels.json', output / 'cache/statistics/ntc-diagnostics.json']
    paths += list((output / 'checkpoints').glob('*'))
    paths += [output / 'evaluation-metrics.parquet', output / 'evaluation-by-context.csv']
    paths += list((output / 'cache').glob('holdout-*/official/reference-bundle/*'))
    paths += list((output / 'cache').glob('holdout-*/official/null-*'))
    paths += list((output / 'cache').glob('holdout-*/official/eval-config.yaml'))
    paths += list((output / 'predictions').glob('holdout-*/mean-metrics.parquet'))
    paths += list((output / 'predictions').glob('holdout-*/*/seed-*/*.json'))
    paths += list((output / 'predictions').glob('holdout-*/*/seed-*/*.csv'))
    paths += list((output / 'predictions').glob('holdout-*/*/seed-*/*.parquet'))
    paths += list((output / 'predictions/official').glob('*.json'))
    paths += list((output / 'predictions/official').glob('*diagnostics.parquet'))
    paths += list((output / 'predictions/official').glob('training-support.parquet'))
    paths = sorted({p for p in paths if p.is_file()})
    for path in paths:
        artifact.add_file(str(path), name=str(path.relative_to(output)))
    logged = tracked.log_artifact(artifact)
    if online:
        logged.wait()
        remote = wandb.Api().artifact(f'yjcyxky/virtual-cell-challenge/{logged.name}', type='model')
        assert remote.digest == logged.digest and len(remote.manifest.entries) == len(paths)
        result = {'status': 'verified_online', 'name': logged.name, 'version': logged.version, 'digest': logged.digest, 'files': len(paths)}
    else:
        result = {'status': 'pending_offline_sync', 'name': artifact.name, 'files': len(paths)}
    result['metrics_sha256'] = metrics_digest
    write_json(marker, result)
    return result


def run(run_id, template, submit=False):
    from run import tracking
    if not re.fullmatch(r'[A-Za-z0-9_-]+', run_id):
        raise ValueError('invalid_run_id')
    output = EXPERIMENT / 'outputs' / run_id
    output.mkdir(parents=True, exist_ok=True)
    for name in ['cache', 'checkpoints', 'predictions']:
        (output / name).mkdir(exist_ok=True)
    with (output / '.run.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config = config_identity(template, run_id)
        config_path = output / 'config.yaml'
        if config_path.exists():
            saved = json.loads(config_path.read_text())
            current = dict(saved); current.pop('pipeline_commit')
            assert current == config, 'run_conditions_changed_new_run_required'
            config = saved
        else:
            config['pipeline_commit'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
            write_json(config_path, config)
        if (output / 'complete.json').exists() and (not submit or (output / 'predictions/official/official-summary.json').exists()):
            print((output / 'complete.json').read_text()); return
        with (output / 'train.log').open('a', buffering=1) as log:
            original_out, original_err = sys.stdout, sys.stderr
            sys.stdout, sys.stderr = Tee(sys.stdout, log), Tee(sys.stderr, log)
            tracked, tracking_info = tracking(output, config)
            write_json(output / 'tracking.json', tracking_info)
            try:
                if not (output / 'cache/tests-passed.json').exists():
                    with (output / 'cache/tests.log').open('w') as test_log:
                        subprocess.run([sys.executable, '-m', 'pytest', '-q', str(EXPERIMENT / 'tests')], cwd=ROOT,
                                       stdout=test_log, stderr=subprocess.STDOUT, check=True)
                    write_json(output / 'cache/tests-passed.json', {'sha256': hash_file(output / 'cache/tests.log')})
                source = ROOT / config['cache_source']
                report = json.loads((source / 'complete.json').read_text())
                verify_cache(source, report)
                write_json(output / 'cache/data-reference.json', {'path': str(source), 'sha256': hash_file(source / 'complete.json'), 'source_run_id': config['source_run_id']})
                data = CountData(source)
                stats = prepare_statistics(data, config, output / 'cache/statistics')
                results, means, nulls = [], [], []
                for held in config['contexts']:
                    fitted = load_or_fit(stats, [c for c in config['contexts'] if c != held], config, output, 'holdout-' + held)
                    directory = output / 'predictions' / ('holdout-' + held)
                    means.append(evaluate_means(stats, fitted, held, config, directory))
                    evaluator = PanelEvaluation(data, stats, held, config, output)
                    evaluator.prepare(); nulls.extend(evaluator.null_diagnostics())
                    for arm in ARMS:
                        result = evaluator.evaluate(fitted, arm, tracked)
                        results.append(result)
                        write_json(directory / (arm + '-summary.json'), result)
                    evaluator.close()
                    tracked.log({'progress/completed_folds': len(means)})
                macro = {arm: float(np.mean([r['score'] for r in results if r['arm'] == arm])) for arm in ARMS}
                # Select between deployable template models; NB remains the distribution ablation.
                selected = max(['shared', 'conditional'], key=lambda arm: (macro[arm], arm == 'shared'))
                final = load_or_fit(stats, config['contexts'], config, output, 'final')
                library_path = output / 'checkpoints/response-library.npz'
                save_response_library(stats, library_path)
                # Final inference deliberately reloads the delivered model aggregates.
                library = ResponseLibrary(library_path, final)
                official_dir = export_official(library, final, selected, config, output)
                frame = pd.concat(means, ignore_index=True); frame.to_parquet(output / 'evaluation-metrics.parquet', index=False)
                full = frame.loc[frame.stratum.eq('all')]
                average = full.groupby(['context', 'method'])[['mse', 'mae', 'zero_mse', 'correlation', 'direction_accuracy']].mean().reset_index()
                average.to_csv(output / 'evaluation-by-context.csv', index=False)
                metrics = {'status': 'trained_and_locally_evaluated', 'official_method_panel_macro': macro,
                           'selected_arm': selected, 'selection_is_development_feedback': True, 'folds': results,
                           'mean_metrics': average.to_dict('records'), 'null_diagnostics': nulls,
                           'all_training_contexts': config['contexts'], 'completed_outer_folds': len(means),
                           'official_submission': False, 'validation_panel_policy': 'fixed identity-hash panel, max 300 targets/context'}
                write_json(output / 'metrics.json', metrics)
                if submit:
                    metrics['official_submission'] = True
                    metrics['official_result'] = submit_and_score(config, official_dir, tracked)
                    write_json(output / 'metrics.json', metrics)
                delivery = publish_artifact(tracked, config, output, tracking_info['mode'] == 'online')
                tracked.summary.update({'pipeline_status': 'completed', 'selected_arm': selected,
                                         'selected_local_official_score': macro[selected], 'artifact_status': delivery['status'],
                                         'completed_outer_folds': len(means)})
                tracked.finish()
                write_json(output / 'complete.json', {'status': 'completed', 'run_id': run_id, 'artifact': delivery,
                                                      'wandb': tracking_info, 'official_submission': bool(submit)})
            except BaseException as error:
                write_json(output / 'failure.json', {'error': str(error), 'type': type(error).__name__, 'traceback': traceback.format_exc(), 'training_complete': False})
                tracked.summary['pipeline_status'] = 'failed'; tracked.finish(exit_code=1)
                raise
            finally:
                sys.stdout, sys.stderr = original_out, original_err
