"""Issue #32 entry: one identity from validation through training and local evaluation."""
import argparse
import base64
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import traceback
from datetime import datetime, timezone

import torch
import yaml

from data import ROOT, EXPERIMENT, CountData, prepare, hash_file, write_json
from priors import prepare_priors
from training import experiment
from comparison import compare_trials, comparison_sources


class Tee:
    def __init__(self, stream, log):
        self.stream, self.log = stream, log
    def write(self, value):
        self.stream.write(value); self.log.write(value); self.log.flush()
        return len(value)
    def flush(self):
        self.stream.flush(); self.log.flush()


def committed_pipeline():
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    guarded = [str(EXPERIMENT), 'scripts/dossier/rna.py', 'experiments/exp001-context-pair-xgb/src/features.py']
    subprocess.run(['git', 'diff', '--quiet', 'HEAD', '--', *guarded], cwd=ROOT, check=True)
    untracked = subprocess.check_output(['git', 'ls-files', '--others', '--exclude-standard', '--', *guarded], cwd=ROOT, text=True)
    assert not untracked.strip(), 'formal_training_requires_committed_source'
    return commit


def completion_summary(tracked, result, delivery):
    # W&B SummaryDict.update accepts one mapping, unlike dict.update(**kwargs).
    tracked.summary.update({'pipeline_status': 'completed', 'artifact_status': delivery['status'], 'local_only': True,
                            'macro_response_mse': result['macro']['response_mse'],
                            'mse_improvement_vs_zero': result['mse_improvement_vs_zero_fraction']})


def verify_remote_artifact(output, delivery):
    import wandb
    assert delivery['status'] == 'verified_online', 'completion_retry_requires_uploaded_artifact'
    remote = wandb.Api().artifact(f'yjcyxky/virtual-cell-challenge/{delivery["name"]}', type='model')
    assert remote.digest == delivery['digest'] and remote.version == delivery['version'], 'artifact_version_changed'
    assert len(remote.manifest.entries) == delivery['files']
    for name, entry in remote.manifest.entries.items():
        path = (output / name).resolve()
        assert path.is_relative_to(output.resolve()) and path.is_file(), 'artifact_file_missing'
        digest = hashlib.md5()
        with path.open('rb') as stream:
            while block := stream.read(8 << 20):
                digest.update(block)
        assert base64.b64encode(digest.digest()).decode() == entry.digest, 'archived_result_changed:' + name


def finalize_run(run_id):
    """Retry bookkeeping only after all fits, evaluation and artifact upload succeeded."""
    import wandb
    from runtime import environment_identity
    output = EXPERIMENT / 'outputs' / run_id
    config = yaml.safe_load((output / 'config.yaml').read_text())
    assert config['run_id'] == run_id and config['experiment_id'] == EXPERIMENT.name
    assert environment_identity() == config['environment_identity'], 'completion_retry_environment_changed'
    commit = committed_pipeline()
    if (output / 'complete.json').exists():
        print((output / 'complete.json').read_text())
        return
    result = json.loads((output / 'metrics.json').read_text())
    assert result['status'] == 'trained_and_locally_evaluated' and result['contexts'] == len(config['contexts'])
    assert not result['official_submission']
    checkpoints = {}
    for key in ['holdout-' + c for c in config['contexts']] + ['final']:
        path = output / 'checkpoints' / f'{key}-last.pt'
        state = torch.load(path, map_location='cpu', weights_only=False)
        assert {'model', 'optimizer', 'scheduler', 'sampler', 'random', 'progress'} <= state.keys()
        assert state['progress']['complete'], 'cannot_finalize_incomplete_fit:' + key
        checkpoints[key] = hash_file(path)
        if key != 'final':
            assert (output / 'predictions' / key / 'complete.json').is_file()
    delivery = json.loads((output / 'artifact.json').read_text())
    verify_remote_artifact(output, delivery)
    tracked = wandb.init(entity='yjcyxky', project='virtual-cell-challenge', group=EXPERIMENT.name,
                         id=run_id, name=f'{config["variant"]}-{run_id}', dir=str(output),
                         resume='must', mode='online', config=config, settings=wandb.Settings(init_timeout=30))
    try:
        completion_summary(tracked, result, delivery)
        tracked.summary['finalization_commit'] = commit
        tracked.summary['training_commit'] = config['pipeline_commit']
        tracked.log({'recovery/finalization_only': 1, 'recovery/optimizer_steps_repeated': 0})
        tracked.finish()
    except BaseException:
        tracked.finish(exit_code=1)
        raise
    completion = {'status': 'completed', 'wandb_sync': 'online', 'artifact': delivery,
                  'training_commit': config['pipeline_commit'], 'finalization_commit': commit,
                  'finalization_only_retry': True, 'optimizer_steps_repeated': 0,
                  'complete_training_checkpoint_sha256': checkpoints,
                  'original_failure_record_preserved': (output / 'failure.json').exists()}
    write_json(output / 'complete.json', completion)
    with (output / 'train.log').open('a') as stream:
        stream.write(json.dumps({'stage': 'finalization_recovered', **completion}) + '\n')
    print(json.dumps(completion), flush=True)


def artifact(tracked, output, online):
    import wandb
    marker = output / 'artifact.json'
    if marker.exists():
        saved = json.loads(marker.read_text())
        if saved['status'] == 'verified_online':
            return saved
    paths = [output / n for n in ['config.yaml', 'metrics.json', 'evaluation-metrics.parquet', 'evaluation-by-context.csv', 'evaluation-by-stratum.csv']]
    paths += sorted((output / 'checkpoints').glob('*-best.pt'))
    paths += sorted((output / 'cache').glob('*/state.npz'))
    paths += sorted((output / 'cache').glob('*/training.json'))
    paths += sorted((output / 'cache').glob('*/prior-evidence.parquet'))
    paths += sorted((output / 'cache').glob('*/batch-diagnostics.parquet'))
    paths += sorted((output / 'cache').glob('*/evidence-scope.json'))
    paths += sorted((output / 'cache').glob('*/state-scope.json'))
    paths += sorted((output / 'cache').glob('*/split.json'))
    paths += sorted((output / 'cache').glob('*/reused-preprocessing.json'))
    paths += sorted((output / 'cache').glob('*/prior-strength.npy'))
    paths += sorted((output / 'predictions').glob('*/example-*.npz'))
    paths += [output / 'cache' / 'priors' / n for n in ['modules.npz', 'modules.json', 'complete.json']]
    paths += [output / 'cache' / n for n in ['data-reference.json', 'readout-support.json', 'tests.log']]
    paths += sorted((output / 'cache').glob('config-before-validation-repair-*.yaml'))
    paths += sorted(output.glob('comparison*.csv')) + sorted(output.glob('comparison*.parquet')) + sorted(output.glob('comparison.json'))
    item = wandb.Artifact(EXPERIMENT.name + '-' + output.name, type='model', metadata={
        'pipeline_commit': tracked.config['pipeline_commit'], 'local_only': True, 'raw_cells_uploaded': False,
        'prediction_storage': 'two generated count examples per background; all-task prediction summaries retained locally'})
    md5 = {}
    for path in paths:
        assert path.is_file(), str(path)
        name = str(path.relative_to(output))
        item.add_file(str(path), name=name)
        digest = hashlib.md5()
        with path.open('rb') as fh:
            while block := fh.read(8 << 20):
                digest.update(block)
        md5[name] = base64.b64encode(digest.digest()).decode()
    logged = tracked.log_artifact(item)
    if not online:
        result = {'status': 'pending_offline_sync', 'name': item.name, 'files': len(paths)}
    else:
        logged.wait()
        remote = wandb.Api().artifact(f'yjcyxky/virtual-cell-challenge/{logged.name}', type='model')
        assert remote.digest == logged.digest
        entries = remote.manifest.entries
        assert set(entries) == set(md5)
        for name, digest in md5.items():
            assert entries[name].digest == digest, name
        result = {'status': 'verified_online', 'name': logged.name, 'version': logged.version,
                  'digest': logged.digest, 'files': len(paths), 'file_digests_verified': True}
    write_json(marker, result)
    return result


def main(args):
    if not re.fullmatch(r'[A-Za-z0-9_-]+', args.run_id):
        raise ValueError('invalid_run_id')
    if args.finalize_only:
        assert not any(getattr(args, name, None) is not None for name in ['seed', 'variant', 'cache_source', 'reuse_run', 'compare_runs', 'repair_validation_reason', 'restart_from_run']), 'completion_retry_cannot_change_conditions'
        assert Path(args.config).resolve() == EXPERIMENT / 'configs/default.yaml'
        finalize_run(args.run_id)
        return
    output = EXPERIMENT / 'outputs' / args.run_id
    output.mkdir(parents=True, exist_ok=True)
    (output / 'cache').mkdir(exist_ok=True)
    log = (output / 'train.log').open('a', buffering=1)
    sys.stdout, sys.stderr = Tee(sys.stdout, log), Tee(sys.stderr, log)
    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text())
    for name in ['seed', 'variant', 'cache_source']:
        value = getattr(args, name)
        if value is not None:
            config[name] = value
    if args.reuse_run is not None:
        assert re.fullmatch(r'[A-Za-z0-9_-]+', args.reuse_run)
        source_run = EXPERIMENT / 'outputs' / args.reuse_run
        assert (source_run / 'complete.json').exists(), 'reuse_requires_completed_source_run'
        source_config = yaml.safe_load((source_run / 'config.yaml').read_text())
        source_data = source_config.get('cache_source', str(source_run / 'cache' / 'data'))
        if config.get('cache_source'):
            assert str(Path(config['cache_source']).resolve()) == str(Path(source_data).resolve())
        config['cache_source'], config['reuse_run'] = source_data, args.reuse_run
    if config.get('cache_source'):
        config['cache_source'] = str(Path(config['cache_source']).resolve())
    if args.restart_from_run is not None:
        assert re.fullmatch(r'[A-Za-z0-9_-]+', args.restart_from_run) and args.restart_from_run != args.run_id
        source_path = EXPERIMENT / 'outputs' / args.restart_from_run / 'config.yaml'
        source_config = yaml.safe_load(source_path.read_text())
        assert source_config['seed'] == config['seed'] and source_config['variant'] == config['variant']
        config['restart_source'] = {'run_id': args.restart_from_run, 'config_sha256': hash_file(source_path),
                                    'pipeline_commit': source_config['pipeline_commit'], 'training_state_reused': False}
    if args.compare_runs is not None:
        assert args.run_id not in args.compare_runs, 'comparison_source_cannot_be_current_run'
        config['comparison_sources'] = comparison_sources(args.compare_runs)
    commit = committed_pipeline()
    assert torch.cuda.is_available(), 'GPU_required_no_silent_CPU_fallback'
    torch.set_num_threads(8)
    torch.set_float32_matmul_precision('high')
    torch.use_deterministic_algorithms(True)
    from runtime import environment_identity, verify_environment
    environment = environment_identity()
    verify_environment(EXPERIMENT / '.venv' / 'experiment-environment.json', environment)
    config.update(experiment_id=EXPERIMENT.name, run_id=args.run_id, pipeline_commit=commit,
                  environment_identity=environment,
                  uv_lock_sha256=hash_file(EXPERIMENT / 'uv.lock'), configuration_sha256=hash_file(config_path),
                  source_collection_sha256=hash_file(ROOT / config['collection'] / 'report.json'),
                  gene_axis_sha256=hash_file(ROOT / config['gene_axis']),
                  prior_source_sha256=hash_file(ROOT / 'data/raw/networks/SOURCE.json'),
                  versions={p: importlib.metadata.version(p) for p in ['torch', 'numpy', 'scipy', 'pandas', 'h5py', 'wandb', 'scikit-learn']},
                  runtime={'python': sys.version, 'executable': sys.executable, 'cuda': torch.version.cuda,
                           'gpu': torch.cuda.get_device_name(), 'architecture': os.uname().machine,
                           'deterministic_algorithms': True, 'float32_matmul_precision': 'high'},
                  official_submission=False,
                  source_data_run='20260922-c', source_cache_run=None if not config.get('cache_source') else str(Path(config['cache_source']).parents[1]))
    if (output / 'config.yaml').exists():
        previous = yaml.safe_load((output / 'config.yaml').read_text())
        if '_pretraining_revisions' in previous:
            config['_pretraining_revisions'] = previous['_pretraining_revisions']
        if previous != config:
            assert args.repair_validation_reason, 'changed_conditions_require_new_run'
            assert not (output / 'cache' / 'optimization-started.json').exists(), 'training_started_code_changes_require_new_run'
            assert not list((output / 'checkpoints').glob('*.pt')), 'existing_training_state_requires_new_run'
            assert {k: v for k, v in previous.items() if k != 'pipeline_commit'} == {k: v for k, v in config.items() if k != 'pipeline_commit'}, 'validation_repair_cannot_change_configuration'
            number = len(config.get('_pretraining_revisions', [])) + 1
            old_bytes = (output / 'config.yaml').read_bytes()
            (output / 'cache' / f'config-before-validation-repair-{number}.yaml').write_bytes(old_bytes)
            config['_pretraining_revisions'] = [*config.get('_pretraining_revisions', []), {
                'previous_commit': previous['pipeline_commit'], 'new_commit': commit,
                'reason': args.repair_validation_reason, 'optimizer_started': False,
                'timestamp_utc': datetime.now(timezone.utc).isoformat(),
                'previous_config_sha256': hashlib.sha256(old_bytes).hexdigest()}]
            (output / 'config.yaml').write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True))
    else:
        (output / 'config.yaml').write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True))
    (output / 'wandb').mkdir(exist_ok=True)
    (output / 'cache').mkdir(exist_ok=True)
    import wandb
    try:
        tracked = wandb.init(entity='yjcyxky', project='virtual-cell-challenge', group=EXPERIMENT.name,
                             id=args.run_id, name=f'{config["variant"]}-{args.run_id}', dir=str(output),
                             resume='allow', mode='online', config=config, allow_val_change=True, settings=wandb.Settings(init_timeout=30))
        online = True
    except Exception as error:
        print(json.dumps({'stage': 'wandb_offline_fallback', 'reason': str(error)}), flush=True)
        tracked = wandb.init(entity='yjcyxky', project='virtual-cell-challenge', group=EXPERIMENT.name,
                             id=args.run_id, name=f'{config["variant"]}-{args.run_id}', dir=str(output),
                             mode='offline', config=config)
        online = False
    try:
        tracked.summary['pipeline_status'] = 'validating'
        with (output / 'cache' / 'tests.log').open('a') as testlog:
            subprocess.run([sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider', str(EXPERIMENT / 'tests')], cwd=EXPERIMENT,
                           stdout=testlog, stderr=subprocess.STDOUT, check=True)
        cache = prepare(config, output)
        data = CountData(cache)
        write_json(output / 'cache' / 'readout-support.json', data.readout_support)
        write_json(output / 'cache' / 'data-reference.json', {'path': str(cache), 'sha256': hash_file(cache / 'complete.json'),
                   'data_spec_hash': data.report['data_spec_hash'], 'source_inputs': data.report['inputs']})
        priors = prepare_priors(config, data.genes, output)
        tracked.log({'preparation/contexts': len(data.masks), 'preparation/tasks': len(data.tasks),
                     'preparation/common_genes': int(data.common.sum()), 'preparation/cached_cells': len(data.cells)})
        tracked.summary['pipeline_status'] = 'training_and_evaluating'
        result = experiment(data, config, output, priors, tracked)
        comparison = compare_trials(data, config, output)
        if comparison is not None:
            result['comparison'] = comparison
            tracked.log({'comparison/completed_runs': comparison['run_count'], 'comparison/task_instances': comparison['task_instances']})
        result['wandb_url'] = f'https://wandb.ai/yjcyxky/virtual-cell-challenge/runs/{args.run_id}'
        result['wandb_sync'] = 'online' if online else 'pending_offline_sync'
        result['official_submission'] = False
        write_json(output / 'metrics.json', result)
        delivery = artifact(tracked, output, online)
        completion_summary(tracked, result, delivery)
        tracked.finish()
        write_json(output / 'complete.json', {'status': 'completed', 'wandb_sync': result['wandb_sync'], 'artifact': delivery})
        print(json.dumps(result), flush=True)
    except BaseException as error:
        write_json(output / 'failure.json', {'error': type(error).__name__, 'message': str(error), 'traceback': traceback.format_exc(),
                                           'status': 'incomplete', 'resume_run_id': args.run_id})
        tracked.summary['pipeline_status'] = 'interrupted_or_failed'; tracked.finish(exit_code=1)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--finalize-only', action='store_true', help='Retry completion metadata only; requires all fits and an unchanged verified online artifact.')
    parser.add_argument('--config', default=str(EXPERIMENT / 'configs/default.yaml'))
    parser.add_argument('--seed', type=int)
    parser.add_argument('--variant', choices=['true_prior', 'random_prior', 'no_prior', 'no_state', 'no_context', 'no_residual', 'no_module'])
    parser.add_argument('--cache-source')
    parser.add_argument('--reuse-run', help='Reuse explicitly versioned data/PCA/prior diagnostics from a completed same-protocol run.')
    parser.add_argument('--restart-from-run', help='Record the fixed source of an independent from-scratch restart; does not load training state.')
    parser.add_argument('--compare-runs', nargs='+', help='Completed trial IDs to compare with this trial before final artifact delivery.')
    parser.add_argument('--repair-validation-reason', help='Document a same-config code repair before any optimization; old config is preserved.')
    main(parser.parse_args())
