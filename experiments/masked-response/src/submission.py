"""Official evaluation of the frozen shared + low-depth QC fit."""
import argparse
import fcntl
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import subprocess
import runtime  # Keep the masked predictor ahead of the frozen provider modules.
from research import git, ref_error

EXPERIMENT = Path(__file__).resolve().parents[1]
ROOT = EXPERIMENT.parents[1]
CONFIG = EXPERIMENT / 'configs/official-qc-s01.json'
_spec = importlib.util.spec_from_file_location('frozen_submission_io', ROOT / 'experiments/init-linear/src/submission.py')
io = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(io)


def gate(root=ROOT, config_path=CONFIG):
    """Validate this single shared arm before creating outputs or connecting W&B."""
    config = json.loads(config_path.read_text())
    node = json.loads((root / 'docs/research/experiment_dag.json').read_text())['nodes'][config['run_id']]
    stage = node['official_evaluation']
    if config_path.resolve() != (root / stage['config_ref']['path']).resolve():
        raise ValueError('export configuration must be the registered file')
    if node['status'] != 'completed' or stage['comparison_id'] != config['comparison_id']:
        raise ValueError('export requires completed registered training')
    code = [stage['config_ref'], *stage['code_refs'], node['config_ref'], *node['code_refs']]
    for item in [*code, node['metrics_ref'], config['checkpoint_ref']]:
        if problem := ref_error(root, item):
            raise ValueError(f'{item["path"]}: {problem}')
    for item in code:
        if hashlib.sha256(git(root, 'show', 'HEAD:' + item['path']).stdout).hexdigest() != item['sha256']:
            raise ValueError('export code/config must be committed: ' + item['path'])
    frozen = json.loads(git(root, 'show', 'HEAD:docs/research/experiment_dag.json').stdout)['nodes'][config['run_id']]['official_evaluation']
    if any(stage[k] != frozen[k] for k in ('config_ref', 'code_refs', 'comparison_id')):
        raise ValueError('export registration must be committed')
    metrics = json.loads((root / node['metrics_ref']['path']).read_text())
    if metrics['checkpoint_ref'] != config['checkpoint_ref'] or not metrics['evaluation_completed']:
        raise ValueError('checkpoint must be the completed local evaluation checkpoint')
    training = node['expected_config']
    if (training['representation']['kind'] != 'shared' or training['training_selection']['kind'] != 'low_depth'
            or config['arms'] != ['linear'] or config['seed'] != training['seed']):
        raise ValueError('export requires the frozen shared + low-depth QC arm and seed')
    if any(config[k] != training[k] for k in ('generation', 'data')):
        raise ValueError('export must preserve trained normalization/generator configuration')
    return config, metrics, git(root, 'rev-parse', 'HEAD').stdout.decode().strip()


def generate(*args):
    """Reuse the frozen streaming writer; label masked-model semantics explicitly.

    `linear` is the historical evaluator interface, not a conditional model.
    The writer imports this Experiment's model.predict through runtime.py.
    """
    from data import write_json
    result = io.generate(*args)
    result.update(model_kind='shared_low_depth_qc',
                  unseen_policy='zero delta for unseen targets; zero delta outside training measured union')
    write_json(args[2] / 'generation.json', result)
    return result


def load_model(config, metrics, genes):
    import numpy as np
    model = dict(np.load(ROOT / config['checkpoint_ref']['path']))
    if (model['genes'].tolist() != genes or str(model['config_sha256']) != metrics['research']['effective_config_sha256']
            or str(model['training_state']) != 'analytic_fit_complete'
            or not np.array_equal(model['effects'], model['shared'])
            or np.any(model['effects'][:, ~model['trained_mask']])):
        raise ValueError('checkpoint identity/shared response does not match export')
    return model


def replay(config, model, directory, official, manifest, targets):
    """Check actual serialized counts against the frozen predictor/generator."""
    import anndata as ad
    from data import write_json, ref
    from model import predict
    from evaluation import generate_counts
    prediction = ref(directory / 'predictions.h5ad')
    record = directory / 'export-replay.json'
    if record.exists():
        if json.loads(record.read_text())['prediction_ref'] != prediction:
            raise ValueError('replayed prediction changed')
        return
    indices = {0, len(targets)//2, len(targets)-1}
    trained = set(model['targets'].astype(str))
    unseen = next((i for i, target in enumerate(targets) if target not in trained), None)
    if unseen is not None:
        indices.add(unseen)
    cells = config['generation']['cells_per_target']
    actual = ad.read_h5ad(directory / 'predictions.h5ad', backed='r')
    checks = []
    try:
        for ci, context in enumerate(manifest['contexts']):
            controls = ad.read_h5ad(official / f'context_{context}.h5ad')
            delta = predict(model, {}, targets, 'linear')
            for i in sorted(indices):
                start = (ci * len(targets) + i) * cells
                expected = generate_counts(controls.X, delta[i], config['seed']+i, cells, config['data']['target_sum'])
                if (actual.X[start:start+cells] != expected).nnz:
                    raise ValueError(f'export replay mismatch: {context}/{targets[i]}')
                checks.append({'context': context, 'target': targets[i], 'target_index': i, 'cells': cells, 'exact': True})
    finally:
        actual.file.close()
    write_json(record, {'prediction_ref': prediction, 'checkpoint_ref': config['checkpoint_ref'], 'checks': checks})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gate', action='store_true')
    parser.add_argument('--submit', action='store_true')
    args = parser.parse_args()
    config, metrics, commit = gate()
    if args.gate:
        print('Completed shared + low-depth QC official export gate passed', flush=True)
        return
    import yaml
    import wandb
    from data import write_json, ref
    output = EXPERIMENT / 'outputs' / config['run_id']
    training = yaml.safe_load((output / 'config.yaml').read_text())
    versions = {p: importlib.metadata.version(p) for p in training['runtime_versions']}
    if versions != training['runtime_versions'] or ref(EXPERIMENT / 'uv.lock') != training['lock_ref']:
        raise ValueError('export numerical environment differs from completed training')
    if subprocess.check_output([config['vcc_cli'], '--version'], text=True).splitlines()[0] != 'vcc ' + config['vcc_cli_version']:
        raise ValueError('official CLI version changed')
    helper = io.helpers()
    official, manifest, genes, targets, identities = helper.official_inputs(
        {'official': config['official'], 'cells_per_perturbation': config['generation']['cells_per_target']})
    if manifest['panel_id'] != config['panel_id']:
        raise ValueError('official panel changed')
    model = load_model(config, metrics, genes)
    directory = output / 'predictions/official-abc'
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / '.export.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        seal = {'config_ref': ref(CONFIG), 'checkpoint_ref': config['checkpoint_ref'], 'research': metrics['research'],
                'runtime_versions': versions, 'lock_ref': training['lock_ref'], 'official_inputs': identities,
                'code_refs': json.loads((ROOT / 'docs/research/experiment_dag.json').read_text())['nodes'][config['run_id']]['official_evaluation']['code_refs']}
        identity = directory / 'export-identity.json'
        if identity.exists() and json.loads(identity.read_text())['binding'] != seal:
            raise ValueError('official export identity changed on resume')
        if not identity.exists():
            write_json(identity, {'binding': seal, 'export_git_commit': commit, 'config': config})
        tracked = wandb.init(entity='yjcyxky', project='virtual-cell-challenge', group='masked-response',
                             id=config['run_id'], name=config['run_id'], dir=str(output), resume='must',
                             config={'official_export': json.loads(identity.read_text())})
        try:
            destination = directory / 'linear'
            destination.mkdir(exist_ok=True)
            generate(config, model, destination, official, manifest, genes, targets, 'linear', helper)
            replay(config, model, destination, official, manifest, targets)
            io.package(config, destination, official)
            if args.submit:
                io.submit(config, destination, 'linear', tracked)
                marker = destination / 'artifact.json'
                if not marker.exists():
                    artifact = wandb.Artifact(config['run_id'] + '-official-shared-qc', type='evaluation', metadata=seal)
                    artifact.add_file(str(identity), name='export-identity.json')
                    for name in ['predictions.vcc', 'official-summary.json', 'official-status.json',
                                 'submission-entry.json', 'submission-file.json', 'generation.json',
                                 'prediction-audit.json', 'export-replay.json', 'prep.json']:
                        artifact.add_file(str(destination / name), name=name)
                    logged = tracked.log_artifact(artifact)
                    logged.wait()
                    write_json(marker, {'artifact': logged.qualified_name})
            tracked.summary['official/status'] = 'published' if args.submit else 'prepared'
            tracked.finish()
        except BaseException:
            tracked.finish(exit_code=1)
            raise


if __name__ == '__main__':
    main()
