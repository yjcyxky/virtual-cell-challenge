"""Issue #31: context/relationship ML, held-background ablations and submission."""
import argparse
import importlib.metadata
import json
import re
import subprocess
import sys
from pathlib import Path

from dataset import ROOT, prepare
from training import evaluate, final_fit
from profile_responses import write_json
from rna import hash_file
from vcc_submission import official_inputs, generate, audit_prediction, package, submit_and_score

EXPERIMENT = Path(__file__).resolve().parents[1]


def log_artifact(tracked, output, name, kind, paths, identity_file):
    import wandb
    if (output / identity_file).exists():
        return
    artifact = wandb.Artifact(name, type=kind)
    for path in paths:
        artifact.add_file(str(path), name=str(path.relative_to(output)))
    logged = tracked.log_artifact(artifact)
    logged.wait()
    write_json(output / identity_file, {'name': logged.name, 'version': logged.version, 'digest': logged.digest})


def run(run_id, submit):
    if not re.fullmatch(r'[A-Za-z0-9_-]+', run_id):
        raise ValueError('invalid_run_id')
    config_path = EXPERIMENT / 'configs/context-relation.json'
    config = json.loads(config_path.read_text())
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    subprocess.run(['git', 'diff', '--quiet', 'HEAD', '--', str(EXPERIMENT),
                    str(ROOT / 'experiments/exp002-response-transfer-validation/src')], cwd=ROOT, check=True)
    config.update(experiment_id=EXPERIMENT.name, run_id=run_id, pipeline_commit=commit,
                  model_name=config['model_name_prefix'] + '-' + run_id,
                  uv_lock_sha256=hash_file(EXPERIMENT / 'uv.lock'), configuration_sha256=hash_file(config_path),
                  source_collection_sha256=hash_file(ROOT / config['collection'] / 'report.json'),
                  versions={p: importlib.metadata.version(p) for p in ['xgboost','numpy','scipy','pandas','anndata','vcc-cli','wandb']})
    output = EXPERIMENT / 'outputs' / run_id
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'config.yaml').exists():
        assert json.loads((output / 'config.yaml').read_text()) == config, 'changed_run_identity_requires_new_run'
    write_json(output / 'config.yaml', config)
    import wandb
    tracked = wandb.init(entity='yjcyxky', project='virtual-cell-challenge', group=EXPERIMENT.name,
                         id=run_id, name=EXPERIMENT.name + '-' + run_id, resume='allow', mode='online',
                         dir=str(output), config=config)
    try:
        with (output / 'tests.log').open('w') as log:
            for directory in [EXPERIMENT / 'tests', ROOT / 'experiments/exp002-response-transfer-validation/tests']:
                subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', str(directory), '-v'],
                               cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        metadata = prepare(config, output)
        evaluated = evaluate(config, output, metadata, tracked)
        trained = final_fit(config, output, metadata, evaluated, tracked)
        model_paths = ([output / name for name in ['config.yaml','model.npz','evaluation.json','final-training.json',
                        'evaluation-metrics.parquet','evaluation-summary.parquet','evaluation-by-context.parquet',
                        'evaluation-strata.parquet','input-sensitivity.json','tests.log']]
                       + sorted((output / 'checkpoints').glob('*'))
                       + sorted((output / 'prepared/priors').glob('*'))
                       + [output / 'prepared' / name for name in ['preparation.json','tasks.parquet','excluded-tasks.parquet','input-identities.json']])
        log_artifact(tracked, output, EXPERIMENT.name + '-' + run_id, 'model', model_paths, 'model-artifact.json')
        official, manifest, genes, targets, identities = official_inputs(config)
        write_json(output / 'official-inputs.json', identities)
        generated = generate(config, output, official, manifest, genes, targets)
        if not (output / 'prediction-audit.json').exists():
            write_json(output / 'prediction-audit.json', audit_prediction(output / 'predictions.h5ad', genes, targets,
                       manifest['contexts'], config['cells_per_perturbation'], config['max_nnz'], config['max_counts_per_cell']))
        identity = package(config, output, official)
        result = {'status':'prepared', 'run_id':run_id, 'issue':config['issue'], 'pipeline_commit':commit,
                  'evaluation':evaluated, 'training':trained, 'generation':generated, 'submission_file':identity}
        write_json(output / 'result.json', result)
        if submit:
            result['official'] = submit_and_score(config, output, tracked)
            result['status'] = 'completed'
            write_json(output / 'result.json', result)
            paths = [output / n for n in ['official-summary.json','submission-entry.json','submission-file.json',
                     'prediction-audit.json','generation.json','emission-diagnostics.parquet','prep.json']]
            log_artifact(tracked, output, EXPERIMENT.name + '-' + run_id + '-official-score', 'evaluation', paths, 'score-artifact.json')
        tracked.finish()
        print(json.dumps({'status':result['status'], 'output':str(output)}), flush=True)
    except Exception as error:
        write_json(output / 'pipeline-failure.json', {'status':'failed','error':repr(error),'pipeline_commit':commit})
        tracked.finish(exit_code=1)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--submit', action='store_true')
    args = parser.parse_args()
    run(args.run_id, args.submit)
