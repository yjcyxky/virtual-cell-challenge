"""One reproducible run: collect, independent holdout groups, audit, summarize."""
import argparse
import json
import netrc
import os
from pathlib import Path
import subprocess
import sys
import time

EXPERIMENT = Path(__file__).resolve().parents[1]
ROOT = EXPERIMENT.parents[1]
sys.path.insert(0, str(ROOT / 'scripts/dossier'))
from profile_responses import write_json
from rna import hash_file
from evaluate import context_groups


def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def execute(module, arguments, log):
    with log.open('a') as stream:
        subprocess.run([sys.executable, str(EXPERIMENT / 'src' / module), *map(str, arguments)],
                       cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)


def tracking(output, config):
    import wandb
    credential = bool(os.environ.get('WANDB_API_KEY'))
    try:
        credential |= bool(netrc.netrc().authenticators('api.wandb.ai'))
    except (OSError, netrc.NetrcParseError):
        pass
    options = dict(entity='yjcyxky', project='virtual-cell-challenge', group=EXPERIMENT.name,
                   id=output.name, name=output.name, dir=str(output), config=config)
    if config.get('run_type') == 'trained_baseline_official_validation_submission':
        options['resume'] = 'allow'
    mode = 'online' if credential else 'offline'
    try:
        run = wandb.init(**options, mode=mode, settings=wandb.Settings(init_timeout=30))
        return run, {'mode': mode, 'url': run.url if mode == 'online' else None,
                     'offline_reason': None if credential else 'W&B credential not configured'}
    except Exception as error:
        run = wandb.init(**options, mode='offline')
        return run, {'mode': 'offline', 'url': None, 'offline_reason': repr(error)}


def run(run_id, attach_collector=None):
    output = EXPERIMENT / 'outputs' / run_id
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'report.json').exists() and 'posthoc_zero_variance' in json.loads((output / 'report.json').read_text()):
        print(json.dumps({'status': 'already_completed', 'output': str(output)}), flush=True)
        return
    config = {'experiment_id': EXPERIMENT.name, 'run_id': run_id,
              'issue': 'https://github.com/yjcyxky/virtual-cell-challenge/issues/28',
              'pipeline_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              'uv_lock_sha256': hash_file(EXPERIMENT / 'uv.lock'),
              'seeds': [20260921, 20260922, 20260923], 'collection': str(output / 'collection'),
              'run_type': 'retrospective_diagnostic_prediction_experiment', 'base_counts_read_only': True}
    write_json(output / 'config.yaml', config)  # JSON is also valid YAML.
    tracked, tracking_info = tracking(output, config)
    write_json(output / 'tracking.json', tracking_info)
    child, collector_log = None, None
    try:
        with (output / 'tests.log').open('w') as log:
            subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', str(EXPERIMENT / 'tests'), '-v'],
                           cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        if not (output / 'collection/report.json').exists() and attach_collector is None:
            collector_log = (output / 'collect.log').open('a')
            child = subprocess.Popen([sys.executable, str(EXPERIMENT / 'src/collect.py'), '--output', str(output / 'collection')],
                                     cwd=ROOT, stdout=collector_log, stderr=subprocess.STDOUT)
        while True:
            reports = [json.loads(p.read_text()) for p in (output / 'collection').glob('*/report.json')]
            groups, _ = context_groups(reports)
            ready = [name for name, members in groups.items() if len(members) ==
                     (8 if name == 'cross_study_sensitivity' else 6 if name.startswith('Jiang_') else 3)]
            pending = [name for name in ready if not (output / 'evaluation' / name / 'report.json').exists()]
            if pending:
                name = sorted(pending)[0]
                print(json.dumps({'evaluate_group': name, 'collected_tasks': sum(r['tasks'] for r in reports)}), flush=True)
                execute('evaluate.py', ['--collection', output / 'collection', '--output', output / 'evaluation', '--only', name], output / 'evaluate.log')
                tracked.log({'progress/completed_prediction_groups': len(list((output / 'evaluation').glob('*/report.json'))),
                             'progress/collected_tasks': sum(r['tasks'] for r in reports)})
                continue
            if (output / 'collection/report.json').exists():
                assert len(ready) == 15, 'registered prediction groups incomplete'
                break
            if child is not None and child.poll() is not None:
                raise RuntimeError('collector exited without completion: ' + str(child.returncode))
            if attach_collector is not None and not alive(attach_collector):
                raise RuntimeError('attached collector exited without completion')
            time.sleep(15)
        if child is not None:
            assert child.wait() == 0
        if not (output / 'audit/report.json').exists():
            execute('audit.py', ['--collection', output / 'collection', '--output', output / 'audit'], output / 'audit.log')
        if not (output / 'report.json').exists():
            execute('summarize.py', ['--output', output], output / 'summarize.log')
        if not (output / 'posthoc-zero-variance/report.json').exists():
            execute('audit_zero_variance.py', ['--run-folder', output], output / 'posthoc-zero-variance.log')
        execute('summarize.py', ['--output', output, '--attach-posthoc'], output / 'summarize.log')
        summary = json.loads((output / 'report.json').read_text())
        for row in summary['prediction_summary']:
            if row['support_subset'] == 'all' and row['scale'] == 'control_only':
                tracked.summary[row['family'] + '/' + row['model'] + '/relative_MSE_improvement'] = row['relative_MSE_improvement']
        if tracking_info['mode'] == 'online':
            import wandb
            artifact = wandb.Artifact(EXPERIMENT.name + '-' + run_id, type='evaluation')
            for name in ['report.json', 'report.html', 'prediction-summary.parquet', 'config.yaml']:
                artifact.add_file(str(output / name), name=name)
            for p in (output / 'evaluation').glob('*/hyperparameters.json'):
                artifact.add_file(str(p), name=str(p.relative_to(output)))
            tracked.log_artifact(artifact)
        tracked.finish()
        print(json.dumps({'status': 'completed', 'output': str(output)}), flush=True)
    except Exception as error:
        write_json(output / 'pipeline-failure.json', {'status': 'failed', 'error': repr(error), 'config': config})
        tracked.finish(exit_code=1)
        raise
    finally:
        if collector_log:
            collector_log.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--attach-collector', type=int)
    parser.add_argument('--scope', choices=['original', 'genetic_only', 'vcc2026'], default='original')
    parser.add_argument('--submit', action='store_true', help='Submit the prepared vcc2026 run and wait for its official score.')
    args = parser.parse_args()
    if args.submit and args.scope != 'vcc2026':
        parser.error('--submit requires --scope vcc2026')
    if args.scope == 'vcc2026':
        from vcc_submission import run as submission_run
        submission_run(args.run_id, args.submit)
    elif args.scope == 'genetic_only':
        from genetic import run as genetic_run
        genetic_run(args.run_id)
    else:
        run(args.run_id, args.attach_collector)
