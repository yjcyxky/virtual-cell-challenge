"""Complete registered evaluator acceptance batch, with no predictive fitting."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts'), str(ROOT / 'scripts/dossier')]

import wandb
import yaml
from research import bind, execution_metadata
from challenge import scorer_contract
from vcc_task.common import ref, verified, write_json
from vcc_task.capability_controls import audit_panel, audit_cell_count_power


class Tee:
    def __init__(self, original, stream):
        self.original, self.stream = original, stream
    def write(self, value):
        self.original.write(value); self.stream.write(value); self.stream.flush()
    def flush(self):
        self.original.flush(); self.stream.flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    config = json.loads(parser.parse_args().config.read_text())
    resume = os.environ.get('VCC_RESEARCH_RESUME') == '1'
    research = bind(ROOT, config['run_id'], config, resume=resume)
    output = ROOT / 'experiments' / config['experiment_id'] / 'outputs' / config['run_id']
    for name in ('cache', 'predictions', 'checkpoints'):
        (output / name).mkdir(parents=True, exist_ok=True)
    stream = (output / 'train.log').open('a', buffering=1)
    sys.stdout, sys.stderr = Tee(sys.stdout, stream), Tee(sys.stderr, stream)
    frozen = dict(config, research=research, lock_ref=ref(output.parents[1] / 'uv.lock'),
        runtime_versions={p: importlib.metadata.version(p) for p in ['numpy', 'scipy', 'torch', 'gpudge', 'cell-eval2', 'wandb']})
    config_path = output / 'config.yaml'
    if resume and yaml.safe_load(config_path.read_text()) != frozen:
        raise ValueError('Resume requires identical config, binding and runtime')
    config_path.write_text(yaml.safe_dump(frozen, allow_unicode=True, sort_keys=False))
    if scorer_contract() != json.loads(verified(config['benchmark']['scorer']).read_text()):
        raise ValueError('Official scorer contract changed')
    run = wandb.init(entity='yjcyxky', project='virtual-cell-challenge', group=config['experiment_id'],
        id=config['run_id'], name=config['run_id'], config=frozen, dir=str(output),
        resume='allow' if resume else 'never', mode=config['tracking']['mode'],
        settings=wandb.Settings(init_timeout=120))
    started = time.monotonic()
    try:
        results, power = {}, {}
        for context in config['contexts']:
            power[context] = audit_cell_count_power(output, config, context)
            source = config['observation_refs'][context]
            observation = json.loads(verified(source).read_text())
            for panel_id in sorted(observation['panels']):
                key = panel_id + '/' + context
                result = audit_panel(output, config, context, panel_id, source)
                results[key] = result
                run.log({key + '/acceptance_fraction': result['acceptance_fraction'],
                         key + '/defined_official_scores': int(result['checks']['official_six_scores_defined'])})
        if len(results) != config['expected_panels']:
            raise ValueError('Frozen evaluator panel budget was not completed')
        checkpoint = output / 'checkpoints' / 'control-specification.json'
        write_json(checkpoint, {'kind': 'analytic_instrument_controls_no_predictor_fit',
            'conditions': config['conditions'], 'research': research, 'completed_panels': list(results)})
        manifest = output / 'predictions' / 'manifest.json'
        write_json(manifest, {'kind': 'evaluator_controls_not_submission',
            'panels': {k: ref(output / 'predictions' / k / 'panel-result.json') for k in results},
            'power_diagnostics': {c: ref(output / 'predictions/cell-count-power' / c / 'result.json') for c in power}})
        checks = [value for result in results.values() for value in result['checks'].values()]
        metrics = {'status': 'completed', 'research': research, 'execution': execution_metadata(ROOT, config['run_id']),
            'evaluation_completed': True, 'acceptance': {'fraction': sum(checks)/len(checks),
                'passed': sum(checks), 'total': len(checks)},
            'panels': {k: {'checks': r['checks'], 'conditions': r['conditions'],
                          'measured_genes': r['measured_genes'], 'real_cell_counts': r['real_cell_counts']}
                       for k, r in results.items()},
            'power_diagnostics': power,
            'checkpoint_ref': ref(checkpoint), 'predictions_ref': ref(manifest),
            'wandb_url': f'https://wandb.ai/yjcyxky/virtual-cell-challenge/runs/{config["run_id"]}',
            'wandb_sync': config['tracking']['mode'], 'elapsed_seconds_this_attempt': time.monotonic() - started,
            'max_rss_gib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
            'limitations': 'Five contexts, 15 S2/S3/S4 context-panels, three studies, no predictive fits. Scores of oracle controls are instrument acceptance, never transferable model performance.'}
        write_json(output / 'metrics.json', metrics)
        artifact = wandb.Artifact(config['run_id'] + '-diagnostics', type='evaluation', metadata={'research': research})
        for path in [output / 'metrics.json', checkpoint, manifest, config_path]:
            artifact.add_file(str(path), name=path.name)
        for key in results:
            path = output / 'predictions' / key / 'panel-result.json'
            artifact.add_file(str(path), name=key + '/panel-result.json')
        logged = run.log_artifact(artifact)
        if config['tracking']['mode'] == 'online':
            logged.wait(); metrics['artifact'] = logged.qualified_name
            write_json(output / 'metrics.json', metrics)
        run.summary['status'] = 'completed'
        run.summary['acceptance/fraction'] = metrics['acceptance']['fraction']
        run.finish()
    except BaseException:
        run.finish(exit_code=1)
        raise


if __name__ == '__main__':
    main()
