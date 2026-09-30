"""Registered no-predictor-fit observation and generation calibration."""
import argparse
import importlib.metadata
import json
import logging
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
from vcc_task.observations import prepare_context
from vcc_task.calibration import calibrate


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
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(message)s')
    frozen = dict(config, research=research, lock_ref=ref(output.parents[1] / 'uv.lock'),
        runtime_versions={p: importlib.metadata.version(p) for p in ['numpy', 'scipy', 'torch', 'gpudge', 'cell-eval2', 'wandb']})
    config_path = output / 'config.yaml'
    if resume and yaml.safe_load(config_path.read_text()) != frozen:
        raise ValueError('Resume requires identical config, binding and runtime')
    config_path.write_text(yaml.safe_dump(frozen, allow_unicode=True, sort_keys=False))
    actual_scorer = scorer_contract()
    expected_scorer = json.loads(verified(config['benchmark']['scorer']).read_text())
    if actual_scorer != expected_scorer:
        raise ValueError('Official scorer contract changed')
    run = wandb.init(entity='yjcyxky', project='virtual-cell-challenge', group=config['experiment_id'],
        id=config['run_id'], name=config['run_id'], config=frozen, dir=str(output),
        resume='allow' if resume else 'never', mode=config['tracking']['mode'],
        settings=wandb.Settings(init_timeout=120))
    started = time.monotonic()
    try:
        observations, calibrations = {}, {}
        for context in config['contexts']:
            observations[context] = prepare_context(output, config, context)
            calibrations[context] = calibrate(output, config, context, observations[context])
            run.log({f'calibration/{context}/acceptance_fraction': calibrations[context]['acceptance_fraction'],
                     f'data/{context}/measured_genes': observations[context]['measured_genes']})
        checkpoint = output / 'checkpoints' / 'analytic-emitter.json'
        write_json(checkpoint, {'kind': 'fixed_analytic_emitter_no_predictor_fit', 'emitter': config['emitter'],
            'calibration': config['calibration'], 'resume_state': 'all_context_stages_complete', 'research': research})
        manifest = output / 'predictions' / 'manifest.json'
        write_json(manifest, {'kind': 'diagnostic_predictions_not_submission',
            'files': [ref(output / 'predictions' / c / 'calibration.json') for c in config['contexts']]})
        metrics = {'status': 'completed', 'research': research, 'execution': execution_metadata(ROOT, config['run_id']),
            'evaluation_completed': True,
            'acceptance': {'fraction': sum(v['acceptance_fraction'] for v in calibrations.values()) / len(calibrations)},
            'observations': {c: ref(output / 'cache' / c / 'observations.json') for c in config['contexts']},
            'calibrations': {c: ref(output / 'predictions' / c / 'calibration.json') for c in config['contexts']},
            'checkpoint_ref': ref(checkpoint), 'predictions_ref': ref(manifest),
            'wandb_url': f'https://wandb.ai/yjcyxky/virtual-cell-challenge/runs/{config["run_id"]}',
            'wandb_sync': config['tracking']['mode'], 'elapsed_seconds_this_attempt': time.monotonic() - started,
            'max_rss_gib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
            'limitations': 'Data/generator calibration only; no predictive ability or six-metric Overall is claimed.'}
        write_json(output / 'metrics.json', metrics)
        artifact = wandb.Artifact(config['run_id'] + '-diagnostics', type='evaluation', metadata={'research': research})
        for path in [output / 'metrics.json', checkpoint, manifest, config_path]:
            artifact.add_file(str(path), name=path.name)
        for context in config['contexts']:
            for path in [output / 'cache' / context / 'observations.json', output / 'predictions' / context / 'calibration.json']:
                artifact.add_file(str(path), name=context + '-' + path.name)
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
