"""Shared run identity, tracking and completion for new registered task runs."""
import importlib.metadata
import json
import os
import resource
import sys
import time

import wandb
import yaml
from research import execution_metadata
from challenge import scorer_contract

from .common import ROOT, ref, verified, write_json


class Tee:
    def __init__(self, original, stream):
        self.original, self.stream = original, stream

    def write(self, value):
        self.original.write(value)
        self.stream.write(value)
        self.stream.flush()

    def flush(self):
        self.original.flush()
        self.stream.flush()


class RegisteredRun:
    def __init__(self, config, research):
        self.config = config
        self.resume = os.environ.get('VCC_RESEARCH_RESUME') == '1'
        self.research = research
        self.output = ROOT / 'experiments' / config['experiment_id'] / 'outputs' / config['run_id']
        for name in ('cache', 'predictions', 'checkpoints', 'diagnostics', 'wandb'):
            (self.output / name).mkdir(parents=True, exist_ok=True)
        stream = (self.output / 'train.log').open('a', buffering=1)
        sys.stdout, sys.stderr = Tee(sys.stdout, stream), Tee(sys.stderr, stream)
        frozen = dict(config, research=self.research, lock_ref=ref(self.output.parents[1] / 'uv.lock'),
                      runtime_versions={p: importlib.metadata.version(p) for p in
                                        ['numpy', 'scipy', 'torch', 'gpudge', 'cell-eval2', 'wandb']})
        self.config_path = self.output / 'config.yaml'
        if self.resume and self.config_path.exists() and yaml.safe_load(self.config_path.read_text()) != frozen:
            raise ValueError('Resume configuration/runtime mismatch')
        self.config_path.write_text(yaml.safe_dump(frozen, allow_unicode=True, sort_keys=False))
        if scorer_contract() != json.loads(verified(config['benchmark']['scorer']).read_text()):
            raise ValueError('Frozen official scorer changed')
        self.run = wandb.init(entity='yjcyxky', project='virtual-cell-challenge', group=config['experiment_id'],
            id=config['run_id'], name=config['run_id'], config=frozen, dir=str(self.output),
            resume='allow' if self.resume else 'never', mode=config['tracking']['mode'],
            settings=wandb.Settings(init_timeout=120))
        self.started = time.monotonic()

    def complete(self, metrics, checkpoint, predictions, artifact_files=()):
        metrics.update(status='completed', research=self.research,
            execution=execution_metadata(ROOT, self.config['run_id']), evaluation_completed=True,
            checkpoint_ref=ref(checkpoint), predictions_ref=ref(predictions),
            wandb_url=f'https://wandb.ai/yjcyxky/virtual-cell-challenge/runs/{self.config["run_id"]}',
            wandb_sync=self.config['tracking']['mode'],
            elapsed_seconds_this_attempt=time.monotonic() - self.started,
            max_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20)
        path = self.output / 'metrics.json'
        write_json(path, metrics)
        artifact = wandb.Artifact(self.config['run_id'] + '-results', type='evaluation',
                                 metadata={'research': self.research})
        for item in [path, self.config_path, checkpoint, predictions, *artifact_files]:
            artifact.add_file(str(item), name=str(item.relative_to(self.output)))
        logged = self.run.log_artifact(artifact)
        if self.config['tracking']['mode'] == 'online':
            logged.wait()
            metrics['artifact'] = logged.qualified_name
            write_json(path, metrics)
        self.run.summary['status'] = 'completed'
        self.run.finish()
