"""One registered fit, generation and official evaluation per W&B/run identity."""
import gc
import importlib.metadata
import json
import logging
import resource
import sys
import time
import yaml
import wandb
import torch
import model as model_module

from data import ROOT, write_json, ref
from research import execution_metadata
from vcc_mechanism.inputs import prepare, release_file_cache
from model import fit, generate
from evaluation import evaluate


class Tee:
    def __init__(self, original, stream):
        self.original, self.stream = original, stream
    def write(self, value):
        self.original.write(value); self.stream.write(value); self.stream.flush()
    def flush(self):
        self.original.flush(); self.stream.flush()


def run_bound(config, research, resume):
    output = ROOT/'experiments'/config['experiment_id']/'outputs'/config['run_id']
    log = (output/'train.log').open('a', buffering=1)
    sys.stdout, sys.stderr = Tee(sys.stdout, log), Tee(sys.stderr, log)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(message)s')
    frozen = dict(config, research=research, lock_ref=ref(output.parents[1]/'uv.lock'),
                  runtime_versions={p: importlib.metadata.version(p) for p in ['numpy','scipy','torch','gpudge','cell-eval2','wandb']})
    history = []
    if resume:
        previous = yaml.safe_load((output/'config.yaml').read_text()); history = previous.pop('execution_history', [])
        if previous != frozen:
            raise ValueError('Resume runtime/configuration mismatch')
    execution = execution_metadata(ROOT, config['run_id'])
    if execution not in history:
        history.append(execution)
    frozen['execution_history'] = history
    (output/'config.yaml').write_text(yaml.safe_dump(frozen, sort_keys=False, allow_unicode=True))
    run = wandb.init(entity='yjcyxky', project='virtual-cell-challenge', group=config['experiment_id'],
                     id=config['run_id'], name=config['run_id'], config=frozen, dir=str(output),
                     resume='allow' if resume else 'never', mode=config['tracking']['mode'], allow_val_change=True,
                     settings=wandb.Settings(init_timeout=120))
    started = time.monotonic()
    try:
        axis, split, prepared = prepare(output, config)
        (output/'checkpoints').mkdir(exist_ok=True)
        checkpoint, model, fit_audit = fit(output, axis, split, config, run)
        fit_elapsed = time.monotonic()-started
        generate(output, model, axis, split, config)
        from vcc_mechanism.diagnostics import ntc_audit
        decoder = (lambda ntc: model_module.diagnostic_counts(output, model, ntc, config)) if hasattr(model_module, 'diagnostic_counts') else None
        ntc_audit(output, config, decoder)
        del decoder
        del model
        gc.collect(); torch.cuda.empty_cache(); release_file_cache(output)
        result = evaluate(output, split, config, run)
        metrics = {'status': 'completed', 'research': research, 'execution': execution,
                   'evaluation_completed': True, 'S2': {'Overall': result['all']['linear']['Overall']},
                   'evaluation': result, 'checkpoint_ref': ref(checkpoint),
                   'predictions_ref': ref(output/'predictions/manifest.json'), 'preparation': prepared,
                   'fit_ref': ref(output/'cache/fit.json'), 'control_provenance_ref': ref(output/'cache/control-provenance.json'),
                   'ntc_diagnostics_ref': ref(output/'cache/ntc-diagnostics.json'),
                   'wandb_url': f'https://wandb.ai/yjcyxky/virtual-cell-challenge/runs/{config["run_id"]}',
                   'wandb_sync': 'online', 'elapsed_seconds_this_attempt': time.monotonic()-started,
                   'fit_and_preparation_seconds_this_attempt': fit_elapsed,
                   'resources': {'max_rss_gib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20}}
        write_json(output/'metrics.json', metrics)
        artifact = wandb.Artifact(config['run_id']+'-result', type='model', metadata={'research': research})
        for path in [checkpoint, output/'metrics.json', output/'config.yaml', output/'predictions/manifest.json',
                     output/'cache/fit.json', output/'cache/control-provenance.json', output/'cache/ntc-diagnostics.json']:
            artifact.add_file(str(path), name=path.name)
        logged = run.log_artifact(artifact); logged.wait()
        metrics['artifact'] = logged.qualified_name
        write_json(output/'metrics.json', metrics)
        run.summary['status'] = 'completed'; run.summary['S2.Overall'] = metrics['S2']['Overall']
        run.log({'resources/max_rss_gib': metrics['resources']['max_rss_gib'], 'train/complete': 1})
        run.finish()
        print(json.dumps({'status': 'completed', 'S2': metrics['S2'], 'artifact': metrics['artifact']}), flush=True)
    except BaseException:
        run.finish(exit_code=1)
        raise
