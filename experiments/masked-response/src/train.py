"""Shared-response completion, generation and official evaluation in one run."""
import runtime  # Frozen providers; local model takes precedence.
from pathlib import Path
import argparse
import importlib.metadata
import json
import logging
import os
import resource
import sys
import time
import numpy as np
import yaml
import wandb
from data import ROOT, prepare, write_json, ref, verify_reference_reuse
from model import fit
from evaluation import generate, evaluate
from selection import prepare_selected

sys.path.insert(0, str(ROOT/'scripts'))
from research import bind, digest, execution_metadata


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    resume = os.environ.get('VCC_RESEARCH_RESUME') == '1'
    research = bind(ROOT, config['run_id'], config, resume=resume)
    output = ROOT/'experiments'/config['experiment_id']/'outputs'/config['run_id']
    log = (output/'train.log').open('a', buffering=1)
    sys.stdout, sys.stderr = Tee(sys.stdout, log), Tee(sys.stderr, log)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(message)s')
    frozen = dict(config, research=research, lock_ref=ref(output.parents[1]/'uv.lock'),
                  runtime_versions={p: importlib.metadata.version(p) for p in ['numpy','scipy','torch','gpudge','cell-eval2','wandb']})
    history = []
    if resume:
        previous = yaml.safe_load((output/'config.yaml').read_text())
        history = previous.pop('execution_history', [])
        if frozen != previous:
            raise ValueError('resume runtime/configuration mismatch')
    execution = execution_metadata(ROOT, config['run_id'])
    if not history:
        history.append({'git_commit': research['git_commit'], 'scope': 'original bound training'})
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
        print(f'Bound {config["run_id"]} at {research["git_commit"]}; resume={resume}', flush=True)
        axis, split, prepared = prepare(output, config)
        if 'training_selection' in config:
            prepared['selection'] = prepare_selected(output, config, axis, split)
        checkpoint_dir = output/'checkpoints'
        checkpoint_dir.mkdir(exist_ok=True)
        checkpoint = checkpoint_dir/'response.npz'
        if checkpoint.exists():
            model = dict(np.load(checkpoint))
            if str(model['config_sha256']) != digest(config) or str(model['training_state']) != 'analytic_fit_complete':
                raise ValueError('checkpoint training identity/state mismatch')
            print('Restored complete analytic training state; evaluation continues', flush=True)
        else:
            model = fit(output/'cache', axis, split, config)
            model.update(config_sha256=np.asarray(digest(config)), training_state=np.asarray('analytic_fit_complete'))
            with (checkpoint_dir/'response.tmp').open('wb') as stream:
                np.savez_compressed(stream, **model)
            (checkpoint_dir/'response.tmp').replace(checkpoint)
        references = verify_reference_reuse(output, model, config) if 'reference_reuse' in config else {}
        run.log({'train/masked_mse': float(model['training_mse']), 'train/tasks': int(model['training_tasks']),
                 'train/complete': 1, 'resources/max_rss_gib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20})
        print(f'Response fit complete: masked MSE={float(model["training_mse"]):.8f}', flush=True)
        generate(output, model, axis, split, config)
        del model
        result = evaluate(output, split, config, run)
        for panel, arms in references.items():
            result[panel].update(arms)
            run.log({f'reference/{panel}/{arm}/Overall': item['Overall'] for arm, item in arms.items()})
        metrics = {'status': 'completed', 'research': research, 'execution': execution, 'evaluation_completed': True,
                   'S2': {'Overall': result['all']['linear']['Overall']}, 'evaluation': result,
                   'checkpoint_ref': ref(checkpoint), 'predictions_ref': ref(output/'predictions/manifest.json'),
                   'wandb_url': f'https://wandb.ai/yjcyxky/virtual-cell-challenge/runs/{config["run_id"]}',
                   'wandb_sync': 'online', 'preparation': prepared,
                   'elapsed_seconds_this_attempt': time.monotonic()-started,
                   'resources': {'max_rss_gib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20}}
        for name in ('representation', 'reference-reuse', 'selection', 'control-provenance'):
            path = output/'cache'/f'{name}.json'
            if path.exists():
                metrics[name.replace('-', '_')+'_ref'] = ref(path)
        write_json(output/'metrics.json', metrics)
        artifact = wandb.Artifact(config['run_id']+'-result', type='model', metadata={'research': research})
        artifact.add_file(str(checkpoint), name='response.npz')
        artifact.add_file(str(output/'metrics.json'), name='metrics.json')
        artifact.add_file(str(output/'config.yaml'), name='config.yaml')
        artifact.add_file(str(output/'predictions/manifest.json'), name='prediction-manifest.json')
        for name in ('representation', 'reference-reuse', 'selection', 'control-provenance'):
            path = output/'cache'/f'{name}.json'
            if path.exists():
                artifact.add_file(str(path), name=f'{name}.json')
        logged = run.log_artifact(artifact)
        logged.wait()
        metrics['artifact'] = logged.qualified_name
        write_json(output/'metrics.json', metrics)
        run.summary['status'] = 'completed'
        run.summary['S2.Overall'] = metrics['S2']['Overall']
        run.finish()
        print(json.dumps({'status': 'completed', 'S2': metrics['S2'], 'artifact': metrics['artifact']}), flush=True)
    except BaseException:
        run.finish(exit_code=1)
        raise


if __name__ == '__main__':
    main()
