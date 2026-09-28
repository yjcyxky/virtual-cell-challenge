"""Export an existing H1-selected checkpoint to the official A/B/C panel."""
import importlib.metadata
import importlib.util
import contextlib
import json
from pathlib import Path
import subprocess
import sys

from runtime import configure_memory_policy
configure_memory_policy()

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from scipy import sparse
import wandb
import xgboost as xgb
import yaml

from common import EXPERIMENT, ROOT, digest, event, write_json
from data import Data, GeneIdentity, NTC, inspect_panel, prepare_context
from features import Features
from generation import Generator
from training import fit_id, predict


def helpers():
    """Reuse the existing full-file audit, official CLI and resumable upload."""
    source = ROOT / 'experiments/exp002-response-transfer-validation/src'
    sys.path.insert(0, str(source))
    spec = importlib.util.spec_from_file_location('exp002_submission', source / 'vcc_submission.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_official_data(training, official, config, output, identities):
    directory = output / 'cache/official-export-data'
    directory.mkdir(parents=True, exist_ok=True)
    marker = directory / 'complete.json'
    identity = {'sources': identities, 'training_data_sha256': digest(training.directory / 'complete.json')}
    if marker.exists():
        if json.loads(marker.read_text())['export_inputs'] != identity:
            raise ValueError('official_export_inputs_changed')
    else:
        canonical = GeneIdentity()
        for context in ['A', 'B', 'C']:
            path = official / f'context_{context}.h5ad'
            obs, mapping = inspect_panel(path, context, canonical)
            if set(obs.target) != {NTC}:
                raise ValueError('official_input_must_contain_only_controls')
            prepare_context(context, [(path, obs, mapping)], training.genes,
                            training.lookup, config, directory)
        write_json(marker, {**training.metadata, 'contexts': ['A', 'B', 'C'], 'export_inputs': identity})
    return Data(directory)


def generate(data, features, booster, genes, targets, config, directory, legacy):
    contexts = ['A', 'B', 'C']
    canonical = GeneIdentity()
    internal = [canonical.canonical(t) for t in targets]
    if any(t not in data.lookup for t in internal):
        raise ValueError('official_target_without_gene_features')
    size = config['cells_per_prediction']
    rows = len(contexts) * len(targets) * size
    final = directory / 'predictions.h5ad'
    marker = directory / 'generation.json'
    if marker.exists():
        result = json.loads(marker.read_text())
        if digest(final) != result['sha256']:
            raise ValueError('exported_prediction_changed')
        return result
    obs = pd.DataFrame({'context': np.repeat(contexts, len(targets) * size),
                        'target_gene': np.tile(np.repeat(targets, size), len(contexts))},
                       index=[f'exp007-{i}' for i in range(rows)])
    for column in obs:
        obs[column] = pd.Categorical(obs[column])
    temporary = directory / 'predictions.partial.h5ad'
    ad.AnnData(sparse.csr_matrix((rows, len(genes)), dtype=np.int32), obs=obs,
               var=pd.DataFrame(index=pd.Index(genes, name='gene_name'))).write_h5ad(temporary)
    pointer = 0
    with h5py.File(temporary, 'r+') as handle:
        del handle['X']
        matrix = handle.create_group('X')
        matrix.attrs.update({'encoding-type': 'csr_matrix', 'encoding-version': '0.1.0',
                             'shape': [rows, len(genes)]})
        for name in ['data', 'indices']:
            matrix.create_dataset(name, (0,), maxshape=(None,), chunks=(1048576,),
                                  dtype='int32', compression='gzip', compression_opts=1)
        matrix.create_dataset('indptr', (rows + 1,), dtype='int64')
        for context in contexts:
            axis = data.contexts[context]['gene_indices']
            positions = {int(g): i for i, g in enumerate(axis)}
            if set(positions) != set(range(len(genes))):
                raise ValueError('official_control_axis_incomplete')
            reorder = np.array([positions[i] for i in range(len(genes))])
            response = predict(booster, features, context, internal, axis)
            np.savez_compressed(directory / f'{context}-response.npz',
                                targets=np.array(targets), response=response[:, reorder])
            generator = Generator(data, context, config)
            for index, target in enumerate(internal):
                counts = generator.generate(target, response[index])[:, reorder]
                legacy.append_sparse(matrix, counts.astype(np.int32), pointer)
                pointer += size
                if len(matrix['data']) > 4750000000:
                    raise ValueError('official_prediction_density_limit')
                if index % 25 == 0:
                    event('official_export', context=context, target_index=index, cells=pointer)
            handle.flush()
    if pointer != rows:
        raise ValueError('incomplete_official_export')
    temporary.replace(final)
    audit = legacy.audit_prediction(final, genes, targets, contexts, size, 4750000000, 1000000)
    write_json(directory / 'prediction-audit.json', audit)
    result = {'sha256': digest(final), 'bytes': final.stat().st_size, 'audit': audit}
    write_json(marker, result)
    return result


def _run(run_id, iteration, submit=False):
    output = EXPERIMENT / 'outputs' / run_id
    frozen = yaml.safe_load((output / 'config.yaml').read_text())
    result = json.loads((output / 'metrics.json').read_text())
    if result['status'] != 'completed':
        raise ValueError('wait_for_original_run_to_finish_before_export')
    config = frozen['configuration']
    metric = result['validation']['H1'][str(iteration)]
    checkpoint = output / 'checkpoints' / fit_id(metric['training_contexts']) / f'round-{iteration:04d}.ubj'
    state = json.loads((checkpoint.parent / 'state.json').read_text())
    if state['iteration'] != iteration or state['checkpoint_sha256'] != digest(checkpoint):
        raise ValueError('checkpoint_does_not_match_recorded_final_training_state')
    # Added packaging dependencies must not change the numerical inference stack.
    for name, version in frozen['identity']['runtime'].items():
        if importlib.metadata.version(name) != version:
            raise ValueError(f'training_runtime_changed:{name}')
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    subprocess.run(['git', 'diff', '--exit-code', 'HEAD', '--', 'experiments/exp007/src',
                    'experiments/exp007/pyproject.toml', 'experiments/exp007/uv.lock'], cwd=ROOT, check=True)
    for name in ['features.py', 'generation.py', 'training.py', 'common.py']:
        original = subprocess.check_output(['git', 'show', f'{frozen["identity"]["git_commit"]}:experiments/exp007/src/{name}'], cwd=ROOT)
        if original != (EXPERIMENT / 'src' / name).read_bytes():
            raise ValueError(f'trained_inference_code_changed:{name}')
    directory = output / 'predictions' / f'leaderboard-round-{iteration:04d}-seed-{config["prediction_seed"]}'
    directory.mkdir(parents=True, exist_ok=True)
    legacy = helpers()
    official, manifest, genes, targets, inputs = legacy.official_inputs({
        'official': 'data/raw/arc_vcc2026_controls', 'cells_per_perturbation': config['cells_per_prediction']})
    identity = {'run_id': run_id, 'round': iteration, 'checkpoint': str(checkpoint.relative_to(output)),
                'checkpoint_sha256': digest(checkpoint), 'training_commit': frozen['identity']['git_commit'],
                'training_lock_sha256': frozen['identity']['uv_lock_sha256'], 'export_commit': commit,
                'export_lock_sha256': digest(EXPERIMENT / 'uv.lock'), 'official_inputs': inputs,
                'manifest': manifest, 'configuration': config, 'training_contexts': metric['training_contexts'],
                'selection': 'User selected round 512 following H1 development validation; no refit',
                'vcc_cli_version': importlib.metadata.version('vcc-cli')}
    identity_path = directory / 'export-identity.json'
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError('export_identity_changed')
    write_json(identity_path, identity)
    tracked = wandb.init(entity='yjcyxky', project='virtual-cell-challenge', group='exp007',
                         id=run_id, name=f'exp007-{run_id}', dir=str(output / 'wandb'), resume='must')
    tracked.config.update({'official_export': identity}, allow_val_change=True)
    try:
        event('official_export_start', round=iteration, checkpoint_sha256=identity['checkpoint_sha256'])
        training = Data(output / 'cache/data')
        data = load_official_data(training, official, config, output, inputs)
        features = Features(data, output / 'cache/priors')
        booster = xgb.Booster()
        booster.load_model(checkpoint)
        booster.set_param({'nthread': config['threads'], 'device': 'cpu'})
        if booster.num_boosted_rounds() != iteration:
            raise ValueError('checkpoint_round_mismatch')
        generate(data, features, booster, genes, targets, config, directory, legacy)
        packaged = legacy.package(config, directory, official)
        tracked.summary['official/export_round'] = iteration
        tracked.summary['official/submission_sha256'] = packaged['sha256']
        if submit:
            legacy.submit_and_score({**config, 'model_name': f'Exp007-Log2FC-r{iteration}-{run_id}',
                'submission_description': 'XGBoost predicts log2 fold changes of arithmetic per-cell CPM means. Trained on K562, RPE1, HepG2 and Jurkat; checkpoint selected using H1 development validation, no refit. A/B/C NTC-conditioned STRING, Reactome and CollecTRI features; NTC template count generation.'}, directory, tracked)
        artifact = wandb.Artifact(f'exp007-{run_id}-official-r{iteration}', type='predictions', metadata=identity)
        for name in ['predictions.vcc', 'export-identity.json', 'generation.json', 'prediction-audit.json',
                     'prep.json', 'submission-file.json', 'submission-entry.json', 'official-summary.json']:
            path = directory / name
            if path.exists():
                artifact.add_file(str(path), name=name)
        published = tracked.log_artifact(artifact)
        published.wait()
        write_json(directory / 'complete.json', {'status': 'published' if submit else 'packaged',
                   'artifact': published.qualified_name, 'checkpoint_sha256': identity['checkpoint_sha256'],
                   'submission_sha256': packaged['sha256']})
        tracked.summary['official/artifact'] = published.qualified_name
        report_path = EXPERIMENT / 'REPORT.md'
        report = report_path.read_text().split('\n## 官方 A/B/C 提交')[0]
        report += '\n## 官方 A/B/C 提交\n\n'
        report += f'用户在 H1 开发验证后指定第 {iteration} 轮直接导出；沿用四背景 checkpoint，不重训。\n\n'
        report += f'导出 commit `{commit}`；checkpoint SHA-256 `{identity["checkpoint_sha256"]}`。\n\n'
        report += f'文件：`{(directory / "predictions.vcc").relative_to(EXPERIMENT)}`；SHA-256 `{packaged["sha256"]}`。\n\n'
        report += f'W&B Artifact：`{published.qualified_name}`。\n\n'
        if submit:
            summary = json.loads((directory / 'official-summary.json').read_text())
            report += f'官方 entry `{summary["entry_id"]}`；状态 published；Overall **{summary["score_avg"]:.9f}**。\n\n'
            report += '| PDS | MSE | NMAE | Fidelity | Reach | Jaccard |\n|---:|---:|---:|---:|---:|---:|\n'
            report += '| ' + ' | '.join(f'{summary[k]:.9f}' for k in ['score_pds','score_mse','score_nmae','score_fid','score_reach','score_jac']) + ' |\n\n'
            report += f'面板 `{summary["panel_id"]}`；anchors `{summary["anchor_version"]}`。该反馈为开发用途，不与本地 H1 分数直接换算。\n'
        report_path.write_text(report)
        tracked.finish(exit_code=0)
    except BaseException:
        tracked.finish(exit_code=1)
        raise


def run(run_id, iteration, submit=False):
    if Path(run_id).name != run_id:
        raise ValueError('invalid_run_id')
    with (EXPERIMENT / 'outputs' / run_id / 'train.log').open('a', buffering=1) as log:
        class Tee:
            def __init__(self, stream): self.stream = stream
            def write(self, value): self.stream.write(value); log.write(value)
            def flush(self): self.stream.flush(); log.flush()
        with contextlib.redirect_stdout(Tee(sys.stdout)), contextlib.redirect_stderr(Tee(sys.stderr)):
            _run(run_id, iteration, submit)
