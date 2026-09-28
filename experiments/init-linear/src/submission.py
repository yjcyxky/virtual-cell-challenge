"""Evaluate the completed fit on official A/B/C, without changing its identity."""
import argparse
import fcntl
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys
import time

EXPERIMENT = Path(__file__).resolve().parents[1]
ROOT = EXPERIMENT.parents[1]
CONFIG = EXPERIMENT / 'configs/official-s01.json'
sys.path.insert(0, str(ROOT / 'scripts'))
from research import digest, git, ref_error


def gate(root=ROOT, config_path=CONFIG):
    """Check the registered, committed export before creating files or W&B runs."""
    config = json.loads(config_path.read_text())
    dag = json.loads((root / 'docs/research/experiment_dag.json').read_text())
    node = dag['nodes'][config['run_id']]
    stage = node['official_evaluation']
    if config_path.resolve() != (root / stage['config_ref']['path']).resolve():
        raise ValueError('export configuration must be the registered file')
    if node['status'] != 'completed' or stage['comparison_id'] != config['comparison_id']:
        raise ValueError('export requires completed registered training')
    refs = [stage['config_ref'], *stage['code_refs'], node['config_ref'], *node['code_refs'],
            node['metrics_ref'], config['checkpoint_ref']]
    for item in refs:
        if problem := ref_error(root, item):
            raise ValueError(f'{item["path"]}: {problem}')
    committed = [stage['config_ref'], *stage['code_refs'], node['config_ref'], *node['code_refs']]
    for item in committed:
        if hashlib.sha256(git(root, 'show', 'HEAD:' + item['path']).stdout).hexdigest() != item['sha256']:
            raise ValueError('export code/config must be committed: ' + item['path'])
    committed_dag = json.loads(git(root, 'show', 'HEAD:docs/research/experiment_dag.json').stdout)
    frozen_stage = committed_dag['nodes'][config['run_id']]['official_evaluation']
    if any(stage[k] != frozen_stage[k] for k in ('config_ref', 'code_refs', 'comparison_id')):
        raise ValueError('export registration must be committed')
    metrics = json.loads((root / node['metrics_ref']['path']).read_text())
    if metrics['checkpoint_ref'] != config['checkpoint_ref'] or not metrics['evaluation_completed']:
        raise ValueError('checkpoint must be the completed local evaluation checkpoint')
    training = node['expected_config']
    if config['arms'] != ['linear', 'shared'] or config['seed'] != training['seed']:
        raise ValueError('export arms/seed differ from registered comparison')
    if config['generation'] != training['generation'] or config['data'] != training['data']:
        raise ValueError('export must preserve trained normalization/generator configuration')
    return config, metrics, git(root, 'rev-parse', 'HEAD').stdout.decode().strip()


def helpers():
    """Only source verification, CSR writing and file auditing are reused."""
    source = ROOT / 'experiments/exp002-response-transfer-validation/src'
    sys.path.append(str(source))
    spec = importlib.util.spec_from_file_location('vcc_file_helpers', source / 'vcc_submission.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def official_statistics(controls, genes, context, expected_cells, data_config):
    import numpy as np
    from data import normalized, NTC
    if controls.var_names.tolist() != list(genes):
        raise ValueError('official gene identity/order mismatch')
    if (controls.n_obs != expected_cells or set(controls.obs.context.astype(str)) != {context}
            or set(controls.obs.target_gene.astype(str)) != {NTC}):
        raise ValueError('official context/control identity mismatch')
    total = np.zeros(len(genes), dtype=np.float64)
    for start in range(0, controls.n_obs, data_config['chunk_rows']):
        total += np.asarray(normalized(controls.X[start:start + data_config['chunk_rows']],
                                       data_config['target_sum']).sum(axis=0)).ravel()
    return {'positions': np.arange(len(genes)), 'mean': (total / controls.n_obs)[None, :]}


def generate(config, model, directory, official, manifest, genes, targets, arm, helper):
    import anndata as ad
    import h5py
    import numpy as np
    import pandas as pd
    from scipy import sparse
    from data import write_json, ref
    from model import predict
    from evaluation import generate_counts
    done = directory / 'generation.json'
    final = directory / 'predictions.h5ad'
    if done.exists():
        result = json.loads(done.read_text())
        if ref(final) != result['file_ref']:
            raise ValueError('completed prediction changed')
        return result
    if final.exists():
        raise ValueError('unsealed prediction exists; audit it before recovery')
    contexts = manifest['contexts']
    cells = config['generation']['cells_per_target']
    rows = len(contexts) * len(targets) * cells
    obs = pd.DataFrame({'context': np.repeat(contexts, len(targets)*cells),
                        'target_gene': np.tile(np.repeat(targets, cells), len(contexts))},
                       index=[f'{config["run_id"]}-{arm}-{i}' for i in range(rows)])
    for column in obs:
        obs[column] = pd.Categorical(obs[column])
    partial = directory / 'predictions.partial.h5ad'
    # Only an unfinished, deterministic export of this sealed identity is restarted.
    ad.AnnData(sparse.csr_matrix((rows, len(genes)), dtype=np.int32), obs=obs,
               var=pd.DataFrame(index=pd.Index(genes, name='gene_name'))).write_h5ad(partial)
    pointer, diagnostics = 0, []
    with h5py.File(partial, 'r+') as handle:
        del handle['X']
        matrix = handle.create_group('X')
        matrix.attrs.update({'encoding-type': 'csr_matrix', 'encoding-version': '0.1.0',
                             'shape': [rows, len(genes)]})
        for name in ['data', 'indices']:
            matrix.create_dataset(name, (0,), maxshape=(None,), chunks=(1048576,),
                                  dtype='int32', compression='gzip', compression_opts=1)
        matrix.create_dataset('indptr', (rows + 1,), dtype='int64')
        for context in contexts:
            controls = ad.read_h5ad(official / f'context_{context}.h5ad')
            stats = official_statistics(controls, genes, context,
                                       manifest['per_context'][context]['control_cells'], config['data'])
            delta = predict(model, stats, targets, arm)
            diagnostics.append({'context': context, 'input_ntc_cells': controls.n_obs,
                                'mean_delta': float(delta.mean()), 'rms_delta': float(np.sqrt((delta**2).mean()))})
            for i, target in enumerate(targets):
                block = generate_counts(controls.X, delta[i], config['seed'] + i,
                                        cells, config['data']['target_sum'])
                helper.append_sparse(matrix, block, pointer)
                pointer += cells
                if len(matrix['data']) > config['max_nnz']:
                    raise ValueError('official nonzero cap exceeded')
                if (i + 1) % 25 == 0:
                    print(json.dumps({'stage': 'generate', 'arm': arm, 'context': context,
                                      'targets': i+1, 'cells': pointer, 'nnz': len(matrix['data'])}), flush=True)
            del controls
        if pointer != rows:
            raise ValueError('incomplete generation')
    audit = helper.audit_prediction(partial, genes, targets, contexts, cells,
                                    config['max_nnz'], config['max_counts_per_cell'])
    partial.replace(final)
    write_json(directory / 'prediction-audit.json', audit)
    trained = set(model['targets'].astype(str))
    result = {'file_ref': ref(final), 'arm': arm, 'cells': rows, 'genes': len(genes),
              'targets': targets, 'seen_targets': [t for t in targets if t in trained],
              'unseen_targets': [t for t in targets if t not in trained],
              'untrained_output_genes': int((~model['trained_mask']).sum()),
              'unseen_policy': 'linear: fitted conditional term only; shared: zero delta',
              'rng': 'seed + official target index, paired across arms; repeated per context',
              'diagnostics': diagnostics, 'audit': audit}
    write_json(done, result)
    return result


def cli_json(config, arguments):
    result = subprocess.run([config['vcc_cli'], *map(str, arguments), '--json'],
                            text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError(f'official CLI failed ({result.returncode}); command={arguments[0]}')
    return json.loads(result.stdout)


def package(config, directory, official):
    from data import write_json, ref
    final = directory / 'predictions.vcc'
    seal = directory / 'submission-file.json'
    if seal.exists():
        record = json.loads(seal.read_text())
        if record['file_ref'] != ref(final):
            raise ValueError('sealed VCC changed')
        return record
    report = cli_json(config, ['prep', directory / 'predictions.h5ad', '-g', official / 'gene_names.csv',
                               '--perts', official / 'pert_counts.csv', '-o', final])
    write_json(directory / 'prep.json', report)
    record = {'file_ref': ref(final), 'bytes': final.stat().st_size, 'cli_version': config['vcc_cli_version']}
    write_json(seal, record)
    return record


def pending_entry(config, prediction):
    # Ask the CLI's own environment for entry IDs only, never credential/cache contents.
    code = ('import json,sys; from pathlib import Path; from vcc import auth; '
            'print(json.dumps([v["entry_id"] for v in auth.list_pending_uploads("default").values() '
            'if Path(v.get("local_path", "")).resolve()==Path(sys.argv[1]).resolve()]))')
    entries = json.loads(subprocess.check_output([config['cli_python'], '-c', code, str(prediction)], text=True))
    if len(entries) > 1:
        raise ValueError('ambiguous pending uploads for this VCC')
    return entries[0] if entries else None


def submit(config, directory, arm, tracked):
    from data import write_json, ref
    final = directory / 'predictions.vcc'
    identity = json.loads((directory / 'submission-file.json').read_text())
    if ref(final) != identity['file_ref']:
        raise ValueError('refusing submission of changed VCC')
    receipt = directory / 'submission-entry.json'
    name = config['model_names'][arm]
    if receipt.exists():
        entry = json.loads(receipt.read_text())
        if entry['submission_sha256'] != identity['file_ref']['sha256'] or entry['model_name'] != name:
            raise ValueError('submission receipt does not match current file/model')
    else:
        entry = {'entry_id': pending_entry(config, final), 'model_name': name,
                 'submission_sha256': identity['file_ref']['sha256']}
    if entry['entry_id']:
        write_json(receipt, entry)
    status = cli_json(config, ['status', entry['entry_id']]) if entry['entry_id'] else None
    if status is None or status.get('status') == 'uploading':
        command = [config['vcc_cli'], 'submit', str(final), '-m', name, '-d', config['description']]
        if entry['entry_id']:
            command.extend(['--resume', entry['entry_id']])
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        with (directory / 'submission.log').open('a') as log:
            for line in process.stdout:
                line = re.sub(r'https://\S*\?\S+', '[redacted-capability-url]', line)
                log.write(line); log.flush()
                print(line.rstrip(), flush=True)
                found = re.search(r'^\s*entry:\s+(\S+)', line)
                if found:
                    if entry['entry_id'] and entry['entry_id'] != found.group(1):
                        process.terminate()
                        raise ValueError('CLI returned a different entry during resume')
                    entry['entry_id'] = found.group(1)
                    write_json(receipt, entry)
        if process.wait():
            raise RuntimeError('submission interrupted; resume this same file/entry')
    if not entry['entry_id']:
        raise RuntimeError('missing official entry receipt')
    while True:
        status = cli_json(config, ['status', entry['entry_id']])
        write_json(directory / 'official-status.json', status)
        tracked.summary[f'official/{arm}/status'] = status.get('status')
        tracked.summary[f'official/{arm}/entry_id'] = entry['entry_id']
        print(json.dumps({'stage': 'official_scoring', 'arm': arm, 'entry_id': entry['entry_id'],
                          'status': status.get('status')}), flush=True)
        if status.get('status') == 'published':
            import numpy as np
            keys = ['score_pds', 'score_mse', 'score_nmae', 'score_fid', 'score_reach', 'score_jac']
            scores = np.asarray([status[k] for k in keys], dtype=float)
            if (not np.isfinite(scores).all() or not np.isclose(scores.mean(), status['score_avg'], atol=1e-6)
                    or status['partition'] != 'val' or status['panel_id'] != config['panel_id']
                    or not status['anchor_version']):
                raise ValueError('published result violates registered panel/six-metric contract')
            result = {k: status.get(k) for k in keys + ['score_avg', 'rank', 'partition', 'panel_id', 'anchor_version']}
            result.update(entry, status='published')
            write_json(directory / 'official-summary.json', result)
            for key, value in result.items():
                if value is not None:
                    tracked.summary[f'official/{arm}/{key}'] = value
            return result
        if status.get('is_terminal'):
            raise RuntimeError('official evaluation terminated without published score')
        time.sleep(30)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gate', action='store_true')
    parser.add_argument('--submit', action='store_true', help='submit and wait for both registered arms')
    args = parser.parse_args()
    config, metrics, commit = gate()
    if args.gate:
        print('Completed-model official export gate passed', flush=True)
        return
    import numpy as np
    import yaml
    import wandb
    from data import write_json, ref
    output = EXPERIMENT / 'outputs' / config['run_id']
    training = yaml.safe_load((output / 'config.yaml').read_text())
    runtime = {p: importlib.metadata.version(p) for p in training['runtime_versions']}
    if runtime != training['runtime_versions'] or ref(EXPERIMENT / 'uv.lock') != training['lock_ref']:
        raise ValueError('export numerical environment differs from completed training')
    cli_version = subprocess.check_output([config['vcc_cli'], '--version'], text=True)
    if cli_version.splitlines()[0] != 'vcc ' + config['vcc_cli_version']:
        raise ValueError('official CLI version changed')
    helper = helpers()
    official, manifest, genes, targets, identities = helper.official_inputs(
        {'official': config['official'], 'cells_per_perturbation': config['generation']['cells_per_target']})
    if manifest['panel_id'] != config['panel_id']:
        raise ValueError('official panel changed')
    model = dict(np.load(ROOT / config['checkpoint_ref']['path']))
    if (model['genes'].tolist() != genes or str(model['config_sha256']) != metrics['research']['effective_config_sha256']
            or str(model['training_state']) != 'analytic_fit_complete'):
        raise ValueError('checkpoint identity does not match export')
    directory = output / 'predictions/official-abc'
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / '.export.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        seal = {'config_ref': ref(CONFIG), 'checkpoint_ref': config['checkpoint_ref'], 'research': metrics['research'],
                'runtime_versions': runtime, 'lock_ref': training['lock_ref'], 'official_inputs': identities,
                'code_refs': json.loads((ROOT / 'docs/research/experiment_dag.json').read_text())['nodes'][config['run_id']]['official_evaluation']['code_refs']}
        identity = directory / 'export-identity.json'
        if identity.exists() and json.loads(identity.read_text())['binding'] != seal:
            raise ValueError('official export identity changed on resume')
        if not identity.exists():
            write_json(identity, {'binding': seal, 'export_git_commit': commit, 'config': config})
        tracked = wandb.init(entity='yjcyxky', project='virtual-cell-challenge', group='init-linear',
                             id=config['run_id'], name=config['run_id'], dir=str(output), resume='must',
                             config={'official_export': json.loads(identity.read_text())})
        try:
            for arm in config['arms']:
                destination = directory / arm
                destination.mkdir(exist_ok=True)
                generate(config, model, destination, official, manifest, genes, targets, arm, helper)
                package(config, destination, official)
                if args.submit:
                    submit(config, destination, arm, tracked)
                    marker = destination / 'artifact.json'
                    if not marker.exists():
                        artifact = wandb.Artifact(config['run_id'] + '-official-' + arm, type='evaluation', metadata=seal)
                        artifact.add_file(str(directory / 'export-identity.json'), name='export-identity.json')
                        for name in ['predictions.vcc', 'official-summary.json', 'official-status.json',
                                     'submission-entry.json', 'submission-file.json', 'generation.json',
                                     'prediction-audit.json', 'prep.json']:
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
