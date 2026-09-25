"""Export an immutable trained checkpoint to a validated VCC package, without submitting."""
import argparse
import contextlib
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from scipy import sparse
import torch
import yaml

from data import ROOT, EXPERIMENT, hash_file, logcp, write_json, release_read_cache
from evaluation import task_seed
from model import ModuleCVAE
from state import context_features

# Reuse the existing format writer and independent full-file audit. Model-specific
# generation stays here; the earlier template/scaling generator is never called.
sys.path.append(str(ROOT / 'experiments/exp002-response-transfer-validation/src'))
from vcc_submission import official_inputs, append_sparse, audit_prediction


def restore_model(checkpoint, expected_cycle, device):
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    assert saved['cycle'] == expected_cycle, 'checkpoint_cycle_mismatch'
    weights, config = saved['model'], saved['config']
    genes, modules = weights['anchor'].shape
    model = ModuleCVAE(genes, np.ones((genes, modules), np.float32), np.ones(modules),
                       weights['projection'].numpy(), weights['origin'].numpy(),
                       weights['common'].numpy(), weights['seen'][:-1].numpy(), config)
    model.load_state_dict(weights, strict=True)
    assert all(torch.equal(value, model.state_dict()[key]) for key, value in weights.items())
    return model.to(device).eval(), saved


def deployment_state(model, counts, config):
    """All provided NTCs are features; no inferred guide-as-batch or refitted PCA."""
    common = model.common.cpu().numpy().astype(bool)
    projection = model.projection.cpu().numpy()
    origin = model.origin.cpu().numpy()
    # Match the original float32 transformation, in blocks to bound RAM usage.
    coordinates = np.concatenate([logcp(counts[start:start + 512], common) @ projection - origin
                                  for start in range(0, len(counts), 512)])
    states, diagnostics = context_features([counts], [coordinates], np.ones(model.genes, bool), config)
    return states[0], diagnostics[0]


def model_inputs(state, target, count, device):
    inputs = {key: torch.as_tensor(value, dtype=torch.float32, device=device)[None].expand(count, *value.shape)
              for key, value in state.items()}
    inputs['target'] = torch.full((count,), target, dtype=torch.long, device=device)
    return inputs


@torch.no_grad()
def generate(model, config, official, manifest, genes, targets, output, seed, chunk):
    path = output / 'predictions.h5ad'
    marker = output / 'generation.json'
    if marker.exists():
        record = json.loads(marker.read_text())
        assert record['sha256'] == hash_file(path), 'changed_generated_predictions'
        return record
    device = next(model.parameters()).device
    n = len(manifest['contexts']) * len(targets) * manifest['cells_per_pert']
    per_target = manifest['cells_per_pert']
    obs = pd.DataFrame({'context': np.repeat(manifest['contexts'], len(targets) * per_target),
                        'target_gene': np.tile(np.repeat(targets, per_target), len(manifest['contexts']))},
                       index=[f'mcvae-{seed}-{i:06d}' for i in range(n)])
    for key in obs:
        obs[key] = pd.Categorical(obs[key])
    temporary = output / 'predictions.partial.h5ad'
    ad.AnnData(X=sparse.csr_matrix((n, len(genes)), dtype=np.int32), obs=obs,
               var=pd.DataFrame(index=pd.Index(genes, name='gene_name'))).write_h5ad(temporary)
    diagnostics, controls_used = [], []
    gene_index = {g: i for i, g in enumerate(genes)}
    pointer = 0
    with h5py.File(temporary, 'r+') as hf:
        del hf['X']
        x = hf.create_group('X')
        x.attrs.update({'encoding-type': 'csr_matrix', 'encoding-version': '0.1.0', 'shape': [n, len(genes)]})
        for key in ['data', 'indices']:
            x.create_dataset(key, (0,), maxshape=(None,), chunks=(1048576,), dtype='int32',
                             compression='gzip', compression_opts=1)
        x.create_dataset('indptr', (n + 1,), dtype='int64')
        for context in manifest['contexts']:
            controls = ad.read_h5ad(official / f'context_{context}.h5ad')
            assert controls.var_names.tolist() == genes, 'control_gene_order_mismatch'
            assert set(controls.obs.context.astype(str)) == {context}, 'control_context_mismatch'
            assert set(controls.obs.target_gene.astype(str)) == {'non-targeting'}, 'perturbation_labels_in_controls'
            assert len(controls) == manifest['per_context'][context]['control_cells']
            assert set(controls.obs.columns) == {'context', 'target_gene', 'ntc_id'}, 'review_new_control_covariates'
            counts = controls.X.toarray().astype(np.float32)
            assert np.isfinite(counts).all() and (counts >= 0).all() and (counts.sum(1) > 0).all()
            state, diagnostic = deployment_state(model, counts, config)
            del controls, counts
            np.savez_compressed(output / f'context-{context}-features.npz', **state)
            controls_used.append({'context': context, 'cells': manifest['per_context'][context]['control_cells'],
                                  'technical_batch': 'unavailable; pooled NTC', 'ntc_id_used_as_batch': False,
                                  **diagnostic})
            print(json.dumps({'stage': 'export_context_ready', 'context': context}), flush=True)
            for target in targets:
                target_seed = task_seed(context, f'checkpoint-export-{seed}|{target}')
                torch.manual_seed(target_seed)
                total_nnz, low, high, total = 0, 1000000, 0, 0
                for start in range(0, per_target, chunk):
                    size = min(chunk, per_target - start)
                    batch = model_inputs(state, gene_index[target], size, device)
                    emitted = model.generate(batch).cpu().numpy()
                    assert np.isfinite(emitted).all() and (emitted >= 0).all()
                    assert np.array_equal(emitted, np.floor(emitted))
                    depth = emitted.sum(1, dtype=np.float64)
                    assert (depth > 0).all() and depth.max() <= 1000000, 'prediction_exceeds_official_count_limit'
                    total_nnz += append_sparse(x, emitted.astype(np.int32), pointer)
                    pointer += size
                    assert len(x['data']) <= 4750000000, 'prediction_density_exceeds_official_cap'
                    low, high = min(low, int(depth.min())), max(high, int(depth.max()))
                    total += float(depth.sum())
                diagnostics.append({'context': context, 'target_gene': target, 'cells': per_target,
                                    'seed': target_seed, 'nnz': total_nnz, 'minimum_cell_counts': low,
                                    'maximum_cell_counts': high, 'mean_cell_counts': total / per_target,
                                    'target_seen_in_training': bool(model.seen[gene_index[target]].item())})
                if len(diagnostics) % 25 == 0:
                    print(json.dumps({'stage': 'export_generate', 'groups_done': len(diagnostics),
                                      'cells_written': pointer, 'nnz': len(x['data'])}), flush=True)
            hf.flush()
        assert pointer == n
    temporary.replace(path)
    pd.DataFrame(diagnostics).to_parquet(output / 'emission-diagnostics.parquet', index=False)
    write_json(output / 'control-features.json', controls_used)
    record = {'sha256': hash_file(path), 'bytes': path.stat().st_size, 'cells': n, 'genes': len(genes),
              'groups': len(diagnostics), 'seed': seed, 'chunk_cells': chunk,
              'nnz': sum(v['nnz'] for v in diagnostics),
              'method': 'unchanged checkpoint conditional-prior and negative-binomial generation',
              'untrained_readout_policy': 'unchanged model output; no silent NTC fallback',
              'clipping_or_resampling_invalid_cells': False}
    write_json(marker, record)
    return record


def package(output, official, cli_project):
    """Use the existing locked CLI environment as a separate format-checking tool."""
    marker, final = output / 'prep.json', output / 'predictions.vcc'
    if not marker.exists():
        command = ['/home/jy001/micromamba/envs/virtual-cell/bin/python', str(EXPERIMENT / 'src/runtime.py'),
                   '--base-run', 'uv', 'run', '--locked', '--no-sync', '--project', str(cli_project),
                   '--python', '/home/jy001/micromamba/envs/virtual-cell/bin/python', '--no-python-downloads',
                   'vcc', 'prep', str(output / 'predictions.h5ad'), '-g', str(official / 'gene_names.csv'),
                   '--perts', str(official / 'pert_counts.csv'), '-o', str(final), '--json']
        result = subprocess.run(command, text=True, capture_output=True)
        if result.returncode:
            raise RuntimeError('official_prep_failed: ' + result.stdout + result.stderr)
        write_json(marker, json.loads(result.stdout))
    identity = {'file': str(final), 'sha256': hash_file(final), 'bytes': final.stat().st_size,
                'cli_project': str(cli_project), 'cli_lock_sha256': hash_file(cli_project / 'uv.lock'),
                'official_prep_passed': True, 'submitted': False}
    prior = output / 'submission-file.json'
    if prior.exists():
        assert json.loads(prior.read_text()) == identity, 'changed_submission_package'
    write_json(prior, identity)
    return identity


def main(args):
    torch.set_num_threads(args.threads)
    torch.set_float32_matmul_precision('high')
    torch.use_deterministic_algorithms(True)
    run = Path(args.run_dir).resolve()
    checkpoint = (run / 'checkpoints' / args.checkpoint).resolve()
    assert checkpoint.is_relative_to(run / 'checkpoints') and checkpoint.is_file()
    config = yaml.safe_load((run / 'config.yaml').read_text())
    repo = run.parents[3]
    output = run / 'predictions' / f'leaderboard-holdout-H1-cycle-{args.cycle:04d}-seed-{args.seed}'
    output.mkdir(exist_ok=True, parents=True)
    with (output / '.export.lock').open('w') as lock, (output / 'export.log').open('a', buffering=1) as log:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with contextlib.redirect_stdout(log):
            started = time.monotonic()
            subprocess.run(['git', 'diff', '--quiet', 'HEAD', '--', str(EXPERIMENT)], cwd=ROOT, check=True)
            commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
            # The original decoder/generator must remain byte-identical to the trained version.
            original = subprocess.check_output(['git', 'show', config['pipeline_commit'] + ':experiments/exp003-context-module-cvae/src/model.py'], cwd=ROOT)
            assert original == (EXPERIMENT / 'src/model.py').read_bytes(), 'changed_checkpoint_model_implementation'
            checkpoint_hash = hash_file(checkpoint)
            model, saved = restore_model(checkpoint, args.cycle, args.device)
            assert saved['config'] == config, 'checkpoint_run_identity_mismatch'
            assert saved['training_contexts'] == ['K562', 'RPE1', 'HepG2', 'Jurkat'], 'export_fold_mismatch'
            official, manifest, genes, targets, sources = official_inputs({
                'official': str(repo / 'data/raw/arc_vcc2026_controls'), 'cells_per_perturbation': 400})
            assert genes == pd.read_csv(repo / config['gene_axis']).iloc[:, 0].astype(str).tolist()
            identity = {'run_id': config['run_id'], 'checkpoint': str(checkpoint), 'checkpoint_sha256': checkpoint_hash,
                        'cycle': saved['cycle'], 'step': saved['step'], 'local_score': saved['validation_score'],
                        'training_contexts': saved['training_contexts'], 'training_commit': config['pipeline_commit'],
                        'export_commit': commit, 'uv_lock_sha256': hash_file(EXPERIMENT / 'uv.lock'),
                        'seed': args.seed, 'device': args.device, 'chunk_cells': args.chunk_cells,
                        'float32_matmul_precision': 'high', 'deterministic_algorithms': True,
                        'official_sources': sources, 'partition': manifest['partition'], 'panel_id': manifest['panel_id'],
                        'readouts_without_training_supervision': int((~model.trained_readouts).sum().item()),
                        'ntc_policy': 'all supplied controls pooled per context; same trained PCA; no guide-as-batch',
                        'prediction_only': True, 'optimizer_steps': 0, 'submitted': False}
            marker = output / 'export-identity.json'
            if marker.exists():
                assert json.loads(marker.read_text()) == identity, 'changed_export_conditions_require_distinct_export'
            else:
                write_json(marker, identity)
            pd.DataFrame({'gene': genes, 'trained_readout': model.trained_readouts.cpu().numpy(),
                          'primary_delta_axis': model.common.cpu().numpy().astype(bool),
                          'seen_training_target': model.seen[:-1].cpu().numpy().astype(bool)}).to_csv(output / 'gene-support.csv', index=False)
            generated = generate(model, config, official, manifest, genes, targets, output, args.seed, args.chunk_cells)
            audit = audit_prediction(output / 'predictions.h5ad', genes, targets, manifest['contexts'], 400, 4750000000, 1000000)
            write_json(output / 'prediction-audit.json', audit)
            packaged = package(output, official, repo / 'experiments/exp002-response-transfer-validation')
            assert hash_file(checkpoint) == checkpoint_hash, 'source_checkpoint_changed'
            write_json(output / 'complete.json', {'status': 'validated_package_ready_not_submitted',
                       'checkpoint_unchanged': True, 'generation': generated, 'audit': audit,
                       'package': packaged, 'elapsed_seconds': time.monotonic() - started})
            release_read_cache(output / 'predictions.h5ad')
            print(json.dumps({'stage': 'export_complete', **packaged}), flush=True)
    print(json.dumps({'output': str(output), 'status': 'validated_package_ready_not_submitted'}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--cycle', required=True, type=int)
    parser.add_argument('--seed', type=int, default=101)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--chunk-cells', type=int, default=128)
    main(parser.parse_args())
