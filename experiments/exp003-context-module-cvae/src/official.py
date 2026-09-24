"""Fixed whole-dataset vcc2026 references and three-seed checkpoint evaluation.

All metric definitions, replicate estimators, normalization and clamps are delegated
to the pinned official package. Local reference panels retain their own measured
gene axes and cached cell populations; scores are not leaderboard-equivalent.
"""
from dataclasses import replace
import gc
import json
from pathlib import Path
import shutil

import anndata as ad
import numpy as np
import pandas as pd
import polars as pl
import torch

from cell_eval2 import EvalConfig, compute_metrics, aggregate_metrics_wide, score_metrics
from cell_eval2.baseline import build_run_meta, generic_response_profile
from cell_eval2.run import metric_output_names
from cell_eval2.competition import competition_members
from cell_eval2.real_bundle import build_real_bundle, read_real_bundle

from data import hash_file, write_json, release_read_cache
from evaluation import task_seed


def official_config(config, cache):
    preset = EvalConfig.from_preset('vcc2026')
    return replace(preset, device=config['official_device'], num_threads=8,
                   de=replace(preset.de, backend=config['official_de_backend']),
                   cache_real=str(cache / 'real-metric-cache'), pert_chunk=128)


def scored_result(prediction, reference, cfg, bundle, directory):
    directory.mkdir(parents=True, exist_ok=True)
    raw = compute_metrics(prediction, reference, config=cfg)
    raw.write_parquet(directory / 'raw-metrics.parquet')
    aggregate = aggregate_metrics_wide(raw, metrics=metric_output_names(cfg))
    aggregate.write_csv(directory / 'aggregate.csv')
    meta = build_run_meta(cfg, reference, prediction)
    write_json(directory / 'run-meta.json', meta)
    scores = score_metrics(aggregate, real_bundle=str(bundle), user_meta=meta)
    scores.write_csv(directory / 'scores.csv')
    members = list(competition_members())
    selected = scores.filter(pl.col('metric').is_in(members))
    assert set(selected['metric'].to_list()) == set(members)
    components = dict(zip(selected['metric'].to_list(), selected['from_replicate'].to_list()))
    assert all(v is not None and np.isfinite(v) for v in components.values()), 'invalid_official_component'
    average = scores.filter(pl.col('metric') == 'avg_score')['from_replicate'].item()
    assert average is not None and np.isfinite(average)
    assert abs(average - np.mean(list(components.values()))) < 1e-10
    return {'score': float(average), 'components': components}


class OfficialValidation:
    def __init__(self, data, view, context, config, cache, predictions):
        self.data, self.view, self.context, self.config = data, view, context, config
        self.directory = cache / 'official'; self.directory.mkdir(parents=True, exist_ok=True)
        self.predictions = predictions; predictions.mkdir(parents=True, exist_ok=True)
        self.cfg = official_config(config, self.directory)
        self.targets = sorted(t for c, t in data.tasks if c == context)
        self.axis = np.flatnonzero(data.masks[context])
        self.genes = np.asarray(data.genes)[self.axis]
        self.bundle = self.directory / 'reference-bundle'
        self.reference = None
        self.observations = None
        self.blocks = None
        self.last_cycle = None

    def reference_layout(self):
        ids, labels, batches, constructs, blocks = [], [], [], [], []
        for target in self.targets:
            for (construct, batch), rows in sorted(self.data.groups[(self.context, target)].items()):
                start = len(ids)
                ids.extend(rows); labels.extend([target] * len(rows))
                batches.extend([batch] * len(rows)); constructs.extend([construct] * len(rows))
                blocks.append((target, batch, start, len(ids)))
        for (c, batch, half), rows in sorted(self.data.controls.items()):
            if c != self.context or half != 1:
                continue
            start = len(ids)
            ids.extend(rows); labels.extend([self.cfg.control] * len(rows))
            batches.extend([batch] * len(rows)); constructs.extend(['NTC'] * len(rows))
            blocks.append(('__NTC__', batch, start, len(ids)))
        observations = pd.DataFrame({'target': labels, 'batch': batches, 'construct': constructs,
                                     'source_cached_row': np.asarray(ids, dtype=np.int64)},
                                    index=[f'reference-{i}' for i in range(len(ids))])
        return np.asarray(ids, dtype=np.int64), observations, blocks

    def prepare(self):
        ids, self.observations, self.blocks = self.reference_layout()
        count_path = self.directory / 'reference-counts.npy'
        marker = self.directory / 'reference.json'
        signature = {'context': self.context, 'targets': self.targets, 'genes': self.genes.tolist(),
                     'source_data_complete_sha256': hash_file(self.data.directory / 'complete.json'),
                     'rows': len(ids), 'cell_population': 'all admitted cached perturbation cells; reference-half NTC',
                     'technical_composition': 'prediction counts per construct/batch equal the fixed reference allocation',
                     'baseline': 'equal mean across all admitted target means; tiled counts; no extra efficacy filter',
                     'replicate': 'official five disjoint-half splits, base seed 0, including full-gate LFC estimator',
                     'interpretation': 'official-method local validation, not leaderboard-equivalent'}
        if marker.exists():
            saved = json.loads(marker.read_text())
            assert saved['identity'] == signature and saved['sha256'] == hash_file(count_path)
        else:
            temporary = count_path.with_suffix('.tmp.npy')
            array = np.lib.format.open_memmap(temporary, mode='w+', dtype=np.uint16, shape=(len(ids), len(self.axis)))
            for start in range(0, len(ids), 512):
                array[start:start + 512] = self.data.read(ids[start:start + 512])[:, self.axis].astype(np.uint16)
            array.flush(); del array
            temporary.replace(count_path)
            self.observations.to_parquet(self.directory / 'reference-observations.parquet')
            write_json(marker, {'identity': signature, 'sha256': hash_file(count_path)})
        self.reference = ad.AnnData(np.load(count_path, mmap_mode='r'), obs=self.observations,
                                    var=pd.DataFrame(index=self.genes))
        self.cfg.to_yaml(str(self.directory / 'eval-config.yaml'))
        if not self.bundle.exists():
            print(json.dumps({'stage': 'official_reference', 'context': self.context, 'shape': self.reference.shape}), flush=True)
            profile = generic_response_profile(self.reference, pert_col=self.cfg.pert_col,
                                               control=self.cfg.control, exclude_target_gene=False)
            path = self.directory / 'baseline-counts.npy'
            baseline = np.lib.format.open_memmap(path, mode='w+', dtype=np.float32, shape=self.reference.shape)
            for start in range(0, len(ids), 512):
                baseline[start:start + 512] = profile.values
            ctrl = np.flatnonzero(self.observations.target.eq(self.cfg.control))
            baseline[ctrl] = self.reference.X[ctrl]
            baseline.flush()
            arm = ad.AnnData(baseline, obs=self.observations.copy(), var=self.reference.var.copy())
            manifest = build_real_bundle(self.reference, arm, config=self.cfg, outdir=str(self.bundle),
                                         bundle_id='local-' + self.context, base_seed=0, n_splits=5)
            assert manifest['rule_digest'] is not None, manifest['rule_mismatches']
            del arm, baseline; gc.collect(); path.unlink()
            release_read_cache(count_path)
        manifest = read_real_bundle(self.bundle).manifest
        assert manifest['rule_digest'] is not None
        print(json.dumps({'stage': 'official_reference_ready', 'context': self.context,
                          'bundle': str(self.bundle), 'targets': len(self.targets)}), flush=True)

    @torch.no_grad()
    def evaluate(self, model, cycle):
        assert self.reference is not None
        was_training = model.training
        model.eval()
        device = next(model.parameters()).device
        result = []
        cycle_dir = self.directory / f'cycle-{cycle:04d}'; cycle_dir.mkdir(exist_ok=True)
        try:
            for seed in self.config['prediction_seeds']:
                path = self.directory / f'prediction-seed-{seed}.npy'
                prediction = np.lib.format.open_memmap(path, mode='w+', dtype=np.uint32, shape=self.reference.shape)
                with torch.random.fork_rng(devices=[device.index or 0] if device.type == 'cuda' else []):
                    for i, (target, batch, lo, hi) in enumerate(self.blocks):
                        torch.manual_seed(task_seed(self.context, f'official-{seed}|{target}|{batch}|{lo}'))
                        # Chunk size is fixed, hence the same random streams recur at every checkpoint.
                        for start in range(lo, hi, self.config['prediction_batch_cells']):
                            stop = min(start + self.config['prediction_batch_cells'], hi)
                            inputs = self.view.inputs([(self.context, target, batch, stop - start)], device)
                            counts = model.generate(inputs).cpu().numpy()[:, self.axis]
                            assert np.isfinite(counts).all() and counts.min() >= 0
                            assert np.array_equal(counts, np.floor(counts)) and counts.max(initial=0) <= np.iinfo(np.uint32).max
                            assert np.all(counts.sum(1, dtype=np.float64) <= self.cfg.max_counts_per_cell), 'prediction_exceeds_official_count_limit'
                            prediction[start:stop] = counts.astype(np.uint32)
                        if (i + 1) % 1000 == 0:
                            print(json.dumps({'stage': 'official_prediction', 'context': self.context,
                                              'cycle': cycle, 'seed': seed, 'blocks_done': i + 1, 'blocks': len(self.blocks)}), flush=True)
                prediction.flush()
                obs = self.observations.drop(columns='source_cached_row').copy()
                obs.index = [f'generated-{seed}-{i}' for i in range(len(obs))]
                predicted = ad.AnnData(prediction, obs=obs, var=self.reference.var.copy())
                torch.cuda.empty_cache()
                print(json.dumps({'stage': 'official_score_start', 'context': self.context,
                                  'cycle': cycle, 'seed': seed}), flush=True)
                scores = scored_result(predicted, self.reference, self.cfg, self.bundle, cycle_dir / f'seed-{seed}')
                print(json.dumps({'stage': 'official_score', 'context': self.context,
                                  'cycle': cycle, 'seed': seed, **scores}), flush=True)
                result.append({'seed': seed, **scores})
                del predicted, prediction; gc.collect()
                release_read_cache(path)
                try:
                    import cupy
                    cupy.get_default_memory_pool().free_all_blocks()
                finally:
                    torch.cuda.empty_cache()
            values = [r['score'] for r in result]
            summary = {'cycle': cycle, 'mean_score': float(np.mean(values)),
                       'score_sd': float(np.std(values, ddof=1)), 'seed_scores': result}
            write_json(cycle_dir / 'summary.json', summary)
            self.last_cycle = cycle
            return summary
        finally:
            model.train(was_training)

    def promote_best(self, cycle):
        assert self.last_cycle == cycle
        for seed in self.config['prediction_seeds']:
            source = self.directory / f'prediction-seed-{seed}.npy'
            destination = self.predictions / f'best-seed-{seed}.npy'
            temporary = destination.with_suffix('.tmp.npy')
            shutil.copyfile(source, temporary); temporary.replace(destination)
        obs = self.observations.drop(columns='source_cached_row').copy()
        obs.index = [f'generated-{i}' for i in range(len(obs))]
        obs.to_parquet(self.predictions / 'official-observations.parquet')
        write_json(self.predictions / 'official-genes.json', self.genes.tolist())
        summary = json.loads((self.directory / f'cycle-{cycle:04d}' / 'summary.json').read_text())
        write_json(self.predictions / 'best-official.json', summary)

    def record_prediction_files(self):
        files = {f'best-seed-{seed}.npy': hash_file(self.predictions / f'best-seed-{seed}.npy')
                 for seed in self.config['prediction_seeds']}
        write_json(self.predictions / 'prediction-files.json', {
            'files': files, 'gene_axis': 'official-genes.json', 'observations': 'official-observations.parquet',
            'raw_reference_cells_included': False, 'retention': 'local full arrays; hashes in versioned result artifact'})

    def clear_transient(self):
        for seed in self.config['prediction_seeds']:
            path = self.directory / f'prediction-seed-{seed}.npy'
            if path.exists():
                path.unlink()
