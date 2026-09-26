"""Paired distribution comparisons under the unchanged official six-metric rule."""
from dataclasses import replace
import gc
import json
from pathlib import Path
import time

import anndata as ad
import numpy as np
import pandas as pd
import torch

from count_transfer import hash_file, write_json, stable_seed, predict_transfer
from count_generation import generate_counts
from data import release_read_cache
from official import official_config, scored_result
from cell_eval2.baseline import generic_response_profile, build_baseline_prediction
from cell_eval2.real_bundle import build_real_bundle, read_real_bundle
from cell_eval2.de_compute import compute_de


ARMS = {'ntc': ('zero', 'template'), 'shared': ('shared_calibrated', 'template'),
        'conditional': ('conditional', 'template'), 'conditional_nb': ('conditional', 'nb')}


class PanelEvaluation:
    def __init__(self, data, stats, held, config, output):
        self.data, self.stats, self.held, self.config = data, stats, held, config
        self.ci = stats.contexts.index(held)
        self.axis = np.flatnonzero(data.masks[held]); self.genes = np.asarray(data.genes)[self.axis]
        self.directory = Path(output) / 'cache' / ('holdout-' + held) / 'official'
        self.directory.mkdir(parents=True, exist_ok=True)
        self.predictions = Path(output) / 'predictions' / ('holdout-' + held)
        self.predictions.mkdir(parents=True, exist_ok=True)
        self.cfg = replace(official_config(config, self.directory), num_threads=4, pert_chunk=32)
        self.targets = json.loads((stats.directory / 'panels.json').read_text())['contexts'][held]
        self.feature_ids = np.concatenate([ids for (c, b, half), ids in data.controls.items() if c == held and half == 0])
        reference_ids = np.concatenate([ids for (c, b, half), ids in data.controls.items() if c == held and half == 1])
        ids, labels, self.blocks = [], [], []
        for target in self.targets:
            start = len(ids); group = data.tasks[held, target]
            ids.extend(group); labels.extend([target] * len(group)); self.blocks.append((target, start, len(ids)))
        start = len(ids); ids.extend(reference_ids); labels.extend([self.cfg.control] * len(reference_ids))
        self.control_start = start
        self.observations = pd.DataFrame({'target': labels}, index=[f'real-{held}-{i}' for i in range(len(ids))])
        path = self.directory / 'reference-counts.npy'
        identity = {'context': held, 'targets': self.targets, 'source_rows': list(map(int, ids)), 'genes': self.genes.tolist(),
                    'stats_sha256': hash_file(stats.directory / 'complete.json')}
        marker = self.directory / 'reference.json'
        if marker.exists():
            saved = json.loads(marker.read_text())
            assert saved['identity'] == identity and hash_file(path) == saved['sha256']
        else:
            values = np.lib.format.open_memmap(path, mode='w+', dtype=np.uint16, shape=(len(ids), len(self.axis)))
            for begin in range(0, len(ids), 256):
                values[begin:begin + 256] = data.read(ids[begin:begin + 256])[:, self.axis].astype(np.uint16)
            values.flush(); del values
            write_json(marker, {'identity': identity, 'sha256': hash_file(path)})
        self.reference = ad.AnnData(np.load(path, mmap_mode='r'), obs=self.observations, var=pd.DataFrame(index=self.genes))
        self.bundle = self.directory / 'reference-bundle'
        self.cfg.to_yaml(str(self.directory / 'eval-config.yaml'))

    def prepare(self):
        if not self.bundle.exists():
            profile = generic_response_profile(self.reference, pert_col=self.cfg.pert_col,
                                               control=self.cfg.control, exclude_target_gene=False)
            baseline = build_baseline_prediction(profile, self.reference, pert_col=self.cfg.pert_col,
                                                  control=self.cfg.control, emit='tile')
            manifest = build_real_bundle(self.reference, baseline, config=self.cfg, outdir=str(self.bundle),
                                         bundle_id='native-count-panel-' + self.held, base_seed=0, n_splits=5)
            assert manifest['rule_digest'] is not None, manifest.get('rule_mismatches')
            del baseline; gc.collect()
        assert read_real_bundle(self.bundle).manifest['rule_digest'] is not None

    def null_diagnostics(self):
        marker = self.directory / 'null-diagnostics.json'
        if marker.exists():
            return json.loads(marker.read_text())
        rng = np.random.default_rng(stable_seed(self.config['seed'], self.held, 'null'))
        n = min(400, len(self.feature_ids))
        ids = rng.choice(self.feature_ids, n, replace=False)
        templates = self.data.read(ids)[:, self.axis]
        reference = self.reference[self.control_start:].copy()
        mean = self.stats.feature_mean[self.ci, self.axis]; var = self.stats.feature_variance[self.ci, self.axis]
        rows = []
        for method in ['template', 'nb']:
            counts, diagnostic = generate_counts(mean, var, templates, np.zeros(len(self.axis)), 0., method,
                                                  stable_seed(self.config['seed'], self.held, 'null', method), self.config)
            predicted = ad.AnnData(counts, obs=pd.DataFrame({'target': ['null-probe'] * n},
                                   index=[f'null-{method}-{i}' for i in range(n)]), var=reference.var.copy())
            table = compute_de(predicted, backend=self.cfg.de.backend, groupby=self.cfg.pert_col,
                               reference=reference, mean_calc=self.cfg.de.mean_calc, epsilon=self.cfg.de.epsilon,
                               input_type='counts', target_sum=self.cfg.target_sum, clip_value=self.cfg.de.clip_value,
                               filter_gene_min_cpm_cell=self.cfg.filter.filter_gene_min_cpm_cell,
                               fdr_scope=self.cfg.de.fdr_scope, threads=self.cfg.num_threads, device=self.cfg.device)
            table.write_parquet(self.directory / f'null-{method}-de.parquet')
            frame = table.to_pandas()
            padj = frame['p_adj'].to_numpy()
            rows.append({'context': self.held, 'generator': method, 'cells': n, 'tested_genes': int(np.isfinite(padj).sum()),
                         'significant_genes': int((padj < .05).sum()),
                         'significant_fraction': float(np.mean(padj[np.isfinite(padj)] < .05)), **diagnostic})
        write_json(marker, rows)
        return rows

    def evaluate(self, fitted, arm, tracked):
        response_method, generator = ARMS[arm]
        response, depth, donors, support = predict_transfer(self.stats, fitted, self.targets,
                                                           self.stats.feature_mean[self.ci], response_method)
        mu = self.stats.feature_mean[self.ci, self.axis]; var = self.stats.feature_variance[self.ci, self.axis]
        summaries = []
        for seed in self.config['prediction_seeds']:
            destination = self.predictions / arm / f'seed-{seed}'
            destination.mkdir(parents=True, exist_ok=True)
            marker = destination / 'complete.json'; path = destination / 'counts.npy'
            if marker.exists():
                saved = json.loads(marker.read_text())
                assert saved['prediction_sha256'] == hash_file(path), 'prediction_changed'
                for filename, digest in saved['score_files'].items():
                    assert hash_file(destination / filename) == digest
                summaries.append(saved); continue
            begin = time.monotonic()
            predicted = np.lib.format.open_memmap(path, mode='w+', dtype=np.uint32, shape=self.reference.shape)
            diagnostics = []
            for i, (target, lo, hi) in enumerate(self.blocks):
                rng = np.random.default_rng(stable_seed(seed, self.held, target, 'templates'))
                selected = rng.choice(self.feature_ids, hi - lo, replace=hi - lo > len(self.feature_ids))
                templates = self.data.read(selected)[:, self.axis]
                counts, diagnostic = generate_counts(mu, var, templates, response[i, self.axis], float(depth[i]), generator,
                                                       stable_seed(seed, self.held, target, 'counts'), self.config)
                predicted[lo:hi] = counts
                diagnostics.append({'context': self.held, 'target': target, 'arm': arm, 'seed': seed,
                                    'target_readouts_supported': int((donors[i, self.axis] > 0).sum()),
                                    'no_training_measurement': int((~support[i, self.axis]).sum()), **diagnostic})
            rng = np.random.default_rng(stable_seed(seed, self.held, 'predicted-controls'))
            for lo in range(self.control_start, len(predicted), 256):
                hi = min(len(predicted), lo + 256)
                ids = rng.choice(self.feature_ids, hi - lo, replace=True)
                predicted[lo:hi] = self.data.read(ids)[:, self.axis].astype(np.uint32)
            predicted.flush()
            obs = self.observations.copy(); obs.index = [f'pred-{arm}-{seed}-{i}' for i in range(len(obs))]
            adata = ad.AnnData(predicted, obs=obs, var=self.reference.var.copy())
            print(json.dumps({'stage': 'official_score', 'context': self.held, 'arm': arm, 'seed': seed}), flush=True)
            result = scored_result(adata, self.reference, self.cfg, self.bundle, destination)
            pd.DataFrame(diagnostics).to_parquet(destination / 'generation-diagnostics.parquet', index=False)
            raw = pd.read_csv(destination / 'aggregate.csv').set_index('statistic').loc['mean'].to_dict()
            result.update(context=self.held, arm=arm, seed=seed, raw=raw, prediction_sha256=hash_file(path),
                          elapsed_seconds=time.monotonic() - begin,
                          score_files={name: hash_file(destination / name) for name in ['scores.csv', 'aggregate.csv', 'raw-metrics.parquet']})
            write_json(marker, result); summaries.append(result)
            tracked.log({f'validation/{self.held}/{arm}/score': result['score'],
                         f'validation/{self.held}/{arm}/seconds': result['elapsed_seconds'], 'generation_seed': seed})
            del adata, predicted; gc.collect(); release_read_cache(path); torch.cuda.empty_cache()
        return {'context': self.held, 'arm': arm, 'score': float(np.mean([r['score'] for r in summaries])),
                'seed_sd': float(np.std([r['score'] for r in summaries], ddof=1)), 'seeds': summaries,
                'panel_targets': len(self.targets), 'cells': self.reference.n_obs, 'genes': self.reference.n_vars}

    def close(self):
        del self.reference
        gc.collect(); torch.cuda.empty_cache()
        release_read_cache(self.directory / 'reference-counts.npy')
