"""NTC-only state coordinates and batch-matched count offsets; no type ground truth."""
import json
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from data import logcp, write_json


class FoldView:
    def __init__(self, data, training_contexts, config, directory):
        self.data, self.config, self.directory = data, config, directory
        self.controls = {}
        path = directory / 'state.npz'
        dimensions = config['state_dimensions']
        if path.exists():
            saved = np.load(path)
            self.projection, self.origin = saved['projection'], saved['origin']
        else:
            rng = np.random.default_rng(config['data_seed'])
            training = []
            for context in training_contexts:
                # Cached NTC has a common per-layer cap; round-robin batches below
                # avoid unequal low-support layer sizes influencing PCA fitting.
                batches = [v for (c, b, h), v in sorted(data.controls.items()) if c == context and h == 0]
                chosen = [rng.choice(batches[i % len(batches)]) for i in range(1024)]
                training.append(logcp(data.read(chosen)[:, data.common]))
            x = np.concatenate(training)
            pca = PCA(n_components=dimensions, svd_solver='randomized', random_state=config['data_seed']).fit(x)
            scale = np.maximum(np.sqrt(pca.explained_variance_), 0.1)
            self.projection = np.zeros((len(data.genes), dimensions), np.float32)
            self.projection[data.common] = (pca.components_.T / scale).astype(np.float32)
            self.origin = (pca.mean_ @ (pca.components_.T / scale)).astype(np.float32)
            np.savez_compressed(path, projection=self.projection, origin=self.origin, common=data.common,
                                explained_variance=pca.explained_variance_, training_contexts=training_contexts)
        diagnostics = []
        for context in config['contexts']:
            batches = sorted(b for c, b, h in data.controls if c == context and h == 0)
            coordinates, batch_arrays = [], []
            for batch in batches:
                x = data.control(context, batch, 0)
                coordinates.append(self.project(x))
                batch_arrays.append(x)
            z = np.concatenate(coordinates)
            # Equal layer mass in fitting state centers, even when layers have fewer cells.
            weight = np.concatenate([np.full(len(v), 1 / len(v)) for v in coordinates])
            cluster = KMeans(n_clusters=config['state_components'], n_init=5,
                             random_state=config['data_seed']).fit(z, sample_weight=weight)
            centers = cluster.cluster_centers_.astype(np.float32)
            context_mean = np.mean([v.mean(0) for v in coordinates], axis=0).astype(np.float32)
            scales = np.stack([np.maximum(z[cluster.labels_ == k].std(0), 0.15) if (cluster.labels_ == k).sum() > 2
                               else np.maximum(z.std(0), 0.15) for k in range(len(centers))]).astype(np.float32)
            for batch, x, zs in zip(batches, batch_arrays, coordinates):
                mean = x.mean(0); var = x.var(0)
                theta = np.clip(mean ** 2 / np.maximum(var - mean, 0.01), 0.1, 1000)
                ref = data.control(context, batch, 1)
                distances = ((zs[:, None] - centers) ** 2 / np.maximum(scales, 0.2)[None] ** 2).mean(-1)
                probability = np.exp(-distances + distances.min(1, keepdims=True))
                probability /= probability.sum(1, keepdims=True)
                prior_weight = np.maximum(probability.mean(0), 0.01); prior_weight /= prior_weight.sum()
                library = x.sum(1)
                covariate = np.concatenate([context_mean, zs.mean(0) - context_mean,
                                            [np.log1p(library.mean()) / 10, library.std() / max(library.mean(), 1),
                                             data.masks[context].mean()]])
                self.controls[(context, batch)] = {
                    'base': mean.astype(np.float32), 'theta': theta.astype(np.float32),
                    'reference': ref.mean(0), 'reference_log': logcp(ref).mean(0),
                    'reference_log_common': logcp(ref, data.common).mean(0),
                    'centers': centers, 'scales': scales, 'probability': prior_weight.astype(np.float32),
                    'condition': covariate.astype(np.float32), 'mask': data.masks[context],
                }
                diagnostics.append({'context': context, 'batch': batch, 'feature_controls': len(x), 'reference_controls': len(ref),
                                    'mean_observed_depth': float(library.mean()), 'depth_cv': float(library.std() / max(library.mean(), 1)),
                                    'between_batch_state_squared_distance': float(np.mean((zs.mean(0) - context_mean) ** 2)),
                                    'within_batch_state_variance': float(zs.var(0).mean()),
                                    'posterior_cell_type_probability': None, 'state_method': 'train-NTC PCA; NTC-only soft clusters'})
        pd.DataFrame(diagnostics).to_parquet(directory / 'batch-diagnostics.parquet', index=False)
        write_json(directory / 'state-scope.json', {'pca_training_contexts': list(training_contexts),
                   'new_context_allowed_input': 'feature-half NTC only', 'dimension': dimensions,
                   'common_genes': int(data.common.sum()), 'method': 'linear PCA; per-context NTC soft clusters',
                   'interpretation': 'unpaired observational states, not calibrated cell types or causal trajectories'})

    def project(self, x):
        return logcp(x, self.data.common) @ self.projection - self.origin

    def inputs(self, specifications, device):
        import torch
        fields = ['base', 'theta', 'centers', 'scales', 'probability', 'condition', 'mask']
        values = {k: [] for k in fields}
        target_ids = []
        for context, target, batch, count in specifications:
            state = self.controls[(context, batch)]
            for key in fields:
                values[key].extend([state[key]] * count)
            target_ids.extend([self.data.gene_index[target] if target != '__NTC__' else len(self.data.genes)] * count)
        result = {k: torch.as_tensor(np.asarray(v), dtype=torch.float32, device=device) for k, v in values.items()}
        result['target'] = torch.as_tensor(target_ids, dtype=torch.long, device=device)
        return result
