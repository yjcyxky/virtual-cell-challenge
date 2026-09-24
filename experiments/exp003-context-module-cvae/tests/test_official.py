"""Exercise the pinned official six-metric path, including local calibration."""
from pathlib import Path
import sys

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from official import official_config, scored_result
from cell_eval2.baseline import generic_response_profile, build_baseline_prediction
from cell_eval2.competition import competition_members
from cell_eval2.real_bundle import build_real_bundle
from official import OfficialValidation


def test_official_bundle_scores_correct_perturbations_above_label_swap(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip('production DE backend requires CUDA')
    torch.set_num_threads(2)
    cfg = yaml.safe_load((Path(__file__).resolve().parents[1] / 'configs/default.yaml').read_text())
    evaluator = official_config(cfg, tmp_path)
    rng = np.random.default_rng(812)
    genes = [f'G{i}' for i in range(80)]
    targets = [f'G{i}' for i in range(8)]
    arrays = [rng.poisson(5., (128, len(genes)))]
    labels = [evaluator.control] * 128
    for target in targets:
        means = 5 * np.exp(rng.choice([-1., 1.], len(genes)) * 1.2)
        arrays.append(rng.poisson(means, (128, len(genes))))
        labels.extend([target] * 128)
    real = ad.AnnData(np.concatenate(arrays).astype(np.uint16), obs=pd.DataFrame({'target': labels},
                      index=[str(i) for i in range(len(labels))]), var=pd.DataFrame(index=genes))
    original = real.X.copy()
    profile = generic_response_profile(real, pert_col='target', control=evaluator.control, exclude_target_gene=False)
    baseline = build_baseline_prediction(profile, real, pert_col='target', control=evaluator.control, emit='tile')
    bundle = tmp_path / 'bundle'
    manifest = build_real_bundle(real, baseline, config=evaluator, outdir=str(bundle), bundle_id='test')
    assert manifest['rule_digest'] is not None
    perfect = scored_result(real, real, evaluator, bundle, tmp_path / 'perfect')
    swapped = real.copy()
    swapped.obs['target'] = swapped.obs.target.map({**dict(zip(targets, targets[1:] + targets[:1])), evaluator.control: evaluator.control})
    bad = scored_result(swapped, real, evaluator, bundle, tmp_path / 'swapped')
    assert set(perfect['components']) == set(competition_members()) and len(perfect['components']) == 6
    assert perfect['score'] > bad['score']
    assert perfect['components']['pds_cosine'] > bad['components']['pds_cosine']
    assert perfect['components']['expr_mse_unbiased_capped_norm'] == 1.
    np.testing.assert_array_equal(real.X, original)


def test_whole_reference_prediction_and_resume_lifecycle(tmp_path):
    from types import SimpleNamespace
    if not torch.cuda.is_available():
        pytest.skip('production DE backend requires CUDA')
    cfg = yaml.safe_load((Path(__file__).resolve().parents[1] / 'configs/default.yaml').read_text())
    rng = np.random.default_rng(912)
    genes = [f'G{i}' for i in range(80)]
    means = [np.full(80, 5.)] + [5 * np.exp(rng.choice([-1., 1.], 80) * 1.2) for _ in range(8)]
    counts = np.concatenate([rng.poisson(mu, (128, 80)) for mu in means]).astype(np.uint16)
    tasks = {('A', f'G{i}'): np.arange((i+1)*128, (i+2)*128) for i in range(8)}
    source = tmp_path / 'source'; source.mkdir(); (source / 'complete.json').write_text('{}')
    data = SimpleNamespace(directory=source, genes=genes, masks={'A': np.ones(80, bool)},
                           groups={key: {('guide', 'b0'): rows} for key, rows in tasks.items()}, tasks=tasks,
                           controls={('A', 'b0', 1): np.arange(128)}, read=lambda ids: counts[ids].copy())
    def inputs(specifications, device):
        indices = []
        for context, target, batch, n in specifications:
            indices.extend([0 if target == '__NTC__' else int(target[1:]) + 1] * n)
        return {'index': torch.tensor(indices, device=device)}
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.weight = torch.nn.Parameter(torch.ones(1, device='cuda'))
            self.register_buffer('means', torch.tensor(np.stack(means), dtype=torch.float32, device='cuda'))
        def generate(self, batch): return torch.poisson(self.means[batch['index']])
    view = SimpleNamespace(inputs=inputs)
    cache, predictions = tmp_path / 'cache', tmp_path / 'predictions'; cache.mkdir()
    evaluation = OfficialValidation(data, view, 'A', cfg, cache, predictions)
    evaluation.prepare()
    model = Model()
    first = evaluation.evaluate(model, 1)
    evaluation.promote_best(1)
    evaluation.record_prediction_files()
    evaluation.clear_transient()
    recovered = OfficialValidation(data, view, 'A', cfg, cache, predictions)
    recovered.prepare()
    second = recovered.evaluate(model, 2)
    assert first['seed_scores'] == second['seed_scores']
    assert first['mean_score'] == second['mean_score']
    obs = pd.read_parquet(predictions / 'official-observations.parquet')
    assert 'source_cached_row' not in obs and len(obs) == len(counts)
    for seed in cfg['prediction_seeds']:
        array = np.load(predictions / f'best-seed-{seed}.npy')
        assert array.shape == counts.shape and array.dtype == np.uint32
    assert model.training
