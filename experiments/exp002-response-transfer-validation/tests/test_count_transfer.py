"""Behavioral checks for held-context isolation, physical counts and official evaluation."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from count_transfer import (bulk, donor_predictions, fit_transfer, predict_transfer, evaluate_means, write_json,
                            save_response_library, ResponseLibrary, CountData, count_moments)
from count_generation import desired_mean, generate_counts, template_counts


def configuration():
    return json.loads((Path(__file__).resolve().parents[1] / 'configs/native-count-transfer.json').read_text())


def fixture_stats(directory, contexts=4, targets=8, genes=80):
    rng = np.random.default_rng(46)
    directory.mkdir(parents=True, exist_ok=True); (directory / 'complete.json').write_text('{}')
    signal = rng.choice([-1., 1.], (targets, genes)) * .2
    response = np.tile(signal[None], (contexts, 1, 1))
    mean = rng.uniform(5, 10, (contexts, genes))
    masks = np.ones((contexts, genes), bool)
    return SimpleNamespace(directory=directory, genes=[f'G{i}' for i in range(genes)],
                           contexts=[f'C{i}' for i in range(contexts)], targets=[f'G{i}' for i in range(targets)],
                           target_index={f'G{i}': i for i in range(targets)}, gene_index={f'G{i}': i for i in range(genes)},
                           response=response, available=np.ones((contexts, targets), bool),
                           depth_response=np.zeros((contexts, targets)), generic=response.mean(1),
                           feature_mean=mean, feature_variance=2 * mean, reference_mean=mean.copy(),
                           masks=masks, common=masks.all(0), features=bulk(mean))


def test_native_statistics_match_expanded_counts_with_mixed_panels_and_missing_genes():
    data = object.__new__(CountData)
    data.genes = list('ABCDEFG'); data.axes = {0: np.array([4, 1, 3, 5]), 1: np.array([6, 5, 0, 1])}
    data.counts = {0: np.array([[4, 9, 2, 0], [5, 3, 0, 10]], np.uint16),
                   1: np.array([[0, 2, 7, 5], [6, 3, 4, 2]], np.uint16)}
    data.panel_context = {0: 'C0', 1: 'C1'}
    data.masks = {'C0': np.array([False, True, False, True, False, True, False]),
                  'C1': np.array([True, True, False, False, False, True, True])}
    data.panel_indices = np.array([0, 1, 0, 1]); data.cache_rows = np.array([0, 0, 1, 1])
    ids = np.array([3, 0, 2, 3, 1])
    expanded = data.read(ids).astype(np.float64)
    expected = expanded.mean(0), expanded.var(0), (expanded > 0).mean(0), expanded.sum(1)
    for actual, wanted in zip(count_moments(data, ids), expected):
        np.testing.assert_allclose(actual, wanted, rtol=1e-14, atol=1e-14)


def test_preparation_repair_preserves_identity_and_refuses_changes_after_fit(tmp_path):
    from count_pipeline import resolve_config
    (tmp_path / 'checkpoints').mkdir(); (tmp_path / 'predictions').mkdir()
    config = {'seed': 17, 'source_files': {'pipeline.py': 'old'}}
    original = resolve_config(tmp_path, dict(config))
    with pytest.raises(ValueError, match='new_run_required'):
        resolve_config(tmp_path, dict(config, seed=18), 'same preparation')
    repaired = resolve_config(tmp_path, dict(config, source_files={'pipeline.py': 'new'}), 'equivalent native sum')
    assert len(repaired['preparation_revisions']) == 1
    archived = tmp_path / repaired['preparation_revisions'][0]['previous_config']
    assert json.loads(archived.read_text()) == original
    (tmp_path / 'checkpoints/holdout-C0.json').write_text('{}')
    with pytest.raises(ValueError, match='new_run_required'):
        resolve_config(tmp_path, dict(config, source_files={'pipeline.py': 'another'}), 'cannot change fitted source')


def test_external_held_response_never_changes_calibration_or_prediction(tmp_path):
    stats = fixture_stats(tmp_path / 'statistics'); cfg = configuration()
    before = fit_transfer(stats, stats.contexts[1:], cfg)
    expected = predict_transfer(stats, before, stats.targets, stats.feature_mean[0])[0]
    stats.response[0] += 1000; stats.generic[0] += 1000; stats.depth_response[0] += 100
    after = fit_transfer(stats, stats.contexts[1:], cfg)
    assert before == after
    np.testing.assert_array_equal(expected, predict_transfer(stats, after, stats.targets, stats.feature_mean[0])[0])
    assert before['selected']['coefficients'][0] == pytest.approx(1, abs=1e-7)
    assert before['selected']['coefficients'][1] == pytest.approx(0, abs=1e-7)


def test_native_noncommon_gene_is_learned_and_absence_is_not_zero_label(tmp_path):
    stats = fixture_stats(tmp_path / 'statistics')
    stats.response[1, :, 70] = np.nan; stats.generic[1, 70] = np.nan; stats.masks[1, 70] = False
    stats.response[2, :, 70] = .9; stats.generic[2, 70] = .9
    stats.common = stats.masks.all(0); stats.features = bulk(stats.feature_mean[:, stats.common])
    shared, _, count, supported, _ = donor_predictions(stats, ['C1', 'C2'], stats.targets, stats.feature_mean[0], .1)
    np.testing.assert_allclose(shared[:, 70], .9)
    assert (count[:, 70] == 1).all() and supported[:, 70].all()
    stats.response[2, :, 70] = np.nan; stats.generic[2, 70] = np.nan
    shared, _, count, supported, _ = donor_predictions(stats, ['C1', 'C2'], stats.targets, stats.feature_mean[0], .1)
    assert (count[:, 70] == 0).all() and not supported[:, 70].any()
    assert (shared[:, 70] == 0).all()  # explicit prediction fallback; labels stay NaN
    assert np.isnan(stats.response[1:3, :, 70]).all()


def test_zero_intervention_is_exact_ntc_and_replay_is_deterministic():
    rng = np.random.default_rng(61)
    templates = rng.poisson(np.linspace(2, 10, 50), (128, 50)).astype(np.uint16)
    mean, var = templates.mean(0), templates.var(0)
    zero, _ = generate_counts(mean, var, templates, np.zeros(50), 0, 'template', 83, configuration())
    np.testing.assert_array_equal(zero, templates)
    response = rng.normal(0, .2, 50)
    first, diag = generate_counts(mean, var, templates, response, .1, 'template', 83, configuration())
    second, other = generate_counts(mean, var, templates, response, .1, 'template', 83, configuration())
    np.testing.assert_array_equal(first, second); assert diag == other


def test_margin_fit_preserves_library_variation_and_matches_response_mean():
    rng = np.random.default_rng(62)
    rates = np.exp(rng.normal(0, .5, 400))[:, None] * np.linspace(5, 30, 40)
    templates = rng.poisson(rates)
    control = templates.mean(0); response = np.linspace(-.4, .4, 40)
    desired, _ = desired_mean(control, response, .2, configuration())
    counts, diagnostic = generate_counts(control, templates.var(0), templates, response, .2, 'template', 33, configuration())
    assert diagnostic['row_relative_error'] <= configuration()['ipf_tolerance']
    assert np.abs(counts.mean(0) - desired).sum() / desired.sum() < .005
    cv = lambda x: x.sum(1).std() / x.sum(1).mean()
    assert cv(counts) == pytest.approx(cv(templates), rel=.02)
    assert np.issubdtype(counts.dtype, np.integer) and counts.min() >= 0


def test_induction_and_limits_are_explicit():
    templates = np.tile([0, 10, 20, 30], (128, 1))
    cfg = configuration()
    counts, _ = generate_counts(templates.mean(0), templates.var(0), templates, [2., 0., 0., 0.], 0, 'template', 92, cfg)
    assert counts[:, 0].sum() > 0
    with pytest.raises(ValueError, match='official_limit'):
        generate_counts(templates.mean(0), templates.var(0), templates, np.zeros(4), 0, 'template', 1,
                        dict(cfg, max_counts_per_cell=10))
    with pytest.raises(RuntimeError, match='not_converged'):
        template_counts(np.array([[1, 100], [100, 1]]), np.array([10., 100.]), np.random.default_rng(3), 1, 1e-15)


def test_mean_evaluation_includes_ntc_offset_and_all_tasks(tmp_path):
    stats = fixture_stats(tmp_path / 'statistics')
    stats.reference_mean[0, 20] *= 3
    stats.masks[0, 70:] = False
    stats.common = stats.masks.all(0); stats.features = bulk(stats.feature_mean[:, stats.common])
    fitted = fit_transfer(stats, stats.contexts[1:], configuration())
    frame = evaluate_means(stats, fitted, 'C0', configuration(), tmp_path / 'prediction')
    baseline = frame.loc[frame.method.eq('zero') & frame.stratum.eq('all')]
    expected = bulk(stats.feature_mean[0, :70]) - bulk(stats.reference_mean[0, :70])
    for row in baseline.itertuples():
        mask = np.arange(70) != stats.gene_index[row.target]
        truth = stats.response[0, stats.target_index[row.target], :70]
        assert row.mse == pytest.approx(np.mean((truth[mask] - expected[mask]) ** 2))
    assert len(baseline) == len(stats.targets)


def test_delivered_model_predicts_without_statistics_cache_and_checks_identity(tmp_path):
    stats = fixture_stats(tmp_path / 'statistics')
    # One missing task exercises the archive's sparse target indexing.
    stats.response[1, 2] = np.nan; stats.available[1, 2] = False; stats.depth_response[1, 2] = np.nan
    fitted = fit_transfer(stats, stats.contexts[1:], configuration())
    targets = stats.targets + ['unseen-target']
    expected = predict_transfer(stats, fitted, targets, stats.feature_mean[0])
    path = tmp_path / 'model/response-library.npz'
    save_response_library(stats, path)
    library = ResponseLibrary(path, fitted)
    stats.response[:] = np.nan; (stats.directory / 'complete.json').unlink()
    for actual, wanted in zip(predict_transfer(library, fitted, targets, stats.feature_mean[0]), expected):
        np.testing.assert_allclose(actual, wanted)
    assert 0 not in library.tables
    with pytest.raises(AssertionError, match='checkpoint_mismatch'):
        ResponseLibrary(path, dict(fitted, stats_sha256='different-training-run'))


def test_null_de_and_paired_official_panel_use_production_path(tmp_path):
    import torch
    if not torch.cuda.is_available():
        pytest.skip('official gpudge requires CUDA')
    from count_evaluation import PanelEvaluation
    rng = np.random.default_rng(902)
    stats = fixture_stats(tmp_path / 'statistics', contexts=3)
    cfg = dict(configuration(), contexts=stats.contexts, prediction_seeds=[101, 202, 303])
    genes = stats.genes; mean = np.full(len(genes), 5.)
    arrays = [rng.poisson(mean, (128, len(genes))), rng.poisson(mean, (128, len(genes)))]
    tasks = {}; cursor = 256
    for i, target in enumerate(stats.targets):
        response = rng.choice([-1., 1.], len(genes)) * 1.2
        x = rng.poisson(mean * np.exp(response), (128, len(genes)))
        arrays.append(x); tasks['C0', target] = np.arange(cursor, cursor + 128); cursor += 128
        stats.response[1:, i] = bulk(x.mean(0)) - bulk(mean)
    counts = np.concatenate(arrays).astype(np.uint16)
    stats.feature_mean[:] = mean; stats.feature_variance[:] = mean; stats.reference_mean[:] = mean
    stats.generic[:] = stats.response.mean(1); stats.features = bulk(stats.feature_mean)
    write_json(stats.directory / 'panels.json', {'contexts': {'C0': stats.targets}})
    data = SimpleNamespace(genes=genes, masks={'C0': np.ones(len(genes), bool)}, tasks=tasks,
                           controls={('C0', 'b0', 0): np.arange(128), ('C0', 'b0', 1): np.arange(128, 256)},
                           read=lambda ids: counts[np.asarray(ids)].astype(np.float32))
    evaluator = PanelEvaluation(data, stats, 'C0', cfg, tmp_path / 'output')
    evaluator.prepare()
    nulls = evaluator.null_diagnostics()
    assert len(nulls) == 2 and all(r['tested_genes'] > 0 for r in nulls)
    fitted = {'training_contexts': ['C1', 'C2'], 'selected': {'temperature': .1, 'coefficients': [1., 0.],
              'fallback_coefficient': 0., 'shared_coefficient': 1., 'depth_coefficient': 0.}}
    tracker = SimpleNamespace(log=lambda x: None)
    ntc = evaluator.evaluate(fitted, 'ntc', tracker)
    target = evaluator.evaluate(fitted, 'conditional', tracker)
    assert target['score'] > ntc['score']
    assert len(target['seeds'][0]['components']) == 6
    assert evaluator.evaluate(fitted, 'conditional', tracker) == target
    evaluator.close()
