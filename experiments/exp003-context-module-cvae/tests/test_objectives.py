"""Observable behavior of response contrasts, batch mixtures and calibration."""
import copy
import numpy as np
import pytest
import torch

from test_protocol import SmallData, configuration, setup_model
from state import FoldView
from objectives import TaskObjective, COMPONENTS, predictive_losses
from model import masked_logcp


def make_objective(tmp_path):
    data = SmallData(); config = configuration(); config['unseen_target_percent'] = 0
    data.forbidden = set(np.concatenate([v for (c, t), v in data.tasks.items() if c == 'E']))
    view = FoldView(data, list('ABCD'), config, tmp_path)
    return data, config, view, TaskObjective(data, view, list('ABCD'), config, tmp_path)


def test_task_statistics_use_unique_cells_and_preserve_unequal_layer_mass(tmp_path):
    data, config, view, objective = make_objective(tmp_path)
    key = objective.keys[0]; groups = data.groups[key]
    # Thirty cells and three cells receive equal layer mass, not equal cell mass.
    mean = sum(data.read(ids).mean(0) / 2 for ids in groups.values())
    ntc = sum(data.control(key[0], b, 1).mean(0) / 2 for _, b in groups)
    expected = masked_logcp(torch.tensor(mean), torch.ones(24)) - masked_logcp(torch.tensor(ntc), torch.ones(24))
    np.testing.assert_allclose(objective.arrays['delta'][0], expected.numpy(), atol=1e-6)
    import pandas as pd
    support = pd.read_parquet(objective.directory / 'support.parquet').iloc[0]
    assert support.independent_cells == 33
    assert support.effective_cells == pytest.approx(1 / (30*(.5/30)**2 + 3*(.5/3)**2))
    restored = TaskObjective(data, view, list('ABCD'), config, tmp_path)
    assert objective.signature == restored.signature
    assert set(c for c, t in objective.keys) == set('ABCD')


def test_predictive_batch_sampling_matches_weighted_reference_measure(tmp_path):
    _, _, _, objective = make_objective(tmp_path)
    key = objective.keys[0]
    objective.mixtures[key] = (['b0', 'b1'], np.array([.8, .2]))
    specs = objective.batch_specs([key] * 1000, np.random.default_rng(12))
    assert np.mean([b == 'b0' for c, t, b, n in specs]) == pytest.approx(.8, abs=.01)


def test_common_noise_null_response_is_exact_and_ntc_has_gradients():
    model, inputs = setup_model()
    ntc = dict(inputs, target=torch.full_like(inputs['target'], model.genes))
    epsilon = torch.randn(len(inputs['target']), 2, 4, model.d)
    target, _ = model.conditional_moments(inputs, epsilon)
    control, _ = model.conditional_moments(ntc, epsilon)
    torch.testing.assert_close(target, control, rtol=0, atol=0)
    control.sum().backward()
    assert model.baseline_modules[-1].weight.grad.abs().sum() > 0


def test_moments_include_nb_variance_and_ignore_unmeasured_genes(monkeypatch):
    model, batch = setup_model()
    def decode(z, inputs):
        return torch.ones_like(inputs['base']) * 5, torch.ones_like(inputs['base']) * 2, {}
    monkeypatch.setattr(model, 'decode', decode)
    mean, second = model.conditional_moments(batch, torch.zeros(12, 2, 3, model.d))
    torch.testing.assert_close(mean[:, -1], torch.zeros(12))
    # 23 measured independent NB(5,2) readouts plus a constant latent mean.
    torch.testing.assert_close(second, torch.ones(12) * ((23*5)**2 + 23*(5+25/2)))


def observations(target, control):
    mask = torch.ones_like(target)
    depth = torch.stack([target.sum(-1), control.sum(-1)], -1)
    return dict(delta=masked_logcp(target, mask)-masked_logcp(control, mask), ntc=masked_logcp(control, mask),
                response_scale=mask, ntc_scale=mask, response_weight=mask,
                predicted_target_depth=depth[:, 0], predicted_ntc_depth=depth[:, 1],
                depth=torch.cat([depth.log1p(), torch.ones_like(depth).log1p()], -1))


def test_shared_wrong_baseline_is_caught_by_ntc_anchor_not_response():
    reference = torch.tensor([[10., 20., 30., 40.]])
    shifted = torch.tensor([[40., 30., 20., 10.]], requires_grad=True)
    obs = observations(reference, reference)
    second = shifted.sum(-1).square() + 1
    losses = predictive_losses(shifted, shifted, second, second, obs, torch.ones_like(shifted), .25)
    assert losses['response_loss'].item() == 0
    assert losses['ntc_loss'].item() > 0
    losses['ntc_loss'].sum().backward()
    assert shifted.grad.abs().sum() > 0


def test_response_uses_predicted_ntc_and_backpropagates_both_branches():
    control = torch.tensor([[10., 20., 30., 40.]])
    target = torch.tensor([[30., 20., 30., 20.]])
    predicted = target.clone().requires_grad_()
    predicted_ntc = control.roll(1, -1).requires_grad_()
    obs = observations(target, control)
    losses = predictive_losses(predicted, predicted_ntc, predicted.sum(-1).square()+1,
                              predicted_ntc.sum(-1).square()+1, obs, torch.ones_like(target), .25)
    # A loss using the same observed NTC on both sides would be zero here.
    assert losses['response_loss'].item() > 0
    losses['response_loss'].sum().backward()
    assert predicted.grad.abs().sum() > 0 and predicted_ntc.grad.abs().sum() > 0


def test_predictive_loss_reaches_prior_response_baseline_and_dispersion(tmp_path):
    data, config, view, objective = make_objective(tmp_path)
    model, _ = setup_model()
    losses = objective.losses(model, objective.keys[:4], np.random.default_rng(1))
    loss = sum(losses[k].mean() for k in COMPONENTS)
    loss.backward()
    for module in [model.prior, model.response_modules, model.baseline_modules, model.dispersion]:
        assert module[-1].weight.grad.abs().sum() > 0
    before = copy.deepcopy(model.state_dict())
    objective.calibrate(model, list('ABCD'), tmp_path, 'cpu')
    assert all(config['loss_weight_min'] <= v <= config['loss_weight_max'] for v in objective.weights.values())
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)
    assert objective.weights['response_loss'] == 1.
    weights = objective.weights.copy()
    objective.weights = None
    objective.calibrate(model, list('ABCD'), tmp_path, 'cpu')
    assert objective.weights == weights


def test_count_scale_drift_is_invisible_to_delta_but_penalized_by_depth():
    reference = torch.tensor([[10., 20., 30., 40.]])
    scaled = (reference * 10).requires_grad_()
    obs = observations(reference, reference)
    obs['predicted_target_depth'] = scaled.sum(-1)
    obs['predicted_ntc_depth'] = scaled.sum(-1)
    second = scaled.sum(-1).square() + 1
    losses = predictive_losses(scaled, scaled, second, second, obs, torch.ones_like(scaled), .25)
    assert losses['response_loss'].item() == 0
    assert losses['ntc_loss'].item() < 1e-10
    assert losses['depth_loss'].item() > 0
    losses['depth_loss'].sum().backward()
    assert scaled.grad.abs().sum() > 0
