"""Behavioral contracts for shared readouts, response objectives and recovery."""
from pathlib import Path
import sys
import gzip

import numpy as np
import pytest
import torch
import yaml

EXPERIMENT = Path(__file__).resolve().parents[1]
CORE = EXPERIMENT.parent / 'exp003-context-module-cvae'
sys.path[:0] = [str(EXPERIMENT / 'src'), str(CORE / 'src'), str(CORE / 'tests')]
from test_protocol import SmallData, configuration, setup_model
from shared_model import SharedResponseCVAE
from response_objective import ResponseObjective, amplitude_loss, specificity_loss
from state import FoldView
from data import BalancedSampler
from training import train_step, save_checkpoint, load_checkpoint
from training import monitor_training, monitor_plan


def shared_model(config=None):
    original, batch = setup_model()
    overrides = config or {}
    config = dict(configuration(), gene_feature_dimensions=8, relation_rank=2,
                  relation_scale=1., relation_mode='conditional', shared_target=True,
                  response_auxiliary='none', retrieval_candidates=3, retrieval_temperature=.2)
    config.update(overrides)
    rng = np.random.default_rng(35)
    membership = (rng.random((24, 4)) > .5).astype(np.float32)
    model = SharedResponseCVAE(24, membership, np.ones(4), original.projection.numpy(),
                              original.origin.numpy(), original.common.numpy(), np.ones(24, bool),
                              config, features=rng.normal(size=(24, 12)).astype(np.float32))
    return model, batch


def test_low_rank_operator_matches_explicit_and_changes_cross_ratios():
    base = torch.tensor([[1., 2.], [3., 4.]])
    left = torch.tensor([[1.], [2.]])
    right = torch.tensor([[1.], [-1.]])
    values = torch.tensor([[2., 3.], [4., 5.]])
    coefficients = torch.tensor([[0.], [.7]])
    actual = SharedResponseCVAE.relation_decode(values, base, left, right, coefficients)
    expected = torch.stack([v @ (base + (left * c) @ right.T).T for v, c in zip(values, coefficients)])
    torch.testing.assert_close(actual, expected)
    changed = base + .7 * left @ right.T
    ratio = lambda x: x[0, 0] * x[1, 1] / (x[0, 1] * x[1, 0])
    assert not torch.isclose(ratio(base), ratio(changed))


def test_unlabelled_readout_is_updated_through_shared_parameters():
    model, batch = shared_model()
    names = dict(model.named_parameters())
    assert not {'loadings', 'residual_loadings', 'dispersion_loadings', 'gene_gate'} & names.keys()
    # Only readout 0 is supervised. Readout 23 changes through shared mapping.
    before = model.shared_loadings(model.feature_encoder(model.gene_features)).detach().clone()
    optimizer = torch.optim.SGD(model.parameters(), lr=.1)
    model.shared_loadings(model.feature_encoder(model.gene_features))[0].sum().backward()
    optimizer.step()
    after = model.shared_loadings(model.feature_encoder(model.gene_features)).detach()
    assert not torch.equal(before[23], after[23])


def test_ntc_identity_and_missing_measurements():
    model, batch = shared_model()
    null = dict(batch, target=torch.full_like(batch['target'], model.genes))
    noise = torch.randn(len(batch['target']), 2, 4, model.d)
    predicted, _ = model.conditional_moments(batch, noise)
    control, _ = model.conditional_moments(null, noise)
    torch.testing.assert_close(predicted, control, rtol=0, atol=0)
    assert torch.all(predicted[:, -1] == 0)
    samples = model.generate(batch)
    assert torch.isfinite(samples).all() and torch.all(samples >= 0)
    assert torch.all(samples[:, -1] == 0)
    assert torch.equal(samples, samples.round())


def test_conditional_relation_receives_gradient_and_depends_on_context():
    model, batch = shared_model()
    z = torch.randn(len(batch['target']), model.d)
    mean, _, _ = model.decode(z, batch)
    mean.square().mean().backward()
    assert model.relation_context[-1].weight.grad.abs().sum() > 0
    with torch.no_grad():
        model.relation_context[-1].weight.add_(-.01 * model.relation_context[-1].weight.grad)
    coefficients = model.relation_context(torch.randn(5, 3 * model.d + 5))
    assert not torch.equal(coefficients[0], coefficients[1])


def test_amplitude_projection_rewards_correct_sign_and_scale_with_gradient_at_zero():
    observed = torch.tensor([[1., -2., .5]])
    scale, weight = torch.ones_like(observed) * .1, torch.ones_like(observed)
    loss = lambda x: amplitude_loss(x, observed, scale, weight).sum()
    zero = torch.zeros_like(observed, requires_grad=True)
    assert loss(observed) < loss(.5 * observed) < loss(zero) < loss(-observed)
    loss(zero).backward()
    assert torch.isfinite(zero.grad).all() and zero.grad.abs().sum() > 0


def test_specificity_uses_soft_similarities_and_has_gradient_at_zero():
    observed = torch.tensor([[1., -1., .5, -.5]])
    references = torch.stack([observed, observed * .99, -observed], dim=1)
    scale, mask = torch.ones_like(observed) * .1, torch.ones_like(observed)
    loss = lambda x: specificity_loss(x, observed, references, scale, mask).sum()
    zero = torch.zeros_like(observed, requires_grad=True)
    assert abs(loss(observed).item()) < 1e-6
    assert loss(observed) < loss(zero) < loss(-observed)
    loss(zero).backward()
    assert torch.isfinite(zero.grad).all() and zero.grad.abs().sum() > 0


@pytest.mark.parametrize('auxiliary', ['none', 'amplitude', 'specificity'])
@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_training_only_losses_and_exact_full_state_recovery(tmp_path, auxiliary, device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(2)
    model, _ = shared_model({'response_auxiliary': auxiliary})
    config = model.config
    if auxiliary != 'none':
        config['auxiliary_gradient_ratios'][auxiliary + '_loss'] = .05
    data = SmallData()
    data.forbidden = set(np.concatenate([v for (c, _), v in data.tasks.items() if c == 'E']))
    view = FoldView(data, list('ABCD'), config, tmp_path)
    model.projection.copy_(torch.tensor(view.projection))
    model.origin.copy_(torch.tensor(view.origin)); model.common.fill_(1)
    model.to(device)
    objective = ResponseObjective(data, view, list('ABCD'), config, tmp_path)
    assert set(c for c, _ in objective.keys) == set('ABCD')
    objective.calibrate(model, list('ABCD'), tmp_path, device)
    assert set(objective.weights) == set(objective.loss_terms)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1., total_iters=1)
    sampler = BalancedSampler(data, list('ABCD'), 37, 20)
    train_step(model, data, view, sampler, optimizer, scheduler, config, 0, device, objective)
    path = tmp_path / 'last.pt'
    save_checkpoint(path, model, optimizer, scheduler, sampler, {'next_step': 1, 'complete': False})
    expected = train_step(model, data, view, sampler, optimizer, scheduler, config, 1, device, objective)
    expected_state = {k: v.clone() for k, v in model.state_dict().items()}
    recovered, _ = shared_model({'response_auxiliary': auxiliary})
    recovered.to(device)
    opt = torch.optim.AdamW(recovered.parameters(), lr=.001)
    sch = torch.optim.lr_scheduler.ConstantLR(opt, factor=1., total_iters=1)
    sampling = BalancedSampler(data, list('ABCD'), 99, 20)
    progress = load_checkpoint(path, recovered, opt, sch, sampling, device)
    assert progress == {'next_step': 1, 'complete': False}
    actual = train_step(recovered, data, view, sampling, opt, sch, config, 1, device, objective)
    assert actual == expected
    for k, value in recovered.state_dict().items():
        torch.testing.assert_close(value, expected_state[k], rtol=0, atol=0)


def test_matrix_preserves_full_folds_and_independent_objectives():
    matrix = yaml.safe_load((EXPERIMENT / 'configs/matrix.yaml').read_text())
    assert len(matrix['arms']) == 9 and len(matrix['seeds']) == 3
    for arm in matrix['arms']:
        config = yaml.safe_load((EXPERIMENT / 'configs' / f'{arm}.yaml').read_text())
        assert config['prior_mode'] in ['true', 'random', 'none']
        assert config['relation_mode'] in ['off', 'constant', 'conditional']
        assert len(config['contexts']) == 5 and config['unseen_target_percent'] == 0
        assert config['learning_rate'] == .001 and config['learning_rate_schedule'] == 'constant'
        assert config['response_auxiliary'] == (arm if arm in ['amplitude', 'specificity'] else 'none')


@pytest.mark.parametrize('auxiliary', ['none', 'amplitude', 'specificity'])
def test_vectorized_monitor_is_repeatable_and_preserves_training_rng(tmp_path, auxiliary):
    model, _ = shared_model({'response_auxiliary': auxiliary})
    data = SmallData(); config = model.config
    view = FoldView(data, list('ABCD'), config, tmp_path)
    objective = ResponseObjective(data, view, list('ABCD'), config, tmp_path)
    objective.weights = dict.fromkeys(objective.loss_terms, 1.)
    plan = monitor_plan(data, list('ABCD'), config, tmp_path)
    rng = torch.get_rng_state().clone()
    first = monitor_training(model, data, view, plan, list('ABCD'), config, 12, objective)
    second = monitor_training(model, data, view, plan, list('ABCD'), config, 12, objective)
    assert first == second
    assert torch.equal(rng, torch.get_rng_state())
    assert all(np.isfinite(v) for row in first.values() for v in row.values())
    if auxiliary != 'none':
        assert all(auxiliary + '_loss' in row for row in first.values())


def test_annotation_features_have_no_expression_dependency_and_detect_stale_inputs(tmp_path, monkeypatch):
    import shared_model as module
    from scipy import sparse
    priors = tmp_path / 'priors'; priors.mkdir()
    network = tmp_path / 'data/raw/networks'; network.mkdir(parents=True)
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    np.save(priors / 'embedding.npy', np.eye(4, dtype=np.float32))
    sparse.save_npz(priors / 'physical.npz', sparse.csr_matrix(np.eye(4)))
    sparse.save_npz(priors / 'functional.npz', sparse.csr_matrix(np.eye(4)))
    sparse.save_npz(priors / 'pathways.npz', sparse.csr_matrix(np.eye(4)))
    np.savez(priors / 'modules.npz', true=np.eye(4, dtype=np.float32))
    with gzip.open(network / 'goa_human.gaf.gz', 'wt') as stream:
        for gene, qualifier in [('G0', ''), ('G1', 'NOT')]:
            fields = ['DB', 'ID', gene, qualifier, 'GO:123', *['x'] * 10]
            stream.write('\t'.join(fields) + '\n')
    for mode in ['true', 'random', 'none']:
        output = tmp_path / mode; output.mkdir()
        np.save(output / 'prior-strength.npy', np.array([1., .25, .5, .25]))
        config = dict(data_seed=42, prior_mode=mode)
        result = module.gene_features(['G0', 'G1', 'G2', 'G3'], priors, config, output)
        np.testing.assert_array_equal(result, module.gene_features(['G0', 'G1', 'G2', 'G3'], priors, config, output))
        assert np.isfinite(result).all()
        if mode == 'none':
            assert not result.any()
        if mode == 'true':
            np.testing.assert_allclose(np.diag(result[:, -4:]), [2., .5, 1., .5])
            assert result[0, 40:72].any() and not result[1, 40:72].any()
        np.save(output / 'prior-strength.npy', np.zeros(4))
        with pytest.raises(AssertionError):
            module.gene_features(['G0', 'G1', 'G2', 'G3'], priors, config, output)
