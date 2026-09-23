"""Behavioral checks for joint counts, leakage boundaries, balancing and resume."""
from pathlib import Path
import copy
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from data import BalancedSampler, balanced_selection, reserved
from model import ModuleCVAE, log_nb
from priors import degree_matched_random
from state import FoldView
from training import save_checkpoint, load_checkpoint, train_step
from evaluation import observed_task, evaluate_task
from priors import fold_evidence


def configuration():
    return dict(variant='true_prior', state_dimensions=3, hidden_dimensions=12, residual_dimensions=2,
                module_off_support_weight=0.02, state_components=2, data_seed=301, seed=17,
                contexts=['A', 'B', 'C', 'D', 'E'], unseen_target_percent=20, tasks_per_step=2,
                cells_per_task=4, kl_warmup_epochs=2, steps_per_epoch=4, response_weight=1.0)


def setup_model(variant='true_prior', genes=24):
    torch.manual_seed(19)
    rng = np.random.default_rng(99)
    config = configuration(); config['variant'] = variant
    members = np.zeros((genes, 4), np.float32)
    for j in range(4):
        members[j*4:(j+1)*4, j] = 1
    projection = rng.normal(size=(genes, 3)).astype(np.float32) / np.sqrt(genes)
    common = np.ones(genes, bool); common[-1] = False; projection[-1] = 0
    model = ModuleCVAE(genes, members, np.ones(4), projection, np.zeros(3, np.float32), common,
                       np.ones(genes, bool), config)
    n = 12
    batch = {'base': torch.ones(n, genes) * 5, 'theta': torch.ones(n, genes) * 3,
             'centers': torch.randn(n, 2, 3) * 0.2, 'scales': torch.ones(n, 2, 3) * 0.3,
             'probability': torch.ones(n, 2) / 2, 'condition': torch.randn(n, 9) * 0.1,
             'mask': torch.ones(n, genes), 'target': torch.arange(n) % genes}
    batch['mask'][:, -1] = 0
    return model, batch


def test_nb_parameterization_matches_torch():
    x = torch.tensor([0., 1., 3., 10.])
    mu, theta = torch.tensor([0.1, 2., 5., 7.]), torch.tensor([1., 2., 3., 4.])
    reference = torch.distributions.NegativeBinomial(total_count=theta, logits=(mu / theta).log())
    torch.testing.assert_close(log_nb(x, mu, theta), reference.log_prob(x))


def test_unmeasured_counts_do_not_change_encoder_or_likelihood():
    model, batch = setup_model()
    x = torch.poisson(batch['base'])
    changed = x.clone(); changed[:, -1] = 10000
    torch.manual_seed(41); original, _ = model(x, batch)
    torch.manual_seed(41); altered, _ = model(changed, batch)
    torch.testing.assert_close(original, altered, rtol=0, atol=0)
    draws = model.generate(batch)
    assert torch.all(draws[:, -1] == 0)


def test_degree_matched_random_preserves_gene_and_module_degrees():
    rng = np.random.default_rng(3)
    matrix = (rng.random((80, 12)) < 0.25).astype(np.float32)
    families = ['a'] * 6 + ['b'] * 6
    changed = degree_matched_random(matrix, families, 4)
    assert not np.array_equal(matrix, changed)
    np.testing.assert_array_equal(matrix.sum(0), changed.sum(0))
    for columns in [slice(0, 6), slice(6, 12)]:
        np.testing.assert_array_equal(matrix[:, columns].sum(1), changed[:, columns].sum(1))


def test_joint_generation_has_learned_shared_gene_dependence():
    model, batch = setup_model()
    count = 6000
    batch = {k: v[:1].repeat((count,) + (1,) * (v.ndim - 1)) for k, v in batch.items()}
    batch['theta'][:] = 1000
    batch['centers'][:] = 0; batch['scales'][:] = 1
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.baseline_modules[0].weight[0, 0] = 1
        model.baseline_modules[-1].weight[0, 0] = 1
        model.loadings[0, 0] = model.loadings[1, 0] = 0.8
    samples = model.generate(batch).numpy()
    assert np.corrcoef(samples[:, 0], samples[:, 1])[0, 1] > 0.4
    assert abs(np.corrcoef(samples[:, 0], samples[:, 2])[0, 1]) < 0.08
    assert np.isfinite(samples).all() and np.all(samples == np.floor(samples)) and samples.min() >= 0


def test_prior_changes_decoder_and_state_changes_perturbation():
    model, batch = setup_model()
    with torch.no_grad():
        model.response_modules[-1].weight.normal_(std=0.1)
        model.state_gate[-1].weight.normal_(std=0.1)
    z = torch.randn(len(batch['target']), 3)
    first = model.decode(z, batch)[0]
    moved = model.decode(z + 1, batch)[0]
    assert not torch.allclose(first, moved)
    changed = copy.deepcopy(model)
    changed.anchor.copy_(changed.anchor.roll(3, 0))
    assert not torch.allclose(first, changed.decode(z, batch)[0])


def test_key_controls_have_equal_parameter_capacity():
    sizes = []
    for variant in ['true_prior', 'random_prior', 'no_prior']:
        model, _ = setup_model(variant)
        sizes.append(sum(p.numel() for p in model.parameters()))
    assert len(set(sizes)) == 1


def test_module_ablation_cannot_bypass_through_target_shifted_latent():
    model, batch = setup_model('no_module')
    with torch.no_grad():
        model.prior[-1].weight.normal_(std=0.2)
        model.state_gate[-1].weight.normal_(std=0.2)
    batch['base'][:, 1] = 20
    other = {k: v.clone() for k, v in batch.items()}; other['target'][:] = 1
    before = model.prior_parameters(batch)
    after = model.prior_parameters(other)
    for a, b in zip(before, after):
        torch.testing.assert_close(a, b)
    z = torch.randn(len(batch['target']), 3)
    torch.testing.assert_close(model.decode(z, batch)[0], model.decode(z, other)[0])


def test_state_ablation_has_state_invariant_conditional_mean_response():
    model, batch = setup_model('no_state')
    with torch.no_grad():
        model.response_modules[-1].weight.normal_(std=0.4)
        model.response_residual[-1].weight.normal_(std=0.4)
        model.module_interaction.normal_(std=0.3)
        model.state_gate[-1].weight.normal_(std=0.3)
    control = {k: v.clone() for k, v in batch.items()}; control['target'][:] = model.genes
    z = torch.randn(len(batch['target']), 3)
    # The perturbation is a fixed affine map of the state-varying NTC mean:
    # multiplicative shift plus an additive induction term, both state invariant.
    bases = [model.decode(z + shift, control)[0] for shift in [0, 1, 2]]
    predictions = [model.decode(z + shift, batch)[0] for shift in [0, 1, 2]]
    keep = abs(bases[1] - bases[0]) > 1e-4
    slope = (predictions[1] - predictions[0])[keep] / (bases[1] - bases[0])[keep]
    predicted_third = predictions[0][keep] + slope * (bases[2] - bases[0])[keep]
    torch.testing.assert_close(predicted_third, predictions[2][keep], atol=1e-4, rtol=1e-4)


def test_stratified_cache_keeps_rare_construct_and_batch():
    frame = pd.DataFrame({'construct': ['a'] * 100 + ['b'] * 3, 'batch': ['x'] * 99 + ['y'] + ['x', 'y', 'z']})
    selected = balanced_selection(frame, 12, np.random.default_rng(3))
    assert set(map(tuple, selected[['construct', 'batch']].to_numpy())) == set(map(tuple, frame.to_numpy()))
    assert len(selected) == 12 and selected.index.is_unique


class SmallData:
    def __init__(self):
        rng = np.random.default_rng(52)
        self.genes = [f'G{i}' for i in range(24)]
        self.gene_index = {g: i for i, g in enumerate(self.genes)}
        self.common = np.ones(24, bool)
        self.masks = {c: self.common.copy() for c in configuration()['contexts']}
        self.tasks, self.groups, self.controls, self.storage = {}, {}, {}, []
        self.forbidden = set()
        for ci, context in enumerate(self.masks):
            for batch in ['b0', 'b1']:
                for half in [0, 1]:
                    first = len(self.storage)
                    x = rng.poisson(3 + ci, size=(12, 24)).astype(np.float32)
                    self.storage.extend(x)
                    self.controls[(context, batch, half)] = np.arange(first, len(self.storage))
            for target in self.genes[:6]:
                groups = {}
                for construct, batch, n in [('guide-A|guide-B', 'b0', 30), ('guide-C|guide-D', 'b1', 3)]:
                    first = len(self.storage)
                    x = rng.poisson(4 + ci, size=(n, 24)).astype(np.float32)
                    self.storage.extend(x)
                    groups[(construct, batch)] = np.arange(first, len(self.storage))
                self.groups[(context, target)] = groups
                self.tasks[(context, target)] = np.concatenate(list(groups.values()))
        self.storage = np.asarray(self.storage)

    def read(self, ids):
        assert not set(ids) & self.forbidden, 'held_perturbation_leakage'
        return self.storage[ids].copy()

    def control(self, c, b, h):
        return self.read(self.controls[(c, b, h)])

    def task_weights(self, context, target, original=False):
        groups = self.groups[(context, target)]
        if original:
            total = sum(len(v) for v in groups.values())
            return {k: len(v) / total for k, v in groups.items()}
        return {k: 1 / len(groups) for k in groups}


def test_sampler_balances_context_target_and_construct():
    data = SmallData(); sampler = BalancedSampler(data, ['A', 'B'], 8, 20)
    rows = []
    for _ in range(600):
        rows.extend(sampler.sample(2, 4))
    counts = pd.Series([r[0] for r in rows]).value_counts()
    assert counts.A == counts.B
    assert all(not reserved(row[1], 20) for row in rows)
    batch_fraction = np.mean([row[2] == 'b0' for row in rows])
    assert 0.45 < batch_fraction < 0.55  # 30:3 cell imbalance does not become training weight.


def test_state_construction_cannot_read_held_perturbations(tmp_path):
    data = SmallData()
    data.forbidden = set(np.concatenate([ids for (c, t), ids in data.tasks.items() if c in ['D', 'E']]))
    view = FoldView(data, ['A', 'B', 'C'], configuration(), tmp_path)
    assert ('D', 'b0') in view.controls and ('E', 'b1') in view.controls
    assert np.isfinite(view.projection).all()


@pytest.mark.parametrize('device_name', ['cpu', 'cuda'])
def test_complete_resume_matches_uninterrupted_optimization(tmp_path, device_name):
    torch.set_num_threads(2)
    if device_name == 'cuda' and not torch.cuda.is_available():
        pytest.skip('native CUDA required for GPU resume check')
    torch.use_deterministic_algorithms(True)
    device = torch.device(device_name)
    data = SmallData(); config = configuration()
    view = FoldView(data, ['A', 'B', 'C'], config, tmp_path)
    model, _ = setup_model()
    model.projection.copy_(torch.tensor(view.projection)); model.origin.copy_(torch.tensor(view.origin)); model.common.fill_(1)
    model.to(device)
    sampler = BalancedSampler(data, ['A', 'B', 'C'], 37, 20)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=8)
    for step in range(3):
        train_step(model, data, view, sampler, optimizer, scheduler, config, step, device)
    path = tmp_path / 'checkpoint.pt'
    save_checkpoint(path, model, optimizer, scheduler, sampler, {'next_step': 3, 'stale': 2, 'best_epoch': 1})
    future = [train_step(model, data, view, sampler, optimizer, scheduler, config, s, device) for s in range(3, 6)]
    expected = {k: v.clone() for k, v in model.state_dict().items()}
    recovered, _ = setup_model()
    recovered.to(device)
    opt = torch.optim.AdamW(recovered.parameters(), lr=0.001)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=8)
    sampling = BalancedSampler(data, ['A', 'B', 'C'], 99, 20)
    progress = load_checkpoint(path, recovered, opt, sch, sampling, device)
    assert progress == {'next_step': 3, 'stale': 2, 'best_epoch': 1}
    replay = [train_step(recovered, data, view, sampling, opt, sch, config, s, device) for s in range(3, 6)]
    assert replay == future
    for k, value in recovered.state_dict().items():
        torch.testing.assert_close(value, expected[k], rtol=0, atol=0)
    assert sampling.context_queue == sampler.context_queue
    assert sampling.queues == sampler.queues


def test_conditional_mean_trains_mixture_weights():
    model, batch = setup_model()
    batch['centers'][:, 0] = -1; batch['centers'][:, 1] = 1
    mean = model.conditional_mean(batch)
    mean.square().mean().backward()
    gradient = model.prior[-1].weight.grad[-1]
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0


def test_batch_matching_does_not_invent_response_from_composition(tmp_path):
    data = SmallData()
    for j, batch in enumerate(['b0', 'b1']):
        profile = np.ones(24, np.float32); profile[:6] = 1 + 10 * j
        for half in [0, 1]:
            data.storage[data.controls[('A', batch, half)]] = profile
        for (construct, b), ids in data.groups[('A', 'G0')].items():
            if b == batch:
                data.storage[ids] = profile
    view = FoldView(data, ['B', 'C', 'D'], configuration(), tmp_path)
    values, _, _, _ = observed_task(data, view, 'A', 'G0')
    np.testing.assert_allclose(values['log_common'], values['reference_log_common'], atol=1e-6)
    np.testing.assert_allclose(values['original_log_common'], values['original_reference_log_common'], atol=1e-6)


def test_prior_diagnostics_never_read_held_or_reserved_perturbations(tmp_path):
    data = SmallData(); config = configuration(); config['validation_tasks'] = 4
    data.forbidden = set(np.concatenate([ids for (c, t), ids in data.tasks.items() if c in ['D', 'E'] or reserved(t, 20)]))
    rng = np.random.default_rng(4)
    membership = (rng.random((24, 4)) < 0.5).astype(np.float32)
    null = degree_matched_random(membership, ['x'] * 4, 9)
    strength = fold_evidence(data, ['A', 'B', 'C'], membership, null, config, tmp_path)
    assert np.all(strength >= 0.25) and np.all(strength <= 1)


def test_evaluation_outputs_finite_metrics_and_integer_samples(tmp_path):
    data = SmallData(); config = configuration(); config['evaluation_cells'] = 24
    view = FoldView(data, ['A', 'B', 'C'], config, tmp_path)
    model, _ = setup_model()
    metrics, prediction = evaluate_task(model, data, view, 'E', 'G1', config, save_cells=True)
    for value in metrics.values():
        if isinstance(value, float):
            assert np.isfinite(value)
    assert metrics['technical_construct_layers'] == 2
    assert prediction['cells'].dtype == np.uint32
    assert prediction['native_mean_count'].shape == (24,)
