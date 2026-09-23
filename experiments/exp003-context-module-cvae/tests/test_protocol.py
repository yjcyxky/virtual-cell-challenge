"""Behavioral checks for joint counts, leakage boundaries, balancing and resume."""
from pathlib import Path
import copy
import json
import sys
import subprocess
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from data import BalancedSampler, CountData, balanced_selection, reserved
from model import ModuleCVAE, log_nb
from priors import degree_matched_random
from state import FoldView
from training import save_checkpoint, load_checkpoint, train_step
from evaluation import observed_task, evaluate_task, evaluate_context
from runtime import verify_environment
from priors import fold_evidence
import training
from comparison import paired_difference
import comparison
import main as entrypoint


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


def test_encoder_distinguishes_missing_gene_identity_from_observed_zero():
    model, batch = setup_model()
    with torch.no_grad():
        model.posterior[-1].weight.normal_(std=0.3)
    batch['target'][:] = model.genes
    x = torch.poisson(batch['base']); x[:, 0] = 0; x[:, -1] = 0
    alternative = {k: v.clone() for k, v in batch.items()}
    alternative['mask'][:, 0] = 0; alternative['mask'][:, -1] = 1
    outputs = []
    hook = model.posterior.register_forward_hook(lambda module, inputs, output: outputs.append(output.detach().clone()))
    model(x, batch); model(x, alternative); hook.remove()
    assert not torch.allclose(outputs[0], outputs[1])


def test_context_readouts_require_safe_identity_in_reference_NTC(tmp_path):
    genes = ['G0', 'G1', 'G2', 'G3']
    (tmp_path / 'genes.json').write_text(json.dumps(genes))
    panels = []
    for i, n in enumerate([3, 4]):
        folder = tmp_path / f'panel-{i}'; folder.mkdir()
        panels.append({'index': i, 'context': 'H1', 'panel': f'H1:{i}'})
        np.save(folder / 'genes.npy', np.arange(n))
        np.save(folder / 'counts.npy', np.stack([np.arange(n) + 1, np.arange(n) + 11]).astype(np.uint16))
        pd.DataFrame({'panel_index': [i]*2, 'context': ['H1']*2, 'cache_row': [0, 1],
                      'is_NTC': [i == 0]*2, 'target': ['__NTC__' if i == 0 else 'G0']*2,
                      'batch': ['b0']*2, 'half': [0, 1], 'construct': ['one-dual-guide']*2}).to_parquet(folder / 'cells.parquet')
    (tmp_path / 'complete.json').write_text(json.dumps({'panels': panels}))
    pd.DataFrame({'target': [], 'context': []}).to_parquet(tmp_path / 'factor-weights.parquet')
    data = CountData(tmp_path)
    assert data.masks['H1'].tolist() == [True, True, True, False]
    assert data.readout_support[1]['unmatched_safe_readouts_excluded'] == ['G3']
    assert np.all(data.read([2, 3])[:, -1] == 0)
    np.testing.assert_array_equal(np.load(tmp_path / 'panel-1' / 'counts.npy')[:, -1], [4, 14])
    np.testing.assert_array_equal(data.read([3, 0, 2, 3]), [[11, 12, 13, 0], [1, 2, 3, 0], [1, 2, 3, 0], [11, 12, 13, 0]])


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


def test_context_ablation_has_context_invariant_dispersion_response():
    model, batch = setup_model('no_context')
    with torch.no_grad():
        model.dispersion[-1].weight.normal_(std=0.3)
        model.dispersion_loadings.normal_(std=0.3)
    altered = {k: v.clone() for k, v in batch.items()}
    altered['condition'] += 2
    altered['base'] *= 4
    z = torch.randn(len(batch['target']), 3)
    ratios = []
    for inputs in [batch, altered]:
        control = {k: v.clone() for k, v in inputs.items()}
        control['target'][:] = model.genes
        ratios.append(model.decode(z, inputs)[1] / model.decode(z, control)[1])
    torch.testing.assert_close(ratios[0], ratios[1], rtol=0, atol=0)


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
    model, _ = setup_model()
    config = configuration(); config['evaluation_cells'] = 24
    metrics, _ = evaluate_task(model, data, view, 'A', 'G0', config)
    assert metrics['ntc_sample_response_mse'] < 1e-10
    assert metrics['ntc_sample_response_mae'] < 1e-5


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


def test_saved_generated_cells_carry_available_and_trained_readout_masks(tmp_path):
    data = SmallData(); config = configuration(); config['evaluation_cells'] = 24
    data.masks['E'][-1] = False
    view = FoldView(data, ['A', 'B', 'C'], config, tmp_path)
    model, _ = setup_model(); model.trained_readouts[-2] = False
    evaluate_context(model, data, view, 'E', config, tmp_path, {}, np.zeros(data.common.sum()))
    saved = np.load(tmp_path / 'example-0.npz')
    np.testing.assert_array_equal(saved['available_readout_mask'], data.masks['E'])
    np.testing.assert_array_equal(saved['trained_readout_mask'], model.trained_readouts.numpy())
    assert not saved['available_readout_mask'][-1]
    assert not saved['trained_readout_mask'][-2]


def test_evaluation_reads_only_selected_NTC_and_preserves_sample_order(tmp_path):
    data = SmallData(); config = configuration(); config['evaluation_cells'] = 24
    view = FoldView(data, ['A', 'B', 'C'], config, tmp_path)
    model, _ = setup_model()
    reads = []
    original_read = data.read
    def record_read(ids):
        reads.append(np.asarray(ids).copy())
        return original_read(ids)
    data.read = record_read
    metrics, _ = evaluate_task(model, data, view, 'E', 'G1', config)
    assert [len(ids) for ids in reads] == [33, 24]
    # The fixture has two equally weighted construct/batch layers, with 30 and
    # three observed cells. Reconstruct the declared population sampling law.
    from evaluation import task_seed
    rng = np.random.default_rng(task_seed('E', 'G1'))
    rng.choice(33, 24, p=np.r_[np.full(30, 0.5 / 30), np.full(3, 0.5 / 3)])
    rng.choice(24, 24, p=np.full(24, 1 / 24))
    first, second = rng.integers(12, size=12), rng.integers(12, size=12)
    expected = np.r_[data.controls[('E', 'b0', 0)][first], data.controls[('E', 'b1', 0)][second]]
    np.testing.assert_array_equal(reads[-1], expected)
    assert np.isfinite(metrics['ntc_energy_distance'])


def test_environment_verification_rejects_changed_lock_or_package_metadata(tmp_path):
    record = tmp_path / 'environment.json'
    expected = {'uv_lock_sha256': 'locked', 'packages': {'torch': {'version': 'fixed', 'metadata': 'known'}}}
    with pytest.raises(ValueError, match='not_yet_synchronized'):
        verify_environment(record, expected)
    record.write_text(json.dumps(expected))
    verify_environment(record, expected)
    for changed in [dict(expected, uv_lock_sha256='different'),
                    dict(expected, packages={'torch': {'version': 'fixed', 'metadata': 'altered'}})]:
        with pytest.raises(ValueError, match='environment_changed'):
            verify_environment(record, changed)


def test_preprocessing_reuse_checks_fold_and_never_reuses_model_weights(tmp_path, monkeypatch):
    monkeypatch.setattr(training, 'EXPERIMENT', tmp_path)
    monkeypatch.setattr(training, 'preprocessing_identity', lambda commit: {'preparation': 'same'})
    source = tmp_path / 'outputs' / 'source'; source.mkdir(parents=True)
    config = yaml.safe_load((Path(__file__).resolve().parents[1] / 'configs/default.yaml').read_text())
    config.update(pipeline_commit='fixed-code', uv_lock_sha256='fixed-lock')
    (source / 'config.yaml').write_text(yaml.safe_dump(config))
    (source / 'artifact.json').write_text(json.dumps({'name': 'fixed-model:v0', 'digest': 'fixed-digest'}))
    folder = source / 'cache' / 'heldout'; folder.mkdir(parents=True)
    (folder / 'split.json').write_text(json.dumps({'training_targets_by_context': {'A': ['G1'], 'B': ['G2']}}))
    for name in ['state.npz', 'shared-response.npz', 'prior-strength.npy', 'prior-evidence.parquet', 'evidence-scope.json']:
        (folder / name).write_bytes(name.encode())
    (folder / 'model.pt').write_bytes(b'must-not-reuse')
    destination = tmp_path / 'new'; destination.mkdir()
    config.update(reuse_run='source', seed=29)
    with pytest.raises(AssertionError, match='fold_mismatch'):
        training.reuse_preprocessing(config, destination, 'heldout', ['D', 'E'])
    training.reuse_preprocessing(config, destination, 'heldout', ['A', 'B'])
    assert (destination / 'shared-response.npz').is_symlink()
    assert not (destination / 'model.pt').exists()
    assert (destination / 'state.npz').read_bytes() == b'state.npz'
    (folder / 'state.npz').write_bytes(b'unexpected-change')
    with pytest.raises(AssertionError):
        training.reuse_preprocessing(config, destination, 'heldout', ['A', 'B'])


def assert_nested_equal(actual, expected):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(expected, np.ndarray):
        np.testing.assert_array_equal(actual, expected)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_nested_equal(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected):
            assert_nested_equal(a, b)
    else:
        assert actual == expected


@pytest.mark.parametrize('failure_stage', ['first_step', 'epoch_validation'])
def test_fit_resume_replays_incomplete_epoch_exactly(tmp_path, monkeypatch, failure_stage):
    if not torch.cuda.is_available():
        pytest.skip('native CUDA required for full fit recovery')
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    data = SmallData(); config = configuration()
    config.update(epochs=2, minimum_epochs=2, patience=10, learning_rate=0.001, weight_decay=0.0001,
                  validation_tasks=3, evaluation_cells=24, checkpoint_every_steps=2, pipeline_commit='test')
    def construct(data, view, contexts, config, prior_directory, folder):
        model, _ = setup_model()
        model.projection.copy_(torch.tensor(view.projection)); model.origin.copy_(torch.tensor(view.origin))
        model.common.fill_(1)
        return model
    monkeypatch.setattr(training, 'construct_model', construct)
    tracked = SimpleNamespace(log=lambda values: None)
    def execute(folder):
        return training.fit(data, ['A', 'B', 'C'], 'D', config, folder, 'heldout', tmp_path, tracked)
    uninterrupted, interrupted = tmp_path / 'uninterrupted', tmp_path / 'interrupted'
    execute(uninterrupted)
    original = training.train_step if failure_stage == 'first_step' else training.evaluate_context
    name = 'train_step' if failure_stage == 'first_step' else 'evaluate_context'
    def fail(*args, **kwargs):
        raise RuntimeError('injected interruption')
    monkeypatch.setattr(training, name, fail)
    with pytest.raises(RuntimeError, match='injected interruption'):
        execute(interrupted)
    interrupted_state = torch.load(interrupted / 'checkpoints/heldout-last.pt', map_location='cpu', weights_only=False)
    assert interrupted_state['progress']['next_step'] == (0 if failure_stage == 'first_step' else 2)
    monkeypatch.setattr(training, name, original)
    execute(interrupted)
    expected = torch.load(uninterrupted / 'checkpoints/heldout-last.pt', map_location='cpu', weights_only=False)
    actual = torch.load(interrupted / 'checkpoints/heldout-last.pt', map_location='cpu', weights_only=False)
    assert actual['progress']['resume_count'] == 1
    actual['progress']['resume_count'] = 0
    assert_nested_equal(actual, expected)
    selected = torch.load(interrupted / 'checkpoints/heldout-best.pt', map_location='cpu', weights_only=False)
    expected_selected = torch.load(uninterrupted / 'checkpoints/heldout-best.pt', map_location='cpu', weights_only=False)
    assert_nested_equal(selected, expected_selected)
    seen, coverage = comparison.selected_exposure(data, interrupted, config, 'heldout')
    assert seen == set.union(*(set(v) for v in selected['sampled_training_targets'].values()))
    assert all(row['replay_matches_checkpoint'] for row in coverage)


def test_paired_comparison_balances_contexts_and_rejects_missing_tasks():
    identity = pd.DataFrame({'seed': [17]*4, 'context': ['A', 'A', 'A', 'B'], 'target': ['x', 'y', 'z', 'x']})
    reference = identity.assign(response_mse=[2., 2., 2., 8.], response_mae=1., energy_distance=2., original_weight_mse=2.)
    model = identity.assign(response_mse=[1., 1., 1., 2.], response_mae=.5, energy_distance=1., original_weight_mse=1.)
    result = paired_difference(model, reference.iloc[::-1], 100, 42)
    assert result['mean_reference_minus_model']['response_mse'] == 3.5
    assert result['seed_sample_sd'] is None
    json.dumps(result, allow_nan=False)
    with pytest.raises(AssertionError, match='unpaired_tasks'):
        paired_difference(model, reference.iloc[:-1], 100, 42)


def test_trial_comparison_preserves_sources_and_labels_realized_exposure(tmp_path, monkeypatch):
    monkeypatch.setattr(comparison, 'EXPERIMENT', tmp_path)
    monkeypatch.setattr(comparison, 'selected_exposure', lambda *args: ({'x'}, [{'replay_matches_checkpoint': True}]))
    config = yaml.safe_load((Path(__file__).resolve().parents[1] / 'configs/default.yaml').read_text())
    config.update(pipeline_commit='same', uv_lock_sha256='lock', gene_axis_sha256='axis',
                  source_collection_sha256='collection', prior_source_sha256='prior', bootstrap_repeats=10)
    variants = [(v, s) for v in ['true_prior', 'random_prior', 'no_prior'] for s in [17, 29, 43]]
    variants += [(v, 17) for v in ['no_state', 'no_residual', 'no_module', 'no_context']]
    columns = ['response_mse', 'response_mae', 'zero_mse', 'zero_mae', 'shared_mse', 'shared_mae',
               'energy_distance', 'ntc_energy_distance', 'original_weight_mse', 'original_weight_zero_mse',
               'depth_mean_ratio', 'detection_mae', 'covariance_mse', 'state_composition_l1',
               'projected_quantile_mae', 'native_response_mse', 'native_zero_mse',
               'log_mean_raw_count_mse', 'log_mean_raw_count_zero_mse']
    sources = []
    for variant, seed in variants:
        run_id = f'{variant}-{seed}'
        folder = tmp_path / 'outputs' / run_id; folder.mkdir(parents=True)
        actual = dict(config, variant=variant, seed=seed, run_id=run_id)
        rows = []
        for context in config['contexts']:
            for target in ['x', 'y']:
                row = dict.fromkeys(columns, 1.)
                row.update(context=context, target=target, variant=variant, seed=seed,
                           response_mse=1. if variant == 'true_prior' else 2.,
                           reserved_target=target == 'y', target_seen_in_training=target == 'x')
                rows.append(row)
        frame = pd.DataFrame(rows)
        if variant == 'no_context':
            frame['ntc_sample_response_mse'] = .7; frame['ntc_sample_response_mae'] = .5
            actual['comparison_sources'] = comparison.comparison_sources(sources)
        else:
            (folder / 'complete.json').write_text(json.dumps({'status': 'completed', 'artifact': {'name': run_id + ':v0'}}))
            sources.append(run_id)
        frame.to_parquet(folder / 'evaluation-metrics.parquet')
        (folder / 'config.yaml').write_text(yaml.safe_dump(actual))
    priors = folder / 'cache/priors'; priors.mkdir(parents=True)
    np.savez(priors / 'modules.npz', true=np.array([[1.], [0.]]))
    result = comparison.compare_trials(SimpleNamespace(genes=['x', 'y']), actual, folder)
    assert result['run_count'] == 13
    assert result['comparisons']['true_prior_vs_random_prior']['mean_reference_minus_model']['response_mse'] == 1.
    combined = pd.read_parquet(folder / 'comparison-tasks.parquet')
    assert combined.actually_sampled_at_selected_checkpoint.equals(combined.target.eq('x'))
    assert combined.ntc_sample_response_mse.eq(.7).all()
    first = tmp_path / 'outputs' / sources[0] / 'evaluation-metrics.parquet'
    original = pd.read_parquet(first)
    assert 'ntc_sample_response_mse' not in original
    original['zero_mse'] = 99
    original.to_parquet(first)
    with pytest.raises(AssertionError, match='comparison_source_changed'):
        comparison.compare_trials(SimpleNamespace(genes=['x', 'y']), actual, folder)


def test_preprocessing_identity_allows_unrelated_fixes_but_rejects_changed_preparation(tmp_path, monkeypatch):
    monkeypatch.setattr(training, 'ROOT', tmp_path)
    experiment = tmp_path / 'experiments/exp003-context-module-cvae'
    monkeypatch.setattr(training, 'EXPERIMENT', experiment)
    paths = [experiment / 'src' / f'{name}.py' for name in ['data', 'state', 'priors']]
    paths += [tmp_path / 'scripts/dossier/rna.py', tmp_path / 'experiments/exp001-context-pair-xgb/src/features.py']
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True); path.write_text('value = 1\n')
    evaluation = experiment / 'src/evaluation.py'
    evaluation.write_text('import numpy as np\ndef response_summary(): return 1\ndef shared_responses(): return 2\ndef evaluate_task(): return 3\n')
    (experiment / 'src/training.py').write_text('def construct_model(): return 1\n')
    def git(*args):
        return subprocess.check_output(['git', '-c', 'user.name=Protocol Test', '-c', 'user.email=protocol@example.invalid', *args], cwd=tmp_path, text=True).strip()
    git('init', '-q'); git('add', '.')
    git('commit', '-qm', 'original preparation'); first = git('rev-parse', 'HEAD')
    evaluation.write_text(evaluation.read_text().replace('evaluate_task(): return 3', 'evaluate_task(): return 4'))
    (experiment / 'src/model.py').write_text('fixed_ablation = True\n')
    git('add', '.'); git('commit', '-qm', 'unrelated model and evaluation fixes'); second = git('rev-parse', 'HEAD')
    assert training.preprocessing_identity(first) == training.preprocessing_identity(second)
    evaluation.write_text(evaluation.read_text().replace('shared_responses(): return 2', 'shared_responses(): return 5'))
    git('add', '.'); git('commit', '-qm', 'changed baseline'); third = git('rev-parse', 'HEAD')
    assert training.preprocessing_identity(first) != training.preprocessing_identity(third)


def test_completion_summary_uses_the_installed_wandb_API():
    from wandb.sdk.wandb_summary import Summary
    updates = []
    summary = Summary(lambda: {})
    summary._set_update_callback(updates.append)
    entrypoint.completion_summary(SimpleNamespace(summary=summary),
                                 {'macro': {'response_mse': .2}, 'mse_improvement_vs_zero_fraction': -.5},
                                 {'status': 'verified_online'})
    values = {item.key[0]: item.value for record in updates for item in record.update}
    assert values['pipeline_status'] == 'completed'
    assert values['artifact_status'] == 'verified_online'
    assert values['macro_response_mse'] == .2


def test_completion_recovery_rejects_changed_archived_files(tmp_path, monkeypatch):
    import wandb
    import base64
    import hashlib
    path = tmp_path / 'model.pt'; path.write_bytes(b'unchanged-model')
    entry = SimpleNamespace(digest=base64.b64encode(hashlib.md5(path.read_bytes()).digest()).decode())
    remote = SimpleNamespace(digest='fixed', version='v0', manifest=SimpleNamespace(entries={'model.pt': entry}))
    monkeypatch.setattr(wandb, 'Api', lambda: SimpleNamespace(artifact=lambda *args, **kwargs: remote))
    delivery = {'status': 'verified_online', 'name': 'model:v0', 'digest': 'fixed', 'version': 'v0', 'files': 1}
    entrypoint.verify_remote_artifact(tmp_path, delivery)
    path.write_bytes(b'changed')
    with pytest.raises(AssertionError, match='archived_result_changed'):
        entrypoint.verify_remote_artifact(tmp_path, delivery)


@pytest.mark.parametrize('fit_complete', [True, False])
def test_finalization_retry_preserves_training_identity_and_rejects_unfinished_fit(tmp_path, monkeypatch, fit_complete):
    import wandb
    import runtime
    from wandb.sdk.wandb_summary import Summary
    monkeypatch.setattr(entrypoint, 'EXPERIMENT', tmp_path)
    monkeypatch.setattr(entrypoint, 'committed_pipeline', lambda: 'bookkeeping-fix')
    monkeypatch.setattr(runtime, 'environment_identity', lambda: {'locked': True})
    monkeypatch.setattr(entrypoint, 'verify_remote_artifact', lambda *args: None)
    output = tmp_path / 'outputs' / 'original'; (output / 'checkpoints').mkdir(parents=True)
    config = {'run_id': 'original', 'experiment_id': tmp_path.name, 'variant': 'true_prior', 'contexts': ['A'],
              'pipeline_commit': 'original-training', 'environment_identity': {'locked': True}}
    config_path = output / 'config.yaml'; config_path.write_text(yaml.safe_dump(config))
    original_config = config_path.read_bytes()
    (output / 'metrics.json').write_text(json.dumps({'status': 'trained_and_locally_evaluated', 'contexts': 1,
          'official_submission': False, 'macro': {'response_mse': .2}, 'mse_improvement_vs_zero_fraction': -.5}))
    (output / 'artifact.json').write_text(json.dumps({'status': 'verified_online'}))
    (output / 'failure.json').write_text('{"error":"summary update failed"}')
    prediction = output / 'predictions/holdout-A'; prediction.mkdir(parents=True)
    (prediction / 'complete.json').write_text('{}')
    for key in ['holdout-A', 'final']:
        torch.save({'model': {}, 'optimizer': {}, 'scheduler': {}, 'sampler': {}, 'random': {},
                    'progress': {'complete': fit_complete}}, output / 'checkpoints' / f'{key}-last.pt')
    tracked = SimpleNamespace(summary=Summary(lambda: {}), log=lambda values: None, finish=lambda **kwargs: None)
    init_arguments = []
    def initialize(**kwargs):
        init_arguments.append(kwargs); return tracked
    monkeypatch.setattr(wandb, 'init', initialize)
    if not fit_complete:
        with pytest.raises(AssertionError, match='cannot_finalize_incomplete_fit'):
            entrypoint.finalize_run('original')
        assert not init_arguments and not (output / 'complete.json').exists()
        return
    entrypoint.finalize_run('original')
    complete = json.loads((output / 'complete.json').read_text())
    assert complete['training_commit'] == 'original-training'
    assert complete['finalization_commit'] == 'bookkeeping-fix'
    assert complete['optimizer_steps_repeated'] == 0
    assert complete['original_failure_record_preserved']
    assert init_arguments[0]['id'] == 'original' and init_arguments[0]['resume'] == 'must'
    assert config_path.read_bytes() == original_config
