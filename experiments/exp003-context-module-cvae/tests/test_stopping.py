"""Stopping behavior and recovery across full-task evaluation boundaries."""
import copy
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from data import BalancedSampler, validate_cache_protocol, data_spec_hash
from stopping import CoverageCosine, assess_stopping
import training
from test_protocol import SmallData, configuration, setup_model, assert_nested_equal
from state import FoldView


def stop_config():
    return dict(decay_cycles=10, patience_cycles=3, kl_warmup_steps=8,
                validation_min_delta=.001, loss_relative_range=.01, stability_epsilon=1e-8)


def point(cycle, score=.2):
    return {'cycle': cycle, 'step': cycle * 10, 'validation_score': score,
            'training_monitor': {c: {'loss': 1., 'response_loss': .2} for c in 'ABCD'}}


def test_no_stop_before_three_whole_floor_cycles():
    config = stop_config(); history = []; reference = None; stale = 0
    for cycle in range(1, 14):
        history.append(point(cycle))
        result = assess_stopping(history, reference, stale, list('ABCD'), config)
        reference, stale = result['reference'], result['stale']
        assert result['stop'] == (cycle == 13)
    assert len(result['ranges']) == 8


def test_small_improvements_accumulate_independently_of_checkpoint_highs():
    config = stop_config(); history = []; reference = None; stale = 0
    for cycle in range(10, 14):
        history.append(point(cycle, .2 + (cycle - 10) * .0004))
        result = assess_stopping(history, reference, stale, list('ABCD'), config)
        reference, stale = result['reference'], result['stale']
    assert reference == history[-1]['validation_score']
    assert stale == 0 and not result['stop']


@pytest.mark.parametrize('metric', ['loss', 'response_loss'])
def test_one_unstable_dataset_or_oscillation_blocks_stop(metric):
    history = [point(cycle) for cycle in range(10, 14)]
    history[1]['training_monitor']['D'][metric] *= 1.1
    result = assess_stopping(history, .2, 2, list('ABCD'), stop_config())
    assert not result['stable'] and not result['stop']
    # Endpoints are equal: a first-vs-last test would have hidden the oscillation.
    assert history[0]['training_monitor'] == history[-1]['training_monitor']


def test_coverage_clock_requires_every_task_each_cycle_and_restores():
    data = SmallData(); sampler = BalancedSampler(data, list('ABCD'), 12, 0)
    seen = {c: set() for c in sampler.contexts}; previous_time = 0
    for _ in range(250):
        before = sampler.completed_cycles
        for c, t, _, _ in sampler.sample(2, 4):
            seen[c].add(t)
        assert sampler.coverage_time >= previous_time
        previous_time = sampler.coverage_time
        if sampler.completed_cycles > before:
            assert all(seen[c] == set(sampler.targets[c]) for c in seen)
            seen = {c: set() for c in seen}
    state = copy.deepcopy(sampler.state_dict())
    restored = BalancedSampler(data, list('ABCD'), 99, 0); restored.load_state_dict(state)
    for _ in range(30):
        assert_nested_equal(restored.sample(2, 4), sampler.sample(2, 4))
    assert_nested_equal(restored.state_dict(), sampler.state_dict())


def test_cosine_holds_floor_without_budget_or_restart():
    model = torch.nn.Linear(1, 1); opt = torch.optim.AdamW(model.parameters(), lr=.0003)
    scheduler = CoverageCosine(opt, .0003, .00003, 10)
    values = []
    for position in np.linspace(0, 30, 121):
        scheduler.step(position); values.append(scheduler.get_last_lr()[0])
    assert all(a >= b for a, b in zip(values, values[1:]))
    assert all(value == .00003 for value in values[40:])
    recovery = CoverageCosine(opt, .0003, .00003, 10); recovery.load_state_dict(scheduler.state_dict())
    assert recovery.get_last_lr() == scheduler.get_last_lr()


def test_full_training_monitor_is_fixed_weighted_and_never_reads_held_labels(tmp_path):
    torch.set_num_threads(2)
    data = SmallData(); config = configuration(); config['unseen_target_percent'] = 0
    contexts = list('ABCD')
    data.forbidden = set(np.concatenate([v for (c, t), v in data.tasks.items() if c == 'E']))
    view = FoldView(data, contexts, config, tmp_path)
    model, _ = setup_model(); model.projection.copy_(torch.tensor(view.projection)); model.origin.copy_(torch.tensor(view.origin))
    plan = training.monitor_plan(data, contexts, config, tmp_path)
    assert {(row['context'], row['target']) for row in plan} == {(c,t) for c,t in data.tasks if c in contexts}
    assert all(abs(sum(r['weight'] for r in plan if r['context'] == c) - 1) < 1e-10 for c in contexts)
    state = copy.deepcopy(model.state_dict()); rng = training.random_state()
    first = training.monitor_training(model, data, view, plan, contexts, config, 100)
    assert_nested_equal(training.random_state(), rng)
    assert_nested_equal(model.state_dict(), state)
    assert first == training.monitor_training(model, data, view, plan, contexts, config, 100)
    assert training.monitor_plan(data, contexts, config, tmp_path) == plan
    for row in first.values():
        assert row['loss'] == pytest.approx(row['nll'] + row['kl_per_gene'] + row['response_loss'])


@pytest.mark.parametrize('failure_stage', ['first_step', 'validation'])
def test_recovery_matches_uninterrupted_cycles_and_selection(tmp_path, monkeypatch, failure_stage):
    if not torch.cuda.is_available():
        pytest.skip('native CUDA required')
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
    data = SmallData(); config = configuration()
    config.update(unseen_target_percent=0, decay_cycles=1, patience_cycles=1, kl_warmup_steps=1,
                  learning_rate=.001, minimum_learning_rate=.0001, weight_decay=.0001,
                  loss_relative_range=.01, stability_epsilon=1e-8, validation_min_delta=.001,
                  checkpoint_every_steps=2, pipeline_commit='test')
    def construct(data, view, contexts, config, prior_directory, folder):
        model, _ = setup_model()
        model.projection.copy_(torch.tensor(view.projection)); model.origin.copy_(torch.tensor(view.origin)); model.common.fill_(1)
        return model
    class Evaluator:
        def __init__(self, *args): self.targets = ['G0', 'G1', 'G2']
        def prepare(self): pass
        def promote_best(self, cycle): pass
        def clear_transient(self): pass
        def record_prediction_files(self): pass
        def evaluate(self, model, cycle):
            return {'mean_score': -float(cycle), 'score_sd': 0., 'seed_scores': [-float(cycle)] * 3}
    monkeypatch.setattr(training, 'construct_model', construct)
    monkeypatch.setattr(training, 'OfficialValidation', Evaluator)
    monkeypatch.setattr(training, 'monitor_training', lambda *args: {c: {'loss': 1., 'response_loss': .2} for c in 'ABCD'})
    tracked = SimpleNamespace(log=lambda values: None)
    def execute(folder):
        return training.fit(data, list('ABCD'), 'E', config, folder, 'holdout-E', tmp_path, tracked)
    # fit normally receives an existing run directory.
    a, b = tmp_path / 'a', tmp_path / 'b'; a.mkdir(); b.mkdir()
    execute(a)
    owner, name = (training, 'train_step') if failure_stage == 'first_step' else (Evaluator, 'evaluate')
    original = getattr(owner, name)
    def fail(*args, **kwargs): raise RuntimeError('injected interruption')
    monkeypatch.setattr(owner, name, fail)
    with pytest.raises(RuntimeError, match='injected interruption'): execute(b)
    monkeypatch.setattr(owner, name, original)
    execute(b)
    actual = torch.load(b / 'checkpoints/holdout-E-last.pt', map_location='cpu', weights_only=False)
    expected = torch.load(a / 'checkpoints/holdout-E-last.pt', map_location='cpu', weights_only=False)
    assert actual['progress']['resume_count'] == 1
    actual['progress']['resume_count'] = 0
    assert_nested_equal(actual, expected)
    assert actual['progress']['history'][-1]['cycle'] == 2
    assert actual['progress']['best_cycle'] == 1
    assert_nested_equal(torch.load(a / 'checkpoints/holdout-E-best.pt', map_location='cpu', weights_only=False),
                        torch.load(b / 'checkpoints/holdout-E-best.pt', map_location='cpu', weights_only=False))


def test_cached_reservation_exemption_cannot_hide_other_data_changes(tmp_path):
    import yaml
    root = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load((root / 'configs/default.yaml').read_text())
    source = dict(cfg, unseen_target_percent=20, pipeline_commit='fixed')
    producer = tmp_path / 'producer'; cache = producer / 'cache' / 'data'; cache.mkdir(parents=True)
    (producer / 'config.yaml').write_text(yaml.safe_dump(source))
    report = {'data_spec_hash': data_spec_hash(source)}
    validate_cache_protocol(cache, report, cfg, tmp_path / 'consumer')
    for field in ['data_seed', 'maximum_cells_per_target', 'minimum_target_cells']:
        changed = dict(cfg); changed[field] += 1
        with pytest.raises(AssertionError, match='cache_data_conditions_differ'):
            validate_cache_protocol(cache, report, changed, tmp_path / 'consumer')
