"""Prior-only changes: known graph behavior, evidence isolation and cache identity."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse
import torch
import yaml

from test_protocol import SmallData, configuration, setup_model
from priors import annotation_incidence, diffuse_membership, prepare_priors, representation_config
import training


SETTINGS = {'propagation': .5, 'steps': 8, 'minimum': .01}


def test_weighted_network_preserves_seeds_and_does_not_invent_isolated_knowledge():
    graph = sparse.csr_matrix([[0, 3, 1, 0], [3, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]])
    seeds = np.array([[0, 0], [1, 0], [0, 1], [0, 0]], np.float32)
    values = diffuse_membership(seeds, graph, dict(SETTINGS, steps=1))
    np.testing.assert_allclose(values[0], [.375, .125])
    np.testing.assert_array_equal(values[seeds == 1], 1)
    np.testing.assert_array_equal(values[3], 0)
    np.testing.assert_array_equal(diffuse_membership(seeds, graph, dict(SETTINGS, propagation=0)), seeds)


def test_annotation_operator_matches_explicit_term_balanced_off_diagonal_graph():
    terms = [('small', {0, 1}), ('large', {0, 1, 2, 3}), ('singleton', {4})]
    incidence = annotation_incidence(terms, 5)
    assert incidence.shape == (5, 2)
    # Both terms contribute one unit of row mass, independent of their size.
    graph = np.zeros((5, 5), np.float32)
    for _, members in terms[:2]:
        for i in members:
            for j in members - {i}:
                graph[i, j] += 1 / (len(members) - 1)
    seeds = np.eye(5, dtype=np.float32)[:, :3]
    actual = diffuse_membership(seeds, incidence, SETTINGS, annotation=True)
    expected = diffuse_membership(seeds, sparse.csr_matrix(graph), SETTINGS)
    np.testing.assert_allclose(actual, expected, atol=1e-7)
    assert not actual[4].any()


def test_diffusion_reaches_multihop_neighbors_without_crossing_components():
    graph = sparse.diags([np.ones(3), np.ones(3)], [-1, 1], shape=(4, 4)).tolil()
    graph[2, 3] = graph[3, 2] = 0
    seeds = np.array([[1], [0], [0], [0]], np.float32)
    first = diffuse_membership(seeds, graph, dict(SETTINGS, steps=1))
    later = diffuse_membership(seeds, graph, SETTINGS)
    assert first[2, 0] == 0 < later[2, 0] < later[1, 0] < 1
    assert later[3, 0] == 0


def test_diffusion_changes_only_fixed_model_buffers_and_keeps_NTC_zero():
    binary, batch = setup_model()
    soft, _ = setup_model(soft_prior=True)
    assert binary.state_dict().keys() == soft.state_dict().keys()
    changed = {k for k, v in binary.state_dict().items() if not torch.equal(v, soft.state_dict()[k])}
    assert changed == {'anchor', 'target_prior'}
    for (_, a), (_, b) in zip(binary.named_parameters(), soft.named_parameters()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert not soft.target_prior[-1].any()
    assert not binary.target_prior[17].any() and soft.target_prior[17].any()
    torch.manual_seed(1)
    before = binary.decode(torch.ones(12, 3), batch)[0]
    after = soft.decode(torch.ones(12, 3), batch)[0]
    assert not torch.allclose(before, after)


def test_fold_evidence_keeps_binary_modules_while_model_consumes_soft_values(tmp_path, monkeypatch):
    data = SmallData(); config = configuration()
    config.update(prior_representation='restart_diffusion', prior_diffusion_propagation=.5,
                  prior_diffusion_steps=8, prior_diffusion_minimum=.01)
    binary = np.zeros((24, 4), np.float32); binary[:12, 0] = 1
    null = np.roll(binary, 5, 0)
    soft = binary.copy(); soft[12:16, 0] = .25
    priors = tmp_path / 'priors'; priors.mkdir()
    np.savez(priors / 'modules.npz', true=binary, random=null, representation=soft)
    (priors / 'modules.json').write_text(json.dumps([{'family': 'go'}] * 4))
    calls = []
    def evidence(data, contexts, candidate, random, config, folder):
        calls.append(contexts)
        np.testing.assert_array_equal(candidate, binary)
        np.testing.assert_array_equal(random, null)
        return np.full(4, .25, np.float32)
    monkeypatch.setattr(training, 'fold_evidence', evidence)
    view = SimpleNamespace(projection=np.ones((24, 3)), origin=np.zeros(3))
    model = training.construct_model(data, view, ['A', 'B', 'C'], config, priors, tmp_path)
    assert calls == [['A', 'B', 'C']]
    assert model.target_prior[12, 0] == .25
    np.testing.assert_array_equal(model.target_prior[16].numpy(), 0)


def test_cache_rejects_a_different_representation_before_reuse(tmp_path):
    cache = tmp_path / 'cache/priors'; cache.mkdir(parents=True)
    (cache / 'complete.json').write_text(json.dumps({'artifacts': {}, 'representation': {'method': 'binary'}}))
    config = dict(variant='true_prior', prior_representation='restart_diffusion',
                  prior_diffusion_propagation=.5, prior_diffusion_steps=8, prior_diffusion_minimum=.01)
    with pytest.raises(AssertionError, match='prior_representation_changed'):
        prepare_priors(config, [], tmp_path)
    assert prepare_priors({}, [], tmp_path) == cache
    with pytest.raises(ValueError, match='diffusion_requires_true_prior'):
        representation_config(dict(config, variant='random_prior'))


def test_new_configuration_changes_only_the_prior_representation():
    folder = Path(__file__).resolve().parents[1] / 'configs'
    baseline = yaml.safe_load((folder / 'default.yaml').read_text())
    candidate = yaml.safe_load((folder / 'prior-diffusion.yaml').read_text())
    assert {k: v for k, v in candidate.items() if k in baseline} == baseline
    assert set(candidate) - set(baseline) == {
        'prior_representation', 'prior_diffusion_propagation', 'prior_diffusion_steps', 'prior_diffusion_minimum'}
