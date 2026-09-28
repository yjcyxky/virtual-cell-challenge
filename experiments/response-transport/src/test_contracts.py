"""Behavior checks for biological boundaries, analytic moments and full resume."""
import copy
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from scipy import sparse
import model
from vcc_mechanism.inputs import accumulate_moments, matching_populations
from vcc_mechanism.population import composition_counts, negative_binomial_counts, dispersion


SPEC = json.loads((Path(__file__).resolve().parents[1]/'configs/transport-mlp-s01.json').read_text())['transport']


def test_zero_response_anchor_and_ntc_interactions():
    net = model.Transport(SPEC)
    torch.nn.init.normal_(net.net[-1].weight)
    one = torch.ones(8)
    x = model.features(torch.zeros(8), one, one*2, one, one, one, one, one*100, one, SPEC)
    torch.testing.assert_close(net(x), torch.zeros(8), rtol=0, atol=0)
    x = model.features(one*.1, one, one*2, one, one, one, one, one*100, one, SPEC)
    y = model.features(one*.1, one, one*4, one, one, one, one, one*100, one, SPEC)
    assert not torch.equal(net(x), net(y))
    # Public feature constructor has no destination perturbed labels argument.
    assert x.shape == (8, 17)


def test_population_reuse_accepts_decoder_change_rejects_data_or_eval():
    c = json.loads((Path(__file__).resolve().parents[1]/'configs/transport-mlp-s01.json').read_text())
    changed = copy.deepcopy(c); changed['generation']['kind'] = 'different_registered_decoder'
    matching_populations(c, changed)
    changed['data']['target_sum'] = 50000
    with pytest.raises(ValueError, match='data'):
        matching_populations(c, changed)
    changed = copy.deepcopy(c); changed['evaluation']['runtime']['device'] = 'cpu'
    with pytest.raises(ValueError, match='Evaluation'):
        matching_populations(c, changed)


def test_moments_are_per_cell_cpm_not_pooled_counts():
    raw = sparse.csr_matrix([[1., 1.], [90., 10.], [2., 0.]])
    sums, squares, logs = [np.zeros((2, 2)) for _ in range(3)]
    inv, counts = np.zeros(2), np.zeros(2, dtype=int)
    accumulate_moments(sums, squares, logs, inv, counts, raw, np.array([0,0,1]), 10000)
    np.testing.assert_allclose(sums/counts[:, None], [[7000,3000],[10000,0]])
    np.testing.assert_allclose(logs/counts[:, None], [[(np.log1p(5000)+np.log1p(9000))/2,
                                                     (np.log1p(5000)+np.log1p(1000))/2], [np.log1p(10000),0]])
    np.testing.assert_allclose(inv/counts, [.255,.5])


def test_decoder_can_activate_template_zeros_and_nb_has_overdispersion():
    raw = np.tile([1000., 0.], (20000,1)); profile = np.array([.7,.3])
    generated, error = composition_counts(raw[:400], profile, np.random.default_rng(1), {'template_smoothing':.01,'ipf_iterations':64})
    assert error < 1e-10
    assert generated[:, 1].mean() > 295
    np.testing.assert_allclose(generated.sum(1), 1000, atol=1)
    counts, _ = negative_binomial_counts(raw, profile, np.array([.2,0]), np.random.default_rng(1), {'poisson_threshold':1e-8})
    np.testing.assert_allclose(counts.mean(0), [700,300], rtol=.02)
    assert counts[:,0].var() > 80000 and counts[:,1].var() < 400


def test_single_cell_dispersion_uses_ntc_prior():
    s = {'mean':np.array([[10.,0.],[20.,0.]]),'variance':np.array([[100.,0.],[0.,0.]]),
         'inverse_library':np.array([.0001,.0001]),'counts':np.array([100,1])}
    phi = dispersion(s, {'target_sum':10000,'dispersion_max':10,'dispersion_prior_cells':50})
    np.testing.assert_allclose(phi, [[.9,0],[.9,0]])


def test_resume_restores_network_optimizer_sampler_and_torch_rng(tmp_path):
    torch.manual_seed(91)
    net = model.Transport(SPEC); optimizer = model.optimizer_for(net, SPEC); rng = np.random.default_rng(4)
    config = {'transport':SPEC,'test':'resume'}
    def step(n, opt, random):
        x = torch.tensor(random.normal(size=(32,17)), dtype=torch.float32)
        y = torch.rand(32)
        opt.zero_grad(set_to_none=True); ((n(x)-y)**2).mean().backward(); opt.step()
    step(net, optimizer, rng)
    path = tmp_path/'state.pt'; model.save_state(path, net, optimizer, rng, 1, config)
    step(net, optimizer, rng)
    expected = copy.deepcopy(net.state_dict())
    restored = model.Transport(SPEC); opt = model.optimizer_for(restored, SPEC); random = np.random.default_rng(999)
    assert model.restore_state(path, restored, opt, random, config) == 1
    step(restored, opt, random)
    for k,v in restored.state_dict().items():
        torch.testing.assert_close(v, expected[k], rtol=0, atol=0)


def test_prediction_uses_only_available_donors_and_zero_for_unseen(tmp_path):
    genes = ['G','T','U']
    np.savez(tmp_path/'A-statistics.npz', positions=[0,1], labels=['non-targeting','T'],
             mean=np.array([[1,2],[1.2,1.6]],dtype=np.float32), counts=[50,100])
    np.savez(tmp_path/'B-statistics.npz', positions=[1,2], labels=['non-targeting','T'],
             mean=np.array([[2,1],[1.9,1.6]],dtype=np.float32), counts=[50,200])
    data = model.context_data(tmp_path,['A','B'],genes,'cpu')
    m = dict(network=model.Transport(SPEC), data=data, genes=genes, spec=SPEC)
    s = dict(positions=np.array([0,1,2]), mean=np.array([[3,2,1]],dtype=np.float32))
    result = model.predict(m,s,['T','new'],'linear')
    np.testing.assert_allclose(result, [[.2,-.2,.6],[0,0,0]], atol=1e-6)
