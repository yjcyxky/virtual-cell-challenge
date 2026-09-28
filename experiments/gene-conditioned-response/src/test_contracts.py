import copy
import json
from pathlib import Path
import numpy as np
import torch
import model
from vcc_mechanism.learning import save_state, restore_state, optimizer_for

SPEC=json.loads((Path(__file__).resolve().parents[1]/'configs/gene-decoder-s01.json').read_text())['decoder']


def network():
    rng=np.random.default_rng(1)
    return model.GeneDecoder(rng.normal(size=(4,5)),np.ones((5,4)),np.ones((5,4)),[3,0],np.ones((5,2)),SPEC)


def test_source_residual_anchor_and_missing_readout_inference():
    net=network().eval()
    # Identical NTC states give identical fitted functions, so the residual
    # construction must exactly preserve the observed source response.
    data={'positions':torch.tensor([0,1]),'targets':torch.tensor([1]),'labels':np.array(['T']),
          'delta':torch.tensor([[.2,-.3]]),'counts':np.array([50])}
    m=dict(network=net,spec=SPEC,training=[data],targets=['non-targeting','T'],genes=['T','G','missing'])
    y=model.predict(m,{},['T','new'],'linear')
    np.testing.assert_allclose(y[0,:2],[.2,-.3],atol=1e-7)
    with torch.no_grad():
        expected=float(net(torch.tensor([1]),torch.tensor([2]),4)[0])
    np.testing.assert_allclose(y[0,2],expected,rtol=1e-5,atol=1e-8)
    assert expected!=0
    assert np.isfinite(y[1]).all() and np.any(y[1]!=0)
    assert net.target_positions.tolist()==[3,0]  # Inference never mutates buffers.


def test_zero_ntc_and_no_free_readout_parameters():
    net=network(); genes=torch.arange(3)
    torch.testing.assert_close(net(torch.zeros(3,dtype=torch.long),genes,0),torch.zeros(3))
    # Adding a new output gene only extends frozen feature/state buffers.
    before=sum(p.numel() for p in net.parameters())
    larger=model.GeneDecoder(np.ones((20,5)),np.ones((5,20)),np.ones((5,20)),[19,0],np.ones((5,2)),SPEC)
    assert sum(p.numel() for p in larger.parameters())==before


def test_full_training_resume_restores_sampling_and_optimizer(tmp_path):
    torch.manual_seed(20); net=network(); opt=optimizer_for(net,SPEC); rng=np.random.default_rng(11)
    def step(n,o,r):
        genes=torch.tensor(r.integers(0,3,32)); target=torch.ones(32,dtype=torch.long)
        truth=torch.rand(32); o.zero_grad(); ((n(target,genes,0)-truth)**2).mean().backward();o.step()
    step(net,opt,rng); path=tmp_path/'state.pt'; save_state(path,net,opt,rng,1,SPEC)
    step(net,opt,rng); expected=copy.deepcopy(net.state_dict())
    restored=network(); opt2=optimizer_for(restored,SPEC); rng2=np.random.default_rng(99)
    assert restore_state(path,restored,opt2,rng2,SPEC)==1
    step(restored,opt2,rng2)
    for k,v in restored.state_dict().items(): torch.testing.assert_close(v,expected[k],rtol=0,atol=0)
