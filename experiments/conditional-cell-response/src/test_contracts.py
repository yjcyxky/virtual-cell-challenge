import copy
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import pytest
from scipy import sparse
import model
from vcc_mechanism.cell_inputs import choose_cells
from vcc_mechanism.learning import save_state,restore_state,optimizer_for

SPEC=json.loads((Path(__file__).resolve().parents[1]/'configs/cell-cvae-aligned-s01.json').read_text())['cvae']


def test_ntc_encoder_handles_missing_heldout_input_without_shrinking_output():
    training=[{'positions':np.array([0,1,2]),'labels':['non-targeting','T']}] * 4
    heldout={'positions':np.array([1,2,3]),'labels':['non-targeting']}
    np.testing.assert_array_equal(model.encoder_positions(training+[heldout]),[1,2])
    np.testing.assert_array_equal(heldout['positions'],[1,2,3])
    with pytest.raises(ValueError,match='only input NTC'):
        model.encoder_positions(training+[dict(heldout,labels=['non-targeting','forbidden'])])


def network():
    return model.CellVAE(np.ones((5,6)),np.ones((5,6))*.2,np.ones((5,3)),np.zeros((4,6)),SPEC)


def test_likelihood_pool_excludes_scoring_NTC_and_disallowed_targets():
    frame=pd.DataFrame({'target':['non-targeting']*5+['T']*10+['excluded']*10,'ntc_pool':['input']*3+['score']*2+['none']*20,
                        'eligible':[True]*25,'source_row':range(25)})
    selected=choose_cells(frame,['non-targeting','T'],1,4,2)
    assert selected.target.value_counts().to_dict()=={'T':4,'non-targeting':2}
    assert not set(selected.index)&{3,4,*range(15,25)}
    pd.testing.assert_frame_equal(selected,choose_cells(frame,['non-targeting','T'],1,4,2))


def test_nb_likelihood_matches_torch_distribution():
    x=torch.tensor([[0.,1.,4.],[1.,2.,3.]])
    mean=torch.tensor([[1.,2.,3.],[2.,3.,4.]]);phi=torch.tensor([.1,.2,.3])
    dist=torch.distributions.NegativeBinomial(total_count=1/phi,logits=torch.log(mean*phi))
    torch.testing.assert_close(model.nb_nll(x,mean,phi),-dist.log_prob(x).mean(),rtol=1e-5,atol=1e-5)


def test_generation_uses_prior_and_masks_untrained_output_weights():
    net=network().eval(); targets=torch.ones(2,dtype=torch.long);z=torch.randn(2,SPEC['latent']);pos=torch.arange(6)
    cond=net.condition(targets,4); libraries=torch.ones(2)*100
    a=net.decode(z,cond,targets,4,pos,libraries,torch.zeros(6))
    with torch.no_grad():
        net.mean_genes.fill_(100);net.residual_genes.fill_(100);net.log_dispersion.fill_(3)
    b=net.decode(z,cond,targets,4,pos,libraries,torch.zeros(6))
    for x,y in zip(a,b):torch.testing.assert_close(x,y)
    net.posterior_sample=lambda *args: (_ for _ in ()).throw(AssertionError('posterior leakage'))
    m=dict(network=net,targets=['non-targeting','T','U','V'],genes=list('ABCDEF'),union=np.zeros(6,dtype=bool),spec=SPEC)
    counts=model.draw(m,sparse.csr_matrix(np.ones((10,6))*100),np.arange(6),'T',1,128,{'data':{'target_sum':10000}})
    assert counts.shape==(128,6) and np.all(counts.data==np.floor(counts.data)) and np.all(counts.data>=0)


def test_stochastic_posterior_resume_is_exact(tmp_path):
    torch.manual_seed(2);net=network();opt=optimizer_for(net,SPEC);rng=np.random.default_rng(5)
    def step(n,o,r):
        x=torch.tensor(r.uniform(.5,2,(8,3)),dtype=torch.float32);ids=torch.ones(8,dtype=torch.long)
        z,cond,kl=n.posterior_sample(x,ids,0)
        mu,phi,_=n.decode(z,cond,ids,0,torch.arange(6),torch.ones(8)*100)
        loss=model.nb_nll(torch.ones(8,6)*10,mu,phi)+.01*kl
        o.zero_grad();loss.backward();o.step();n.context_steps[0]+=1
    step(net,opt,rng);p=tmp_path/'state.pt';save_state(p,net,opt,rng,1,SPEC)
    step(net,opt,rng);expected=copy.deepcopy(net.state_dict())
    restored=network();opt2=optimizer_for(restored,SPEC);rng2=np.random.default_rng(8)
    restore_state(p,restored,opt2,rng2,SPEC);step(restored,opt2,rng2)
    for k,v in restored.state_dict().items():torch.testing.assert_close(v,expected[k],rtol=0,atol=0)
