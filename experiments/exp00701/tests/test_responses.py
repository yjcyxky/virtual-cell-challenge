"""Observed source effects must transfer without recipient labels or false zero imputation."""
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from scipy import sparse

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from responses import ResponseEvidence,module_basis


def evidence_data():
    genes=['P','Q','G','Z'];contexts={};de={}
    for ctx,y in [('a',2.),('b',-4.),('H1',100.)]:
        contexts[ctx]={'targets':np.array(['P']),'gene_indices':np.arange(3),
                       'response':np.array([[0.,0.,y]],np.float32)}
        de[ctx]={'sign':np.array([[0,0,np.sign(y)]],np.int8),'tested':np.array([[False,True,True]]),
                 'evidence':np.array([[0,0,3.]],np.float32),'cells':np.array([100])}
    data=SimpleNamespace(genes=genes,lookup={g:i for i,g in enumerate(genes)},contexts=contexts)
    return data,de,np.array([[1,0],[0,1],[1,1],[0,0]],np.float32)


@pytest.mark.parametrize('mode',['continuous','deg','modules'])
def test_recipient_and_h1_responses_never_enter_features(mode):
    data,de,basis=evidence_data()
    before=ResponseEvidence(data,mode,de,basis,('a','b')).matrix('a',[0],[2])
    # Even rebuilding the encoder cannot expose recipient labels through modules.
    data.contexts['a']['response'][:]=9000;data.contexts['H1']['response'][:]=-9999
    de['a']['sign'][:]=-1;de['H1']['sign'][:]=1
    after=ResponseEvidence(data,mode,de,basis,('a','b')).matrix('a',[0],[2])
    np.testing.assert_array_equal(before,after)
    assert after[0,0]==-4
    data.contexts['b']['response'][0,2]=-8
    changed=ResponseEvidence(data,mode,de,basis,('a','b')).matrix('a',[0],[2])
    assert changed[0,0]==-8
    with pytest.raises(ValueError,match='H1_response'):
        ResponseEvidence(data,mode,de,basis,('a','H1'))


def test_missing_readout_and_missing_target_differ_from_observed_zero():
    data,de,basis=evidence_data();features=ResponseEvidence(data,'continuous',de,basis,('a','b'))
    actual=features.matrix('H1',[0,0,1],[1,3,2])
    # All three means are zero; support identifies measured zero, missing g, missing p.
    np.testing.assert_array_equal(actual[:,0],[0,0,0])
    np.testing.assert_array_equal(actual[:,5],[2,0,0])
    np.testing.assert_array_equal(actual[:,6],[2,2,0])
    assert np.isfinite(actual).all()


def test_consensus_keeps_opposite_signs_and_nonsignificance_distinct():
    data,de,basis=evidence_data();features=ResponseEvidence(data,'deg',de,basis,('a','b'))
    v=features.matrix('H1',[0],[2])[0]
    np.testing.assert_allclose(v[:7],[-1,3,-4,2,3,2,2])
    np.testing.assert_allclose(v[7:14],[2,1,1,.5,.5,0,1])
    de['b']['sign'][0,2]=0
    v=features.matrix('H1',[0],[2])[0]
    np.testing.assert_allclose(v[7:14],[2,1,0,.5,0,.5,0])
    assert v[0]==-1  # non-DE does not overwrite a measured continuous effect


def test_module_profiles_are_target_specific_and_use_only_sources():
    data,de,basis=evidence_data();features=ResponseEvidence(data,'modules',de,basis,('a','b'))
    p=features.matrix('H1',[0,1],[2,2])
    assert p.shape[1]==17+5*2+2
    assert np.any(p[0,17:23]!=0)
    np.testing.assert_array_equal(p[1,17:23],0)
    # H1 labels cannot change a source signature.
    old=p.copy();de['H1']['sign'][:]=0;data.contexts['H1']['response'][:]=0
    np.testing.assert_array_equal(ResponseEvidence(data,'modules',de,basis,('a','b')).matrix('H1',[0,1],[2,2]),old)


def test_fixed_module_basis_is_deterministic_and_label_free():
    rng=np.random.default_rng(5);pathways=sparse.csr_matrix((rng.random((50,20))>.7).astype(np.float32))
    cfg={'module_min_genes':5,'module_max_genes':30,'module_dimensions':4,'seed':17}
    a=module_basis(pathways,cfg);b=module_basis(pathways,cfg)
    np.testing.assert_array_equal(a,b);np.testing.assert_allclose(np.linalg.norm(a,axis=0),1,atol=1e-6)
