import itertools
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from common import log_profile,stratified_sample,write_json
from data import GeneIdentity
from features import Features
from generation import generate_counts
from training import Rows,fit,fit_id,select_rounds


class Tracker:
    def log(self,*args,**kwargs): pass


def fixture_data(tmp_path):
    rng=np.random.default_rng(19)
    genes=['A','B','C','D','E','F','G','H']
    contexts={}
    for name,offset in [('a',0),('b',1),('c',2),('d',3),('e',4)]:
        contexts[name]={'targets':np.array(genes[:offset+2]),'gene_indices':np.arange(len(genes)),
                        'features':rng.random((3,3,len(genes))).astype(np.float32),
                        'response':rng.normal(size=(offset+2,len(genes))).astype(np.float32)}
    write_json(tmp_path/'complete.json',{'test':'immutable synthetic input'})
    data=SimpleNamespace(directory=tmp_path,genes=genes,lookup={g:i for i,g in enumerate(genes)},contexts=contexts)
    prior=tmp_path/'priors';prior.mkdir()
    np.save(prior/'embedding.npy',rng.normal(size=(len(genes),3)).astype(np.float32))
    for name in ['functional','physical','activation','repression']:
        sparse.save_npz(prior/f'{name}.npz',sparse.csr_matrix(rng.random((len(genes),len(genes)))))
    sparse.save_npz(prior/'pathways.npz',sparse.csr_matrix(rng.random((len(genes),4))>.4))
    return data,Features(data,prior)


def test_unique_gene_identity_and_independent_official_slots():
    identity=GeneIdentity()
    assert identity.canonical('ENSG00000215271.2')=='HOMEZ'
    assert identity.canonical('TMEM104')=='SLC38A12'
    assert identity.canonical('TIAF1')=='TIAF1'
    assert identity.canonical('MYO18A')=='MYO18A'
    ambiguous=next(g for g,values in identity.aliases.items() if len(values)>1 and g not in identity.approved and g!='TIAF1')
    assert identity.canonical(ambiguous) is None


def test_ntc_strata_and_generation_preserve_expected_response():
    strata=np.repeat(np.arange(46),400)
    indices=stratified_sample(strata,400,np.random.default_rng(4))
    assert len(indices)==len(np.unique(indices))==400
    assert set(np.bincount(strata[indices]))=={8,9}
    rng=np.random.default_rng(3)
    cells=rng.poisson(rng.uniform(1,30,80),size=(400,80))
    baseline=log_profile(cells.sum(0))
    delta=np.zeros(80);delta[:10]=.7;delta[10:20]=-.6
    counts=generate_counts(cells,baseline,delta,np.random.default_rng(7))
    np.testing.assert_array_equal(counts,generate_counts(cells,baseline,delta,np.random.default_rng(7)))
    desired=np.expm1(np.maximum(0,baseline+delta));expected=log_profile(desired)
    np.testing.assert_allclose(log_profile(counts.sum(0)),expected,atol=.02)
    assert np.issubdtype(counts.dtype,np.integer)
    np.testing.assert_allclose(counts.sum(1),cells.sum(1),rtol=.02)
    zero=generate_counts(cells,baseline,np.zeros(80),np.random.default_rng(8))
    np.testing.assert_allclose(log_profile(zero.sum(0)),baseline,atol=.02)


def test_rows_are_context_balanced_and_features_do_not_use_responses(tmp_path):
    data,features=fixture_data(tmp_path)
    config={'seed':17,'training_readouts_per_target':8}
    totals={}
    for context in ['a','b','c']:
        totals[context]=sum(weights.sum(dtype=np.float64) for _,_,weights in Rows(data,features,[context],config).batches())
    np.testing.assert_allclose(list(totals.values()),1e6,rtol=1e-6)
    before=features.matrix('d',[0,1],[2,3])
    data.contexts['d']['response'][:]=10000
    np.testing.assert_array_equal(features.matrix('d',[0,1],[2,3]),before)
    assert np.isfinite(before).all()


def test_nested_selection_never_uses_outer_labels():
    contexts=list('abcde');config={'checkpoints':[1,4]};results={}
    for training in itertools.combinations(contexts,3):
        for validation in set(contexts)-set(training):
            for iteration in config['checkpoints']:
                results[(fit_id(training),validation,iteration)]={'score':iteration}
    selected=select_rounds(contexts,config,results)
    assert selected['a']['rounds']==4
    for key in results:
        if key[1]=='a':results[key]['score']=1e9 if key[2]==1 else -1e9
    assert select_rounds(contexts,config,results)['a']==selected['a']


def test_full_state_resume_matches_uninterrupted_training(tmp_path):
    data,features=fixture_data(tmp_path)
    config={'seed':17,'training_readouts_per_target':8,'threads':2,'checkpoints':[1,4,8],
            'model':{'objective':'reg:squarederror','tree_method':'hist','device':'cpu','nthread':2,
                     'max_bin':16,'max_depth':3,'subsample':1,'colsample_bytree':1,
                     'seed_per_iteration':True,'base_score':0}}
    uninterrupted=fit(data,features,['a','b','c'],8,config,tmp_path/'full',Tracker(),lambda *_:None)
    def interrupt(model,iteration):
        if iteration==4:raise RuntimeError('simulated interruption after checkpoint')
    with pytest.raises(RuntimeError,match='simulated interruption'):
        fit(data,features,['a','b','c'],8,config,tmp_path/'resumed',Tracker(),interrupt)
    resumed=fit(data,features,['a','b','c'],8,config,tmp_path/'resumed',Tracker(),lambda *_:None)
    assert uninterrupted.get_dump(dump_format='json')==resumed.get_dump(dump_format='json')
    x=features.matrix('d',np.repeat(np.arange(5),8),np.tile(np.arange(8),5))
    np.testing.assert_array_equal(uninterrupted.inplace_predict(x),resumed.inplace_predict(x))
