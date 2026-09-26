import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from common import log_profile,stratified_sample,write_json
from data import GeneIdentity,CountBlock,RawReader,prepare_context,NTC
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


def test_loco_selection_uses_global_equal_context_mean_and_earliest_tie():
    contexts=list('abcde');config={'checkpoints':[1,4,8]};results={}
    # Four contexts prefer round 1, but the global mean prefers round 4.
    # Round 8 ties round 4 and must lose to the earlier checkpoint.
    for validation in contexts:
        training=[c for c in contexts if c!=validation]
        for iteration in config['checkpoints']:
            score=(0 if validation=='e' else 2) if iteration==1 else (10 if validation=='e' else 1)
            results[(fit_id(training),validation,iteration)]={'score':score,'targets':100000 if validation=='a' else 1}
    selected=select_rounds(contexts,config,results)
    assert selected['rounds']==4
    assert selected['mean_scores']=={1:1.6,4:2.8,8:2.8}
    assert set(selected['fold_scores'][4])==set(contexts)
    # A missing held-out fold cannot silently become a four-context average.
    missing=(fit_id(list('abcd')),'e',4)
    value=results.pop(missing)
    with pytest.raises(KeyError):select_rounds(contexts,config,results)
    results[missing]={**value,'score':float('nan')}
    with pytest.raises(ValueError,match='nonfinite'):select_rounds(contexts,config,results)


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


@pytest.mark.parametrize('encoding',['dense','csr'])
def test_reusable_count_buffers_match_direct_counts(tmp_path,encoding):
    import h5py
    import anndata as ad
    rng=np.random.default_rng(11)
    raw=rng.poisson(3,(517,35)).astype(np.float32)
    file=tmp_path/'input.h5ad'
    ad.AnnData(raw if encoding=='dense' else sparse.csr_matrix(raw)).write_h5ad(file)
    columns=np.arange(35)[::-2];sums=np.zeros((5,len(columns)),np.float64);expected=np.zeros_like(sums)
    codes=rng.integers(0,5,len(raw))
    with h5py.File(file) as handle:
        reader=RawReader(handle);block=CountBlock(columns,raw.dtype)
        for start in range(0,len(raw),512):
            stop=min(start+512,len(raw));rows=np.arange(stop-start)[::2]
            got=block.select(reader.read(start,stop),rows)
            want=raw[start:stop][rows][:,columns]
            np.testing.assert_array_equal(got,want)
            block.accumulate(sums,codes[start:stop][rows])
            np.add.at(expected,codes[start:stop][rows],want)
    np.testing.assert_array_equal(sums,expected)
    for invalid in [np.nan,np.inf,-1,65536,.5]:
        changed=raw[:2].copy();changed[0,columns[0]]=invalid
        with pytest.raises(ValueError):block.select(changed,np.arange(2))


def test_whole_context_preparation_preserves_counts_and_full_ntc_baseline(tmp_path):
    import anndata as ad
    import pandas as pd
    rng=np.random.default_rng(2);genes=['A','B','C','D']
    counts=rng.poisson(5,(21,4)).astype(np.float32)
    path=tmp_path/'input.h5ad';ad.AnnData(counts).write_h5ad(path)
    targets=[NTC]*7+['A']*6+['B']*8
    obs=pd.DataFrame({'target':targets,'original_target':targets,'batch':'one','guide':'guide',
                      'barcode':[str(i) for i in range(21)],'source_row':np.arange(21)})
    mapping=pd.DataFrame({'source_position':range(4),'gene':genes,'valid':True})
    config={'minimum_target_cells':2,'ntc_bag_cells':4,'ntc_augmentation_bags':2,
            'seed':17,'reference_cells_per_target':4}
    prepare_context('toy',[(path,obs,mapping)],genes,{g:i for i,g in enumerate(genes)},config,tmp_path)
    np.testing.assert_array_equal(np.load(tmp_path/'toy/counts.npy'),counts)
    with np.load(tmp_path/'toy/statistics.npz') as stats:
        np.testing.assert_array_equal(stats['baseline'],log_profile(counts[:7].sum(0)))
        np.testing.assert_array_equal(stats['sums'],np.array([counts[7:13].sum(0),counts[13:].sum(0),counts[:7].sum(0)]))
        np.testing.assert_array_equal(stats['response'],log_profile(stats['sums'][:2])-stats['baseline'])
    features=np.load(tmp_path/'toy/context-features.npy')
    np.testing.assert_allclose(features[0,1],(counts[:7]>0).mean(0),rtol=1e-6)
    expected=np.log1p(((counts[:7].astype(np.float64)/counts[:7].sum(1,keepdims=True)*50000)**2).mean(0))
    np.testing.assert_allclose(features[0,2],expected,rtol=1e-6)
    rows=np.load(tmp_path/'toy/evaluation-rows.npy')
    assert set(range(7))<=set(rows) and len(rows)==7+4+4

def test_cross_validation_fits_four_contexts_and_scores_only_held_out(tmp_path,monkeypatch):
    import main
    contexts=list('abcde')
    config={'contexts':contexts,'checkpoints':[1,4]}
    fits=[];scores=[];persisted=[]
    def fake_fit(data,features,training,limit,config,output,tracked,evaluate):
        assert len(training)==4 and limit==4
        fits.append(training)
        for iteration in config['checkpoints']:evaluate(tuple(training),iteration)
    def score(context,training,iteration,booster,features,kind='model'):
        assert context not in training
        if kind=='model':assert booster==tuple(training)
        scores.append((context,kind,iteration))
        return {'score':float(iteration),'round':iteration,'context':context}
    monkeypatch.setattr(main,'fit',fake_fit)
    result={}
    selected=main.cross_validate(None,None,SimpleNamespace(score=score),config,tmp_path,Tracker(),result,
                                lambda:persisted.append(len(scores)))
    assert selected==4 and len(fits)==5
    assert len(scores)==20  # 5 folds x (2 checkpoints + 2 baselines).
    assert all(set(training)==set(contexts)-{held} for held,training in zip(contexts,fits))
    assert set(result['validation'])==set(contexts)
    assert result['selection']['rounds']==result['final_rounds']==4
    assert (tmp_path/'cache/selection.json').is_file() and persisted
