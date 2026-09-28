"""Catch the observed dense COO conversion at the actual official bulk seam."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import anndata as ad
from scipy import sparse
from cell_eval2.streaming_bulk import inmem_pseudobulk
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import anchor_audit


def test_dense_tile_conversion_is_bounded_and_official_moments_match(monkeypatch):
    rng=np.random.default_rng(4)
    values=np.tile(rng.uniform(0.1,8,64).astype(np.float32),(320,1))
    values[:40]=rng.poisson(2,(40,64))
    values[:,0]=0
    obs=pd.DataFrame({'target_gene':['non-targeting']*40+['a']*140+['b']*140})
    original=ad.AnnData(values,obs=obs)
    kwargs=dict(pert_col='target_gene',norms=['bulk_lognorm'],target_sum=1e6,
                device='cpu',with_moments=True,bulk_target_sum=50000)
    expected,expected_moments=inmem_pseudobulk(original,**kwargs)
    maximum=[]
    constructor=sparse.csr_matrix
    class ObservedCSR(constructor):
        def __init__(self,arg1,*args,**kwargs):
            if isinstance(arg1,np.ndarray) and arg1.ndim==2:
                maximum.append(arg1.shape[0])
            super().__init__(arg1,*args,**kwargs)
    monkeypatch.setattr(sparse,'csr_matrix',ObservedCSR)
    prepare=getattr(anchor_audit,'prepare_baseline',lambda x,chunk_rows:x)
    actual=prepare(original.copy(),chunk_rows=32)
    observed,moments=inmem_pseudobulk(actual,**kwargs)
    assert max(maximum,default=0)<=32, f'unbounded dense conversion: {maximum}'
    np.testing.assert_array_equal(actual.X.toarray(),values)
    np.testing.assert_array_equal(observed['bulk_lognorm'][1],expected['bulk_lognorm'][1])
    np.testing.assert_allclose(moments['bulk_lognorm'].jk,expected_moments['bulk_lognorm'].jk,rtol=0,atol=1e-12)


def test_csr_storage_preserves_official_cuda_raw_metrics():
    from dataclasses import replace
    from cell_eval2 import compute_metrics,aggregate_metrics_wide
    from cell_eval2.baseline import generic_response_profile,build_baseline_prediction,baseline_config
    from cell_eval2.run import metric_output_names
    from cell_eval2 import EvalConfig
    rng=np.random.default_rng(8)
    labels=np.repeat(['non-targeting','g1','g2','g3'],50)
    rates=np.broadcast_to(rng.uniform(2,10,128),(200,128)).copy()
    rates[50:100,:20]*=2; rates[100:150,10:30]*=.25;rates[150:,30:60]*=3
    real=ad.AnnData(sparse.csr_matrix(rng.poisson(rates).astype(np.float32)),
                    obs=pd.DataFrame({'target_gene':labels},index=[str(i) for i in range(200)]),
                    var=pd.DataFrame(index=[f'g{i}' for i in range(128)]))
    profile=generic_response_profile(real,pert_col='target_gene',control='non-targeting')
    dense=build_baseline_prediction(profile,real,pert_col='target_gene',control='non-targeting',emit='tile')
    csr=anchor_audit.prepare_baseline(dense.copy(),chunk_rows=32)
    cfg=baseline_config(replace(EvalConfig.from_preset('vcc2026'),pert_col='target_gene',device='cuda',num_threads=8,pert_chunk=16))
    first=compute_metrics(dense,real,config=cfg).sort(['perturbation','metric'])
    second=compute_metrics(csr,real,config=cfg).sort(['perturbation','metric'])
    assert first.drop('value').equals(second.drop('value'))
    np.testing.assert_allclose(first['value'].to_numpy(),second['value'].to_numpy(),rtol=1e-10,atol=1e-12,equal_nan=True)
    a=aggregate_metrics_wide(first,metrics=metric_output_names(cfg))
    b=aggregate_metrics_wide(second,metrics=metric_output_names(cfg))
    assert a['statistic'].to_list()==b['statistic'].to_list()
    np.testing.assert_allclose(a.drop('statistic').to_numpy(),b.drop('statistic').to_numpy(),rtol=1e-10,atol=1e-12,equal_nan=True)
