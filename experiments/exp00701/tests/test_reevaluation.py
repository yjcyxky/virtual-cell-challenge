"""Isolation and alignment contracts for fixed-model exp004 reevaluation."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from data import NTC
from reevaluation import aligned_genes,check_control_partition,generated_blocks,input_statistics


@pytest.mark.parametrize('keys',[['input_sha256','row_index'],['source_batch','source_barcode']])
def test_control_partition_uses_both_raw_identity_keys(keys):
    inputs=pd.DataFrame({'is_NTC':[True,True],'half':[0,0],'input_sha256':['a','a'],
                         'row_index':[1,2],'source_batch':['b','b'],'source_barcode':['c1','c2']})
    reference=inputs.copy();reference['half']=1;reference['row_index']=[3,4];reference['source_barcode']=['c3','c4']
    check_control_partition(inputs,reference)
    for key in keys:reference.loc[0,key]=inputs.loc[0,key]
    with pytest.raises(ValueError,match='control_overlap'):check_control_partition(inputs,reference)


def test_legacy_gene_mapping_preserves_order_and_rejects_ambiguity():
    lookup={'NEW':5,'OTHER':2};aliases={'OLD':'NEW','OTHER':'OTHER','NEW':'NEW'}
    axis,mapped=aligned_genes(['OTHER','OLD'],lookup,aliases.get)
    np.testing.assert_array_equal(axis,[2,5]);assert mapped==['OTHER','NEW']
    with pytest.raises(ValueError,match='unmapped'):aligned_genes(['UNKNOWN'],lookup,aliases.get)
    with pytest.raises(ValueError,match='collision'):aligned_genes(['OLD','NEW'],lookup,aliases.get)


def test_input_features_keep_arithmetic_cpm_separate_from_count_sum_profile():
    counts=np.array([[9,1,0],[0,20,80]],np.uint16)
    original=counts.copy();features,baseline=input_statistics(counts)
    np.testing.assert_allclose(baseline,[450000,150000,400000])
    np.testing.assert_allclose(features[0,0],np.log1p(50000*counts.sum(0)/counts.sum()),rtol=1e-6)
    np.testing.assert_allclose(features[0,1],[.5,1.,.5])
    proportions=counts/counts.sum(1,keepdims=True)
    np.testing.assert_allclose(features[0,2],np.log1p(((proportions*50000)**2).mean(0)),rtol=1e-6)
    np.testing.assert_array_equal(counts,original)
    with pytest.raises(ValueError,match='empty_input'):input_statistics(np.zeros((2,3)))


def test_prediction_allocation_uses_only_matching_input_batches(monkeypatch):
    import reevaluation
    counts=np.array([[10,1],[20,2],[30,3],[40,4]],np.uint16)
    inputs=pd.DataFrame({'batch':['a','a','b','b']})
    obs=pd.DataFrame({'target':['P','P',NTC,NTC],'batch':['b','a','a','b']})
    captured=[]
    def generate(templates,*args):
        captured.append(templates.copy());return templates.astype(np.uint32)
    monkeypatch.setattr(reevaluation,'generate_counts',generate)
    kwargs=dict(counts=counts,inputs=inputs,observations=obs,targets=['P'],response=np.zeros((1,2)),
                baseline=np.array([1.,1.]),prediction_seed=101,epsilon=1e-9)
    first=np.concatenate(list(generated_blocks(**kwargs)))
    second=np.concatenate(list(generated_blocks(**kwargs)))
    np.testing.assert_array_equal(first,second)
    assert first.shape==(4,2) and first.dtype==np.uint32 and len(captured)==2
    assert first[0,0] in (30,40) and first[1,0] in (10,20)
    assert first[2,0] in (10,20) and first[3,0] in (30,40)
    bad=obs.copy();bad.loc[0,'batch']='missing'
    with pytest.raises(ValueError,match='no_input_controls'):list(generated_blocks(**{**kwargs,'observations':bad}))
    bad=obs.iloc[[0,2,1,3]].reset_index(drop=True)
    with pytest.raises(ValueError,match='noncontiguous'):list(generated_blocks(**{**kwargs,'observations':bad}))
