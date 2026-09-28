import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
import runtime
from selection import select_cells, accumulate, validate_control_config, STRATA
from model import fit


def test_exact_matching_protects_small_strata_and_is_order_invariant():
    frame = pd.DataFrame({'target':['a']*4+['b']*3+['c'], 'guide':['x']*4+['y']*3+['z'],
                          'batch':['one']*8, 'physical_id':list('abcdefgh'),
                          'depth':[1.,2.,20.,30.,1.,2.,3.,1.]})
    spec = {'depth_quantile':.7, 'selection_seed':1}
    got = select_cells(frame, 'K562', spec)
    assert got.qc_keep.tolist() == [False,False,True,True,False,False,True,True]
    sizes = got.groupby(STRATA)[['qc_keep','random_keep']].sum()
    np.testing.assert_array_equal(sizes.qc_keep, sizes.random_keep)
    again = select_cells(frame.sample(frac=1,random_state=3), 'K562', spec).sort_index()
    pd.testing.assert_frame_equal(got, again)
    # Singleton c was flagged low-depth, but retains coverage in both arms.
    assert got.iloc[-1].flagged and got.iloc[-1].qc_keep and got.iloc[-1].random_keep


def test_depth_thresholds_do_not_pool_batches():
    frame = pd.DataFrame({'target':['a']*8, 'guide':['x']*8, 'batch':['a']*4+['b']*4,
                          'physical_id':list('abcdefgh'), 'depth':[1.,2.,3.,4.,100.,200.,300.,400.]})
    got = select_cells(frame,'K562',{'depth_quantile':.25,'selection_seed':1})
    assert got.qc_keep.tolist() == [False,True,True,True,False,True,True,True]


def test_accumulation_matches_direct_retained_cell_normalization():
    counts = np.zeros(3, dtype=int); sums = np.zeros((3,2))
    matrix = sparse.csr_matrix([[1.,3.],[8.,0.],[5.,5.],[0.,6.]])
    groups = np.array([1,2,1,2])
    accumulate(sums,counts,matrix[:2],groups[:2],10000)
    accumulate(sums,counts,matrix[2:],groups[2:],10000)
    np.testing.assert_array_equal(counts,[0,2,2])
    direct = np.log1p(matrix.toarray()/matrix.toarray().sum(1)[:,None]*10000)
    np.testing.assert_allclose(sums[1],direct[[0,2]].sum(0))
    np.testing.assert_allclose(sums[2],direct[[1,3]].sum(0))


def test_historical_control_checks_every_scientific_field():
    keys = ('benchmark','data','fit_scope','seed','generation','evaluation','model','representation','reuse')
    old = {k:{'value':k} for k in keys}
    current = dict(old, training_selection={'kind':'low_depth'})
    validate_control_config(old,current)
    for key in keys:
        with pytest.raises(ValueError,match=key):
            validate_control_config(old,dict(current,**{key:{'changed':True}}))


def test_selected_fit_reads_only_selected_training_statistics(tmp_path, monkeypatch):
    import model
    (tmp_path/'selected').mkdir()
    np.savez(tmp_path/'selected/A-statistics.npz',labels=['NTC','p'],positions=[0,1],
             mean=np.array([[1.,1.],[2.,3.]],dtype=np.float32),counts=[10,4])
    np.savez(tmp_path/'selected/B-statistics.npz',labels=['NTC','q'],positions=[1],
             mean=np.array([[1.],[4.]],dtype=np.float32),counts=[10,4])
    (tmp_path/'A-statistics.npz').write_bytes(b'base must not be used for selected fit')
    (tmp_path/'selected/H1-statistics.npz').write_bytes(b'heldout forbidden')
    monkeypatch.setattr(model,'pathway_incidence',lambda *a:(np.arange(2),['m'],sparse.csr_matrix([[1],[1]])))
    cfg={'training_selection':{},'model':{'weight_cell_unit':100,'gene_chunk':2,'completion_alpha':.1},
         'representation':{'kind':'shared','dimensions':1,'pathways':{},'identities':{}}}
    result=fit(tmp_path,['g','h','unmeasured'],{'training_contexts':['A','B']},cfg)
    np.testing.assert_array_equal(result['effects'],[[1.,2.,0.],[0.,3.,0.]])
