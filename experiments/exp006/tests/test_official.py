"""Exercise the real six-metric CUDA/gpudge path, including official anchors."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import polars as pl
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from common import log_profile
from evaluation import Evaluation, EVALUATION_PROTOCOL, read_count_matrix, write_count_matrix


@pytest.fixture
def toy_inputs(tmp_path):
    rng=np.random.default_rng(17)
    genes=[f'GENE{i}' for i in range(80)];targets=genes[:8]
    base=np.full(len(genes),10.)
    groups=[rng.poisson(base,(400,len(genes)))]
    for i in range(len(targets)):
        mean=base.copy()
        changed=rng.permutation(np.arange(8,len(genes)))[:40]
        mean[changed[:20]]*=1.8
        mean[changed[20:]]*=.2
        groups.append(rng.poisson(mean,(100,len(genes))))
    counts=np.concatenate(groups).astype(np.uint16)
    folder=tmp_path/'cache/data/Toy';folder.mkdir(parents=True)
    np.save(folder/'counts.npy',counts)
    np.save(folder/'evaluation-rows.npy',np.arange(len(counts)))
    labels=['non-targeting']*400+list(np.repeat(targets,100))
    frame=pd.DataFrame({'target':labels,'batch':'one','guide':np.tile(np.arange(4),len(counts)//4).astype(str)})
    baseline=log_profile(groups[0].sum(0))
    stats={'folder':folder,'targets':np.array(targets),'gene_indices':np.arange(len(genes)),
           'baseline':baseline,'response':log_profile(np.array([g.sum(0) for g in groups[1:]]))-baseline,
           'control_rows':np.arange(400)}
    data=SimpleNamespace(genes=genes,lookup={g:i for i,g in enumerate(genes)},contexts={'Toy':stats},
                         cells=lambda _:frame,counts=lambda _:np.load(folder/'counts.npy',mmap_mode='r'))
    config={'threads':2,'official_device':'cuda','official_de_backend':'gpudge','replicate_seed':0,
            'replicate_splits':5,'reference_cells_per_target':400,'cells_per_prediction':400,'prediction_seed':101,
            'evaluation_protocol':EVALUATION_PROTOCOL,'baseline_seed':0}
    return data,config


@pytest.mark.skipif(not torch.cuda.is_available(),reason='official production backend requires CUDA')
def test_official_memmap_full_score_and_cached_replay(tmp_path,toy_inputs):
    data,config=toy_inputs
    tracker=SimpleNamespace(log=lambda *args,**kwargs:None)
    evaluation=Evaluation(data,config,tmp_path,tracker)
    score=evaluation.score('Toy',[],0,None,None,kind='zero',keep=True)
    assert len(score['components'])==6 and np.isfinite(score['score'])
    assert score==evaluation.score('Toy',[],0,None,None,kind='zero',keep=True)
    directory=tmp_path/'predictions/baseline/Toy/zero-0000'
    matrix=read_count_matrix(directory/'predicted-counts')
    assert matrix.shape==(9*400,80) and np.issubdtype(matrix.dtype,np.integer)
    scored=pl.read_csv(directory/'scores.csv')
    assert score['score']==scored.filter(pl.col('metric')=='avg_score')['from_replicate'].item()
    manifest=json.loads((tmp_path/'cache/official/Toy/bundle/manifest.json').read_text())
    assert manifest['resolved_device']=='cuda'
    assert manifest['resolved_de_backend']=='gpudge'


def test_bundle_baseline_matches_official_supported_defaults(tmp_path,toy_inputs,monkeypatch):
    import evaluation
    from scipy import sparse
    from cell_eval2.baseline import generic_response_profile,build_baseline_prediction
    data,config=toy_inputs
    tracker=SimpleNamespace(log=lambda *args,**kwargs:None)
    captured=[]
    class Checked(Exception):pass
    def check(real,arm,**kwargs):
        profile=generic_response_profile(real,pert_col='target',control='non-targeting')
        expected=build_baseline_prediction(profile,real,pert_col='target',control='non-targeting')
        actual_values=arm.X.toarray() if sparse.issparse(arm.X) else np.asarray(arm.X)
        expected_values=expected.X.toarray() if sparse.issparse(expected.X) else np.asarray(expected.X)
        np.testing.assert_array_equal(actual_values,expected_values)
        assert arm.uns['baseline_emission']['emit']=='dispersed'
        assert sparse.isspmatrix_csr(real.X) and sparse.isspmatrix_csr(arm.X)
        pd.testing.assert_frame_equal(arm.obs,real.obs)
        pd.testing.assert_frame_equal(arm.var,real.var)
        assert not (tmp_path/'cache/official/Toy/mean-response-baseline.npy').exists()
        captured.append(True)
        raise Checked
    monkeypatch.setattr(evaluation,'build_real_bundle',check)
    with pytest.raises(Checked):evaluation.Evaluation(data,config,tmp_path,tracker).prepare('Toy')
    assert captured


def test_count_matrix_roundtrip_preserves_counts_and_empty_rows(tmp_path):
    from scipy import sparse
    counts=np.array([[0,2,0,5],[0,0,0,0],[65535,0,1,0],[3,4,5,6],[0,0,0,9]],np.uint16)
    result=write_count_matrix(tmp_path/'matrix',(counts[i:i+2] for i in range(0,len(counts),2)),counts.shape,counts.dtype)
    assert sparse.isspmatrix_csr(result) and result.has_canonical_format
    assert isinstance(result.data,np.memmap) and not result.data.flags.writeable
    np.testing.assert_array_equal(result.toarray(),counts)
    np.testing.assert_array_equal(read_count_matrix(tmp_path/'matrix')[[4,0,3]].toarray(),counts[[4,0,3]])
    empty=np.zeros((3,4),np.uint32)
    np.testing.assert_array_equal(write_count_matrix(tmp_path/'empty',[empty],empty.shape,empty.dtype).toarray(),empty)


def test_old_score_and_config_cannot_be_reused(tmp_path,toy_inputs):
    data,config=toy_inputs;tracker=SimpleNamespace(log=lambda *args,**kwargs:None)
    old={k:v for k,v in config.items() if k!='evaluation_protocol'}
    with pytest.raises(ValueError,match='new_run_required'):Evaluation(data,old,tmp_path,tracker)
    directory=tmp_path/'predictions/baseline/Toy/zero-0000';directory.mkdir(parents=True)
    (directory/'metrics.json').write_text(json.dumps({'score':123,'round':0}))
    with pytest.raises(ValueError,match='new_run_required'):
        Evaluation(data,config,tmp_path,tracker).score('Toy',[],0,None,None,kind='zero')


def test_streaming_pointers_cross_int32_limit(tmp_path,monkeypatch):
    import evaluation
    # Simulate three large CSR blocks without allocating billions of entries.
    # The actual streaming writer must widen block-local pointers BEFORE addition.
    block_nnz=2**30+1
    block=SimpleNamespace(data=np.array([],np.uint16),indices=np.array([],np.int32),
                          indptr=np.array([0,block_nnz],np.int32),nnz=block_nnz)
    monkeypatch.setattr(evaluation.sparse,'csr_matrix',lambda _:block)
    monkeypatch.setattr(evaluation,'read_count_matrix',lambda directory:np.load(directory/'indptr.npy'))
    pointers=write_count_matrix(tmp_path/'large',[np.zeros((1,1),np.uint16)]*3,(3,1),np.uint16)
    np.testing.assert_array_equal(pointers,np.arange(4,dtype=np.int64)*block_nnz)
