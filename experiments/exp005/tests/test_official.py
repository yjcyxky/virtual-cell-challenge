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
from evaluation import Evaluation


@pytest.mark.skipif(not torch.cuda.is_available(),reason='official production backend requires CUDA')
def test_official_memmap_full_score_and_cached_replay(tmp_path):
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
            'replicate_splits':5,'reference_cells_per_target':400,'cells_per_prediction':400,'prediction_seed':101}
    tracker=SimpleNamespace(log=lambda *args,**kwargs:None)
    evaluation=Evaluation(data,config,tmp_path,tracker)
    score=evaluation.score('Toy',[],0,None,None,kind='zero',keep=True)
    assert len(score['components'])==6 and np.isfinite(score['score'])
    assert score==evaluation.score('Toy',[],0,None,None,kind='zero',keep=True)
    directory=tmp_path/'predictions/baseline/Toy/zero-0000'
    matrix=np.load(directory/'predicted-counts.npy',mmap_mode='r')
    assert matrix.shape==(9*400,80) and np.issubdtype(matrix.dtype,np.integer)
    scored=pl.read_csv(directory/'scores.csv')
    assert score['score']==scored.filter(pl.col('metric')=='avg_score')['from_replicate'].item()
    manifest=json.loads((tmp_path/'cache/official/Toy/bundle/manifest.json').read_text())
    assert manifest['resolved_device']=='cuda'
    assert manifest['resolved_de_backend']=='gpudge'
