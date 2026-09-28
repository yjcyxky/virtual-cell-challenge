import sys
from pathlib import Path
import numpy as np
import pytest
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from representation import randomize_incidence, randomized_pca, ntc_projection
from model import fit, predict


def test_random_prior_preserves_each_gene_and_module_degree():
    rng = np.random.default_rng(13)
    matrix = sparse.csr_matrix(rng.uniform(size=(40, 16)) < .25, dtype=float)
    random, audit = randomize_incidence(matrix, 7, 5)
    np.testing.assert_array_equal(random.sum(axis=0), matrix.sum(axis=0))
    np.testing.assert_array_equal(random.sum(axis=1), matrix.sum(axis=1))
    assert random.data.max() == 1 and audit['edge_overlap_fraction'] < .7
    again, _ = randomize_incidence(matrix, 7, 5)
    np.testing.assert_array_equal(random.toarray(), again.toarray())


def test_randomized_pca_recovers_known_subspace():
    rng = np.random.default_rng(3)
    left = rng.normal(size=(80, 4)); right = rng.normal(size=(4, 15))
    matrix = left @ right
    basis, _ = randomized_pca(matrix, 4, 8, 6, 2)
    np.testing.assert_allclose(matrix @ basis.T @ basis, matrix, atol=1e-10)
    np.testing.assert_allclose(basis @ basis.T, np.eye(4), atol=1e-12)


def test_projection_excludes_heldout_ntc_and_response_labels(tmp_path, monkeypatch):
    import representation
    import anndata as ad
    import pandas as pd
    from data import normalized
    genes = [f'g{i}' for i in range(8)]
    rng = np.random.default_rng(42)
    stats = {}
    for context in ['A', 'B', 'C']:
        counts = sparse.csr_matrix(rng.poisson(3, (12, 8)).astype(float))
        mean = np.asarray(normalized(counts, 10000).mean(axis=0)).ravel()
        stats[context] = dict(labels=np.array(['non-targeting','p']), positions=np.arange(8),
                              mean=np.vstack([mean, mean+rng.normal(size=8)]), counts=np.array([12, 20]))
        ad.AnnData(counts, obs=pd.DataFrame({'target_gene':['non-targeting']*12}, index=[str(i) for i in range(12)]),
                   var=pd.DataFrame(index=genes)).write_h5ad(tmp_path/f'{context}-input.h5ad')
    matrix = sparse.csr_matrix(np.array([[1,1,0,0],[1,0,1,0],[0,1,1,0],[0,0,1,1],
                                        [1,0,0,1],[0,1,0,1],[1,1,1,0],[0,1,1,1]], dtype=float))
    monkeypatch.setattr(representation, 'pathway_incidence', lambda *args: (np.arange(8), ['a','b','c','d'], matrix))
    config = {'data':{'chunk_rows':5,'target_sum':10000},
              'representation':{'kind':'pca','dimensions':2,'fit_seed':5,'pca_oversample':3,
                                'pca_power_iterations':2,'pathways':{},'identities':{},'swaps_per_edge':2}}
    baseline = ntc_projection(stats, ['A','B'], genes, tmp_path, config)
    stats['C']['mean'][:] = 1e8
    stats['A']['mean'][1:] = -1e9
    for got, expected in zip(ntc_projection(stats, ['A','B'], genes, tmp_path, config), baseline):
        np.testing.assert_array_equal(got, expected)
    assert not (tmp_path/'H1-input.h5ad').exists()
    for kind in ['raw','program','random_program']:
        config['representation']['kind'] = kind
        common, center, basis, scale = ntc_projection(stats, ['A','B'], genes, tmp_path, config)
        means = np.array([stats[c]['mean'][0] for c in ['A','B']])
        assert np.mean(np.sum((((means-center)@basis.T)/scale)**2,axis=1)) == pytest.approx(1)
        if kind != 'raw':
            assert basis.shape == (2,8)


def test_ridge_is_invariant_to_orthogonal_ntc_coordinates(tmp_path, monkeypatch):
    import representation
    genes = ['g0','g1','g2','g3']
    stats = {}
    for i, context in enumerate(['A','B','C']):
        stats[context] = dict(labels=np.array(['non-targeting','p','q']), positions=np.arange(4),
                             mean=np.array([[1+i,2-i,3,2],[3+i,1,4,3],[2,4-i,3,1.]],dtype=float),
                             counts=np.array([10,20+i,30+i]))
        np.savez(tmp_path/f'{context}-statistics.npz', **stats[context])
    center = np.array([2,1,3,2.])
    basis = np.eye(4)[:2]
    monkeypatch.setattr(representation,'ntc_projection',lambda *args: (np.arange(4),center,basis,2.))
    config={'representation':{},'model':{'alpha':1,'weight_cell_unit':100,'gene_chunk':2,'ntc_conditioning':True}}
    first=fit(tmp_path,genes,{'training_contexts':list(stats)},config)
    basis=np.array([[0.,1.],[-1.,0.]])@basis
    second=fit(tmp_path,genes,{'training_contexts':list(stats)},config)
    for context in stats.values():
        np.testing.assert_allclose(predict(first,context,['p','q'],'linear'),predict(second,context,['p','q'],'linear'),atol=1e-6)
    np.testing.assert_array_equal(first['shared'],second['shared'])


def test_reference_reuse_rejects_changed_model_scores_or_protocol(tmp_path, monkeypatch):
    import json
    import yaml
    import data
    monkeypatch.setattr(data, 'ROOT', tmp_path)
    source = tmp_path/'source'; source.mkdir()
    output = tmp_path/'candidate'; (output/'cache').mkdir(parents=True)
    binding = {'node_id':'source'}
    config = {key: {'same':True} for key in ('benchmark','data','fit_scope','generation')}
    config.update(seed=1,evaluation={'single_source':'K562','panels':['all','official_overlap'],
                                    'runtime':{'device':'cuda'},'arms':['linear']})
    (source/'config.yaml').write_text(yaml.safe_dump(dict(config, research=binding)))
    model = {'shared':np.array([[.1,.2]]),'targets':np.array(['p']),
             'genes':np.array(['x','y']),'trained_mask':np.array([True,True])}
    np.savez(source/'model.npz', **model)
    files=[]; scores=[]; evaluation={}
    for arm in ('zero','shared','source'):
        p=source/f'{arm}.h5ad'; p.write_bytes(b'frozen prediction'); files.append(data.ref(p))
    (source/'predictions.json').write_text(json.dumps({'files':files}))
    for panel in config['evaluation']['panels']:
        evaluation[panel]={}
        for arm in ('zero','shared','source'):
            directory=source/f'{panel}-{arm}'; directory.mkdir()
            for name in ('raw.parquet','aggregate.csv','scores.csv','run_meta.json'):
                p=directory/name; p.write_bytes(b'frozen score'); scores.append(data.ref(p))
            evaluation[panel][arm]={'Overall':.2,'normalized':{'m':.2},'result_directory':str(directory.relative_to(tmp_path))}
    (source/'scores.json').write_text(json.dumps({'files':scores}))
    (source/'metrics.json').write_text(json.dumps({'status':'completed','evaluation_completed':True,
                                                  'research':binding,'evaluation':evaluation}))
    config['reference_reuse']={k:data.ref(source/v) for k,v in
        {'config':'config.yaml','metrics':'metrics.json','checkpoint':'model.npz',
         'predictions':'predictions.json','score_manifest':'scores.json'}.items()}
    result=data.verify_reference_reuse(output,model,config)
    assert result['all']['shared']['reused_from']==config['reference_reuse']['metrics']
    changed=dict(model, shared=model['shared']+1)
    with pytest.raises(ValueError, match='shared response changed'):
        data.verify_reference_reuse(output,changed,config)
    config['seed']=2
    with pytest.raises(ValueError, match='changes seed'):
        data.verify_reference_reuse(output,model,config)
    config['seed']=1
    (tmp_path/scores[0]['path']).write_bytes(b'changed')
    with pytest.raises(ValueError, match='score checksum changed'):
        data.verify_reference_reuse(output,model,config)
