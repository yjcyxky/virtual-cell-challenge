from pathlib import Path
import sys
import numpy as np
from scipy import sparse
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from model import solve_ridge, fit
from evaluation import generate_counts


def test_evaluator_adds_input_controls_without_changing_target_predictions():
    import anndata as ad
    import pandas as pd
    import pytest
    from cell_eval2.io import validate_pair
    import evaluation
    genes = pd.DataFrame(index=['G0','G1'])
    pred = ad.AnnData(sparse.csr_matrix([[1.,2.],[3.,4.]]), obs=pd.DataFrame({'target_gene':['G0','G1']}, index=['p0','p1']), var=genes)
    control = ad.AnnData(sparse.csr_matrix([[5.,6.]]), obs=pd.DataFrame({'target_gene':['non-targeting']}, index=['input-control']), var=genes)
    real = ad.concat([control,pred])
    with pytest.raises(ValueError, match='perturbation sets differ'):
        validate_pair(pred,real,pert_col='target_gene',control='non-targeting')
    adapted = evaluation.with_input_controls(pred, control)
    validate_pair(adapted,real,pert_col='target_gene',control='non-targeting')
    np.testing.assert_array_equal(adapted[pred.obs_names].X.toarray(),pred.X.toarray())
    np.testing.assert_array_equal(adapted[control.obs_names].X.toarray(),control.X.toarray())


def test_schur_ridge_matches_dense_normal_equations():
    rng = np.random.default_rng(5)
    target = np.array([0, 1, 2, 0, 2, 1, 3])
    x, y, w = rng.normal(size=(7, 3)), rng.normal(size=(7, 4)), rng.uniform(.2, 2, 7)
    a, beta, shared = solve_ridge(target, x, y, w, 5, .7)
    design = np.column_stack([np.eye(5)[target], x])
    expected = np.linalg.solve(design.T@(w[:, None]*design)+.7*np.eye(8), design.T@(w[:, None]*y))
    np.testing.assert_allclose(np.vstack([a, beta]), expected, atol=1e-12)
    np.testing.assert_array_equal(a[4], 0)
    np.testing.assert_array_equal(shared[4], 0)


def test_fit_respects_missing_genes_and_heldout_labels(tmp_path):
    np.savez(tmp_path/'A-statistics.npz', labels=['non-targeting','p','q'], positions=[0,1],
             mean=[[1,2],[2,5],[1,3]], counts=[10,20,30])
    np.savez(tmp_path/'B-statistics.npz', labels=['non-targeting','p','q'], positions=[1,2],
             mean=[[3,1],[4,2],[3,5]], counts=[10,20,30])
    np.savez(tmp_path/'H1-statistics.npz', mean=np.full((20,4), 1e9))
    config = {'model': {'weight_cell_unit':100, 'gene_chunk':2, 'alpha':1}}
    result = fit(tmp_path, ['p','q','r','unmeasured'], {'training_contexts':['A','B']}, config)
    assert result['trained_mask'].tolist() == [True,True,True,False]
    assert result['training_tasks'] == 4
    np.testing.assert_array_equal(result['effects'][:,3], 0)
    np.testing.assert_array_equal(result['beta'][:,3], 0)
    np.savez(tmp_path/'H1-statistics.npz', mean=np.full((20,4), -1e9))
    again = fit(tmp_path, ['p','q','r','unmeasured'], {'training_contexts':['A','B']}, config)
    np.testing.assert_array_equal(result['effects'], again['effects'])


def test_zero_response_resamples_exact_ntc_counts():
    ntc = sparse.csr_matrix(np.array([[10,3,0],[2,5,1],[4,4,2]],dtype=np.float32))
    generated = generate_counts(ntc, np.zeros(3), 1, 40, 10000).toarray()
    assert all(any(np.array_equal(row, src) for src in ntc.toarray()) for row in generated)
    np.testing.assert_array_equal(generated, generate_counts(ntc, np.zeros(3), 1, 40, 10000).toarray())


def test_count_generator_has_finite_integer_nonnegative_output():
    ntc = sparse.csr_matrix(np.array([[10,3,0],[2,5,1],[4,4,2]],dtype=np.float32))
    out = generate_counts(ntc, np.array([-.8,.1,.2]), 1, 40, 10000).toarray()
    assert np.isfinite(out).all() and (out>=0).all() and np.equal(out,np.floor(out)).all()


def test_no_condition_fit_matches_penalized_target_and_intercept_design(tmp_path):
    from model import predict
    genes = ['p', 'q', 'unmeasured']
    stats = {}
    for context, mean, counts in [('A', [[1,2],[2,5],[1,3]], [10,20,30]),
                                  ('B', [[3,1],[4,2],[3,5]], [10,40,50])]:
        stats[context] = dict(labels=np.array(['non-targeting','p','q']), positions=np.array([0,1]),
                              mean=np.array(mean, dtype=float), counts=np.array(counts))
        np.savez(tmp_path/f'{context}-statistics.npz', **stats[context])
    config = {'model': {'weight_cell_unit':100, 'gene_chunk':2, 'alpha':1, 'ntc_conditioning':False}}
    model = fit(tmp_path, genes, {'training_contexts':['A','B']}, config)
    design = np.column_stack([np.tile(np.eye(2), (2,1)), np.ones(4)])
    w = np.array([.2,.3,.4,.5])
    y = np.concatenate([s['mean'][1:]-s['mean'][0] for s in stats.values()])
    coef = np.linalg.solve(design.T@(w[:,None]*design)+np.eye(3), design.T@(w[:,None]*y))
    np.testing.assert_allclose(np.vstack([model['effects'][:,:2],model['beta'][0,:2]]), coef, rtol=1e-6)
    np.testing.assert_array_equal(model['beta'][1:], 0)
    np.testing.assert_array_equal(model['effects'][:,2], 0)
    expected = coef[:2]+coef[2]
    for s in stats.values():
        np.testing.assert_allclose(predict(model, s, ['p','q'], 'linear')[:,:2], expected, rtol=1e-6)
    config['model'].pop('ntc_conditioning')
    legacy = fit(tmp_path, genes, {'training_contexts':['A','B']}, config)
    config['model']['ntc_conditioning'] = True
    enabled = fit(tmp_path, genes, {'training_contexts':['A','B']}, config)
    for key in legacy:
        np.testing.assert_array_equal(legacy[key], enabled[key])


def test_reused_inputs_reject_changed_scope_or_file_and_never_link_source(tmp_path, monkeypatch):
    import json
    import yaml
    import pytest
    import data
    monkeypatch.setattr(data, 'ROOT', tmp_path)
    source = tmp_path/'experiments/e/outputs/source'
    destination = tmp_path/'experiments/e/outputs/candidate'
    (source/'cache').mkdir(parents=True)
    config = {key: {'fixed': True} for key in ('benchmark','data','fit_scope','generation','evaluation')}
    config['seed'] = 1
    config['evaluation']['arms'] = ['zero','shared','source','linear']
    binding = {'node_id': 'source'}
    (source/'metrics.json').write_text(json.dumps({'status':'completed', 'evaluation_completed':True, 'research':binding}))
    (source/'config.yaml').write_text(yaml.safe_dump(dict(config, research=binding)))
    cached = source/'cache/A-statistics.npz'
    cached.write_bytes(b'frozen input statistics')
    manifest = {'source_run_id':'source', 'source_metrics':data.ref(source/'metrics.json'),
                'source_config':data.ref(source/'config.yaml'),
                'files':[{'ref':data.ref(cached), 'destination':'cache/A-statistics.npz'}]}
    path = tmp_path/'reuse.json'; path.write_text(json.dumps(manifest))
    config['reuse'] = data.ref(path)
    config['evaluation']['arms'] = ['linear']
    data.import_cached_inputs(destination, config)
    copied = destination/'cache/A-statistics.npz'
    assert copied.read_bytes() == cached.read_bytes() and copied.stat().st_ino != cached.stat().st_ino
    config['evaluation']['new_protocol'] = True
    with pytest.raises(ValueError, match='evaluation protocol'):
        data.import_cached_inputs(destination, config)
    config['evaluation'].pop('new_protocol')
    config['seed'] = 2
    with pytest.raises(ValueError, match='frozen seed'):
        data.import_cached_inputs(destination, config)
    config['seed'] = 1
    copied.write_bytes(b'tampered destination')
    with pytest.raises(ValueError, match='destination checksum'):
        data.import_cached_inputs(destination, config)
    assert cached.read_bytes() == b'frozen input statistics'
    cached.write_bytes(b'tampered source')
    with pytest.raises(ValueError, match='source checksum'):
        data.import_cached_inputs(destination, config)
