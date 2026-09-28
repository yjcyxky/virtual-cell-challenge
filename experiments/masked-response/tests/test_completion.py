import sys
from pathlib import Path
import importlib.util
import numpy as np
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
import model
from model import shared_responses, response_basis, complete_responses, predict


def test_shared_matches_weighted_estimator_and_historical_numerics(tmp_path):
    stats = {
        'A': dict(labels=np.array(['NTC', 'p', 'q']), positions=np.array([0, 1]),
                  mean=np.array([[1, 2], [3, 2], [0, 6]], dtype=np.float32), counts=np.array([10, 20, 10])),
        'B': dict(labels=np.array(['NTC', 'p', 'r']), positions=np.array([1, 2]),
                  mean=np.array([[3, 1], [6, 5], [2, 1]], dtype=np.float32), counts=np.array([10, 40, 30]))}
    got = shared_responses(stats, ['A', 'B'], ['a', 'b', 'c', 'never'], 100, 1)
    np.testing.assert_array_equal(got['shared'], [[2, 2, 4, 0], [-1, 4, 0, 0], [0, -1, 0, 0]])
    np.testing.assert_array_equal(got['observed'], [[1,1,1,0], [1,1,0,0], [0,1,1,0]])
    # A measured zero (r,c) is supervised; (q,c) is not, despite equal stored values.
    assert got['observed'][2,2] and not got['observed'][1,2]
    for c, s in stats.items():
        np.savez(tmp_path/f'{c}-statistics.npz', **s)
    source = Path(__file__).resolve().parents[2]/'init-linear/src/model.py'
    spec = importlib.util.spec_from_file_location('historical_model', source)
    historical = importlib.util.module_from_spec(spec); spec.loader.exec_module(historical)
    old = historical.fit(tmp_path, ['a','b','c','never'], {'training_contexts':['A','B']},
                         {'model': {'weight_cell_unit':100, 'gene_chunk':1, 'alpha':1}})
    np.testing.assert_array_equal(got['shared'], old['shared'])


def test_completion_recovers_known_rank_one_and_preserves_measured_zero():
    basis = np.ones((1, 3))/np.sqrt(3)
    shared = np.array([[2., 0., 0.], [0., 0., 0.], [5., -3., 7.]])
    mask = np.array([[1,0,0], [1,0,0], [1,1,1]], dtype=bool)
    result = complete_responses(shared, mask, basis, 0.)
    np.testing.assert_allclose(result[0], [2,2,2], atol=1e-12)
    np.testing.assert_array_equal(result[mask], shared[mask])
    np.testing.assert_array_equal(result[1], [0,0,0])
    shrunk = complete_responses(shared, mask, basis, 1/3)
    np.testing.assert_allclose(shrunk[0], [2,1,1], atol=1e-12)


def test_pca_ignores_missing_placeholders_and_recovers_training_subspace():
    rng = np.random.default_rng(31)
    complete = rng.normal(size=(12, 2)) @ rng.normal(size=(2, 8))
    response = np.vstack([complete, np.arange(8)])
    mask = np.ones_like(response, dtype=bool); mask[-1, 4:] = False
    spec = {'kind':'pca', 'dimensions':2, 'fit_seed':7, 'pca_oversample':3, 'pca_power_iterations':2}
    basis, audit = response_basis(response, mask, None, spec)
    np.testing.assert_allclose(complete @ basis.T @ basis, complete, atol=1e-10)
    response[-1] = 1e20
    again, _ = response_basis(response, mask, None, spec)
    np.testing.assert_array_equal(basis, again)
    assert audit['complete_training_targets'] == 12


def test_true_random_have_matched_rank_coverage_and_randomization_degrees():
    rng = np.random.default_rng(51)
    incidence = sparse.csr_matrix(rng.uniform(size=(30, 15)) < .4, dtype=float)
    shared = rng.normal(size=(10, 30)); observed = np.ones_like(shared, dtype=bool)
    spec = {'kind':'program', 'dimensions':4, 'fit_seed':7, 'swaps_per_edge':5}
    real, _ = response_basis(shared, observed, incidence, spec)
    random, audit = response_basis(shared, observed, incidence, dict(spec, kind='random_program'))
    for b in (real, random):
        np.testing.assert_allclose(b @ b.T, np.eye(4), atol=1e-12)
        assert np.all(np.linalg.norm(b, axis=0) > 0)
    assert audit['randomization']['gene_degrees_and_module_sizes_preserved']
    assert audit['randomization']['edge_overlap_fraction'] < .8
    assert not np.allclose(real.T @ real, random.T @ random)


def test_fit_excludes_heldout_labels_and_preserves_global_fallback(tmp_path, monkeypatch):
    for context, positions, labels in [('A',[0,1,2],['NTC','p','q','s','t']),
                                       ('B',[1,2,3],['NTC','q','s','t'])]:
        rng = np.random.default_rng(len(labels))
        np.savez(tmp_path/f'{context}-statistics.npz', labels=np.array(labels), positions=positions,
                 mean=rng.normal(size=(len(labels),len(positions))).astype(np.float32),
                 counts=np.full(len(labels),10))
    matrix = sparse.csr_matrix(np.array([[1,1,0],[1,0,1],[0,1,1],[1,1,1]], dtype=float))
    monkeypatch.setattr(model, 'pathway_incidence', lambda *args: (np.arange(4), ['a','b','c'],matrix))
    cfg = {'model':{'weight_cell_unit':100, 'gene_chunk':2, 'completion_alpha':.1},
           'representation':{'kind':'pca','dimensions':1,'fit_seed':7,'pca_oversample':1,
                             'pca_power_iterations':2,'pathways':{},'identities':{}}}
    first = model.fit(tmp_path, ['a','b','c','d','never'], {'training_contexts':['A','B']}, cfg)
    (tmp_path/'H1-statistics.npz').write_bytes(b'forbidden heldout data')
    second = model.fit(tmp_path, ['a','b','c','d','never'], {'training_contexts':['A','B']}, cfg)
    for key in ('effects','shared','basis','observed'):
        np.testing.assert_array_equal(first[key], second[key])
    np.testing.assert_array_equal(first['effects'][first['observed']], first['shared'][first['observed']])
    assert not first['effects'][:,-1].any()
    output = predict(first, None, ['p','unknown'], 'linear')
    assert not output[1].any()


def test_fixed_evaluator_resolves_local_model_and_checkpoint_roundtrip(tmp_path):
    import evaluation
    import data
    assert evaluation.predict is model.predict
    assert Path(data.__file__).parent == Path(__file__).resolve().parents[2]/'init-linear/src'
    original = dict(targets=np.array(['p']), genes=np.array(['a','b']), effects=np.array([[1.,2.]]))
    np.savez_compressed(tmp_path/'state.npz', **original)
    restored = dict(np.load(tmp_path/'state.npz'))
    np.testing.assert_array_equal(predict(original, None, ['p'], 'linear'),
                                  predict(restored, None, ['p'], 'linear'))
