from pathlib import Path
import sys
import numpy as np
from scipy import sparse
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from model import solve_ridge, fit
from evaluation import generate_counts


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
