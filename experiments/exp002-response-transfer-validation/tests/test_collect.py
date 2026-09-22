import sys
from pathlib import Path
import unittest
import numpy as np
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from collect import moments, split_cells, geometry


class CollectionTests(unittest.TestCase):
    def test_shared_control_identity_and_disjointness(self):
        keys = np.array([f'barcode{i}' for i in range(13)])
        strata = np.array(['a'] * 4 + ['b'] * 9)
        halves = split_cells(keys, strata, 'shared H1', 20260921)
        order = np.arange(13)[::-1]
        replica = split_cells(keys[order], strata[order], 'shared H1', 20260921)
        np.testing.assert_array_equal(replica, halves[order])
        for b in ['a', 'b']:
            n = np.bincount(halves[strata == b], minlength=2)
            self.assertLessEqual(abs(n[0] - n[1]), 1)
            self.assertGreaterEqual(min(n), 2)
        self.assertFalse(set(np.where(halves == 0)[0]) & set(np.where(halves == 1)[0]))

    def test_dense_sparse_moments_and_missing_support(self):
        x = np.array([[0, 3], [2, 0], [4, 6], [0, 0]], dtype=np.float32)
        membership = np.array([[1, 1, 1, 1], [1, 0, 1, 0], [0, 0, 0, 0], [0, 1, 0, 0]], bool)
        for source in [x, sparse.csr_matrix(x)]:
            n, mean, variance = moments(source, np.arange(4), membership)
            np.testing.assert_allclose(mean[0], x.mean(0))
            np.testing.assert_allclose(variance[0], x.var(0, ddof=1) / 4)
            np.testing.assert_allclose(mean[1], [2, 4.5])
            np.testing.assert_allclose(variance[1], [4, 2.25])
            self.assertTrue(np.isnan(mean[2]).all())
            self.assertTrue(np.isnan(variance[3]).all())

    def test_high_absolute_correlation_does_not_require_zero_effect(self):
        baseline = np.linspace(0, 10, 200)
        target = baseline + np.tile([-.2, .2], 100)
        result = geometry(target, baseline, np.ones(200, bool))
        self.assertGreater(result['NTC_target_correlation'], .99)
        self.assertEqual(result['fraction_abs_delta_gt_0.1'], 1)


if __name__ == '__main__':
    unittest.main()
