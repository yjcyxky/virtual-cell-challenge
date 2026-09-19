import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'dossier'))
from response import grouped_moments, matched_effect, conditional_de, disjoint_control_indices, control_half_means
from profile_responses import task_result, build_cache, serial
from rna import hash_file
from test_rna import h5ad


class ResponseTests(unittest.TestCase):
    def test_batch_matching_removes_composition_difference(self):
        target = np.array([[1, 2]] * 30 + [[10, 4]] * 10, dtype=float)
        control = np.array([[1, 2]] * 10 + [[10, 4]] * 30, dtype=float)
        tb = np.array(['a'] * 30 + ['b'] * 10)
        cb = np.array(['a'] * 10 + ['b'] * 30)
        t = grouped_moments(target, tb, ['a', 'b'])
        c = grouped_moments(control, cb, ['a', 'b'])
        effect, metadata = matched_effect(t, c)
        np.testing.assert_allclose(effect, 0)
        self.assertNotEqual(target[:, 0].mean(), control[:, 0].mean())
        de, dm = conditional_de(t, c)
        np.testing.assert_allclose(de['q_bh'], 1)
        self.assertEqual(dm['unit'], 'observed_cells_conditional_on_sample')
        self.assertIsNone(dm['independent_biological_replicates'])

    def test_control_cannot_be_borrowed_from_other_batch(self):
        t = grouped_moments(np.ones((30, 3)), np.array(['a'] * 30), ['a', 'b'])
        c = grouped_moments(np.ones((30, 3)), np.array(['b'] * 30), ['a', 'b'])
        effect, meta = matched_effect(t, c)
        self.assertIsNone(effect)
        self.assertEqual(meta['reason'], 'no_same_batch_control')
        de, meta = conditional_de(t, c)
        self.assertIsNone(de)
        self.assertEqual(meta['status'], 'not_estimable')

    def test_null_draws_disjoint_matched_and_seeded(self):
        tb, cb = np.array(['a'] * 2 + ['b'] * 9), np.array(['a'] * 10 + ['b'] * 6)
        a, b = disjoint_control_indices(tb, cb, ['a', 'b'], np.random.default_rng(7))
        a2, b2 = disjoint_control_indices(tb, cb, ['a', 'b'], np.random.default_rng(7))
        self.assertEqual(len(set(a) & set(b)), 0)
        self.assertEqual(len(a), 5)
        self.assertEqual((cb[a] == 'a').sum(), 2)
        np.testing.assert_array_equal(a, a2)
        np.testing.assert_array_equal(b, b2)

    def test_known_effect_and_single_construct_limit_in_complete_task(self):
        rng = np.random.default_rng(9)
        control = rng.uniform(.5, 2, (80, 4))
        target = control.copy()
        target[:, 1] += 2
        batches = np.array(['a'] * 40 + ['b'] * 40)
        controls = grouped_moments(control, batches, ['a', 'b'])
        halves = control_half_means(control, batches, ['a', 'b'], 20, 4)
        rows = pd.DataFrame({'source_batch': batches, 'source_guide_id': ['gA|gB'] * 80,
                             'computed_total_counts': [100] * 80, 'computed_detected_genes': [4] * 80})
        result, genes, samples = task_result('target', target, target, rows, control, batches, ['a', 'b'],
                                            controls, controls, halves, ['target', 'response', 'null1', 'null2'], set(), 42)
        self.assertEqual(result['DE']['status'], 'completed')
        self.assertEqual(result['DEG_BH_excluding_target'], 1)
        self.assertAlmostEqual(genes.loc[1, 'effect_all_matched_cells'], 2)
        self.assertLess(genes.loc[1, 'q_bh'], .001)
        self.assertEqual(result['consistency']['guide']['status'], 'not_estimable')
        self.assertEqual(len(samples), 20)
        self.assertTrue(all(s['null_reference_intersection'] == 0 for s in samples))
        self.assertFalse(result['stability']['confidence_interval'])
        json.dumps(serial(result), allow_nan=False)

    def test_cache_preserves_real_h5ad_and_rejects_changed_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / 'counts.h5ad'
            h5ad(path, [[2, 3], [5, 0], [0, 0]])
            digest = hash_file(path)
            log, thin = build_cache(path, root / 'cache', digest, 42, chunk=1)
            np.testing.assert_allclose(np.expm1(log[:2]).sum(axis=1), [10000, 10000], rtol=1e-5)
            np.testing.assert_allclose(log, thin)
            self.assertEqual(hash_file(path), digest)
            del log, thin
            with (root / 'cache/logcp.npy').open('ab') as f:
                f.write(b'corruption')
            with self.assertRaisesRegex(ValueError, 'cache changed'):
                build_cache(path, root / 'cache', digest, 42, chunk=1)


if __name__ == '__main__':
    unittest.main()

class CachedControlPoolTests(unittest.TestCase):
    def test_cached_pools_preserve_seeded_draws_and_disjointness(self):
        from response import disjoint_control_indices
        target_batches=np.array(['a']*4+['b']*7)
        control_batches=np.array(['a']*12+['b']*10)
        pools={b:np.flatnonzero(control_batches==b) for b in ['a','b']}
        for seed in range(3):
            expected=disjoint_control_indices(target_batches,control_batches,['a','b'],np.random.default_rng(seed))
            cached=disjoint_control_indices(target_batches,control_batches,['a','b'],np.random.default_rng(seed),pools)
            for a,b in zip(expected,cached):np.testing.assert_array_equal(a,b)
            self.assertEqual(len(np.intersect1d(*cached)),0)
            self.assertEqual(len(cached[0]),9)


if __name__ == "__main__":
    unittest.main()
