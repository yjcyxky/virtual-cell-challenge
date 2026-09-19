from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'dossier'))
from profile_official_controls import validate_context, pooled_without, guide_halves
from response import grouped_moments


class OfficialControlTests(unittest.TestCase):
    def test_swapped_context_and_gene_order_are_rejected(self):
        obs = pd.DataFrame({'context': ['A'] * 400, 'target_gene': ['non-targeting'] * 400, 'ntc_id': ['ntc-1'] * 400})
        source = SimpleNamespace(obs=obs, var=pd.DataFrame(index=['g1', 'g2']), shape=(400, 2))
        manifest = {'n_genes': 2, 'control_label': 'non-targeting', 'per_context': {'A': {'control_cells': 400, 'n_ntc_ids': 1}}}
        validate_context('A', source, ['g1', 'g2'], manifest)
        with self.assertRaisesRegex(ValueError, 'context_identity'):
            validate_context('B', source, ['g1', 'g2'], manifest)
        with self.assertRaisesRegex(ValueError, 'gene_axis'):
            validate_context('A', source, ['g2', 'g1'], manifest)

    def test_pooled_moments_exclude_own_guide_exactly(self):
        rng = np.random.default_rng(1)
        x = rng.normal(size=(12, 5))
        groups = np.repeat(['a', 'b', 'c'], 4)
        m = grouped_moments(x, groups, ['a', 'b', 'c'])
        pooled = pooled_without(m, 1)
        direct = x[groups != 'b']
        self.assertEqual(pooled['n'], len(direct))
        np.testing.assert_allclose(pooled['mean'], direct.mean(axis=0))
        np.testing.assert_allclose(pooled['variance'], direct.var(axis=0, ddof=1))

    def test_disjoint_reference_halves_do_not_contain_own_guide(self):
        x = np.arange(40).reshape(8, 5).astype(float)
        groups = np.repeat(['a', 'b'], 4)
        def reference(x):
            means, sizes = guide_halves(x, groups, ['a', 'b'], 3)
            sums = (means * sizes[:, :, :, None]).sum(axis=0)
            result = (sums - means[0] * sizes[0, :, :, None]) / (sizes.sum(axis=0) - sizes[0])[:, :, None]
            np.testing.assert_allclose(result.mean(axis=1), np.tile(x[4:].mean(axis=0), (20, 1)))
            return result
        first = reference(x)
        x[:4] += 1000
        np.testing.assert_allclose(reference(x), first)


if __name__ == '__main__':
    unittest.main()
