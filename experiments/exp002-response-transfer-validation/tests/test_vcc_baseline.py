import json
from pathlib import Path
import sys
import tempfile
import unittest

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from vcc_baseline import calibrate, outer_prediction, shrink, balanced_templates, gene_factors, emit_counts
from vcc_submission import generate, audit_prediction


class SubmissionBaselineTest(unittest.TestCase):
    def test_calibration_recovers_conserved_signal_and_ignores_missing_labels(self):
        response = np.tile(np.array([[1., -2., np.nan], [2., 1., np.nan]])[None], (4, 1, 1))
        available = np.ones((4, 2), bool)
        fit = calibrate(response, available, [0., .25, .5, .75, 1.])
        self.assertEqual(fit['single']['lambda'], 1.)
        self.assertEqual(fit['multiple']['lambda'], 1.)
        prediction = shrink(np.array([[1., np.nan, -2.]]), np.array([[1, 0, 2]]), fit)
        np.testing.assert_equal(prediction, [[1., 0., -2.]])

    def test_outer_model_does_not_see_held_response(self):
        rng = np.random.default_rng(51)
        response = rng.normal(size=(5, 6, 20))
        available = np.ones((5, 6), bool)
        first = outer_prediction(response, available, 0, [0., .25, .5, .75, 1.])
        response[0] = 9999
        second = outer_prediction(response, available, 0, [0., .25, .5, .75, 1.])
        self.assertEqual(first[0], second[0])
        np.testing.assert_equal(first[2], second[2])

    def test_templates_are_unique_balanced_and_reproducible(self):
        guides = np.repeat([f'guide_{i}' for i in range(46)], 400)
        rows = balanced_templates(guides, 400, 73)
        self.assertEqual(len(set(rows)), 400)
        sizes = pd.Series(guides[rows]).value_counts()
        self.assertEqual(len(sizes), 46)
        self.assertLessEqual(sizes.max() - sizes.min(), 1)
        np.testing.assert_equal(rows, balanced_templates(guides, 400, 73))

    def test_count_emission_is_identity_at_zero_and_preserves_library_in_expectation(self):
        cells = np.tile(np.array([[100, 200, 300]], dtype=np.int32), (500, 1))
        np.testing.assert_equal(emit_counts(cells, np.ones(3), 24), cells)
        changed = emit_counts(cells, np.array([.1, 1., 1.]), 24)
        self.assertTrue(np.issubdtype(changed.dtype, np.integer))
        self.assertGreaterEqual(changed.min(), 0)
        self.assertLess(abs(changed.sum(1).mean() - 600), .2)
        self.assertLess(changed[:, 0].mean(), 15)

    def test_missing_readouts_and_zero_control_do_not_create_infinite_factors(self):
        config = {'maximum_gene_factor': 10, 'on_target_remaining_fraction': .1}
        factors, diag = gene_factors(np.array([0., 1., .01, 1.]), np.array([1., 4., 4., -5.]),
                                     np.array([1, 0, 2, 1]), 3, config)
        np.testing.assert_equal(factors, [1., 1., 10., .1])
        self.assertEqual(diag['zero_control_unsupported_ratio'], 1)
        self.assertEqual(diag['capped_gene_factors'], 1)

    def test_generated_h5ad_roundtrip_and_audit_reject_swapped_axis(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); official = root / 'official'; official.mkdir()
            output = root / 'run'; output.mkdir()
            genes, targets, contexts = ['G1', 'G2', 'G3'], ['G1', 'G2'], ['A', 'B']
            for context in contexts:
                obs = pd.DataFrame({'context': context, 'target_gene': 'non-targeting',
                                    'ntc_id': np.repeat([f'n{i}' for i in range(46)], 400)},
                                   index=[f'{context}_{i}' for i in range(18400)])
                ad.AnnData(X=sparse.csr_matrix(np.tile([[4, 8, 12]], (18400, 1))), obs=obs,
                           var=pd.DataFrame(index=genes)).write_h5ad(official / f'context_{context}.h5ad')
            np.savez(output / 'model.npz', genes=np.array(genes), targets=np.array(targets),
                     response=np.zeros((2, 3)), donor_counts=np.ones((2, 3)))
            config = {'cells_per_perturbation': 4, 'seed': 3, 'maximum_gene_factor': 10,
                      'on_target_remaining_fraction': .1, 'max_nnz': 1000, 'max_counts_per_cell': 1000}
            manifest = {'contexts': contexts, 'per_context': {c: {'control_cells': 18400} for c in contexts}}
            result = generate(config, output, official, manifest, genes, targets)
            audit = audit_prediction(output / 'predictions.h5ad', genes, targets, contexts, 4, 1000, 1000)
            self.assertEqual(result['cells'], 16)
            self.assertEqual(audit['target_contexts'], 4)
            with h5py.File(output / 'predictions.h5ad') as h:
                self.assertEqual(h['X/indptr'].dtype, np.dtype('int64'))
            with self.assertRaises(AssertionError):
                audit_prediction(output / 'predictions.h5ad', genes[::-1], targets, contexts, 4, 1000, 1000)


if __name__ == '__main__':
    unittest.main()
