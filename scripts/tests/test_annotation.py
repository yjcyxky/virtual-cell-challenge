import json
from pathlib import Path
import sys
import unittest
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'dossier'))
from annotation import marker_model, decide, annotate, state_model, validate_records
from render import render
from rna import value_hash


class AnnotationTests(unittest.TestCase):
    def profiles(self):
        return {'A': {'genes': [f'g{i}' for i in range(10)], 'lineage': 'immune'},
                'B': {'genes': [f'g{i}' for i in range(10, 20)], 'lineage': 'immune'}}

    def test_marker_shortage_does_not_force_label(self):
        model = marker_model(['g1', 'g2'], self.profiles())
        result, scores = annotate(np.array([[3., 0.]]), ['g1'], model)
        self.assertEqual(result.inferred_type.iloc[0], 'unknown')
        self.assertEqual(result.inference_status.iloc[0], 'not_estimable')
        self.assertTrue(result.probability_correct.isna().all())

    def test_no_reference_or_unique_axis(self):
        model = marker_model([None, None], {})
        result, _ = annotate(np.zeros((2, 2)), ['x', 'y'], model)
        self.assertEqual(result.inference_reason.tolist(), ['no_applicable_reference'] * 2)
        model = marker_model(['g1', 'g1'], self.profiles())
        self.assertEqual(model['coverage'][0]['measured_markers'], 0)

    def test_method_conflict_retains_only_shared_lineage(self):
        model = marker_model([f'g{i}' for i in range(20)], self.profiles())
        frame = decide(np.array([[.4, .1]]), np.array([[.2, 1.]]), np.array([[8, 8]]), model)
        self.assertEqual(frame.inferred_type.iloc[0], 'unknown')
        self.assertEqual(frame.inferred_lineage.iloc[0], 'immune')
        self.assertTrue(frame.method_conflict.iloc[0])
        model['lineages'][1] = 'neural'
        frame = decide(np.array([[.4, .1]]), np.array([[.2, 1.]]), np.array([[8, 8]]), model)
        self.assertEqual(frame.inferred_lineage.iloc[0], 'unknown')

    def test_clear_markers_remain_uncalibrated_not_ground_truth(self):
        model = marker_model([f'g{i}' for i in range(30)], self.profiles())
        result, scores = annotate(np.array([[3.] * 10 + [0.] * 20]), ['g1'], model)
        self.assertEqual(result.inferred_type.iloc[0], 'A')
        self.assertEqual(result.inferred_subtype.iloc[0], 'unknown')
        self.assertEqual(result.confidence_calibration.iloc[0], 'uncalibrated')
        self.assertTrue(result.probability_correct.isna().all())
        self.assertTrue(result.target_is_reference_marker.iloc[0])
        payload = json.loads(result.to_json(orient='records'))
        page = render({'tables': [{'title': 'test', 'rows': payload}]})
        self.assertIn('uncalibrated', page)
        self.assertIn('"probability_correct": null', page)

    def test_wrong_identity_is_rejected_not_joined(self):
        frame = pd.DataFrame({'row_index': [0], 'input_sha256': ['a'], 'record_id': [value_hash(['a', 0])]})
        validate_records(frame, 'a', 1)
        with self.assertRaisesRegex(ValueError, 'identity_mismatch'):
            validate_records(frame, 'b', 1)
        frame.loc[0, 'record_id'] = 'wrong'
        with self.assertRaisesRegex(ValueError, 'hash_mismatch'):
            validate_records(frame, 'a', 1)

    def test_state_background_fixed_and_missing_is_not_zero(self):
        genes = [f'g{i}' for i in range(1000)]
        sets = {'present': {'genes': genes[:20], 'reference': 'fixed'},
                'absent': {'genes': ['absent'], 'reference': 'fixed'}}
        a, ca = state_model(genes, sets, np.arange(1000))
        b, cb = state_model(genes, sets, np.arange(1000))
        np.testing.assert_array_equal(a, b)
        self.assertEqual(ca, cb)
        self.assertAlmostEqual(float(a[:, 0].sum()), 0, places=6)
        self.assertEqual(ca[1]['status'], 'not_estimable')


if __name__ == '__main__':
    unittest.main()
