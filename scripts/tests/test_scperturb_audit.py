from pathlib import Path
import sys
import tempfile
import unittest
import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'dossier'))
from scperturb_audit import modality, numeric_audit, layer_semantics, label_summary, metadata_for


class ModalityTests(unittest.TestCase):
    def test_mixed_parent_modality_requires_file_resolution(self):
        self.assertEqual(modality(['RNA + protein'], ['CD3']), 'mixed_or_unspecified')
        self.assertEqual(modality(['RNA + protein (protein)'], ['CD3']), 'protein')
        self.assertEqual(modality(['RNA + protein (RNA)'], ['CD3']), 'RNA')
        self.assertEqual(modality([], ['CD3']), 'unknown')

    def test_count_layer_name_does_not_override_numeric_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            with h5py.File(Path(temp) / 'x.h5', 'w') as handle:
                x = handle.create_dataset('counts', data=[[0., 1.], [2., -0.5]])
                result = numeric_audit(x)
                self.assertEqual(result['stored_values_checked'], 4)
                self.assertFalse(result['nonnegative_integer_compatible'])
                self.assertEqual(layer_semantics('RNA', result)['count_methods'], 'not_applicable')
                self.assertEqual(layer_semantics('protein', {**result, 'negative': 0, 'noninteger': 0})['count_methods'], 'not_applicable')

    def test_unknown_is_not_control(self):
        result = label_summary(np.array([None, 'None', 'nan', 'control', 'GFP+GENE'], dtype=object))
        self.assertEqual(result['missing_or_sentinel_rows'], 3)
        self.assertIn('control', result['values'])

    def test_ambiguous_study_rows_not_arbitrarily_chosen(self):
        metadata = [{'Index (=FirstauthorLastauthorYear)': 'Study', 'dataset_index': 'RNA'},
                    {'Index (=FirstauthorLastauthorYear)': 'Study', 'dataset_index': 'protein'}]
        rows, status = metadata_for('Study_unknown', metadata)
        self.assertEqual(len(rows), 2)
        self.assertEqual(status, 'study_only')
        rows, status = metadata_for('Study_RNA', metadata)
        self.assertEqual(len(rows), 1)
        self.assertEqual(status, 'exact_file')


if __name__ == '__main__':
    unittest.main()
