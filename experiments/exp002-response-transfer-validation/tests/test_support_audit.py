import json
from pathlib import Path
import sys
import tempfile
import unittest
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from audit import audit_supplemental_support
from rna import hash_file


class SupportAuditTest(unittest.TestCase):
    def test_full_availability_does_not_imply_half_support_or_change_errors(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            folder = root / 'evaluation/cross_study_sensitivity'
            folder.mkdir(parents=True)
            ledger = pd.DataFrame([{'task_uid': line, 'canonical_target': 'P', 'cell_line': line, 'full_estimable': True}
                                   for line in ['A', 'B', 'C']])
            ledger.to_parquet(folder / 'task-ledger.parquet', index=False)
            rows = pd.DataFrame([{'held_cell_line': line, 'canonical_target': 'P', 'MSE': i + .25,
                                  'all_6_test_halves_supported': True, 'all_training_6_halves_supported': True}
                                 for i, line in enumerate(['A', 'B', 'C'])])
            hashes = {}
            for name in ['metrics.parquet', 'module-metrics.parquet']:
                rows.to_parquet(folder / name, index=False)
                hashes[name] = hash_file(folder / name)
            (folder / 'report.json').write_text(json.dumps({'artifacts': hashes}))
            support = pd.DataFrame([{'task_uid': line, 'variant': str(i),
                                     'status': 'not_estimable' if line == 'B' and i == 0 else 'completed'}
                                    for line in ['A', 'B', 'C'] for i in range(6)])
            result = audit_supplemental_support(root, root / 'audit', support)
            corrected = pd.read_parquet(folder / 'metrics.parquet')
            self.assertEqual(corrected.all_6_test_halves_supported.tolist(), [True, False, True])
            self.assertEqual(corrected.all_training_6_halves_supported.tolist(), [False, True, False])
            pd.testing.assert_series_equal(corrected.MSE, rows.MSE)
            self.assertFalse(result['prediction_and_error_values_changed'])


if __name__ == '__main__':
    unittest.main()
