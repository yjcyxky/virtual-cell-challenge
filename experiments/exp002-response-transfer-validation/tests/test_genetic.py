import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

import h5py
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from genetic import guide_identity, group_specs, CORE
from evaluate import load_group, prediction_experiment
from summarize import balanced_metrics
from collect import VARIANTS


class GeneticScopeTest(unittest.TestCase):
    def test_collapsed_label_cannot_hide_multiple_targets_or_control_roles(self):
        approved = {'NGFR', 'SERPINF1', 'IRF1'}
        self.assertEqual(guide_identity('Frangieh', 'NGFR_3;SERPINF1_3', 'NGFR', approved)[1], 'multiple_targets_or_mixed_roles')
        self.assertEqual(guide_identity('Frangieh', 'NGFR_3;NO_SITE_47', 'NGFR', approved)[0], None)
        self.assertEqual(guide_identity('Frangieh', 'NGFR_1;NGFR_3', 'NGFR', approved), ('NGFR', 'single_gene_target'))
        self.assertEqual(guide_identity('Dixit', 'p_sgIRF1_2', 'IRF1', approved), ('IRF1', 'single_gene_target'))
        self.assertEqual(guide_identity('Dixit', 'p_sgIRF1_2', 'NGFR', approved)[1], 'guide_source_target_disagreement')
        self.assertEqual(guide_identity('Frangieh', 'NO_SITE_47', 'control', approved)[1], 'non_targeting_control_not_used')
        self.assertEqual(guide_identity('Frangieh', 'ONE_NON-GENE_SITE_531', 'control', approved)[1], 'intergenic_control')
        self.assertNotEqual(guide_identity('Frangieh', 'ONE_NON-GENE_SITE_531;NO_SITE_47', 'control', approved)[1], 'intergenic_control')

    def test_primary_groups_do_not_treat_two_K562_panels_as_two_cell_types(self):
        for name, (panels, minimum, time_comparison) in group_specs().items():
            self.assertFalse(any(p.startswith(('GxE', 'Jiang')) for p in panels))
            self.assertTrue(all(p.startswith('scPerturb:') for p in panels) if name.startswith('Cas9') else all(p in CORE for p in panels))
            if 'cross_study_K562' in name:
                self.assertEqual(sum(p.startswith('replogle:K562') for p in panels), 1)
                self.assertEqual(minimum, 2)
            if 'time' in name:
                self.assertTrue(time_comparison)

    def test_background_balance_does_not_reward_more_targets_in_one_line(self):
        frame = pd.DataFrame({'group': ['study'] * 3, 'balance_group': ['a', 'a', 'b'],
                              'canonical_target': ['p', 'q', 'p'], 'MSE': [0., 0., 2.],
                              'zero_MSE': [1., 1., 1.], 'correlation': [0., 0., 0.]})
        self.assertEqual(balanced_metrics(frame, 0)['balanced_MSE'], 1.)

    def test_single_donor_prediction_is_fixed_and_blind_to_test_response(self):
        rng = np.random.default_rng(83)
        control = rng.uniform(0, 1, (2, 7, 110))
        shape = (2, 3, 7, 110)
        data = {'target': control[:, None] + rng.normal(0, .1, shape),
                'matched_control': np.broadcast_to(control[:, None], shape),
                'target_variance_mean': np.full(shape, .001), 'matched_control_variance_mean': np.full(shape, .001),
                'control': control, 'control_var': np.full(control.shape, .001), 'available': np.ones(shape[:3], bool),
                'excluded': np.zeros((3, 110), bool), 'genes': [f'g{i}' for i in range(110)],
                'targets': ['p', 'q', 'r'], 'lines': ['held', 'train']}
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            a, b = Path(temp) / 'a', Path(temp) / 'b'
            a.mkdir(); b.mkdir()
            prediction_experiment(data, a, minimum_training=1, genetic=True)
            data['target'][0] += 10
            prediction_experiment(data, b, minimum_training=1, genetic=True)
            for row in json.loads((a / 'hyperparameters.json').read_text()):
                self.assertEqual(row['lambda'], 1.)
                self.assertIn('not_identifiable', row['details']['status'])
            with h5py.File(a / 'predictions.h5') as first, h5py.File(b / 'predictions.h5') as second:
                for scale in ['control_only', 'source_matched']:
                    root = scale + '/held/'
                    for name in first[scale + '/held']:
                        if name != 'truth':
                            np.testing.assert_allclose(first[root + name][:], second[root + name][:], equal_nan=True)
                    np.testing.assert_equal(first[root + 'train_quadrant'][:], 0)
                    np.testing.assert_equal(first[root + 'shared'][:], first[root + 'quadrant_gated'][:])

    def test_aggregate_variance_and_split_estimand_are_not_invented(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            reports = []
            for ci in range(2):
                folder = root / str(ci); folder.mkdir()
                tasks = pd.DataFrame({'task_uid': [f'{ci}a', f'{ci}b'], 'canonical_target': ['p', 'p']})
                tasks.to_parquet(folder / 'tasks.parquet', index=False)
                support = [{'task_uid': t, 'variant': v, 'status': 'not_estimable' if ci == 0 and t.endswith('b') and v != 'full' else 'completed'}
                           for t in tasks.task_uid for v in VARIANTS]
                pd.DataFrame(support).to_parquet(folder / 'task-support.parquet', index=False)
                with h5py.File(folder / 'moments.h5', 'w') as hf:
                    hf['safe_symbol'] = np.array(['g1', 'g2'], dtype=h5py.string_dtype())
                    hf['target_excluded'] = np.zeros((2, 2), bool)
                    for key in ['target', 'matched_control', 'target_variance_mean', 'matched_control_variance_mean']:
                        hf[key] = np.ones((2, 7, 2))
                np.savez(folder / 'control-moments.npz', mean=np.ones((7, 2)), variance_mean=np.ones((7, 2)))
                reports.append({'context_id': str(ci), 'panel_id': str(ci), 'source_metadata': {'cell_line': str(ci)}})
            data = load_group(root, reports, genetic=True)
            self.assertTrue(np.isnan(data['target_variance_mean'][0, 0]).all())
            self.assertTrue(data['available'][0, 0, 0])
            self.assertFalse(data['available'][0, 0, 1:].any())
