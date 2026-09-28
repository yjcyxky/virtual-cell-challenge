"""Leakage, missing measurement, and official scorer contract regressions."""
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'dossier'))
from challenge import (ChallengeIdentity, build_splits, reference_indices,
                       scorer_config, scorer_contract, split_ntc, target_partition)


class ChallengeContractTests(unittest.TestCase):
    def setUp(self):
        hgnc = pd.DataFrame([
            ['A', 'Approved', 'OLD_A', '', 'ENSG1'],
            ['B', 'Approved', '', 'AMBIG', 'ENSG2'],
            ['C', 'Approved', '', 'AMBIG', 'ENSG3'],
            ['MYO18A', 'Approved', 'TIAF1', '', 'ENSG4'],
        ], columns=['symbol', 'status', 'prev_symbol', 'alias_symbol', 'ensembl_gene_id'])
        self.identity = ChallengeIdentity(['OLD_A', 'B', 'C', 'MYO18A', 'TIAF1'], hgnc)

    def test_official_historical_label_and_order_win_over_hgnc(self):
        self.assertEqual(self.identity.resolve('A', 'ENSG1')[0], 'OLD_A')
        self.assertEqual(self.identity.resolve('TIAF1', 'ENSG_OLD_TIAF1')[0], 'TIAF1')
        self.assertEqual(self.identity.resolve('MYO18A', 'ENSG4')[0], 'MYO18A')
        self.assertIsNone(self.identity.resolve('TIAF1', 'ENSG4')[0])

    def test_conflicting_identity_and_duplicate_columns_are_missing(self):
        self.assertIsNone(self.identity.resolve('B', 'ENSG3')[0])
        self.assertIsNone(self.identity.resolve('AMBIG')[0])
        mapping = self.identity.features(['OLD_A', 'A', 'B', 'UNKNOWN'], ['ENSG1', 'ENSG1', 'ENSG2', ''])
        self.assertEqual(mapping.measured.tolist(), [False, False, True, False])
        self.assertEqual(mapping.loc[2, 'official_position'], 1)

    def test_ntc_split_is_physical_disjoint_capped_and_reorder_invariant(self):
        frame = pd.DataFrame({'physical_id': [f'c{i}' for i in range(120)],
                              'batch': ['b1']*80+['b2']*40})
        first = split_ntc(frame, 99, 24)
        self.assertEqual(first.eq('input').sum(), 24)
        self.assertEqual(first.eq('score').sum(), 60)
        self.assertEqual(set(frame.loc[first.eq('input'), 'physical_id']) &
                         set(frame.loc[first.eq('score'), 'physical_id']), set())
        shuffled = frame.sample(frac=1, random_state=12)
        pd.testing.assert_series_equal(first, split_ntc(shuffled, 99, 24).reindex(frame.index))

    def test_ntc_duplicate_identity_is_not_a_second_observation(self):
        frame = pd.DataFrame({'physical_id': ['x', 'x'], 'batch': ['b', 'b']})
        with self.assertRaisesRegex(ValueError, 'duplicate_physical'):
            split_ntc(frame, 0, 10)

    def test_reference_does_not_upsample_and_diagnostic_leaves_training_rows(self):
        frame = pd.DataFrame({'physical_id': ['c'+str(i) for i in range(11)]})
        self.assertEqual(len(reference_indices(frame, 0)), 11)
        holdout = reference_indices(frame, 0, diagnostic=True)
        self.assertEqual(len(holdout), 5)
        self.assertFalse(set(holdout) & set(frame.index.difference(holdout)))

    def test_global_target_and_study_holdouts_remove_all_matching_labels(self):
        rows = []
        for context in ['H1', 'K562', 'RPE1', 'HepG2', 'Jurkat']:
            for i in range(40):
                target = f'G{i}'
                rows.append({'context': context, 'target': target, 'eligible_cells': 20,
                    'structurally_eligible': True, 'S0_reference_cells': 10,
                    'partition': target_partition(target, 13), 'official_target': i < 3})
        splits = build_splits(pd.DataFrame(rows), {'minimum_reference_cells': 4})
        self.assertEqual(len(splits), 15)
        for s in splits:
            if s['scenario'] in ['S1', 'S3']:
                self.assertFalse(set(s['training_targets']) & set(s['validation_targets']))
                for targets in s['evaluation_targets'].values():
                    self.assertFalse(set(s['training_targets']) & set(targets))
            if s['id'] == 'S4-REPLOGLE-2022':
                self.assertEqual(set(s['evaluation_contexts']), {'K562', 'RPE1'})
                self.assertFalse({'K562', 'RPE1'} & set(s['training_contexts']))

    def test_scientific_preset_changes_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'scientific_preset'):
            scorer_config(control_source='pred')
        config = scorer_config(device='cpu', num_threads=2)
        self.assertEqual(config.control_source, 'real')
        self.assertTrue(config.cache_strict)
        self.assertEqual(config.discrimination.exclusion_scope, 'panel')
        contract = scorer_contract()
        self.assertEqual(len(contract['scored_metrics']), 6)
        self.assertEqual(contract['anchors']['n_splits'], 5)

    def test_official_engine_computes_all_six_members(self):
        import anndata as ad
        from cell_eval2 import compute_metrics, aggregate_metrics_wide
        from challenge import build_reference_bundle, score_prediction
        rng = np.random.default_rng(31)
        genes = ['G'+str(i) for i in range(200)]
        labels = ['non-targeting', 'G0', 'G1', 'G2']
        blocks = []
        for i in range(4):
            means = np.full(200, 30.0)
            if i:
                direction = -1 if i == 2 else 1
                means[:20] += 20*direction
                means[20:40] -= 20*direction
                means[i*40:i*40+20] = 50
                means[i*40+20:(i+1)*40] = 10
            blocks.append(rng.poisson(means, size=(64, 200)))
        real = ad.AnnData(np.concatenate(blocks).astype(np.float32),
            obs=pd.DataFrame({'target_gene': np.repeat(labels, 64)}, index=[f'r{i}' for i in range(256)]),
            var=pd.DataFrame(index=genes))
        pred = real.copy()
        pred.obs_names = ['p'+str(i) for i in range(256)]
        result = compute_metrics(pred, real, config=scorer_config(device='cpu', num_threads=2))
        from cell_eval2.run import metric_output_names
        result = aggregate_metrics_wide(result, metrics=metric_output_names(scorer_config()))
        members = set(scorer_contract()['scored_metrics'])
        self.assertTrue(members <= set(result.columns), result.columns)
        for member in members:
            values = result.filter(result['statistic'] == 'mean')[member].to_numpy()
            self.assertTrue(np.isfinite(values).any(), member)
        # Real bundle, five official replicate splits, and strict normalized score.
        # Synthetic data validates the evaluator interface, never model efficacy.
        with tempfile.TemporaryDirectory(prefix='vcc-scorer-test-') as temp:
            bundle = Path(temp)/'bundle'
            build_reference_bundle(real, bundle, 'synthetic-contract-test', device='cpu', num_threads=2)
            score = score_prediction(pred, real, bundle, Path(temp)/'score', device='cpu', num_threads=2)
            self.assertEqual(set(score['normalized']), members)
            self.assertTrue(np.isfinite(score['Overall']))
            no_perturbations = real[real.obs.target_gene.eq('non-targeting')].copy()
            with self.assertRaises(ValueError):
                build_reference_bundle(no_perturbations, Path(temp)/'invalid-bundle',
                                       'invalid-reference', device='cpu', num_threads=2)


if __name__ == '__main__':
    unittest.main()
