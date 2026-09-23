import json
from pathlib import Path
import sys
import tempfile
import unittest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / 'src'))

import numpy as np
import pandas as pd
from scipy import sparse
import xgboost as xgb

from dataset import safe_axis, target_reserved
from features import PairFeatures, canonical_map, relation_matrices, residualize
from training import Supervision, feature_rows, evaluate, final_fit
from vcc_submission import context_prediction


def feature_fixture(root):
    prior = root / 'priors'; prior.mkdir()
    graph = sparse.csr_matrix(np.ones((4, 4), np.float32) - np.eye(4, dtype=np.float32))
    for name in ['functional', 'physical']:
        sparse.save_npz(prior / (name + '.npz'), graph)
    sparse.save_npz(prior / 'pathways.npz', sparse.csr_matrix(np.ones((4, 2), np.float32)))
    np.save(prior / 'embedding.npy', np.arange(8, dtype=np.float32).reshape(4, 2))
    np.save(root / 'common-measured.npy', np.ones(4, bool))
    for name, rho in [('positive', .8), ('negative', -.8), ('held', .5)]:
        folder = root / 'contexts' / name; folder.mkdir(parents=True)
        state = {k: np.ones(4, np.float32) for k in ['mean','std','detection','neighbor_mean','neighbor_detection','measured']}
        state['global_state'] = np.ones(2, np.float32)
        np.savez(folder / 'state.npz', **state)
        for k, value in [('raw_correlation', rho), ('correlation', rho), ('split_gap', .1)]:
            np.save(folder / (k + '.npy'), np.full((4, 4), value, np.float32))
        labels = np.full((2, 4), rho, np.float32)
        labels[0, 0] = labels[1, 1] = np.nan
        np.savez(folder / 'labels.npz', targets=np.array([0, 1]), response=labels)
    return PairFeatures(root, ['positive', 'negative', 'held'])


class ContextModelTests(unittest.TestCase):
    def test_ambiguous_aliases_and_duplicate_measurements_are_not_merged(self):
        hgnc = pd.DataFrame({'symbol':['A','B'], 'alias_symbol':['amb|oldA','amb'], 'prev_symbol':[None,None]})
        mapping = canonical_map(hgnc, ['A','B'])
        self.assertNotIn('amb', mapping)
        self.assertEqual(mapping['oldA'], 0)
        source, dest = safe_axis(['A','oldA','B'], mapping)
        np.testing.assert_array_equal(source, [2])
        np.testing.assert_array_equal(dest, [1])

    def test_target_reservation_is_independent_of_context_and_order(self):
        targets = ['G' + str(i) for i in range(1000)]
        values = {p:target_reserved(p, 20) for p in targets}
        self.assertEqual(values, {p:target_reserved(p, 20) for p in reversed(targets)})
        self.assertTrue(150 < sum(values.values()) < 250)

    def test_residualization_removes_batch_and_depth_signal(self):
        rng = np.random.default_rng(4)
        batch = np.repeat(['a','b'], 50)
        depth = rng.normal(size=100)
        x = np.column_stack([depth * 3 + (batch == 'b') * 10, rng.normal(size=100)]).astype(np.float32)
        residual, dof = residualize(x, batch, depth)
        self.assertEqual(dof, 97)
        self.assertLess(np.max(np.abs(residual[:, 0])), 3e-6)
        self.assertLess(np.max(np.abs(residual[batch == 'a'].mean(0))), 1e-6)

    def test_zero_variance_and_unmeasured_relation_are_unknown(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            rng = np.random.default_rng(3)
            x = rng.uniform(.1, 2, size=(160, 4)).astype(np.float32)
            x[:, 1] = x[:, 0] * 2; x[:, 2] = 0
            config = {'seed':4,'minimum_relation_detections':10,'minimum_relation_degrees_freedom':20}
            relation_matrices(x, np.repeat('batch',160), rng.normal(size=160), np.array([1,1,1,0],bool), config, folder)
            correlation = np.load(folder / 'correlation.npy')
            self.assertAlmostEqual(float(correlation[0,1]), 1., places=5)
            self.assertTrue(np.isnan(correlation[2]).all())
            self.assertTrue(np.isnan(correlation[3]).all())

    def test_held_labels_cannot_change_training_rows_or_features(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); features = feature_fixture(root)
            metadata = {'genes':['a','b','c','d'],'contexts':['positive','negative','held']}
            config = {'genes_per_task':3,'validation_genes_per_task':2,'seed':4,'unseen_target_percent':0}
            supervision = Supervision(root, metadata)
            r1,y1,w1 = supervision.rows_for(['positive','negative'], config)
            x1,names = feature_rows(features,r1,['positive','negative'],'context_relation')
            supervision.values['held'][:] = 1000000
            r2,y2,w2 = supervision.rows_for(['positive','negative'], config)
            x2,_ = feature_rows(features,r2,['positive','negative'],'context_relation')
            np.testing.assert_array_equal(x1,x2)
            np.testing.assert_array_equal(y1,y2)
            np.testing.assert_array_equal(w1,w2)
            self.assertIn('relation_sign_conformity', names)

    def test_supervised_model_learns_context_dependent_relation_sign(self):
        with tempfile.TemporaryDirectory() as tmp:
            features = feature_fixture(Path(tmp))
            p = np.tile([0,1],100); g = np.tile([2,3],100)
            a,names = features.features('positive',p,g,['positive','negative'],'context_relation')
            b,_ = features.features('negative',p,g,['positive','negative'],'context_relation')
            matrix = xgb.DMatrix(np.vstack([a,b]),label=np.r_[np.ones(len(a)), -np.ones(len(b))],feature_names=names)
            model = xgb.train({'objective':'reg:squarederror','tree_method':'hist','max_depth':2,'eta':.2,'nthread':1,'seed':4},matrix,40)
            self.assertGreater(float(model.inplace_predict(a).mean()), .9)
            self.assertLess(float(model.inplace_predict(b).mean()), -.9)
            static_a,_ = features.features('positive',p,g,['positive','negative'],'static_pair')
            static_b,_ = features.features('negative',p,g,['positive','negative'],'static_pair')
            np.testing.assert_array_equal(static_a,static_b)
            used = model.get_score()
            self.assertTrue(any('correlation' in key or 'interaction' in key or 'conformity' in key for key in used))

    def test_reference_relationships_exclude_current_and_held_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            features = feature_fixture(Path(tmp))
            x,names = features.features('positive',[0],[2],['positive','negative'],'context_relation')
            self.assertAlmostEqual(float(x[0,names.index('reference_correlation')]),-.8,places=6)
            self.assertEqual(x[0,names.index('reference_relation_support')],1)

    def test_submission_uses_context_identity_not_array_position(self):
        model = {'response':np.array([[[1.,2.]],[[3.,4.]]]), 'response_support':np.ones((2,1,2),bool),
                 'contexts':np.array(['B','A'])}
        response,support = context_prediction(model,'A')
        np.testing.assert_array_equal(response,[[3,4]])
        self.assertTrue(support.all())
        with self.assertRaises(ValueError):
            context_prediction(model,'C')
        shared = {'response':np.array([[1.,2.]]),'donor_counts':np.array([[1,0]])}
        np.testing.assert_array_equal(context_prediction(shared,'A')[0],shared['response'])

    def test_complete_nested_training_and_final_context_prediction(self):
        class Tracker:
            summary = {}
            def log(self, value):
                pass
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); prepared = root / 'prepared'; prepared.mkdir()
            feature_fixture(prepared)
            config = json.loads((PROJECT / 'configs/context-relation.json').read_text())
            config.update(num_boost_round=5,early_stopping_rounds=2,bootstrap_repeats=5,
                          genes_per_task=3,validation_genes_per_task=3,unseen_target_percent=0)
            config['model_parameters'].update(nthread=1,max_depth=2,min_child_weight=1)
            metadata = {'genes':['a','b','c','d'],'contexts':['positive','negative','held'],
                        'official_contexts':['positive'],'official_targets':['a','b']}
            evaluation = evaluate(config,root,metadata,Tracker())
            self.assertEqual(evaluation['status'],'completed')
            self.assertEqual(len(evaluation['rounds']),9)
            fitted = final_fit(config,root,metadata,evaluation,Tracker())
            self.assertTrue(fitted['reserved_targets_released'])
            with np.load(root / 'model.npz') as model:
                prediction, support = context_prediction(model,'positive')
                self.assertEqual(prediction.shape,(2,4))
                self.assertTrue(np.isfinite(prediction).all())


if __name__ == '__main__':
    unittest.main()
