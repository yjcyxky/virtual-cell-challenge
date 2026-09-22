import sys
from pathlib import Path
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from estimators import classify, choose_hyperparameters, response_mean, score


class EstimationTests(unittest.TestCase):
    def test_equivalence_requires_precision_and_nonzero_is_separate(self):
        baseline = np.zeros((3, 4))
        response = np.array([[.3, 0, 0, .3], [.3, 0, .4, .3], [.3, 0, .4, .3]])
        variance = np.full_like(response, .000001)
        variance[:, 3] = 1  # Same point estimates, too uncertain for equivalence.
        labels = classify(baseline, np.full_like(baseline, .000001), response, variance, .1)
        np.testing.assert_array_equal(labels['response_state'], [1, 1, 2, 0])
        np.testing.assert_array_equal(labels['conserved_nonzero'], [1, 0, 0, 0])
        np.testing.assert_array_equal(labels['activity'], [2, 1, 2, 0])
        self.assertEqual(labels['quadrant'][2], 2)  # one inactive context retained

    def test_baseline_difference_does_not_preclude_conserved_response(self):
        baseline = np.array([[0., 0.], [.5, .5], [1., 1.]])
        response = np.full_like(baseline, .3)
        labels = classify(baseline, np.full_like(baseline, 1e-6), response, np.full_like(baseline, 1e-6), .1)
        np.testing.assert_array_equal(labels['quadrant'], [3, 3])
        self.assertTrue(labels['conserved_nonzero'].all())

    def test_missing_values_never_become_zero(self):
        x = np.array([[[1., np.nan]], [[3., np.nan]], [[99., 99.]]])
        prediction = response_mean(x, np.array([[1], [1], [0]], bool))
        self.assertEqual(prediction[0, 0], 2)
        self.assertTrue(np.isnan(prediction[0, 1]))

    def test_train_only_tuning_detects_shared_and_context_specific_signals(self):
        rng = np.random.default_rng(91)
        signal = rng.normal(size=(8, 200))
        shared = np.broadcast_to(signal, (3, 8, 200)).copy()
        available = np.ones((3, 8), bool)
        baseline = rng.normal(size=(3, 200))
        lam, _, _ = choose_hyperparameters(shared, available, baseline)
        self.assertEqual(lam, 1)
        prediction = response_mean(shared, available)
        truth = signal + rng.normal(scale=.01, size=signal.shape)
        self.assertGreater(score(truth, prediction)['relative_MSE_improvement'], .99)
        opposing = np.array([signal, -signal])
        lam, temp, detail = choose_hyperparameters(opposing, available[:2], baseline[:2])
        self.assertEqual(lam, 0)
        self.assertEqual(temp, .1)
        self.assertEqual(detail['kernel_status'], 'fixed_insufficient_training_contexts')


if __name__ == '__main__':
    unittest.main()
