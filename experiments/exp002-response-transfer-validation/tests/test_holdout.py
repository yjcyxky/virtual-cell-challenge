import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from evaluate import prediction_experiment


class HoldoutTest(unittest.TestCase):
    def test_held_out_response_cannot_change_its_predictions_or_tuning(self):
        rng = np.random.default_rng(777)
        c, p, g = 3, 4, 120
        control = rng.uniform(0, 1, (c, 7, g)).astype('float32')
        response = rng.normal(0, .2, (p, g))
        target = control[:, None] + response[None, :, None]
        data = {'target': target, 'matched_control': np.broadcast_to(control[:, None], target.shape),
                'target_variance_mean': np.full(target.shape, 1e-5),
                'matched_control_variance_mean': np.full(target.shape, 1e-5),
                'control': control, 'control_var': np.full(control.shape, 1e-5),
                'available': np.ones((c, p, 7), bool), 'excluded': np.zeros((p, g), bool),
                'genes': [f'gene{i}' for i in range(g)], 'targets': [f'target{i}' for i in range(p)],
                'lines': ['held', 'train_a', 'train_b']}
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            a, b = Path(temp) / 'a', Path(temp) / 'b'
            a.mkdir(); b.mkdir()
            prediction_experiment(data, a)
            data['target'] = data['target'].copy()
            data['target'][0] += rng.normal(0, 10, data['target'][0].shape)
            prediction_experiment(data, b)
            old = json.loads((a / 'hyperparameters.json').read_text())
            new = json.loads((b / 'hyperparameters.json').read_text())
            self.assertEqual([r for r in old if r['held_cell_line'] == 'held'],
                             [r for r in new if r['held_cell_line'] == 'held'])
            with h5py.File(a / 'predictions.h5') as first, h5py.File(b / 'predictions.h5') as second:
                for scale in ['control_only', 'source_matched']:
                    for name in first[scale + '/held']:
                        if name != 'truth':
                            np.testing.assert_allclose(first[scale + '/held/' + name][:],
                                                       second[scale + '/held/' + name][:], equal_nan=True)


if __name__ == '__main__':
    unittest.main()
