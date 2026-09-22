import sys
from pathlib import Path
import unittest
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from summarize import balanced_metrics


class SummaryTest(unittest.TestCase):
    def test_large_condition_cannot_dominate_the_family_result(self):
        rows = [{'group': 'large', 'canonical_target': str(i), 'MSE': 4., 'zero_MSE': 4., 'correlation': 0.}
                for i in range(100)]
        rows += [{'group': 'small', 'canonical_target': str(i), 'MSE': 0., 'zero_MSE': 4., 'correlation': 1.}
                 for i in range(2)]
        result = balanced_metrics(pd.DataFrame(rows), bootstraps=0)
        self.assertEqual(result['relative_MSE_improvement'], .5)
        self.assertEqual(result['target_backgrounds'], 102)


if __name__ == '__main__':
    unittest.main()
