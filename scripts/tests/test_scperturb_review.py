from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'dossier'))
from review_scperturb import review


class SourceReviewTests(unittest.TestCase):
    def card(self, name):
        return {'file': name + '.h5ad', 'intervention_obs': ['CRISPR-cas9'],
                'matrices': {'X': {'semantics': {'value': 'original'}}}, 'RNA_deep_assessment': {}}

    def test_primary_interpretation_does_not_overwrite_source_label(self):
        c = review(self.card('SunshineHein2023'))
        self.assertEqual(c['intervention_obs'], ['CRISPR-cas9'])
        self.assertEqual(c['primary_source_review']['intervention'], 'CRISPRi')

    def test_processed_raw_container_and_numeric_codes_not_treated_as_truth(self):
        c = review(self.card('JoungZhang2023_atlas'))
        self.assertEqual(c['matrices']['X']['semantics']['count_methods'], 'not_applicable')
        self.assertEqual(c['RNA_deep_assessment']['target_name_mapping']['status'], 'blocked')
        c = review(self.card('GehringPachter2019'))
        self.assertEqual(c['RNA_deep_assessment']['dose_mapping']['status'], 'blocked')


if __name__ == '__main__':
    unittest.main()
