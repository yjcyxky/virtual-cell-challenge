import unittest
import pandas as pd
from dossier.cross_source_coverage import gene_coverage, target_coverage, resolve_target


class CoverageBoundaryTests(unittest.TestCase):
    def test_target_identity_requires_source_intervention_evidence(self):
        approved={'TAFAZZIN','WWTR1'};aliases={'TAZ':approved,'ENSG1':{'TAFAZZIN'}}
        self.assertIsNone(resolve_target('TAZ',None,approved,aliases)[0])
        self.assertEqual(resolve_target('TAZ','ENSG1',approved,aliases)[0],'TAFAZZIN')
        self.assertEqual(resolve_target('WWTR1','ENSG1',approved,aliases),(None,'source_symbol_Ensembl_conflict'))

    def test_absent_zero_unresolved_and_species_are_distinct(self):
        source = pd.DataFrame({'source_gene': ['A', 'B'], 'mapped_symbol': ['A', 'B']})
        official = pd.DataFrame({'source_gene': ['A', 'B', 'C', 'old?'], 'mapped_symbol': ['A', 'B', 'C', None], 'mapping_status': ['approved'] * 3 + ['unmapped']})
        actual = gene_coverage(source, official, [0, 3])
        self.assertEqual(actual.status.tolist(), ['measured_all_zero', 'measured_nonzero', 'outside_native_panel', 'official_identifier_unresolved'])
        self.assertTrue(gene_coverage(source, official, [0, 3], False).status.eq('not_applicable_species_or_modality').all())
        self.assertTrue(gene_coverage(source, official, [0, 3], None).status.eq('indeterminate_species_or_modality').all())

    def test_conflicting_and_many_to_one_features_not_summed_or_zeroed(self):
        source = pd.DataFrame({'source_gene': ['oldA', 'A2', 'B'], 'mapped_symbol': ['A', 'A', 'B'], 'symbol_vs_ensembl': ['consistent', 'consistent', 'conflict'], 'ensembl_candidates': ['A', 'A', 'C']})
        official = pd.DataFrame({'source_gene': ['A', 'B', 'C'], 'mapped_symbol': ['A', 'B', 'C'], 'mapping_status': ['approved'] * 3})
        self.assertTrue(gene_coverage(source, official, [0, 0, 0]).status.eq('ambiguous_or_conflicting_native_mapping').all())

    def test_collection_copy_and_DE_qualification_are_separate(self):
        targets = pd.DataFrame({'source_gene': ['A', 'B'], 'mapped_symbol': ['A', 'B']})
        tasks = pd.DataFrame({'canonical_target': ['A', 'A', 'A'], 'modality': ['CRISPRi', 'CRISPRi', 'CRISPRa'], 'effect_status': ['completed'] * 3, 'confirmed_collection_copy': [False, True, False], 'DE_status': ['not_estimable', 'not_estimable', 'completed']})
        result = target_coverage('p', targets, tasks)
        self.assertEqual(result.iloc[0].completed_CRISPRi_construct_tasks, 2)
        self.assertEqual(result.iloc[0].completed_CRISPRi_tasks_after_confirmed_copy_exclusion, 1)
        self.assertEqual(result.iloc[0].completed_DE_tasks, 0)
        self.assertEqual(result.iloc[1].status, 'no_observed_single_target_CRISPRi_response')
