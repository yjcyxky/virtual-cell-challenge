import sys
from pathlib import Path
import unittest
import json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'dossier'))
from chemical_design import parse_hash,rt_cell_line


class ChemicalDesignTests(unittest.TestCase):
    def test_combo_components_and_RT_line_are_distinct_from_hash(self):
        result=parse_hash('plate6_0.1_DDR1IN_0.1_rep1','chemical4','A172')
        self.assertEqual(json.loads(result['components']),[['DDR1IN',.1],['Trametinib',.1]])
        only=parse_hash('plate6_1_vehicle_0_rep1','chemical4','A172')
        self.assertEqual(json.loads(only['components']),[['Trametinib',1.]])
        self.assertFalse(only['is_vehicle'])
        self.assertEqual(rt_cell_line('A01_G01_RT_BC_13_Lig_BC_16'),'A172')
        self.assertEqual(rt_cell_line('A01_G01_RT_BC_5_Lig_BC_16'),'T98G')
        self.assertEqual(rt_cell_line('A01_G01_RT_BC_96_Lig_BC_16'),'U87MG')
        self.assertIsNone(rt_cell_line('A01_G01_RT_BC_97_Lig_BC_16'))

    def test_GSC_zero_is_vehicle_without_discarding_source_compound(self):
        result=parse_hash('plate18_H3_GBM4_trametinib_0_rep_1','chemical3')
        self.assertTrue(result['is_vehicle']);self.assertEqual(result['source_hash_compound'],'trametinib')
        self.assertEqual(result['genetic_supervision_role'],'not_applicable_no_genetic_intervention')


if __name__=='__main__':unittest.main()
