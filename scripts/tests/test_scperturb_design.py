from pathlib import Path
import sys
import unittest
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'dossier'))
from scperturb_design import control_rule,gasperini_targets,build_design
from scperturb_adamson import identity_rows


class ScperturbDesignTests(unittest.TestCase):
    def test_literal_controls_and_mixed_guides_do_not_become_NTC(self):
        self.assertFalse(control_rule('XieHon2017',{'perturbation':'control','guide_id':'control'})[0])
        self.assertFalse(control_rule('Schraivogel_test',{'perturbation':'control'})[0])
        self.assertTrue(control_rule('Schraivogel_test',{'perturbation':'non-targeting_001'})[0])
        self.assertFalse(control_rule('Frangieh_test',{'perturbation':'control','guide_id':'STAT1_3;NO_SITE_411'})[0])
        self.assertTrue(control_rule('Frangieh_test',{'guide_id':'ONE_NON-GENE_SITE_290;NO_SITE_411'})[0])
        self.assertFalse(control_rule('Norman_test',{'guide_id':'no_reads_found'})[0])

    def test_enhancer_and_TSS_compositions_are_parsed_without_nperts(self):
        self.assertEqual(gasperini_targets('chr1.10201_top_two_GNPDA1_TSS_scrambled_2'),['GNPDA1_TSS','chr1.10201_top_two'])
        self.assertEqual(gasperini_targets('scrambled_2_scrambled_3'),[])
        self.assertIsNone(gasperini_targets('unknown_chr1.10201_top_two'))
        self.assertEqual(gasperini_targets('chr1:12-33_chr1:12-33'),['chr1:12-33'])

    def test_background_separates_stimulation_and_same_drug_zero_dose(self):
        obs=pd.DataFrame({'perturbation':['A','A','B','control'],'dose_value':['0','1','0',None],
                          'top_oligo':['A0','A1','B0',None],'cell_line':['L']*4})
        d,_=build_design('SrivatsanTrapnell2020_sciplex2',obs,set())
        self.assertEqual(d.control_eligible.tolist(),[True,False,True,False])
        self.assertEqual(d.response_background[0],d.response_background[1]);self.assertNotEqual(d.response_background[0],d.response_background[2])
        obs=pd.DataFrame({'perturbation':['G','G'],'perturbation_2':['stim','unstim'],'guide_id':['G1','G1']})
        d,_=build_design('DatlingerBock2017',obs,{'G'});self.assertNotEqual(*d.response_background.tolist())

    def test_original_GEM_identity_does_not_rewrite_collection_labels(self):
        obs=pd.DataFrame({'perturbation':['G',None]},index=['AAA','AAA-1'])
        meta=pd.DataFrame({'guide identity':['G','H'],'good coverage':[True,True],'UMI count':[10,10],'read count':[30,30]},index=['AAA-1','AAA-2'])
        r=identity_rows(['AAA-1','AAA-2'],meta,obs)
        self.assertEqual(r.original_GEM_group.tolist(),[1,2]);self.assertEqual(r.source_guide_label_concordant.tolist(),[True,False])
        self.assertTrue(pd.isna(r.collection_source_guide_label[1]));self.assertEqual(r.original_GEO_guide_label[1],'H')
        with self.assertRaisesRegex(ValueError,'row_scope'):identity_rows(['BBB-1','AAA-2'],meta,obs)


if __name__=='__main__':unittest.main()
