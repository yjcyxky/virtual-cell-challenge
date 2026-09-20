import sys
from pathlib import Path
import unittest
import tempfile
import h5py
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'dossier'))
from compose_heterogeneity_dossier import joint_summary,comparison_design
from profile_heterogeneity_supplements import association
from profile_heterogeneity_baselines import verify_archived_components


class DecisionDenominatorTests(unittest.TestCase):
    def test_absolute_and_response_summaries_use_identical_observations(self):
        frame=pd.DataFrame({'absolute':[.9,.95,.8,None],'response':[.2,None,.4,.1]})
        result=joint_summary(frame,'absolute','response','paired')
        self.assertEqual(result['joint_finite_rows'],2)
        self.assertEqual(result['missing_at_least_one_metric'],2)
        self.assertAlmostEqual(result['A_quantiles_on_same_rows']['median'],.85)
        self.assertAlmostEqual(result['B_quantiles_on_same_rows']['median'],.3)

    def test_unobserved_or_constant_factors_do_not_become_zero_associations(self):
        d=pd.DataFrame({'factor':[1.]*12,'outcome':np.arange(12,dtype=float)})
        result=association(d,'factor','outcome','source')
        self.assertEqual(result['status'],'not_estimable');self.assertIsNone(result['Spearman_rho']);self.assertIsNone(result['p_value'])
        d.loc[:3,'factor']=None;result=association(d,'factor','outcome','source')
        self.assertEqual(result['complete_observations'],8);self.assertEqual(result['missing_observations'],4)
        self.assertEqual(result['reason'],'fewer_than_10_complete_observations')

    def test_source_condition_and_confounded_time_contrasts_are_distinct(self):
        self.assertEqual(comparison_design('GxE2:["A172","DMSO",0.0]','GxE2:["A172","drug",1.0]'),'GxE2_same_source_line_different_drug_or_dose')
        self.assertEqual(comparison_design('GxE2:["A172","drug",1.0]','GxE2:["U87MG","drug",1.0]'),'GxE2_different_source_line_same_drug_and_dose')
        self.assertIn('time_library_capture_confounded',comparison_design('replogle:K562_essential','replogle:K562_gwps'))

    def test_unestimable_archived_vectors_must_remain_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'components.h5'
            with h5py.File(path,'w') as h:
                h.create_dataset('task_uid',data=np.asarray(['observed','unsupported'],dtype=h5py.string_dtype()))
                g=h.create_group('type')
                for name,value in [('total_common_support',1),('composition',.25),('within',.75),('difference_from_full_matched_effect',0)]:
                    g.create_dataset(name,data=np.array([[value],[np.nan]],dtype=np.float32))
            d=pd.DataFrame({'task_uid':['observed','unsupported'],'native_gene_row':[0,1],'partition':['type','type'],'status':['completed','not_estimable']})
            checked=verify_archived_components(path,d)
            self.assertEqual(checked[0]['unestimable_vectors_all_NaN'],1)
            self.assertEqual(checked[0]['maximum_archived_identity_residual'],0)
            with h5py.File(path,'r+') as h:h['type/within'][1,0]=0
            with self.assertRaisesRegex(ValueError,'missingness'):
                verify_archived_components(path,d)
