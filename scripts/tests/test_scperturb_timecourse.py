from pathlib import Path
import sys
import unittest
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'dossier'))
from scperturb_design import build_design
from refine_scperturb_timecourse import refine_rows


class TimecourseTests(unittest.TestCase):
    def test_source_hash_recovers_DMSO_and_hours_without_rewriting_obs(self):
        tags=['DMSO_3hr','Tram_3hr','DMSO_48hr','multiplet']
        obs=pd.DataFrame({'time':['3, 6, 12, 24, 48']*4,'perturbation':['Trametinib']*4,'perturbation_type':['drug']*4,
            'hash_tag':tags,'hash_assignment':tags,'channel':['A']*4,'dose_value':[.1]*4,'cell_line':['L']*4})
        design,_=build_design('McFarlandTsherniak2020',obs,set());result,summary=refine_rows(obs,design)
        self.assertEqual(result.control_eligible.tolist(),[True,False,True,False]);self.assertEqual(result.condition_identity_supported.tolist(),[True,True,True,False])
        self.assertEqual(result.response_background[0],result.response_background[1]);self.assertNotEqual(result.response_background[0],result.response_background[2])
        self.assertEqual(obs.perturbation.tolist(),['Trametinib']*4);self.assertEqual(summary['explicit_DMSO_hash_reference_records'],2)


if __name__=='__main__':unittest.main()
