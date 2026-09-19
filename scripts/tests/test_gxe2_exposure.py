from pathlib import Path
import sys
import unittest
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'dossier'))
from gxe2_exposure import guide_comparison


class ExposureTests(unittest.TestCase):
    def test_guide_mixture_shift_does_not_create_matched_state_response(self):
        def frame(guides):
            return pd.DataFrame({'source_guide_id':guides,'source_CDS_replicate':['r1']*len(guides),
                                 'state':[0. if g=='a' else 10. for g in guides]})
        a=frame(['a','b','b','b']);b=frame(['a','a','a','b'])
        self.assertEqual(a.state.mean()-b.state.mean(),5.)
        summary,states=guide_comparison(a,b,['state'],'G')
        self.assertEqual(summary['source_guide_replicate_total_variation'],.5)
        self.assertEqual(states[0]['matched_state_mean_difference'],0.)

    def test_missing_replicate_and_ambiguous_sequence_are_not_estimated(self):
        a=pd.DataFrame({'source_guide_id':['a'],'source_CDS_replicate':['r1'],'state':[1.]})
        b=a.assign(source_CDS_replicate='r2')
        self.assertEqual(guide_comparison(a,b,['state'],'G')[0]['status'],'not_estimable')
        self.assertEqual(guide_comparison(a,a,['state'],'PRKCZ')[0]['status'],'not_estimable')
        self.assertEqual(guide_comparison(a.iloc[:0],a,['state'],'G')[0]['status'],'not_estimable')

    def test_nonfinite_dimensions_preserve_available_matching(self):
        a=pd.DataFrame({'source_guide_id':['a','b'],'source_CDS_replicate':['r1','r1'],'state':[3.,np.nan]})
        b=a.assign(state=[1.,4.])
        summary,states=guide_comparison(a,b,['state'],'G')
        self.assertEqual(states[0]['finite_matched_target_cells'],1)
        self.assertEqual(states[0]['matched_state_mean_difference'],2.)


if __name__=='__main__':unittest.main()
