from pathlib import Path
import sys
import tempfile
import unittest
import h5py
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'dossier'))
from refine_tahoe_dossier import repeat_evidence,canonical_condition


class TahoeRepeatTests(unittest.TestCase):
    def test_response_agreement_is_separate_from_background(self):
        with tempfile.TemporaryDirectory() as root:
            source=Path(root)/'source';output=Path(root)/'out';(source/'counts').mkdir(parents=True);output.mkdir()
            pd.DataFrame({'condition_index':range(5),'plate':['plate6','plate14','plate6','plate14','plate6'],'cell_line_id':['line']*5,
                'source_drug':['DMSO_TF','DMSO_TF','drug','drug','missing_other_plate'],'source_drugname_drugconc':["[('DMSO_TF',0,'uM')]","[('DMSO_TF',0,'uM')]","[('drug',1,'uM')]","[('drug',1,'uM')]","[('missing',1,'uM')]"],
                'eligible_profile_cells':[20]*5,'sample':list('abcde')}).to_parquet(source/'counts/conditions.parquet')
            with h5py.File(source/'counts/condition-profiles.h5','w') as h:h['mean_logCP10K']=[[1,2,3],[5,3,1],[2,5,7],[6,6,5],[1,1,1]]
            rows,base=repeat_evidence(source,output);drug=next(r for r in rows if r['complete_drug_dose_unit_key']==canonical_condition("[('drug',1,'uM')]"))
            self.assertAlmostEqual(drug['response_correlation'],1);self.assertAlmostEqual(drug['response_difference_RMS'],0)
            self.assertGreater(base[0]['baseline_difference_RMS'],0)
            self.assertEqual(sum(r['status']=='not_estimable' for r in rows),1)


if __name__=='__main__':unittest.main()
