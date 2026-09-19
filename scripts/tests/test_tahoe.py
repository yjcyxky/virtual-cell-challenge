import sys
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
import pyarrow as pa
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'dossier'))
from tahoe import decode,parse_compounds,metadata_join


class TahoeTests(unittest.TestCase):
    def test_only_verified_CLS_is_removed_and_other_negative_values_flagged(self):
        tokens=np.array([-1,-1,-1,0,1])
        batch=pa.record_batch({'genes':[[1,3,4],[1,4]],'expressions':[[-2.,2.,3.],[-2.,-1.]]})
        matrix,valid,facts=decode(batch,tokens,2)
        self.assertEqual(matrix.toarray().tolist(),[[2.,3.],[0.,-1.]])
        self.assertEqual(valid.tolist(),[True,False]);self.assertEqual(facts['CLS_markers'],2);self.assertEqual(facts['negative'],1)
        wrong=pa.record_batch({'genes':[[1,3]],'expressions':[[-1.,3.]]})
        with self.assertRaisesRegex(ValueError,'CLS_encoding'):decode(wrong,tokens,2)
        duplicate=pa.record_batch({'genes':[[1,3,3]],'expressions':[[-2.,1.,2.]]})
        with self.assertRaisesRegex(ValueError,'duplicate_gene_tokens'):decode(duplicate,tokens,2)

    def test_metadata_conflicts_and_vehicle_codes_remain_explicit(self):
        local=pd.DataFrame({'BARCODE_SUB_LIB_ID':['b1','b2'],'plate':['p1','p1'],'sample':['s1','s1'],'drug':['DMSO_TF']*2,'cell_line_id':['C1']*2})
        observations=pd.DataFrame({'BARCODE_SUB_LIB_ID':['b1'],'plate':['p1'],'sample':['wrong'],'drug':['DMSO_TF'],'cell_line':['C1']})
        samples=pd.DataFrame({'sample':['s1'],'plate':['p1'],'drug':['DMSO_TF']})
        result=metadata_join(local,observations,samples)
        self.assertEqual(result.metadata_obs_present.tolist(),[True,False]);self.assertFalse(result.metadata_condition_consistent.any())
        with self.assertRaisesRegex(ValueError,'nonunique_metadata'):metadata_join(local,pd.concat([observations,observations]),samples)
        dose=parse_compounds("[('DMSO_TF', 0.0, 'uM')]")[0];self.assertTrue(dose['vehicle_code_not_molar_dose'])
