"""Auxiliary-resource behavior: uncertain identity and non-count numerical semantics."""
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'dossier'))
from profile_priors import safe_mapping,Numeric,weights

class PriorTests(unittest.TestCase):
    def test_identifier_conflict_is_not_official_coverage(self):
        hgnc=pd.DataFrame({'status':['Approved']*2,'symbol':['A','B'],'hgnc_id':['H1','H2'],'entrez_id':['1','2'],
            'alias_symbol':['ALIAS','ALIAS'],'prev_symbol':['',''],'ensembl_gene_id':['ENSG1','ENSG2']})
        result=safe_mapping(['A','A','ALIAS','old'],['1','2','missing','2'],'entrez_id',hgnc,['A','B'])
        self.assertEqual(result.safe_symbol.iloc[[0,3]].tolist(),['A','B'])
        self.assertTrue(result.safe_symbol.iloc[[1,2]].isna().all())
        self.assertTrue(result.identifier_conflict.iloc[1])
        self.assertTrue(result.safe_mapping_ambiguous.iloc[2])

    def test_continuous_signed_values_and_missing_are_distinct(self):
        numeric=Numeric(2);numeric.add([[-2,.5],[np.nan,1]])
        frame=numeric.frame(['A','B'])
        self.assertEqual(frame.loc[0,'mean'],-2)
        self.assertEqual(frame.loc[0,'missing_NaN'],1)
        self.assertEqual(frame.loc[1,'noninteger'],1)
        self.assertEqual(numeric.outside_probability,1)
        self.assertEqual(numeric.summary()['values_checked'],4)

    def test_weight_scan_detects_nonfinite_without_claiming_gene_features(self):
        with tempfile.TemporaryDirectory() as root:
            root=Path(root);path=root/'model.safetensors';output=root/'report';output.mkdir()
            header=json.dumps({'tensor':{'dtype':'F32','shape':[2],'data_offsets':[0,8]}}).encode()
            path.write_bytes(struct.pack('<Q',len(header))+header+np.asarray([0,np.inf],dtype='<f4').tobytes())
            result=weights(path,output)
            self.assertEqual(result['nonfinite'],1)
            self.assertFalse(result['gene_features_computed'])
            self.assertIsNone(result['official_axis_covered'])
            path.write_bytes(path.read_bytes()[:-1])
            with self.assertRaisesRegex(ValueError,'invalid_tensor_shape_or_offsets'):weights(path,output)

if __name__=='__main__':unittest.main()
