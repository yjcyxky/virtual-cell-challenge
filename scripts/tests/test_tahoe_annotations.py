import json
from pathlib import Path
import sys
import tempfile
import unittest
import h5py
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'dossier'))
from tahoe_annotations import prepare_models


class TahoeBackgroundTests(unittest.TestCase):
    def test_local_vehicle_background_never_borrows_another_plate(self):
        with tempfile.TemporaryDirectory() as root:
            counts=Path(root)/'counts';refs=Path(root)/'refs';output=Path(root)/'out'
            for p in [counts,refs,output]:p.mkdir()
            pd.DataFrame({'mapped_symbol':['G'+str(i) for i in range(20)],'many_to_one_mapping':False,'symbol_vs_ensembl':'match'}).to_parquet(counts/'gene-mapping.parquet')
            pd.DataFrame({'condition_index':[0,1,2],'plate':['p1','p1','p2'],'cell_line_id':['line']*3,'source_drug':['DMSO_TF','drug','drug'],
                'sample':['a','b','c'],'eligible_profile_cells':[10,100,20]}).to_parquet(counts/'conditions.parquet')
            with h5py.File(counts/'condition-profiles.h5','w') as h:h['mean_logCP10K']=np.array([np.arange(20),np.arange(20)+100,np.arange(20)+200],np.float32)
            (refs/'gene_sets.json').write_text(json.dumps({'profiles':{'p':{'lineage':'p','genes':['G'+str(i) for i in range(15)]}},
                'states':{'s':{'genes':['G'+str(i) for i in range(5)],'reference':'fixture'}}}))
            groups=prepare_models(counts,refs,output);baselines=np.load(output/'background-means.npz')['mean_logCP10K']
            np.testing.assert_array_equal(baselines[0],np.arange(20));np.testing.assert_array_equal(baselines[1],np.arange(20)+200)
            self.assertEqual(groups[0]['n_DMSO'],10);self.assertEqual(groups[1]['n_DMSO'],0)
            self.assertIn('fallback',groups[1]['background_method'])


if __name__=='__main__':unittest.main()
