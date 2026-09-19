from pathlib import Path
import sys
import tempfile
import unittest
import json
import h5py
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'dossier'))
from profile_tahoe import analyze_background


class TahoeConditionTests(unittest.TestCase):
    def test_vehicle_comparison_excludes_itself_and_resume_verifies_artifacts(self):
        with tempfile.TemporaryDirectory() as root:
            counts=Path(root)/'counts';annotations=Path(root)/'annotations';output=Path(root)/'out'
            for p in [counts,annotations,output,output/'conditions',output/'temporary-vectors']:p.mkdir()
            folder=output/'temporary-vectors';x=np.r_[np.zeros((10,2)),np.full((10,2),10.),np.full((25,2),5.)]
            np.save(folder/'vectors.npy',x);np.save(folder/'types.npy',np.zeros(45,int));np.save(folder/'order.npy',np.arange(45));np.save(folder/'pointers.npy',[0,10,20,45])
            (folder/'metadata.json').write_text(json.dumps({'names':['state__one','state__two'],'types':['unknown']}))
            pd.DataFrame({'condition_index':[0,1,2],'background_index':0,'source_drug':['DMSO_TF','DMSO_TF','drug'],
                'observed_cells':[10,10,25],'eligible_profile_cells':[10,10,25]}).to_parquet(annotations/'conditions.parquet')
            with h5py.File(counts/'condition-profiles.h5','w') as h:h['mean_logCP10K']=[[0.,0.],[10.,10.],[5.,5.]]
            background={'background_index':0,'n_DMSO':20};identity={'fixture':'frozen'}
            result=analyze_background(background,str(counts),str(annotations),str(output),identity)
            with h5py.File(output/'conditions/000/gene-effects.h5','r') as h:np.testing.assert_array_equal(h['effect_vs_plate_line_DMSO'][:],[[-10,-10],[10,10],[0,0]])
            frame=pd.read_parquet(output/'conditions/000/diagnostics.parquet');self.assertEqual(frame.n_control.tolist(),[10,10,20]);self.assertEqual(result['resampling_rows'],20)
            self.assertEqual(analyze_background(background,str(counts),str(annotations),str(output),identity),result)
            with open(output/'conditions/000/diagnostics.parquet','ab') as f:f.write(b'changed')
            with self.assertRaises(ValueError):analyze_background(background,str(counts),str(annotations),str(output),identity)


if __name__=='__main__':unittest.main()
