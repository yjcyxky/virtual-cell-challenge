import gzip
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'dossier'))
from gxe2_capture import guide_calls,CaptureText


class CaptureTests(unittest.TestCase):
    def test_source_rank_one_exception_and_full_read_denominator(self):
        names=np.asarray(['A','B','C','D','E','F','G','H','I','J'])
        # No guide exceeds 30%, but first/second >=2 permits the first guide.
        values=np.array([20,10,9,9,9,9,9,9,9,9])
        calls,maximum,ratio,tie=guide_calls(np.arange(10),values,names)
        assert calls==['A'] and maximum==20 and np.isclose(ratio,20/102) and tie
        assert guide_calls(np.array([],int),np.array([],int),names)==([],None,None,False)
    
    
    def test_multiple_high_fraction_guides_are_preserved(self):
        calls,maximum,ratio,tie=guide_calls(np.arange(3),np.array([4,4,2]),np.array(['A','B','C']))
        assert calls==['A','B'] and maximum==4 and ratio==.4 and tie
    
    
    def test_literal_nul_is_audited_before_csv_parser(self):
        with tempfile.TemporaryDirectory() as temporary:
            self.check_nul(Path(temporary))
    
    def check_nul(self,tmp_path):
        path=tmp_path/'capture.gz'
        with gzip.open(path,'wt') as f:f.write('sample\tbarcode\tguide\t\x00\t2\n')
        with CaptureText(path) as stream:
            frame=pd.read_csv(stream,sep='\t',header=None,keep_default_na=False)
            assert stream.nuls==1 and stream.lines==1
            assert frame.iloc[0,3]=='' and frame.iloc[0,4]==2


if __name__=="__main__":unittest.main()
