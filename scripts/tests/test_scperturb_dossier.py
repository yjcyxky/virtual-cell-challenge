from pathlib import Path
import sys
import tempfile
import unittest
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'dossier'))
from compose_scperturb_dossier import checked_copy,annotation_contract
from rna import hash_file


class DossierTests(unittest.TestCase):
    def test_inference_confidence_and_complete_source_rows_are_enforced(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);summary={'cells':2,'file':'study','input_sha256':'raw','pooled_types':{'unknown':2}}
            cells=pd.DataFrame({'row_index':[0,1],'record_id':['x','y'],'input_sha256':['raw']*2,
                'probability_correct':[None,None],'confidence_calibration':['uncalibrated']*2,'inferred_type':['unknown']*2,'source_obs__label':['A','B'],
                **{'state__'+str(i):[0.,1.] for i in range(12)}})
            cells.to_parquet(root/'cells.parquet',index=False)
            self.assertEqual(annotation_contract(root,summary)['records'],2)
            cells.loc[1,'probability_correct']=0.9;cells.to_parquet(root/'cells.parquet',index=False)
            with self.assertRaisesRegex(ValueError,'inference_contract'):annotation_contract(root,summary)
            digest=hash_file(root/'cells.parquet');checked_copy(root,root/'copy',{'cells.parquet':digest})
            (root/'cells.parquet').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'artifact_changed'):checked_copy(root,root/'copy2',{'cells.parquet':digest})


if __name__=='__main__':unittest.main()
