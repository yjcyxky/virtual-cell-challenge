from pathlib import Path
import json
import sys
import tempfile
import unittest
import h5py
import numpy as np
import pandas as pd
from scipy import sparse
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'dossier'))
from scperturb_response_core import deep_responses,describe_conditions
from convert_seurat_cache import frame


class ScperturbResponseTests(unittest.TestCase):
    def test_complete_condition_responses_separate_stimulus_and_unmatched_targets(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);n=122;genes=['G','B','C','D'];x=np.full((n,4),2.)
            x[30:60,0]=0;x[90:120,0]=10;x[120,0]=5000;x[121]=0
            path=root/'source.h5ad';sx=sparse.csc_matrix(x)
            with h5py.File(path,'w') as h:
                frame(h.create_group('obs'),pd.DataFrame(index=['c'+str(i) for i in range(n)]));frame(h.create_group('var'),pd.DataFrame(index=genes))
                m=h.create_group('X');m.attrs['encoding-type']='csc_matrix';m.attrs['shape']=sx.shape;m['data']=sx.data;m['indices']=sx.indices;m['indptr']=sx.indptr
            control=np.array([True]*30+[False]*30+[True]*30+[False]*32)
            background=np.array([0]*60+[1]*62)
            d=pd.DataFrame({'row_index':range(n),'biological_background_index':background,'single_target':np.where(control,None,'G'),
                'target_kind':'single_gene_source_target','deep_response_candidate':~control,'control_eligible':control,
                'response_background':['a']*60+['b']*60+['unmatched','b'],
                'source_guide_for_consistency':['g1' if i%2 else 'g2' for i in range(n)],'condition_identity_supported':True,
                'analysis_condition':np.where(control,'control','G')})
            c=pd.DataFrame({'computed_numeric_valid':True,'computed_total_expression':x.sum(axis=1),'computed_detected_genes':(x>0).sum(axis=1),
                'inferred_type':'unknown','state__test':x[:,0]})
            mapping=pd.DataFrame({'source_gene':genes,'mapped_symbol':genes,'many_to_one_mapping':False,'symbol_vs_ensembl':'consistent'})
            result=deep_responses('synthetic',path,c,d,mapping,root/'deep',workers=1)
            self.assertEqual(result['actual_tasks'],2);self.assertEqual(result['task_status_counts'],{'completed':2});self.assertEqual(result['resampling_rows'],40)
            self.assertEqual(result['null_reference_intersections'],0);self.assertFalse(list((root/'deep').glob('temporary-*')))
            tasks=json.loads((root/'deep/tasks.json').read_text());bybg={t['biological_background_index']:t for t in tasks}
            self.assertEqual(bybg[1]['n_source_candidate_cells'],32);self.assertEqual(bybg[1]['n_matched_target_cells'],30)
            self.assertEqual(bybg[0]['target_RNA']['RNA_ratio'],0.)
            self.assertAlmostEqual(bybg[1]['target_RNA']['RNA_ratio'],2.5,places=5)
            description=describe_conditions(c,d,root/'description')
            self.assertEqual(description['all_conditions'],5);self.assertEqual(description['null_reference_intersections'],0)
            rows=pd.read_parquet(root/'description/condition-contrasts.parquet')
            self.assertTrue((rows.loc[rows.analysis_condition=='control','status']=='not_applicable').all())


if __name__=='__main__':unittest.main()
