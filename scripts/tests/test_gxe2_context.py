from pathlib import Path
import json
import sys
import tempfile
import unittest
import h5py
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'dossier'))
from convert_seurat_cache import frame
from gxe2_context import matched_rows,threshold_RNA,assess_context


class Gxe2ContextTests(unittest.TestCase):
    def test_threshold_is_separate_from_source_support_and_matching(self):
        cells=pd.DataFrame({'analysis_target':['NTC','G','NTC','G','G'],'source_CDS_gRNA_maxCount':[10,11,1,2,12],
            'source_assignment_supported':[True]*5,'source_batch':['a','a','b','b','c']})
        used,controls,observed=matched_rows(cells,'G',10)
        np.testing.assert_equal(used,[1]);np.testing.assert_equal(controls,[0]);self.assertEqual(observed,2)
        log=np.log1p(np.array([[10],[5],[20],[10],[2]],float))
        rows=threshold_RNA(cells,log,'G',0)
        self.assertAlmostEqual(rows[0]['RNA_ratio'],.5);self.assertAlmostEqual(rows[-1]['RNA_ratio'],.5)
        self.assertEqual(rows[0]['matched_target_cells'],2);self.assertEqual(rows[-1]['matched_target_cells'],1)

    def test_full_context_preserves_ineligible_cells_and_missing_declared_tasks(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);path=root/'counts.h5ad';n,g=32,12;gene_names=['g'+str(i) for i in range(g)]
            counts=np.full((n,g),2.);counts[16:28,0]=0
            with h5py.File(path,'w') as h:
                frame(h.create_group('obs'),pd.DataFrame({'sample':['s']*n},index=['c'+str(i) for i in range(n)]))
                frame(h.create_group('var'),pd.DataFrame(index=gene_names));h['X']=counts
            cells=pd.DataFrame({'CDS_view_row':range(n),'row_index':range(n),'source_barcode':['c'+str(i) for i in range(n)],
                'source_context':['test']*n,'source_assignment_supported':[True]*30+[False]*2,'source_CDS_gRNA_maxCount':[20]*n,
                'analysis_target':['NTC']*16+['g0']*12+['random']*2+[None]*2,'source_batch':['a']*n,'source_guide_id':['id']*n,
                'assignment_limitation':['supported']*30+['unassigned']*2,'computed_total_counts':counts.sum(axis=1),'computed_detected_genes':(counts>0).sum(axis=1)})
            genes=pd.DataFrame({'source_gene':gene_names});mapping=pd.DataFrame({'source_gene':gene_names,'mapped_symbol':gene_names,'many_to_one_mapping':[False]*g,'symbol_vs_ensembl':['consistent']*g})
            profiles={name:{'lineage':lineage,'genes':gene_names} for name,lineage in [('pluripotent_like','pluripotent'),('neuron_like','neural')]}
            reference={'profiles':profiles,'states':{'state':{'genes':gene_names[:4],'reference':'synthetic_test_reference'}}}
            vehicle={'weights':np.zeros((g,1)),'coverage':[{'status':'not_estimable'}]}
            result=assess_context(path,cells,genes,mapping,reference,['g0','missing','random'],vehicle,root/'out','test',{'test':True},workers=1)
            self.assertEqual(result['n_cells'],32);self.assertEqual(result['tasks'],3)
            tasks=json.loads((root/'out/tasks.json').read_text())
            self.assertEqual(next(t for t in tasks if t['target']=='missing')['status'],'not_estimable')
            self.assertEqual(next(t for t in tasks if t['target']=='random')['target_RNA']['status'],'not_applicable')
            annotated=pd.read_parquet(root/'out/cells.parquet')
            self.assertEqual(len(annotated),n);self.assertFalse(annotated.probability_correct.notna().any())
            self.assertFalse((root/'out/temporary-view').exists())


if __name__=='__main__':unittest.main()
