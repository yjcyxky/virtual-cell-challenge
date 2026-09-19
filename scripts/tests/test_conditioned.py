"""Full conditional pipeline: source selection, control boundaries and annotation provenance."""
from pathlib import Path
import json
import sys
import tempfile
import unittest
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'dossier'))
from conditioned import assess_context,normalize_selection
from rna import scan,hash_file,mapping_audit
from test_rna import h5ad

class ConditionedTests(unittest.TestCase):
    def test_source_fields_preserved_and_condition_conflicts_not_silently_mixed(self):
        from profile_jiang import source_design
        frame=pd.DataFrame({'source_cell_type':['BXPC3']*3,'source_pathway':['IFNB']*3,'source_Batch_info':['Rep1','Rep2','Rep1'],
            'source_bc1_well':['B11']*3,'source_sample_ID':['sample_1']*3,'source_guide':['NTg1','KNTC1g1','g2'],
            'source_gene':['NT','KNTC1','G2'],'source_sample':['unknown_unknown','BXPC3_IFNB','BXPC3_IFNB']})
        result=source_design(frame,'IFNB')
        self.assertEqual(result.source_target_gene.tolist(),['non-targeting','KNTC1','G2'])
        self.assertEqual(result.source_label_inconsistent.tolist(),[True,False,False])
        self.assertEqual(result.source_sample.tolist(),frame.source_sample.tolist())
        self.assertNotEqual(result.source_batch.iloc[0],result.source_batch.iloc[1])
        with self.assertRaisesRegex(ValueError,'stimulus_file_vs_source_label_conflict'):source_design(frame,'IFNG')

    def test_full_context_retains_uncertainty_and_rejects_mixed_conditions(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);source=root/'source.h5ad';genes=[f'G{i}' for i in range(20)]
            values=np.random.default_rng(9).poisson(2,size=(64,20)).astype(float);values[40:,0]=0
            h5ad(source,values,genes=genes);digest=hash_file(source)
            qc,cells,var=scan(source,digest,'fixture')
            cells['source_context']='A_IFNB';cells['source_batch']='rep1_well1_library1';cells['source_task']=['NTC']*40+['G0']*24
            cells['source_target_gene']=['non-targeting']*40+['G0']*24;cells['source_guide_id']=['NT1']*40+['g1']*12+['g2']*12
            cells['source_label_inconsistent']=False;cells.loc[0,'source_label_inconsistent']=True;cells.loc[40,'source_label_inconsistent']=True
            mapping=pd.DataFrame({'source_gene':genes,'mapped_symbol':genes,'many_to_one_mapping':False,'symbol_vs_ensembl':'source_ensembl_unavailable'})
            ref=root/'references';ref.mkdir();(ref/'gene_sets.json').write_text(json.dumps({'profiles':{'missing':{'lineage':'epithelial','genes':['absent']*12}},'states':{'cycle':{'genes':genes[:6],'reference':'fixture'}},'local_inputs':{}}))
            (ref/'SHA256SUMS').write_text(hash_file(ref/'gene_sets.json')+'  gene_sets.json\n')
            result=assess_context(source,cells,var,mapping,ref,root/'out','A_IFNB',['epithelial'],'fixture',{'source_sha256':digest},workers=1)
            self.assertEqual(result['actual_tasks'],1);self.assertEqual(result['n_cells'],64)
            self.assertEqual(result['pooled_types'],{'unknown':64})
            saved=pd.read_parquet(root/'out/cells.parquet');self.assertTrue(saved.probability_correct.isna().all());self.assertFalse(saved.available_before_endpoint.any())
            self.assertEqual(saved.source_label_inconsistent.sum(),2)
            sensitivity=json.loads((root/'out/label-sensitivity.json').read_text());self.assertEqual(sensitivity[0]['status'],'completed')
            self.assertFalse((root/'out/temporary-view').exists())
            cells.loc[0,'source_context']='B_IFNG'
            with self.assertRaisesRegex(ValueError,'mixed_condition_context'):
                assess_context(source,cells,var,mapping,ref,root/'wrong','A_IFNB',['epithelial'],'fixture',{'source_sha256':digest},workers=1)
            with self.assertRaisesRegex(ValueError,'unique_sorted'):
                normalize_selection(source,[1,1],root/'duplicate',5)
            self.assertEqual(digest,hash_file(source))

if __name__=='__main__':unittest.main()
