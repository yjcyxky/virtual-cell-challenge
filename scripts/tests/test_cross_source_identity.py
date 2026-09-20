from pathlib import Path
import sys
import unittest
import numpy as np
import pandas as pd
from scipy import sparse
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'dossier'))
from cross_source_identity import compare_pair,projected_fingerprints,author_library_lookup,resolve_library,barcode_core


class IdentityTests(unittest.TestCase):
    def test_same_barcode_and_count_does_not_establish_cross_study_identity(self):
        a=pd.DataFrame({'source_barcode':['A'],'gene_axis_sha256':['axis'],'computed_count_sha256':['same'],'record_id':['x']});b=a.assign(record_id='y')
        r=compare_pair(a,b,{'study_A':'study1','study_B':'study2','RNA_A':True,'RNA_B':True})
        self.assertFalse(r.confirmed_exact_RNA_expression_copy.iloc[0]);self.assertTrue(r.full_native_axis_and_counts_equal.iloc[0])
        relation={'study_A':'study','study_B':'study','sample_A':'sample','sample_B':'sample','capture_A':'lib','capture_B':'lib','RNA_A':True,'RNA_B':True}
        self.assertTrue(compare_pair(a,b,relation).confirmed_exact_RNA_expression_copy.iloc[0])
        b['computed_count_sha256']='different';r=compare_pair(a,b,relation)
        self.assertTrue(r.confirmed_capture_barcode_link.iloc[0]);self.assertFalse(r.confirmed_exact_RNA_expression_copy.iloc[0])
        b['computed_count_sha256']='same';self.assertFalse(compare_pair(a,b,{**relation,'RNA_B':False}).confirmed_exact_RNA_expression_copy.iloc[0])
        self.assertFalse(compare_pair(a,b,{**relation,'capture_B':'other'}).confirmed_capture_barcode_link.iloc[0])

    def test_projection_reorders_measured_genes_but_refuses_unmeasured_zero_fill(self):
        x=sparse.csr_matrix([[3,0,2],[0,4,1]])
        a=projected_fingerprints(x,['G','H','I'],['I','G'])
        b=projected_fingerprints(x[:,[2,0]],['I','G'],['I','G'])
        self.assertEqual(a[0],b[0]);np.testing.assert_array_equal(a[1],[5,1])
        with self.assertRaisesRegex(ValueError,'zero_filled'):projected_fingerprints(x,['G','H','I'],['J'])
        self.assertEqual(barcode_core(['ACGTACGTACGTACGT-27','ACGTACGTACGTACGT','not_10x']).tolist()[:2],['ACGTACGTACGTACGT']*2)

    def test_author_manifest_controls_GEM_not_filename_numeric_guess(self):
        manifest=pd.DataFrame([{'library':'KD8_p1_78','gemgroup':73,'read':'KD8_seq2_p1_mRNA_78_S79_L003_R1_001.fastq.gz'}])
        lookup=author_library_lookup({'K562_gwps':manifest},{})
        result=resolve_library({'supervised_overlap':['replogle2022'],'library_name':'KD8_seq2_p1_mRNA_78_S79_L003','experiment_accession':'SRX1'},lookup)
        self.assertEqual(result['GEM'],73);self.assertEqual(result['modality'],'RNA')


if __name__=='__main__':unittest.main()
