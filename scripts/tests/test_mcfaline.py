"""Coordinate indexing and source condition/guide conflicts must remain visible."""
from pathlib import Path
import sys
import tempfile
import unittest
import pandas as pd
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'dossier'))
from mcfaline import coordinate_cache,compare_CDS,assigned_design,parse_gxe1_hash
from rna import RNAFile
from test_rna import h5ad

class McfalineTests(unittest.TestCase):
    def test_one_based_coordinate_axes_and_full_CDS_comparison(self):
        with tempfile.TemporaryDirectory() as root:
            root=Path(root);(root/'cells.tsv').write_text('cell1\tsample\ncell2\tsample\n');(root/'genes.tsv').write_text('ENSG1\tG1\nENSG2\tG2\n')
            (root/'counts.tsv').write_text('1\t1\t1\n2\t1\t2\n1\t2\t3\n2\t2\t4\n')
            result=coordinate_cache(root/'counts.tsv',root/'cells.tsv',root/'genes.tsv',root/'raw.h5ad',chunk=3)
            self.assertEqual(result['coordinate_count_sum'],10)
            h5ad(root/'CDS.h5ad',[[4,3],[2,1]],genes=['ENSG2','ENSG1'],barcodes=['cell2','cell1'])
            compared=compare_CDS(root/'raw.h5ad',root/'CDS.h5ad');self.assertEqual(compared.differing_values.sum(),0)
            self.assertFalse(compared.independent_observation.any())
            (root/'bad.tsv').write_text('0\t1\t2\n')
            with self.assertRaisesRegex(ValueError,'one_based_coordinate_out_of_bounds'):coordinate_cache(root/'bad.tsv',root/'cells.tsv',root/'genes.tsv',root/'bad.h5ad')
            (root/'bad.tsv').write_text('1\t1\t2.5\n')
            with self.assertRaises((ValueError,TypeError)):coordinate_cache(root/'bad.tsv',root/'cells.tsv',root/'genes.tsv',root/'fractional.h5ad')

    def test_source_assignment_not_repaired_into_false_single_target(self):
        common={'top_oligo_W':'plate10_A1_A172_CRISPRi_MMR_100_temozolomide_96','CRISPR_hash':'CRISPRi','gRNA_library':'MMR','dose':100,'treatment':'temozolomide',
            'hash_umis_W':10,'top_to_second_best_ratio_W':3,'gene_id':'MSH2','CRISPR_gRNA':'CRISPRi'}
        frame=pd.DataFrame([common,{**common,'CRISPR_gRNA':'CRISPRa'},{**common,'gene_id':'MSH2,MSH6'},{**common,'dose':1}])
        result=assigned_design(frame)
        self.assertEqual(result.genetic_response_assignment_status.tolist(),['eligible','not_estimable','not_estimable','not_estimable'])
        self.assertEqual(result.assignment_limitation.iloc[1],'hash_vs_guide_effector_conflict')
        self.assertEqual(frame.dose.iloc[3],1)
        with self.assertRaisesRegex(ValueError,'unrecognized_GxE1_hash_condition'):parse_gxe1_hash('other_background')

if __name__=='__main__':unittest.main()
