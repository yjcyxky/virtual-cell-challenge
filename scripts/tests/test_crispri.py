"""Source-level behavior: intervention identity, control matching and aggregation."""
from pathlib import Path
import sys
import tempfile
import unittest
import h5py
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'dossier'))
from crispri import task_metadata, control_coverage, bulk_comparison
from rna import hash_file
from test_rna import h5ad


class CrispriTests(unittest.TestCase):
    def test_promoters_are_distinct_and_dual_guides_not_replicates(self):
        obs=pd.DataFrame({'gene':['A','A','non-targeting','KNTC1'],
            'gene_id':['ENSG1','ENSG1','NTC','ENSG2'],'transcript':['P1','P2','NTC','P1'],
            'gene_transcript':['A_P1','A_P2','control','KNTC1_P1'],
            'sgID_AB':['a|b','c|d','e|f','g|h'],'gem_group':[1,2,1,1]},index=['a','b','c','d'])
        result=task_metadata(obs).set_index('task')
        self.assertEqual(len(result),4)
        self.assertEqual(result.is_control.sum(),1)
        self.assertTrue((result.guide_components==2).all())
        self.assertTrue((result.independent_constructs==1).all())
        self.assertTrue(result.biological_replicates.isna().all())
        coverage=control_coverage(obs).set_index('task')
        self.assertFalse(coverage.loc['A_P2','matched'])
        self.assertTrue(coverage.loc['KNTC1_P1','matched'])
        conflicting=pd.concat([obs,obs.iloc[:1].rename(index={'a':'different_cell'})])
        conflicting.iloc[-1,conflicting.columns.get_loc('gene')]='WRONG'
        with self.assertRaisesRegex(ValueError,'construct_identity_conflict'):
            task_metadata(conflicting)

    def test_bulk_means_are_not_new_cells_and_unavailable_groups_not_zero(self):
        with tempfile.TemporaryDirectory() as root:
            sc,bulk=Path(root)/'sc.h5ad',Path(root)/'bulk.h5ad'
            h5ad(sc,[[1,2],[2,4],[8,6]])
            h5ad(bulk,[[1.5,3],[7,6],[2,2]],barcodes=['A','B','C'])
            with h5py.File(sc,'a') as f:
                f['obs'].create_dataset('gene_transcript',data=['A','A','B'],dtype=h5py.string_dtype())
            with h5py.File(bulk,'a') as f:
                f['obs'].create_dataset('num_cells_filtered',data=[2,1,0])
                f['obs'].create_dataset('num_cells_unfiltered',data=[3,1,2])
            before=[hash_file(sc),hash_file(bulk)]
            frame,summary=bulk_comparison(sc,bulk,chunk=1)
            self.assertEqual(summary['numeric_violations']['noninteger'],1)
            self.assertEqual(summary['rows_matching_local_singlecell_mean'],1)
            self.assertEqual(summary['groups_without_local_singlecell_records'],1)
            self.assertFalse(frame.independent_observation.any())
            self.assertEqual(frame.loc[2,'reason'],'no_singlecell_rows_in_local_release')
            self.assertTrue(pd.isna(frame.loc[2,'max_absolute_error_to_singlecell_mean']))
            self.assertEqual(before,[hash_file(sc),hash_file(bulk)])


if __name__=='__main__':unittest.main()
