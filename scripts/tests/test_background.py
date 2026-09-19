"""scBaseCount native CSC behavior, modality qualification and inference contract."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import h5py
import numpy as np
import pandas as pd
from scipy import sparse
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'dossier'))
from profile_background import inference_eligibility,assess_file
from rna import hash_file
from test_rna import h5ad


class BackgroundTests(unittest.TestCase):
    def test_assay_specific_identity_overrides_generic_RNA_strategy(self):
        common={'upstream_tax_id':'9606','library_source':'TRANSCRIPTOMIC','library_strategy':'RNA-Seq'}
        self.assertEqual(inference_eligibility(dict(common,library_name='donor_ADT_lib',experiment_title='CITE-seq'))['status'],'not_applicable')
        self.assertEqual(inference_eligibility(dict(common,library_name='donor_GEX_lib',experiment_title='CITE-seq GEX'))['status'],'conditional')
        self.assertEqual(inference_eligibility(dict(common,library_name='GSM123',experiment_title='10x TCR VDJ RNA-Seq'))['status'],'not_applicable')
        self.assertEqual(inference_eligibility(dict(common,upstream_tax_id='10090',library_name='GEX'))['reason'],'upstream_species_conflicts_with_human_reference_axis')

    def test_all_layers_original_labels_zero_library_and_resume_identity(self):
        with tempfile.TemporaryDirectory() as root:
            root=Path(root);raw=root/'data/raw/fixture.h5ad';raw.parent.mkdir(parents=True)
            values=np.array([[1,2],[0,3],[0,0]],dtype=float)
            h5ad(raw,values)
            with h5py.File(raw,'a') as f:
                del f['X'];node=f.create_group('X');m=sparse.csc_matrix(values)
                node.attrs.update({'encoding-type':'csc_matrix','shape':m.shape})
                for name in ['data','indices','indptr']:node.create_dataset(name,data=getattr(m,name))
                f['var'].create_dataset('gene_symbols',data=['TP53','GAPDH'],dtype=h5py.string_dtype())
                for key,val in {'SRX_accession':['SRXTEST']*3,'cell_type':['source_type','','']}.items():
                    f['obs'].create_dataset(key,data=val,dtype=h5py.string_dtype())
                f['obs'].create_dataset('umi_count_Unique',data=[3,3,0]);f['obs'].create_dataset('gene_count_Unique',data=[2,1,0])
                f.create_group('layers').create_dataset('UniqueAndMult-EM',data=values+.5)
            hgnc=root/'data/raw/networks/hgnc_complete_set.txt';hgnc.parent.mkdir()
            hgnc.write_text('status\tsymbol\tensembl_gene_id\nalias\tBAD\tBAD\nApproved\tTP53\tTP53\nApproved\tGAPDH\tGAPDH\n')
            axis=root/'data/raw/arc_vcc2026_controls/gene_names.csv';axis.parent.mkdir();axis.write_text('gene_name\nTP53\nGAPDH\n')
            refs=root/'refs';refs.mkdir();(refs/'gene_sets.json').write_text(json.dumps({'profiles':{'type':{'genes':['TP53'],'lineage':'lineage'}},'states':{'cycle':{'genes':['TP53'],'reference':'fixture'}}}))
            digest=hash_file(raw);out=root/'out';out.mkdir()
            row={'source_file':'data/raw/fixture.h5ad','source_file_sha256_from_inventory':digest,
                'experiment_accession':'SRXTEST','upstream_tax_id':'9606','library_source':'TRANSCRIPTOMIC',
                'library_name':'GEX','experiment_title':'GEX','sample_accessions':np.array(['SAMN1']),
                'study_accessions':np.array(['PRJ1']),'source_cells_from_metadata':3,
                'flags':np.array(['upstream_flag']),'supervised_overlap':np.array([],dtype=str)}
            with patch('profile_background.ROOT',root):
                result=assess_file(row,str(out),str(refs),{'fixture':1})
                self.assertEqual(result['n_cells'],3)
                self.assertEqual(result['numeric']['X']['stored_values_checked'],3)
                self.assertEqual(result['numeric']['layers/UniqueAndMult-EM']['noninteger'],6)
                self.assertEqual(result['zero_libraries'],1)
                self.assertEqual(result['source_QC_disagreements']['source_umi_count_Unique'],0)
                cells=pd.read_parquet(out/'SRXTEST/cells.parquet')
                self.assertEqual(cells.source_cell_type.iloc[0],'source_type')
                self.assertTrue((cells.inferred_type=='unknown').all())
                self.assertTrue(cells.probability_correct.isna().all())
                self.assertEqual(cells.inference_reason.iloc[2],'zero_library')
                self.assertEqual(hash_file(raw),digest)
                self.assertEqual(assess_file(row,str(out),str(refs),{'fixture':1})['n_cells'],3)
                with h5py.File(raw,'a') as f:f['X/data'][0]=99
                with self.assertRaisesRegex(ValueError,'source_inventory_hash_mismatch'):
                    assess_file(row,str(out),str(refs),{'fixture':1})


if __name__=='__main__':unittest.main()
