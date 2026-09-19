from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest.mock import patch
import h5py
import numpy as np
import pandas as pd
from scipy import sparse
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'dossier'))
import scperturb_cells
from scperturb_cells import biological_background,expression_view,scan_file
from convert_seurat_cache import frame
from rna import hash_file


class ScperturbCellTests(unittest.TestCase):
    def test_stimulation_is_background_but_not_endpoint_celltype(self):
        obs=pd.DataFrame({'cell_line':['K']*3,'perturbation_2':['stim','stim','unstim'],'celltype':['a','b','a']})
        codes,groups=biological_background('Frangieh_RNA.h5ad',obs)
        self.assertEqual(codes[0],codes[1]);self.assertNotEqual(codes[0],codes[2]);self.assertEqual(len(groups),2)

    def test_continuous_values_are_not_logged_twice(self):
        x=sparse.csr_matrix([[1.,3.],[0.,0.]])
        np.testing.assert_equal(expression_view(x,False).toarray(),x.toarray())
        np.testing.assert_allclose(expression_view(x,True).toarray()[0],np.log1p([2500,7500]),rtol=1e-6)
        np.testing.assert_equal(expression_view(x,True).toarray()[1],[0,0])

    def test_full_source_records_and_unverified_scale_downgrade(self):
        with tempfile.TemporaryDirectory() as temporary,patch.object(scperturb_cells,'ROOT',Path(temporary)):
            root=Path(temporary);raw=root/'data/raw';(raw/'scperturb').mkdir(parents=True);(raw/'networks').mkdir();(raw/'arc_vcc2026_controls').mkdir()
            symbols=['G'+str(i) for i in range(12)];ids=['ENSG'+str(i).zfill(11) for i in range(12)]
            pd.DataFrame({'symbol':symbols,'status':['Approved']*12,'alias_symbol':['']*12,'prev_symbol':['']*12,'ensembl_gene_id':ids}).to_csv(raw/'networks/hgnc_complete_set.txt',sep='\t',index=False)
            pd.DataFrame({'gene_name':symbols}).to_csv(raw/'arc_vcc2026_controls/gene_names.csv',index=False)
            path=raw/'scperturb/Joung_test.h5ad';x=sparse.csc_matrix(np.array([[1.5]*12,[0.]*12,[2.5]*12]))
            with h5py.File(path,'w') as h:
                frame(h.create_group('obs'),pd.DataFrame({'cell_line':['line']*3,'perturbation':['G0','control',None]},index=['a','b','c']))
                frame(h.create_group('var'),pd.DataFrame({'ensembl_id':ids},index=symbols))
                m=h.create_group('X');m.attrs['encoding-type']='csc_matrix';m.attrs['shape']=x.shape;m['data']=x.data;m['indices']=x.indices;m['indptr']=x.indptr
            references=root/'references';references.mkdir();profiles={n:{'lineage':l,'genes':symbols} for n,l in [('neuron_like','neural'),('pluripotent_like','pluripotent')]}
            (references/'gene_sets.json').write_text(json.dumps({'profiles':profiles,'states':{'state':{'genes':symbols[:4],'reference':'synthetic'}}}))
            result=scan_file({'file':path.name,'input_sha256':hash_file(path),'species':['human'],'matrices':{'X':{'nonnegative_integer_compatible':False}}},str(root/'out'),str(references),str(references),{'test':True})
            cells=pd.read_parquet(root/'out/Joung_test/cells.parquet');self.assertEqual(len(cells),3);self.assertEqual(result['zero_expression_records'],1)
            self.assertTrue((cells.inferred_type=='unknown').all());self.assertTrue((cells.inference_status=='not_estimable').all());self.assertFalse(cells.probability_correct.notna().any())
            np.testing.assert_allclose(np.load(root/'out/Joung_test/state-backgrounds.npz')['mean_expression'],np.full((1,12),4/3),rtol=1e-6)
            self.assertEqual(cells.source_barcode.tolist(),['a','b','c']);self.assertEqual(cells.computed_on_source_identity_axis_sha256.str.len().tolist(),[64,64,64])


if __name__=='__main__':unittest.main()
