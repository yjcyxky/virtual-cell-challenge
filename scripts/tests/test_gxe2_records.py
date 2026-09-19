from pathlib import Path
import sys
import tempfile
import unittest
import json
import h5py
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'dossier'))
from convert_seurat_cache import frame
from gxe2_records import run
from rna import hash_file,RNAFile


class Gxe2RecordsTests(unittest.TestCase):
    def test_all_barcodes_preserved_and_source_cell_priority(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);cache=root/'cache';cache.mkdir();(cache/'CDS-1').mkdir()
            path=cache/'coordinate-counts.h5ad'
            with h5py.File(path,'w') as h:
                frame(h.create_group('obs'),pd.DataFrame({'sample':['sample']*4},index=['a','b','c','d']))
                frame(h.create_group('var'),pd.DataFrame({'gene_name':['A','B']},index=['g1','g2']))
                x=h.create_group('X');x.attrs['encoding-type']='csr_matrix';x.attrs['shape']=(4,2)
                x['indptr']=[0,1,2,3,3];x['indices']=[0,0,1];x['data']=np.array([500,1,500],float)
            pd.DataFrame({'source_barcode':['b'],'n.umi':[1]}).to_parquet(cache/'all-CDS-source-metadata.parquet')
            pd.DataFrame({'source_barcode':['b'],'source_row':[1],'differing_values':[0]}).to_parquet(cache/'CDS-1/raw-count-comparison.parquet')
            (cache/'identity.json').write_text(json.dumps({'coordinate':{'coordinate_sha256':'a'*64,'cache_sha256':hash_file(path),'shape':[4,2],'coordinate_count_sum':1001}}))
            report=run(cache,root/'result',chunk=2)
            self.assertEqual(report['coverage']['records'],4)
            self.assertEqual(report['coverage']['source_CDS_below_500'],1)
            self.assertEqual(report['coverage']['non_CDS_source_threshold_candidates'],2)
            self.assertEqual(report['coverage']['not_established_source_cell'],1)
            self.assertEqual(report['source_CDS_UMI_disagreements'],0)
            with RNAFile(root/'result/non-CDS-candidates.h5ad') as source:
                self.assertEqual(list(source.obs.index),['a','c'])
                matrix=next(source.blocks())[1]
                np.testing.assert_equal(matrix.toarray(),[[500,0],[0,500]])


if __name__=='__main__':unittest.main()
