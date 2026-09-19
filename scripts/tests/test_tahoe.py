import sys
from pathlib import Path
import unittest
import tempfile
import json
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'dossier'))
from tahoe import decode,parse_compounds,metadata_join,condition_eligibility
from tahoe_counts import scan_shard
from rna import hash_file
from scipy import sparse


class TahoeTests(unittest.TestCase):
    def test_full_shard_checkpoint_and_profile_preserve_source_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);cache=root/'cache';output=root/'output'
            (cache/'shards').mkdir(parents=True);(output/'shards').mkdir(parents=True)
            source=root/'source.parquet';pq.write_table(pa.table({'genes':[[1,3,4],[1,4]],'expressions':[[-2.,2.,3.],[-2.,5.]],'BARCODE_SUB_LIB_ID':['b1','b2']}),source)
            frame=pd.DataFrame({'BARCODE_SUB_LIB_ID':['b1','b2'],'plate':['p1']*2,'sample':['s1']*2,'drug':['DMSO_TF']*2,'cell_line_id':['C1']*2,'row_index':[0,1]})
            obs=pd.DataFrame({'BARCODE_SUB_LIB_ID':['b1','b2'],'plate':['p1']*2,'sample':['s1']*2,'drug':['DMSO_TF']*2,'cell_line':['C1']*2,'tscp_count':[5,5],'gene_count':[2,1]})
            samples=pd.DataFrame({'sample':['s1'],'plate':['p1'],'drug':['DMSO_TF'],'drugname_drugconc':["[('DMSO_TF',0,'uM')]"]})
            metadata_join(frame,obs,samples).to_parquet(cache/'shards/000.parquet',index=False)
            pd.DataFrame({'token_id':[3,4],'ensembl_id':['ENSG1','ENSG2'],'gene_symbol':['G1','G2']}).to_parquet(output/'genes.parquet',index=False)
            record={'shard_index':0,'file':str(source),'sha256':hash_file(source),'rows':2};identity={'test':'fixed'}
            first=scan_shard(record,str(cache),str(output),identity);second=scan_shard(record,str(cache),str(output),identity)
            self.assertEqual(first,second);self.assertEqual(first['numeric']['CLS_markers'],2);self.assertEqual(first['source_UMI_QC_disagreements'],0)
            expected=np.log1p(np.array([[4000.,6000.],[0.,10000.]])).sum(axis=0)
            np.testing.assert_allclose(sparse.load_npz(output/'shards/000/condition-log-sums.npz').toarray()[0],expected)
            with self.assertRaisesRegex(ValueError,'checkpoint_method'):scan_shard(record,str(cache),str(output),{'test':'changed'})
            (output/'shards/000/gene-totals.npz').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'checkpoint_artifact'):scan_shard(record,str(cache),str(output),identity)

    def test_only_verified_CLS_is_removed_and_other_negative_values_flagged(self):
        tokens=np.array([-1,-1,-1,0,1])
        batch=pa.record_batch({'genes':[[1,3,4],[1,4]],'expressions':[[-2.,2.,3.],[-2.,-1.]]})
        matrix,valid,facts=decode(batch,tokens,2)
        self.assertEqual(matrix.toarray().tolist(),[[2.,3.],[0.,-1.]])
        self.assertEqual(valid.tolist(),[True,False]);self.assertEqual(facts['CLS_markers'],2);self.assertEqual(facts['negative'],1)
        wrong=pa.record_batch({'genes':[[1,3]],'expressions':[[-1.,3.]]})
        with self.assertRaisesRegex(ValueError,'CLS_encoding'):decode(wrong,tokens,2)
        duplicate=pa.record_batch({'genes':[[1,3,3]],'expressions':[[-2.,1.,2.]]})
        with self.assertRaisesRegex(ValueError,'duplicate_gene_tokens'):decode(duplicate,tokens,2)

    def test_metadata_conflicts_and_vehicle_codes_remain_explicit(self):
        local=pd.DataFrame({'BARCODE_SUB_LIB_ID':['b1','b2'],'plate':['p1','p1'],'sample':['s1','s1'],'drug':['DMSO_TF']*2,'cell_line_id':['C1']*2})
        observations=pd.DataFrame({'BARCODE_SUB_LIB_ID':['b1'],'plate':['p1'],'sample':['wrong'],'drug':['DMSO_TF'],'cell_line':['C1']})
        samples=pd.DataFrame({'sample':['s1'],'plate':['p1'],'drug':['DMSO_TF']})
        result=metadata_join(local,observations,samples)
        self.assertEqual(result.metadata_obs_present.tolist(),[True,False]);self.assertFalse(result.metadata_condition_consistent.any())
        with self.assertRaisesRegex(ValueError,'nonunique_metadata'):metadata_join(local,pd.concat([observations,observations]),samples)
        dose=parse_compounds("[('DMSO_TF', 0.0, 'uM')]")[0];self.assertTrue(dose['vehicle_code_not_molar_dose'])
        observations.loc[0,'sample']='s1';observations.loc[0,'drug']='DMSO_TF '
        result=metadata_join(local,observations,samples);valid,reasons=condition_eligibility(result)
        self.assertEqual(valid.tolist(),[True,False]);self.assertEqual(reasons[0],'explicit_trailing_whitespace_alias')
        self.assertEqual(result.source_obs_drug.iloc[0],'DMSO_TF ')
