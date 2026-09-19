from pathlib import Path
import sys
import unittest
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'dossier'))
from profile_scperturb_responses import verify_copied_rows


class CopyTests(unittest.TestCase):
    def test_full_count_identity_and_intervention_fields_are_required(self):
        a=pd.DataFrame({'source_barcode':['a','b'],'gene_axis_sha256':['axis']*2,'computed_count_sha256':['x','y'],
            'source_task':['t1','t2'],'source_guide_id':['g1','g2'],'source_batch':['1','2'],'source_target_gene':['G','H']})
        b=a.rename(columns={'gene_axis_sha256':'source_identity_axis_sha256','computed_count_sha256':'computed_on_source_identity_axis_sha256',
            'source_task':'source_obs__gene_transcript','source_guide_id':'source_obs__guide_id','source_batch':'source_obs__batch','source_target_gene':'source_obs__gene'})
        self.assertEqual(sum(verify_copied_rows(a,b).values()),0)
        for field in ['computed_on_source_identity_axis_sha256','source_obs__guide_id','source_obs__gene_transcript','source_obs__batch']:
            broken=b.copy();broken.loc[1,field]='different'
            with self.assertRaisesRegex(ValueError,'differs'):verify_copied_rows(a,broken)


if __name__=='__main__':unittest.main()
