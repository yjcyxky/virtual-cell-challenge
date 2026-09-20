from pathlib import Path
import sys
import unittest
import tempfile
import json
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'dossier'))
from profile_scperturb_responses import verify_copied_rows,verified_completed_files
from rna import hash_file


class CopyTests(unittest.TestCase):
    def test_completed_component_reuse_requires_unchanged_inputs_and_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);part=root/'example';part.mkdir();(part/'evidence').write_text('frozen')
            identity={'design_report_sha256':'design','cell_phase_report_sha256':'cells','uv_lock_sha256':'lock',
                'code':{'profile_scperturb_responses.py':'old-driver','scperturb_response_core.py':'old-gate','response.py':'same-method'}}
            (root/'identity.json').write_text(json.dumps(identity))
            report={'status':'completed','identity':identity,'artifacts':{'evidence':hash_file(part/'evidence')}}
            (part/'report.json').write_text(json.dumps(report))
            current={**identity,'code':{**identity['code'],'scperturb_response_core.py':'new-gate'}}
            self.assertIn('example',verified_completed_files(root,current)['completed_file_reports'])
            with self.assertRaisesRegex(ValueError,'input_identity'):verified_completed_files(root,{**current,'design_report_sha256':'different'})
            with self.assertRaisesRegex(ValueError,'analysis_method'):verified_completed_files(root,{**current,'code':{**current['code'],'response.py':'changed'}})
            (part/'evidence').write_text('modified')
            with self.assertRaisesRegex(ValueError,'artifact_changed'):verified_completed_files(root,current)

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
