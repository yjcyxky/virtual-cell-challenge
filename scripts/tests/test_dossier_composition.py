"""Composition rejects changed or incomplete evidence instead of publishing it."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'dossier'))
from compose_crispri_dossier import verified_copy
from rna import hash_file

class CompositionTests(unittest.TestCase):
    def test_only_verified_artifacts_are_published_and_changed_inputs_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);source=root/'source';source.mkdir()
            report=source/'report.json';report.write_text(json.dumps({'status':'completed','inputs_unchanged':True}))
            (source/'unpublished-cache').write_text('large derived matrix')
            (source/'SHA256SUMS').write_text(hash_file(report)+'  report.json\n')
            verified_copy(source,root/'published')
            self.assertFalse((root/'published/unpublished-cache').exists())
            report.write_text(json.dumps({'status':'completed','inputs_unchanged':False}))
            with self.assertRaisesRegex(ValueError,'incomplete_or_changed_component'):
                verified_copy(source,root/'changed')
            report.write_text(json.dumps({'status':'completed','inputs_unchanged':True,'extra':1}))
            with self.assertRaisesRegex(ValueError,'component_artifact_hash_mismatch'):
                verified_copy(source,root/'unverified')

if __name__=='__main__':unittest.main()
