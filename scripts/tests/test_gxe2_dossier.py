from pathlib import Path
import json
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'dossier'))
from compose_gxe2_dossier import copy_phase
from rna import hash_file


class Gxe2DossierTests(unittest.TestCase):
    def test_chemical_checkpoint_content_is_verified_before_copy(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);source=root/'source';(source/'condition').mkdir(parents=True)
            identity={'seed':1};row={'directory':'condition','genotype_tasks':2}
            (source/'identity.json').write_text(json.dumps(identity));(source/'condition/result.txt').write_text('complete')
            (source/'condition/identity.json').write_text(json.dumps({'identity':identity,'summary':row,'artifacts':{'result.txt':hash_file(source/'condition/result.txt')}}))
            (source/'report.json').write_text(json.dumps({'status':'completed','phase':'all_fixed_source_genotype_chemical_exposure_descriptions','identity':identity,'conditions':[row]}))
            copy_phase(source,root/'ok');self.assertEqual((root/'ok/condition/result.txt').read_text(),'complete')
            (source/'condition/result.txt').write_text('changed')
            with self.assertRaisesRegex(ValueError,'source_component_changed'):copy_phase(source,root/'changed')


if __name__=='__main__':unittest.main()
