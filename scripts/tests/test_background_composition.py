import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'dossier'))
from compose_background_dossier import archive_reference


class BackgroundCompositionTests(unittest.TestCase):
    def test_archive_aliases_do_not_inflate_studies_and_unknowns_are_not_pooled(self):
        record={'experiment_accession':'ERX1','study_ref':'ERP1','sample_ref':'ERS1','study_accessions':['ERP1','PRJEB1','GSE1']}
        self.assertEqual(archive_reference(record,'study'),'ERP1')
        self.assertEqual(archive_reference(record,'sample'),'ERS1')
        self.assertNotEqual(archive_reference({'experiment_accession':'NRX1'},'study'),archive_reference({'experiment_accession':'NRX2'},'study'))
