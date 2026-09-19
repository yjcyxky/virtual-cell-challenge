import importlib.util
from pathlib import Path
import unittest
import json
import tempfile
import xml.etree.ElementTree as ET

spec = importlib.util.spec_from_file_location("scbase_audit", Path(__file__).resolve().parents[1] / "scbasecount_tools/audit.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class ProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.local = {"srx_accession": "SRX1", "organism": "Homo sapiens", "perturbation": "none", "obs_count": 10}
        self.exp = ET.fromstring('<EXPERIMENT accession="SRX1"><TITLE>RPE1 sgRNA library</TITLE><DESIGN><LIBRARY_DESCRIPTOR><LIBRARY_STRATEGY>OTHER</LIBRARY_STRATEGY></LIBRARY_DESCRIPTOR></DESIGN></EXPERIMENT>')
        self.sample = ET.fromstring('<SAMPLE accession="SRS1"><SAMPLE_NAME><TAXON_ID>9606</TAXON_ID><SCIENTIFIC_NAME>Homo sapiens</SCIENTIFIC_NAME></SAMPLE_NAME></SAMPLE>')
        self.study = ET.fromstring('<STUDY accession="SRP376262"><DESCRIPTOR><STUDY_TITLE>Perturb-seq</STUDY_TITLE></DESCRIPTOR></STUDY>')

    def test_species_and_library_conflict_do_not_overwrite_source(self):
        self.sample.find('.//TAXON_ID').text = '10090'
        self.sample.find('.//SCIENTIFIC_NAME').text = 'Mus musculus'
        row = audit.classify(self.local, self.exp, self.sample, self.study)
        self.assertEqual(row['source_organism'], 'Homo sapiens')
        self.assertEqual(row['upstream_species'], 'Mus musculus')
        self.assertIn('species_conflict', row['flags'])
        self.assertIn('non_GEX_title_signal', row['flags'])
        self.assertEqual(row['supervised_overlap'], ['replogle2022'])
        self.assertIsNone(row['independent_biological_repeats'])
        self.assertEqual(self.local['perturbation'], 'none')

    def test_missing_archive_is_not_certified_untreated(self):
        row = audit.classify(self.local, None, None, None)
        self.assertEqual(row['status'], 'blocked')
        self.assertEqual(row['missing_records'], ['experiment', 'sample', 'study'])
        self.assertEqual(row['baseline_eligibility'], 'not_certified_from_archive_metadata')
        self.assertEqual(row['library_assessment'], 'unknown')

    def test_identifier_aliases_are_not_distinct_samples(self):
        sample = ET.fromstring('<SAMPLE accession="SRS1"><IDENTIFIERS><PRIMARY_ID>SRS1</PRIMARY_ID><EXTERNAL_ID namespace="BioSample">SAMN1</EXTERNAL_ID></IDENTIFIERS></SAMPLE>')
        self.assertEqual(audit.identifiers(sample), ['SAMN1', 'SRS1'])
        self.assertEqual(len(set(audit.identifiers(sample))), 2)

    def test_protocol_names_do_not_assign_library_modality(self):
        self.exp.find('TITLE').text = 'mRNA gene expression'
        ET.SubElement(self.exp, 'DESIGN_DESCRIPTION').text = 'mRNA and sgRNA libraries sequenced separately'
        row = audit.classify(self.local, self.exp, self.sample, self.study)
        self.assertEqual(row['library_assessment'], 'GEX_title_signal')
        self.assertNotIn('non_GEX_title_signal', row['flags'])

    def test_uid_lookup_must_match_requested_experiment(self):
        data = b'<EXPERIMENT_PACKAGE_SET><EXPERIMENT_PACKAGE><EXPERIMENT accession="SRX_WRONG"/></EXPERIMENT_PACKAGE></EXPERIMENT_PACKAGE_SET>'
        exp, sample, study, error = audit.ncbi_package(data, 'SRX1')
        self.assertIsNone(exp)
        self.assertEqual(error, 'requested_accession_not_returned')

    def test_frozen_failure_is_replayed_without_changing_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            cache = Path(temporary)
            url = audit.ENDPOINT + 'SRX1'
            entry = {'url': url, 'status': 'failed', 'error': 'HTTP 400', 'requested': ['SRX1'], 'retrieved_at': 'fixed-time'}
            path = cache / ('experiment-' + audit.digest(url.encode())[:20] + '.json')
            path.write_text(json.dumps(entry))
            original = path.read_bytes()
            self.assertEqual(audit.fetch_batch('EXPERIMENT', ['SRX1'], cache), entry)
            self.assertEqual(path.read_bytes(), original)


if __name__ == '__main__':
    unittest.main()
