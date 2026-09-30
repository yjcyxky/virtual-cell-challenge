"""A partial or provenance-less source must never enter the finalization stages."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from finish_crispri_acquisition import acquisition_progress, complete_provenance


class AcquisitionTests(unittest.TestCase):
    def test_partial_bytes_and_provenance_gate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base = root / "data/raw/example"
            base.mkdir(parents=True)
            source = {"id": "example", "files": [{"name": "raw.bin", "bytes": 4, "url": "https://source/raw.bin"}]}
            (base / "raw.bin").write_bytes(b"12")
            progress = acquisition_progress(root, source)
            self.assertEqual(progress["bytes_present"], 2)
            self.assertEqual(progress["files_at_expected_size"], 0)
            with self.assertRaises(FileNotFoundError):
                complete_provenance(root, source)
            row = {**source["files"][0], "sha256": "a" * 64}
            (base / "SOURCE.json").write_text(json.dumps({"files": [row]}))
            with self.assertRaisesRegex(ValueError, "incomplete_source_file"):
                complete_provenance(root, source)
            (base / "raw.bin").write_bytes(b"1234")
            self.assertEqual(complete_provenance(root, source), base)
            row["sha256"] = ""
            (base / "SOURCE.json").write_text(json.dumps({"files": [row]}))
            with self.assertRaisesRegex(ValueError, "incomplete_source_file"):
                complete_provenance(root, source)


if __name__ == "__main__":
    unittest.main()
