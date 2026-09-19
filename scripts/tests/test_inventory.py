"""External audit behavior on real CSV/JSON/TSV input files."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / "audit_data_inventory.py"
spec = importlib.util.spec_from_file_location("inventory", SCRIPT)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class InventoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.directory = self.root / "data/raw/demo"
        self.directory.mkdir(parents=True)
        (self.root / "docs").mkdir()
        (self.root / "docs/README.md").write_text("Test scope")
        self.file = self.directory / "counts.csv"
        self.file.write_bytes(b"cell,gene,count\na,TP53,2\n")
        digest = audit.sha256(self.file.read_bytes())
        self.entry = {"name": "counts.csv", "bytes": self.file.stat().st_size,
                      "checksum": "sha256:" + digest, "url": "https://example.test/counts.csv"}
        self.source = {"id": "demo", "files": [self.entry]}
        for name in ["sources.json", "registry.lock.json"]:
            (self.root / "data" / name).write_text(json.dumps({"sources": [self.source]}))
        self.provenance = [{**self.entry, "sha256": digest}]
        (self.directory / "SOURCE.json").write_text(json.dumps({"files": self.provenance}))
        relative = "data/raw/demo/counts.csv"
        self.latest = {relative: {"sha256": digest}}
        (self.root / "data/MANIFEST.tsv").write_text(f"file\tsha256\n{relative}\t{digest}\n")

    def cli(self, output):
        return subprocess.run([sys.executable, str(SCRIPT), "--root", str(self.root),
                               "--output", str(output)], capture_output=True, text=True)

    def file_audit(self, progress=lambda n: None):
        return audit.audit_file(self.root, self.source, self.entry, self.provenance, self.latest, progress)

    def test_success_readonly_and_repeat_refused(self):
        before = {p: p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        output = self.root / "results/report.json"
        run = self.cli(output)
        self.assertEqual(run.returncode, 0, run.stderr)
        report = json.loads(output.read_text())
        self.assertEqual(report["status"], "completed")
        self.assertEqual(report["bytes_read"], self.entry["bytes"])
        self.assertTrue(report["registry_inputs_unchanged"])
        self.assertIn("data/raw/demo/SOURCE.json", report["input_sha256"])
        for path, value in before.items():
            self.assertEqual(path.read_bytes(), value)
        old = output.read_bytes()
        self.assertNotEqual(self.cli(output).returncode, 0)
        self.assertEqual(output.read_bytes(), old)
        self.assertTrue(output.with_suffix(".html").is_file())

    def test_missing_and_changed_contents_fail(self):
        self.file.unlink()
        self.assertIn("missing_file", self.file_audit()["issues"])
        self.file.write_bytes(b"cell,gene,count\na,TP53,3\n")
        result = self.file_audit()
        self.assertEqual(result["checksum_status"], "failed")
        self.assertIn("source_sha256_mismatch", result["issues"])
        self.assertIn("manifest_sha256_mismatch", result["issues"])

    def test_changed_during_actual_read(self):
        def mutate(n):
            # The audit invokes progress after reading bytes but before final stat.
            if self.file.stat().st_size == self.entry["bytes"]:
                with self.file.open("ab") as fh:
                    fh.write(b"b,TP53,4\n")
        self.assertIn("changed_during_read", self.file_audit(mutate)["issues"])

    def test_unavailable_checksum_is_not_failure(self):
        self.entry["checksum"] = None
        result = self.file_audit()
        self.assertEqual(result["checksum_status"], "unavailable")
        self.assertEqual(result["status"], "completed")

    def test_source_change_is_reported(self):
        original = audit.audit_file
        def with_mutation(*args):
            result = original(*args)
            (self.directory / "SOURCE.json").write_text("{}")
            return result
        with patch.object(audit, "audit_file", side_effect=with_mutation):
            report = audit.run_audit(self.root)
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["changed_inputs"], ["data/raw/demo/SOURCE.json"])

    def test_output_inside_source_or_input_symlink_refused(self):
        output = self.directory / "new.json"
        self.assertNotEqual(self.cli(output).returncode, 0)
        self.assertFalse(output.exists())
        link = self.root / "symlink.json"
        link.symlink_to(self.root / "data/registry.lock.json")
        before = link.read_bytes()
        self.assertNotEqual(self.cli(link).returncode, 0)
        self.assertEqual(link.read_bytes(), before)

    def test_json_in_html_cannot_execute_input_markup(self):
        html = audit.render({"malicious": "</script><script>alert(1)</script>"})
        self.assertNotIn("</script><script>alert(1)", html)


if __name__ == "__main__":
    unittest.main()
