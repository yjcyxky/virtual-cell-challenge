"""Download budgets follow source destinations and account for resumable files."""
import importlib.util
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("fetch", Path(__file__).parents[1] / "fetch_data.py")
fetch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fetch)
lock_spec = importlib.util.spec_from_file_location("build_lock", Path(__file__).parents[1] / "build_lock.py")
lock = importlib.util.module_from_spec(lock_spec)
lock_spec.loader.exec_module(lock)


class StorageTests(unittest.TestCase):
    def test_conditional_download_and_registration_reject_changed_etag(self):
        body = b"native raw counts fixture"

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def do_HEAD(self):
                self.send_response(200)
                self.send_header("ETag", '"snapshot"')
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()

            def do_GET(self):
                if self.headers.get("If-Match") != '"snapshot"':
                    self.send_error(412)
                    return
                self.do_HEAD()
                self.wfile.write(body)

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            url = f"http://127.0.0.1:{server.server_port}/counts"
            self.assertEqual(lock.head_size(url, '"snapshot"'), len(body))
            with self.assertRaisesRegex(ValueError, "ETag changed"):
                lock.head_size(url, '"old"')
            with tempfile.TemporaryDirectory() as temp, patch.object(fetch, "ROOT", Path(temp)):
                entry = {"url": url, "bytes": len(body), "etag": '"snapshot"',
                         "checksum": "sha256:" + hashlib.sha256(body).hexdigest()}
                path = Path(temp) / "counts"
                result = fetch.fetch_one(entry, path, True)
                self.assertEqual(result["status"], "downloaded")
                self.assertEqual(path.read_bytes(), body)
                self.assertEqual(result["checksum_verified"], "yes")
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def test_symlink_destination_and_same_volume_are_budgeted_together(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "repo"
            raw = root / "data/raw"
            raw.mkdir(parents=True)
            nas = Path(temp) / "nas"
            nas.mkdir()
            (raw / "a").symlink_to(nas, target_is_directory=True)
            (nas / "partial").write_bytes(b"123")
            sources = [{"id": "a", "files": [{"name": "partial", "bytes": 10}]},
                       {"id": "b", "files": [{"name": "new", "bytes": 20}]}]
            with patch.object(fetch, "ROOT", root), patch.object(fetch.shutil, "disk_usage", return_value=SimpleNamespace(free=100)) as usage:
                volumes = fetch.disk_requirements(sources)
            self.assertEqual(len(volumes), 1)
            self.assertEqual(volumes[0]["remaining"], 27)
            self.assertEqual(volumes[0]["path"], str(nas))
            usage.assert_called_once_with(nas)

    def test_oversized_file_does_not_cancel_another_downloads_budget(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            directory = root / "data/raw/a"
            directory.mkdir(parents=True)
            (directory / "oversized").write_bytes(b"12345")
            source = {"id": "a", "files": [{"name": "oversized", "bytes": 2},
                                              {"name": "new", "bytes": 10}]}
            with patch.object(fetch, "ROOT", root):
                self.assertEqual(fetch.disk_requirements([source])[0]["remaining"], 10)


if __name__ == "__main__":
    unittest.main()
