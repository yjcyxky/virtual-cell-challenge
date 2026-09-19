"""Archive splitting must preserve complete data and reject unregistered artifacts."""
import gzip
import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'dossier'))
from publish_assessment import SplitWriter,verified_files
from rna import hash_file


class ReleaseTests(unittest.TestCase):
    def test_split_stream_reassembles_and_rejects_changed_or_unregistered_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);bundle=root/'bundle';bundle.mkdir();parts=root/'parts';parts.mkdir()
            (bundle/'report.json').write_text(json.dumps({'status':'completed','bundle_id':'fixture'}))
            (bundle/'report.html').write_text('<html>fixture</html>')
            (bundle/'SHA256SUMS').write_text(''.join(f'{hash_file(p)}  {p.name}\n' for p in sorted(bundle.iterdir())))
            files=verified_files(bundle);stream=SplitWriter(parts,limit=37)
            with gzip.GzipFile(fileobj=stream,mode='wb') as compressed:
                with tarfile.open(fileobj=compressed,mode='w|') as tar:
                    for name in files:tar.add(bundle/name,arcname=name)
            stream.close();self.assertGreater(len(stream.parts),1)
            combined=b''.join(p.read_bytes() for p in stream.parts)
            with tarfile.open(fileobj=io.BytesIO(combined),mode='r:gz') as tar:
                self.assertEqual(tar.extractfile('report.html').read(),b'<html>fixture</html>')
            (bundle/'unregistered').write_text('oops')
            with self.assertRaisesRegex(ValueError,'unregistered'):verified_files(bundle)
            (bundle/'unregistered').unlink();(bundle/'report.html').write_text('changed')
            with self.assertRaisesRegex(ValueError,'hash_mismatch'):verified_files(bundle)


if __name__=='__main__':unittest.main()
