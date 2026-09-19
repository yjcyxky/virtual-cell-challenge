"""Archive splitting must preserve complete data and reject unregistered artifacts."""
import gzip
import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'dossier'))
from publish_assessment import SplitWriter,verified_files,verify_remote
from rna import hash_file


class ReleaseTests(unittest.TestCase):
    def test_draft_is_found_by_ID_and_published_only_after_asset_digest_verification(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);(root/'payload').write_text('result');(root/'RELEASE-SHA256SUMS').write_text(hash_file(root/'payload')+'  payload\n')
            draft={'id':7,'tag_name':'test','draft':True,'html_url':'draft-url'}
            assets=[{'name':p.name,'size':p.stat().st_size,'digest':'sha256:'+hash_file(p)} for p in root.iterdir()]
            with patch('publish_assessment.subprocess.check_output',side_effect=[json.dumps([[draft]]),json.dumps([assets]),json.dumps({**draft,'draft':False,'html_url':'published-url'})]),patch('publish_assessment.subprocess.run') as run:
                self.assertEqual(verify_remote(root,'test'),'published-url');run.assert_called_once()
            assets[0]['digest']='sha256:bad'
            with patch('publish_assessment.subprocess.check_output',side_effect=[json.dumps([[draft]]),json.dumps([assets])]),patch('publish_assessment.subprocess.run') as run:
                with self.assertRaisesRegex(ValueError,'digest_or_size'):verify_remote(root,'test')
                run.assert_not_called()

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
