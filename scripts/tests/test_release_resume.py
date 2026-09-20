from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'dossier'))
from resume_assessment_upload import missing_assets


class ResumeTests(unittest.TestCase):
    def test_verified_assets_skipped_and_conflicts_never_overwritten(self):
        expected={'a':{'size':3,'sha256':'abc'},'b':{'size':4,'sha256':'def'}}
        remote={'a':{'size':3,'digest':'sha256:abc','state':'uploaded'}}
        self.assertEqual(missing_assets(expected,remote),['b'])
        for change in [{'size':2},{'digest':'sha256:different'},{'state':'starter'}]:
            with self.assertRaisesRegex(ValueError,'not_verified'):missing_assets(expected,{'a':{**remote['a'],**change}})
        with self.assertRaisesRegex(ValueError,'unexpected'):missing_assets(expected,{'extra':remote['a']})


if __name__=='__main__':unittest.main()
