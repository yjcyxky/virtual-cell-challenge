"""Frozen evidence accepts only untracked artifacts in the exact NAS mirror."""
import hashlib
import importlib.util
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch


spec = importlib.util.spec_from_file_location('research_paths', Path(__file__).parents[1] / 'research.py')
research = importlib.util.module_from_spec(spec)
spec.loader.exec_module(research)


class ArtifactPathsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.root = self.base / 'repo'
        self.root.mkdir()
        subprocess.run(['git', 'init', '-q', str(self.root)], check=True)
        self.mirror = self.base / 'mirror'
        self.relative = 'experiments/example/outputs/run/predictions/counts.h5ad'
        self.target = self.mirror / self.relative
        self.target.parent.mkdir(parents=True)
        self.target.write_bytes(b'frozen prediction')
        self.link = self.root / self.relative
        self.link.parent.mkdir(parents=True)
        self.link.symlink_to(self.target)
        self.record = {'path': self.relative, 'sha256': hashlib.sha256(self.target.read_bytes()).hexdigest()}
        self.patch = patch.object(research, 'ARTIFACT_MIRROR', self.mirror, create=True)
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def test_matching_untracked_mirror_is_hashed_in_worktree_and_staged_checks(self):
        for staged in (False, True):
            self.assertIsNone(research.ref_error(self.root, self.record, staged))
        self.target.write_bytes(b'changed')
        self.assertIn('hash changed', research.ref_error(self.root, self.record))

    def test_tracked_external_link_is_not_an_artifact_exception(self):
        subprocess.run(['git', '-C', str(self.root), 'add', self.relative], check=True)
        self.assertIn('escapes repository', research.ref_error(self.root, self.record))

    def test_other_external_file_and_wrong_mirror_location_are_rejected(self):
        for other in (self.base / 'outside', self.mirror / 'different.h5ad'):
            other.write_bytes(b'frozen prediction')
            self.link.unlink()
            self.link.symlink_to(other)
            self.assertIn('escapes repository', research.ref_error(self.root, self.record))

    def test_source_code_cannot_use_mirror_exception(self):
        source = self.root / 'src/model.py'
        target = self.mirror / 'src/model.py'
        source.parent.mkdir(); target.parent.mkdir()
        target.write_bytes(b'frozen prediction'); source.symlink_to(target)
        self.assertIn('escapes repository', research.ref_error(self.root, dict(self.record, path='src/model.py')))

    def test_absolute_and_parent_traversal_are_rejected(self):
        for name in (str(self.target), '../mirror/' + self.relative):
            self.assertIsNotNone(research.ref_error(self.root, dict(self.record, path=name)))

    def test_missing_mirror_is_not_accepted(self):
        self.target.unlink()
        self.assertIsNotNone(research.ref_error(self.root, self.record))


if __name__ == '__main__':
    unittest.main()
