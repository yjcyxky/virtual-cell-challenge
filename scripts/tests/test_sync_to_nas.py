"""End-to-end storage checks: preserve bytes before offloading local artifacts."""
import importlib.util
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('sync_to_nas', Path(__file__).parents[1] / 'sync_to_nas.py')
sync = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sync)


class SyncTests(unittest.TestCase):
    def setUp(self):
        # Exercise actual copy/hash/replace with small fixtures. A separate sparse
        # file test checks the real decimal-GB production boundary.
        threshold = patch.object(sync, 'OFFLOAD_THRESHOLD', 100)
        threshold.start()
        self.addCleanup(threshold.stop)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.source = Path(self.temp.name) / 'source'
        self.destination = Path(self.temp.name) / 'nas'
        self.source.mkdir()
        self.destination.mkdir()
        subprocess.run(['git', 'init', '-q', str(self.source)], check=True)
        (self.source / 'large.bin').write_bytes(b'raw counts' * 30)
        (self.source / 'small.txt').write_text('abc')

    def run_sync(self, **kwargs):
        return sync.synchronize(self.source, self.destination, **kwargs)

    def test_real_threshold_requires_untracked_and_strictly_more_than_one_gb(self):
        sizes = {'exact.bin': 1_000_000_000, 'below.bin': 999_999_999,
                 'above.bin': 1_000_000_001, 'tracked.bin': 1_000_000_001}
        for name, size in sizes.items():
            with (self.source / name).open('wb') as handle:
                handle.truncate(size)
        # Mark tracked without hashing a GB of zeros into the fixture repository.
        blob = subprocess.check_output(['git', '-C', str(self.source), 'hash-object', '-w', '--stdin'], input=b'fixture').strip().decode()
        subprocess.run(['git', '-C', str(self.source), 'update-index', '--add', '--cacheinfo', '100644', blob, 'tracked.bin'], check=True)
        with patch.object(sync, 'OFFLOAD_THRESHOLD', 1_000_000_000):
            plan = sync.scan(self.source, self.destination)
        self.assertEqual([path for path, _ in plan['candidates']], ['above.bin'])

    def test_newly_tracked_file_during_hash_is_not_replaced(self):
        self.run_sync(copy_only=True)
        path = self.source / 'large.bin'
        original = sync.sha_handle
        calls = 0

        def track_after_hash(handle):
            nonlocal calls
            digest = original(handle)
            calls += 1
            if calls == 2:
                subprocess.run(['git', '-C', str(self.source), 'add', 'large.bin'], check=True)
            return digest

        with patch.object(sync, 'sha_handle', side_effect=track_after_hash):
            result = sync.offload_file(self.source, self.destination, 'large.bin', sync.signature(path.stat()), lambda event: None)
        self.assertIsNone(result)
        self.assertFalse(path.is_symlink())

    def test_nested_repository_tracked_files_stay_local(self):
        nested = self.source / 'nested'
        subprocess.run(['git', 'init', '-q', str(nested)], check=True)
        (nested / 'tracked.bin').write_bytes(b'tracked' * 30)
        (nested / 'untracked.bin').write_bytes(b'untracked' * 30)
        subprocess.run(['git', '-C', str(nested), 'add', 'tracked.bin'], check=True)
        self.run_sync()
        self.assertFalse((nested / 'tracked.bin').is_symlink())
        self.assertTrue((nested / 'untracked.bin').is_symlink())
        self.assertFalse(any(p.is_symlink() for p in (nested / '.git').rglob('*') if p.is_file()))

    def test_uv_cache_git_sentinel_is_not_mistaken_for_repository(self):
        cache = self.source / 'cache'
        cache.mkdir()
        (cache / '.git').write_bytes(b'')
        (cache / 'large.bin').write_bytes(b'cache' * 30)
        self.run_sync()
        self.assertTrue((cache / 'large.bin').is_symlink())

    def test_copy_offload_and_repeat_preserve_nas_only_data_and_git(self):
        (self.source / 'tracked.bin').write_bytes(b'tracked' * 30)
        subprocess.run(['git', '-C', str(self.source), 'add', 'tracked.bin'], check=True)
        (self.destination / 'only-on-nas').write_bytes(b'unfinished download')
        result = self.run_sync()
        self.assertEqual(result['linked_files'], 1)
        self.assertTrue((self.source / 'large.bin').is_symlink())
        self.assertFalse((self.destination / 'large.bin').is_symlink())
        self.assertFalse((self.source / 'tracked.bin').is_symlink())
        self.assertEqual((self.source / 'large.bin').read_bytes(), b'raw counts' * 30)
        self.assertEqual((self.destination / 'only-on-nas').read_bytes(), b'unfinished download')
        before = subprocess.check_output(['git', '-C', str(self.source), 'status', '--porcelain'])
        self.assertEqual(self.run_sync()['linked_files'], 0)
        self.assertEqual(before, subprocess.check_output(['git', '-C', str(self.source), 'status', '--porcelain']))
        self.assertTrue(Path(result['journal']).is_file())

    def test_nas_directory_link_and_glob_characters_never_replace_real_target(self):
        name = 'data [1]*?'
        (self.destination / name).mkdir()
        payload = self.destination / name / 'partial'
        payload.write_bytes(b'active NAS download')
        (self.source / name).symlink_to(self.destination / name, target_is_directory=True)
        self.run_sync()
        self.assertFalse((self.destination / name).is_symlink())
        self.assertEqual(payload.read_bytes(), b'active NAS download')

    def test_dry_run_changes_neither_tree_and_copy_only_retains_original(self):
        self.run_sync(dry_run=True)
        self.assertEqual(list(self.destination.iterdir()), [])
        self.assertFalse((self.source / 'large.bin').is_symlink())
        self.run_sync(copy_only=True)
        self.assertFalse((self.source / 'large.bin').is_symlink())
        self.assertEqual((self.destination / 'large.bin').read_bytes(), (self.source / 'large.bin').read_bytes())

    def test_existing_dangling_nas_link_is_reported_without_creating_self_link(self):
        (self.source / 'historical').symlink_to(self.destination / 'historical')
        result = self.run_sync()
        self.assertEqual(result['missing_nas_links'], ['historical'])
        self.assertTrue((self.source / 'historical').is_symlink())
        self.assertFalse(os.path.lexists(self.destination / 'historical'))

    def test_corrupt_same_size_same_time_copy_never_removes_local_and_checksum_repairs(self):
        self.run_sync(copy_only=True)
        local, remote = self.source / 'large.bin', self.destination / 'large.bin'
        st = remote.stat()
        remote.write_bytes(b'x' * local.stat().st_size)
        os.utime(remote, ns=(st.st_atime_ns, st.st_mtime_ns))
        with self.assertRaisesRegex(ValueError, 'SHA-256 mismatch'):
            self.run_sync()
        self.assertFalse(local.is_symlink())
        self.assertEqual(local.read_bytes(), b'raw counts' * 30)
        self.run_sync(checksum=True)
        self.assertTrue(local.is_symlink())
        backups = list((self.destination / '.nas-sync/backups').glob('*/large.bin'))
        self.assertTrue(any(p.read_bytes() == b'x' * 300 for p in backups))

    def test_failed_rsync_leaves_all_local_files(self):
        with patch.object(sync.subprocess, 'run', side_effect=subprocess.CalledProcessError(23, 'rsync')):
            with self.assertRaises(subprocess.CalledProcessError):
                self.run_sync()
        self.assertFalse((self.source / 'large.bin').is_symlink())

    def test_mutation_during_hash_is_detected(self):
        self.run_sync(copy_only=True)
        path = self.source / 'large.bin'
        expected = sync.signature(path.stat())
        original = sync.sha_handle
        calls = 0

        def corrupt_after_hash(handle):
            nonlocal calls
            result = original(handle)
            calls += 1
            if calls == 2:
                path.write_bytes(b'new' * 100)
            return result

        with patch.object(sync, 'sha_handle', side_effect=corrupt_after_hash):
            with self.assertRaisesRegex(ValueError, 'changed while verifying'):
                sync.offload_file(self.source, self.destination, 'large.bin', expected, lambda event: None)
        self.assertFalse(path.is_symlink())

    def test_writable_handle_blocks_offload(self):
        self.run_sync(copy_only=True)
        path = self.source / 'large.bin'
        with path.open('ab'):
            with self.assertRaisesRegex(ValueError, 'open for writing'):
                sync.offload_file(self.source, self.destination, 'large.bin', sync.signature(path.stat()), lambda event: None)
        self.assertFalse(path.is_symlink())

    def test_hardlinks_and_absolute_internal_symlink(self):
        os.link(self.source / 'large.bin', self.source / 'second.bin')
        (self.source / 'internal').symlink_to(self.source / 'small.txt')
        result = self.run_sync()
        self.assertEqual(result['linked_files'], 2)
        self.assertTrue((self.source / 'large.bin').is_symlink())
        self.assertTrue((self.source / 'second.bin').is_symlink())
        self.assertEqual((self.destination / 'large.bin').stat().st_ino, (self.destination / 'second.bin').stat().st_ino)
        self.assertEqual((self.destination / 'internal').resolve(), self.destination / 'small.txt')

    def test_missing_mount_nested_roots_and_redirected_destination_stop_before_copy(self):
        with self.assertRaisesRegex(ValueError, 'mount'):
            self.run_sync(require_mount=self.destination)
        with self.assertRaisesRegex(ValueError, 'non-nested'):
            sync.validate_roots(self.source, self.source / 'subdirectory')
        (self.source / 'folder').mkdir()
        (self.destination / 'folder').symlink_to(self.source / 'folder')
        with self.assertRaisesRegex(ValueError, 'incompatible type'):
            self.run_sync()
        self.assertFalse((self.source / 'large.bin').is_symlink())


if __name__ == '__main__':
    unittest.main()
