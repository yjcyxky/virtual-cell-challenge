#!/usr/bin/env python3
"""Copy the complete checkout to NAS, then replace verified large files by links.

Run --dry-run first. No destination-only files are deleted. Git-managed files and
.git stay regular locally; only untracked files strictly over 1 GB are offloaded
after a successful rsync and independent SHA-256 comparison. Use a quiet checkout.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile
import uuid

DEFAULT_DESTINATION = Path('/mnt/projects/virtual-cell-challenge')
STATE_DIR = '.nas-sync'
OFFLOAD_THRESHOLD = 1_000_000_000  # GB, decimal; exactly 1 GB stays local.


def signature(st):
    return (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)


def validate_roots(source, destination, require_mount=None):
    source, destination = source.resolve(), destination.resolve()
    if source == destination or source.is_relative_to(destination) or destination.is_relative_to(source):
        raise ValueError('Source and destination must be separate, non-nested directories')
    if not source.is_dir() or not (source / '.git').is_dir() or (source / '.git').is_symlink():
        raise ValueError('Source must be a checkout with its own regular .git directory')
    if require_mount is not None:
        mount = require_mount.absolute()
        if not os.path.ismount(mount) or not destination.is_relative_to(mount.resolve()):
            raise ValueError(f'Required NAS mount is unavailable or does not contain destination: {mount}')
    return source, destination


def git_tracked(source):
    raw = subprocess.check_output(['git', '-C', str(source), 'ls-files', '-z'])
    return {os.fsdecode(p) for p in raw.split(b'\0') if p}


def check_git_locks(root):
    git = root / '.git'
    if git.exists() and any(git.rglob('*.lock')):
        raise ValueError(f'Git is busy or has a stale lock: {git}')


def repository_marker(directory):
    marker = directory / '.git'
    if marker.is_dir():
        return True
    if marker.is_file():
        with marker.open('rb') as handle:
            return handle.read(7) == b'gitdir:'
    # uv caches contain empty .git sentinel files, not nested repositories.
    return False


def is_versioned(source, path):
    repository = source
    for parent in path.parents:
        if parent == source:
            break
        if repository_marker(parent):
            repository = parent
            break
    result = subprocess.run(['git', '--literal-pathspecs', '-C', str(repository), 'ls-files', '--error-unmatch', '--', str(path.relative_to(repository))],
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    if result.returncode not in (0, 1):
        raise ValueError(f'Cannot determine Git tracking state for {path}: {os.fsdecode(result.stderr)}')
    return result.returncode == 0


def scan(source, destination):
    tracked = git_tracked(source)
    plan = {'files': 0, 'bytes': 0, 'already_on_nas': [], 'candidates': [],
            'rewritten_links': [], 'external_links': 0, 'kept_large': 0, 'missing_nas_links': []}
    for directory, dirs, files in os.walk(source, followlinks=False):
        parent = Path(directory)
        if parent == source:
            dirs[:] = [d for d in dirs if d != STATE_DIR]
        relative_parent = parent.relative_to(source)
        if parent != source and '.git' not in relative_parent.parts and '.git' in dirs + files and repository_marker(parent):
            # Nested repositories/submodules have their own tracked file sets.
            tracked.update(str(relative_parent / path) for path in git_tracked(parent))
        dest_parent = destination / relative_parent
        # rsync must never traverse a destination directory redirected elsewhere.
        if dest_parent.is_symlink() or (dest_parent.exists() and not dest_parent.is_dir()):
            raise ValueError(f'Destination directory has an incompatible type: {dest_parent}')
        for name in dirs + files:
            path = parent / name
            relative = path.relative_to(source)
            st = path.lstat()
            if stat.S_ISLNK(st.st_mode):
                target = path.resolve()
                dest = destination / relative
                if target == dest:
                    plan['already_on_nas'].append(str(relative))
                    if not target.exists():
                        # A pre-existing dangling link contains no bytes to copy.
                        # Preserve and report it; never manufacture a self-link on NAS.
                        plan['missing_nas_links'].append(str(relative))
                elif os.path.isabs(os.readlink(path)) and target.is_relative_to(source):
                    plan['rewritten_links'].append((str(relative), str(destination / target.relative_to(source))))
                elif not target.is_relative_to(source) and not target.is_relative_to(destination):
                    plan['external_links'] += 1
            elif stat.S_ISREG(st.st_mode):
                plan['files'] += 1
                plan['bytes'] += st.st_size
                if st.st_size > OFFLOAD_THRESHOLD:
                    if '.git' in relative.parts or str(relative) in tracked:
                        plan['kept_large'] += 1
                    else:
                        plan['candidates'].append((str(relative), signature(st)))
            elif not stat.S_ISDIR(st.st_mode):
                raise ValueError(f'Cannot safely synchronize a socket/device/FIFO: {path}')
    return plan


def filter_pattern(relative):
    # Anchored literal exclusions; protect brackets, wildcards and backslashes.
    if '\n' in relative or '\r' in relative:
        raise ValueError('NAS-backed link paths cannot contain newline characters')
    return '/' + ''.join('\\' + c if c in '\\*?[]' else c for c in relative)


def rsync_command(source, destination, exclusions, backup, *, dry_run=False, checksum=False):
    command = ['rsync', '-aH', '--sparse', '--fsync', '--no-owner', '--no-group',
               '--no-devices', '--no-specials', '--stats', '--backup',
               '--backup-dir=' + str(backup), '--partial-dir=.rsync-partial',
               '--exclude=/' + STATE_DIR, '--exclude-from=' + str(exclusions)]
    if dry_run:
        command.append('--dry-run')
    if checksum:
        command.append('--checksum')
    command += ['--', str(source) + '/', str(destination) + '/']
    return command


def has_writer(path):
    """Detect accessible writable FDs; this is not a lock against future writers."""
    identity = path.stat()
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            if proc.stat().st_uid != os.getuid():
                continue
            for fd in (proc / 'fd').iterdir():
                try:
                    st = fd.stat()
                    if (st.st_dev, st.st_ino) != (identity.st_dev, identity.st_ino):
                        continue
                    info = (proc / 'fdinfo' / fd.name).read_text()
                    flags = next(line.split()[1] for line in info.splitlines() if line.startswith('flags:'))
                    if int(flags, 8) & os.O_ACCMODE != os.O_RDONLY:
                        return True
                except (FileNotFoundError, ProcessLookupError, PermissionError):
                    continue
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    return False


def sha_handle(handle):
    digest = hashlib.sha256()
    while block := handle.read(8 * 1024 * 1024):
        digest.update(block)
    return digest.hexdigest()


def atomic_link(path, target):
    temporary = path.with_name('.' + path.name + '.nas-link-' + uuid.uuid4().hex)
    try:
        temporary.symlink_to(target)
        os.replace(temporary, path)
        fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        if temporary.is_symlink():
            temporary.unlink()


def offload_file(source, destination, relative, expected, record):
    local, remote = source / relative, destination / relative
    if local.is_symlink() or remote.is_symlink() or remote.resolve() != remote:
        raise ValueError(f'Unexpected symlink during verification: {relative}')
    if '.git' in Path(relative).parts or is_versioned(source, local) or local.stat().st_size <= OFFLOAD_THRESHOLD:
        record({'event': 'kept_local', 'path': relative, 'reason': 'versioned_or_not_over_1GB'})
        return None
    # Removing one local hardlink changes ctime of its siblings, without changing
    # their content. The full signature is checked again across both hashes.
    if signature(local.stat())[:4] != tuple(expected)[:4]:
        raise ValueError(f'Source changed since planning; keep local and rerun: {relative}')
    if has_writer(local) or has_writer(remote):
        raise ValueError(f'File is open for writing; keep local and rerun when idle: {relative}')
    with local.open('rb') as left, remote.open('rb') as right:
        before_local, before_remote = signature(os.fstat(left.fileno())), signature(os.fstat(right.fileno()))
        if before_local[2] != before_remote[2] or before_local[:2] == before_remote[:2]:
            raise ValueError(f'Destination is not an independent complete copy: {relative}')
        digest = sha_handle(left)
        if digest != sha_handle(right):
            raise ValueError(f'SHA-256 mismatch; local preserved. Rerun with --checksum: {relative}')
        os.fsync(right.fileno())
        if (signature(local.stat()) != before_local or signature(remote.stat()) != before_remote
                or signature(os.fstat(left.fileno())) != before_local
                or signature(os.fstat(right.fileno())) != before_remote):
            raise ValueError(f'File changed while verifying; local preserved: {relative}')
        event = {'path': relative, 'target': str(remote), 'bytes': before_local[2], 'sha256': digest}
        record({'event': 'verified_before_link', **event})
        # A long transfer/hash may overlap a manual git add. Recheck immediately
        # before replacing the file, even though callers should use an idle tree.
        if is_versioned(source, local):
            record({'event': 'kept_local', 'path': relative, 'reason': 'now_versioned'})
            return None
        if (signature(local.stat()) != before_local or signature(remote.stat()) != before_remote
                or has_writer(local) or has_writer(remote)):
            raise ValueError(f'File changed or acquired a writer before replacement: {relative}')
        # A rename publishes the new symlink atomically; no unlink-then-link gap.
        atomic_link(local, remote)
        record({'event': 'linked', **event})
        return event['bytes']


@contextmanager
def sync_lock(destination):
    state = destination / STATE_DIR
    if state.is_symlink():
        raise ValueError(f'Sync state directory must not be a symlink: {state}')
    state.mkdir(parents=True, exist_ok=True)
    with (state / 'sync.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield state


def synchronize(source, destination, *, dry_run=False, copy_only=False,
                checksum=False, require_mount=None):
    source, destination = validate_roots(source, destination, require_mount)
    check_git_locks(source)
    check_git_locks(destination)
    if shutil.which('rsync') is None:
        raise ValueError('rsync is required')
    print(f'Scanning {source} and checking destination paths in {destination} ...', flush=True)
    plan = scan(source, destination)
    summary = {k: plan[k] for k in ('files', 'bytes', 'kept_large', 'external_links')}
    summary.update(already_on_nas=len(plan['already_on_nas']), large_files=len(plan['candidates']),
                   offload_bytes=sum(item[1][2] for item in plan['candidates']),
                   source=str(source), destination=str(destination), dry_run=dry_run,
                   missing_nas_links=plan['missing_nas_links'],
                   selection='not Git tracked AND size > 1000000000 bytes; exclude .git')
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    run = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:8]
    # No --delete: NAS-only downloads, outputs and previous backups are retained.
    with tempfile.TemporaryDirectory(prefix='vcc-nas-sync-') as temporary:
        exclusions = Path(temporary) / 'exclusions'
        exclusions.write_text(''.join(filter_pattern(p) + '\n' for p in plan['already_on_nas']))
        backup = destination / STATE_DIR / 'backups' / run
        command = rsync_command(source, destination, exclusions, backup, dry_run=dry_run, checksum=checksum)
        if dry_run:
            subprocess.run(command, check=True)
            return summary
        with sync_lock(destination) as state:
            journal_path = state / (run + '.jsonl')
            with journal_path.open('x') as journal:
                def record(event):
                    journal.write(json.dumps({'at': datetime.now(timezone.utc).isoformat(), **event}, ensure_ascii=False) + '\n')
                    journal.flush()
                    os.fsync(journal.fileno())

                record({'event': 'started', **summary})
                try:
                    subprocess.run(command, check=True)
                    check_git_locks(source)
                    check_git_locks(destination)
                    for relative, target in plan['rewritten_links']:
                        # Relocate absolute internal links so NAS does not depend on Downloads.
                        remote = destination / relative
                        if not remote.is_symlink():
                            raise ValueError(f'Expected copied internal symlink: {remote}')
                        atomic_link(remote, target)
                    count = moved = 0
                    if not copy_only:
                        for relative, expected in plan['candidates']:
                            print(f'Verify and offload: {relative}', flush=True)
                            size = offload_file(source, destination, relative, expected, record)
                            if size is not None:
                                moved += size
                                count += 1
                    summary.update(linked_files=count, linked_logical_bytes=moved, journal=str(journal_path))
                    record({'event': 'completed', **summary})
                except BaseException as error:
                    record({'event': 'failed', 'error': str(error)})
                    raise
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--destination', type=Path, default=DEFAULT_DESTINATION)
    parser.add_argument('--dry-run', action='store_true', help='Plan and rsync preview only; do not change either tree')
    parser.add_argument('--copy-only', action='store_true', help='Copy without replacing local files')
    parser.add_argument('--checksum', action='store_true', help='Also compare contents for rsync decisions on all files')
    parser.add_argument('--require-mount', type=Path, help='Require this mount; /mnt/projects is mandatory for the default destination')
    args = parser.parse_args()
    mount = args.require_mount
    if args.destination.absolute().is_relative_to(Path('/mnt/projects')):
        mount = Path('/mnt/projects')
    try:
        synchronize(args.source, args.destination,
                    dry_run=args.dry_run, copy_only=args.copy_only, checksum=args.checksum, require_mount=mount)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f'Sync stopped: {error}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
