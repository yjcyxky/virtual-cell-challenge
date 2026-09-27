"""Verify a uv-synchronized environment without mutating it during active runs."""
import argparse
import ctypes
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


EXPERIMENT = Path(__file__).resolve().parents[1]


def memory_policy():
    libc = ctypes.CDLL(None, use_errno=True)
    libc.prctl.argtypes = [ctypes.c_int] + [ctypes.c_ulong] * 4
    libc.prctl.restype = ctypes.c_int
    disabled = libc.prctl(42, 0, 0, 0, 0)  # PR_GET_THP_DISABLE
    if disabled < 0:
        raise OSError(ctypes.get_errno(), 'cannot query process THP policy')
    return {'transparent_hugepages_disabled': disabled == 1,
            'numpy_madvise_hugepage': os.environ.get('NUMPY_MADVISE_HUGEPAGE')}


def configure_memory_policy():
    """Avoid synchronous THP compaction on GB10; inherited by child processes."""
    libc = ctypes.CDLL(None, use_errno=True)
    libc.prctl.argtypes = [ctypes.c_int] + [ctypes.c_ulong] * 4
    libc.prctl.restype = ctypes.c_int
    if libc.prctl(41, 1, 0, 0, 0) != 0:  # PR_SET_THP_DISABLE; process only
        raise OSError(ctypes.get_errno(), 'cannot disable process transparent hugepages')
    os.environ['NUMPY_MADVISE_HUGEPAGE'] = '0'
    policy = memory_policy()
    if not policy['transparent_hugepages_disabled']:
        raise RuntimeError('process THP policy was not applied')
    return policy


def run_in_base(command):
    """Isolate micromamba's transient registry while preserving the child's caches."""
    if not command:
        raise ValueError('base_command_required')
    configure_memory_policy()
    restore_cache = (['env', 'XDG_CACHE_HOME=' + os.environ['XDG_CACHE_HOME']]
                     if 'XDG_CACHE_HOME' in os.environ else ['env', '-u', 'XDG_CACHE_HOME'])
    with tempfile.TemporaryDirectory(prefix='vcc-mamba-runtime-') as process_cache:
        environment = dict(os.environ, XDG_CACHE_HOME=process_cache)
        # Preserve the shell's intentionally inherited environment-lock FD. All
        # descriptors created by Python remain non-inheritable by default.
        return subprocess.call(['micromamba', 'run', '-n', 'virtual-cell', *restore_cache, *command],
                               env=environment, close_fds=False)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def environment_identity():
    assert Path(sys.prefix).resolve() == (EXPERIMENT / '.venv').resolve(), 'experiment_environment_required'
    packages = {}
    for package in importlib.metadata.distributions():
        name = package.metadata['Name'].lower().replace('_', '-')
        packages[name] = {'version': package.version, 'metadata': {
            file: hashlib.sha256((package.read_text(file) or '').encode()).hexdigest()
            for file in ['METADATA', 'WHEEL', 'RECORD']}}
    return {'uv_lock_sha256': digest(EXPERIMENT / 'uv.lock'),
            'pyproject_sha256': digest(EXPERIMENT / 'pyproject.toml'),
            'python': sys.version, 'base_executable': sys._base_executable,
            'python_executable_sha256': digest(sys._base_executable),
            'pyvenv_sha256': digest(Path(sys.prefix) / 'pyvenv.cfg'),
            'packages': packages}


def verify_environment(path, current):
    if not path.is_file():
        raise ValueError('environment_not_yet_synchronized')
    if json.loads(path.read_text()) != current:
        raise ValueError('locked_environment_changed')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--record', action='store_true', help='Only after successful uv sync under the exclusive environment lock.')
    parser.add_argument('--base-run', nargs=argparse.REMAINDER,
                        help='Launch through the fixed base environment with an isolated process registry.')
    args = parser.parse_args()
    if args.base_run is not None:
        if args.record or not args.base_run:
            parser.error('--base-run requires a command and cannot be combined with --record')
        raise SystemExit(run_in_base(args.base_run))
    identity = environment_identity()
    path = EXPERIMENT / '.venv' / 'experiment-environment.json'
    if args.record:
        path.write_text(json.dumps(identity, sort_keys=True, indent=2))
    else:
        try:
            verify_environment(path, identity)
        except ValueError as error:
            print(str(error), file=sys.stderr)
            raise SystemExit(1) from error


if __name__ == '__main__':
    main()
