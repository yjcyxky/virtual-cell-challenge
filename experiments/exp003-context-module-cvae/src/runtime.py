"""Verify a uv-synchronized environment without mutating it during active runs."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


EXPERIMENT = Path(__file__).resolve().parents[1]


def run_in_base(command):
    """Isolate micromamba's transient registry while preserving the child's caches."""
    if not command:
        raise ValueError('base_command_required')
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


def environment_identity(experiment=None):
    experiment = EXPERIMENT if experiment is None else Path(experiment).resolve()
    assert Path(sys.prefix).resolve() == (experiment / '.venv').resolve(), 'experiment_environment_required'
    packages = {}
    for package in importlib.metadata.distributions():
        name = package.metadata['Name'].lower().replace('_', '-')
        packages[name] = {'version': package.version, 'metadata': {
            file: hashlib.sha256((package.read_text(file) or '').encode()).hexdigest()
            for file in ['METADATA', 'WHEEL', 'RECORD']}}
    return {'uv_lock_sha256': digest(experiment / 'uv.lock'),
            'pyproject_sha256': digest(experiment / 'pyproject.toml'),
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
    parser.add_argument('--experiment', type=Path, default=EXPERIMENT)
    parser.add_argument('--record', action='store_true', help='Only after successful uv sync under the exclusive environment lock.')
    parser.add_argument('--base-run', nargs=argparse.REMAINDER,
                        help='Launch through the fixed base environment with an isolated process registry.')
    args = parser.parse_args()
    if args.base_run is not None:
        if args.record or not args.base_run:
            parser.error('--base-run requires a command and cannot be combined with --record')
        raise SystemExit(run_in_base(args.base_run))
    identity = environment_identity(args.experiment)
    path = args.experiment.resolve() / '.venv' / 'experiment-environment.json'
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
