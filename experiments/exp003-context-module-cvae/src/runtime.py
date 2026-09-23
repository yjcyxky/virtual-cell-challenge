"""Verify a uv-synchronized environment without mutating it during active runs."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys


EXPERIMENT = Path(__file__).resolve().parents[1]


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
    args = parser.parse_args()
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
