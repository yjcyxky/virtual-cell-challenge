"""Freeze a portable scientific bundle after the run and W&B synchronization end."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

EXPERIMENT = Path(__file__).resolve().parents[1]
ROOT = EXPERIMENT.parents[1]
sys.path.insert(0, str(ROOT / 'scripts/dossier'))
from profile_responses import write_json
from rna import hash_file


def run(run_id):
    output = EXPERIMENT / 'outputs' / run_id
    report = json.loads((output / 'report.json').read_text())
    assert report['status'] == 'completed'
    assert json.loads((output / 'audit/report.json').read_text())['all_collection_artifact_hashes_verified']
    groups = list((output / 'evaluation').glob('*/report.json'))
    assert len(groups) == 15
    for path in groups:
        group = json.loads(path.read_text())
        assert group['status'] == 'completed'
        for name, digest in group['artifacts'].items():
            assert hash_file(path.parent / name) == digest, 'evaluation artifact changed: ' + str(path.parent / name)
    runtime = {'python': sys.version, 'platform': platform.platform(),
               'packages': {p: importlib.metadata.version(p) for p in ['numpy', 'scipy', 'pandas', 'h5py', 'pyarrow', 'wandb']},
               'finalizer_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
               'issue': 'https://github.com/yjcyxky/virtual-cell-challenge/issues/28'}
    write_json(output / 'runtime.json', runtime)
    # Run tracking is linked in tracking.json. Do not ship W&B's mutable symlinks,
    # transport cache, credentials, or duplicate latest-run traversal.
    sources = [p for p in output.iterdir() if p.is_file() and p.suffix in ['.json', '.html', '.parquet', '.yaml', '.log']]
    for directory in ['collection', 'evaluation', 'audit']:
        sources.extend(p for p in (output / directory).rglob('*') if p.is_file())
    bundle = output / 'publication'
    bundle.mkdir(exist_ok=False)
    for source in sources:
        assert not source.is_symlink()
        destination = bundle / source.relative_to(output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.link(source, destination)
    reproduction = bundle / 'reproduction'
    reproduction.mkdir()
    for name in ['pyproject.toml', 'uv.lock', 'reproduce.sh']:
        shutil.copy2(EXPERIMENT / name, reproduction / name)
    # Source snapshots are for inspection; the executable entry uses the exact
    # repository commit since verified dataset adapters are shared repo modules.
    shutil.copytree(EXPERIMENT / 'src', reproduction / 'src', ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree(EXPERIMENT / 'tests', reproduction / 'tests', ignore=shutil.ignore_patterns('__pycache__'))
    write_json(reproduction / 'entry.json', {
        'repository': 'https://github.com/yjcyxky/virtual-cell-challenge',
        'commit': runtime['finalizer_commit'],
        'command': 'experiments/exp002-response-transfer-validation/reproduce.sh --run-id independent-reproduction',
        'requires_frozen_source_data': True,
        'frozen_collection_identity': json.loads((output / 'collection/identity.json').read_text()),
        'source_count_matrices_are_not_in_release': True})
    paths = sorted(p for p in bundle.rglob('*') if p.is_file())
    (bundle / 'SHA256SUMS').write_text(''.join(hash_file(p) + '  ' + str(p.relative_to(bundle)) + '\n' for p in paths))
    print(json.dumps({'status': 'completed', 'bundle': str(bundle), 'files': len(paths),
                      'bytes': sum(p.stat().st_size for p in paths)}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-id', required=True)
    run(p.parse_args().run_id)
