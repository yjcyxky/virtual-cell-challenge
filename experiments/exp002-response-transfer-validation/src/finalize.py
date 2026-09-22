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
    assert len(groups) == (9 if report.get('scope') == 'genetic_only' else 15)
    for path in groups:
        group = json.loads(path.read_text())
        assert group['status'] == 'completed'
        for name, digest in group['artifacts'].items():
            assert hash_file(path.parent / name) == digest, 'evaluation artifact changed: ' + str(path.parent / name)
    runtime = {'python': sys.version, 'platform': platform.platform(),
               'packages': {p: importlib.metadata.version(p) for p in ['numpy', 'scipy', 'pandas', 'h5py', 'pyarrow', 'wandb']},
               'finalizer_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
               'issue': report['issue']}
    write_json(output / 'runtime.json', runtime)
    write_json(output / 'artifact-schema.json', {
        'measurement_scale': 'mean of per-cell log1p(count / total_native_counts * 10000)',
        'collection_moments_h5': {
            'axis_order': ['task_uid', 'variant', 'source_gene'],
            'variants': ['full', 's20260921h0', 's20260921h1', 's20260922h0', 's20260922h1', 's20260923h0', 's20260923h1'],
            'target': 'observed target mean on matched source technical support',
            'matched_control': 'NTC technical-stratum means weighted by that variant target-cell fractions',
            'variance_mean': 'variance of the estimated mean; standard error is its square root',
            'target_excluded': 'native readout excluded as intervention gene; also require nonempty safe_symbol',
            'eligibility': 'task-support.parquet determines estimability; a stored mean alone is not an eligible prediction label'},
        'classifications_h5': {
            'axis_order': ['axes.json targets', 'scale', 'variant', 'tolerance', 'axes.json genes'],
            'scales': ['control_only', 'source_matched'], 'tolerances': [.05, .1, .2],
            'baseline_state_and_response_state': {'0': 'unknown', '1': 'equivalent', '2': 'different'},
            'quadrant': {'0': 'unknown_or_excluded_readout', '1': 'baseline_equivalent_response_equivalent',
                         '2': 'baseline_equivalent_response_different', '3': 'baseline_different_response_equivalent',
                         '4': 'baseline_different_response_different'},
            'activity': {'0': 'unknown_or_excluded_readout', '1': 'near_zero_in_all_contexts', '2': 'active_in_at_least_one_context'},
            'conserved_nonzero': 'response equivalent with simultaneous same-sign nonzero support, excluding operational near-zero',
            'context_set': 'classification-counts.parquet: original full-estimable backgrounds fixed across split variants',
            'intervals': 'reconstruct from source means and variance_mean using estimators.py; Bonferroni only over context pairs within (p,g)'},
        'predictions_h5': {
            'groups': 'scale / held_cell_line', 'row_axis': 'target_index selects axes.json targets',
            'column_axis': 'axes.json genes', 'values': 'predicted delta; add control-only C to obtain predicted expression',
            'evaluation_mask': 'intersection of finite truth and all model predictions; at least 100 genes',
            'train_quadrant_and_train_conserved_nonzero': 'computed using training backgrounds only',
            'models': 'hyperparameters.json records training cell lines and inner-CV selections'},
        'module_metrics': {
            'observed_and_predicted_values': 'fixed arithmetic mean across eligible member genes',
            'random_membership': 'use stored random candidate order, remove common evaluation exclusions, take prefix equal to actual reference set size',
            'relative_improvement': '1 - balanced mean squared error / balanced zero-response squared error',
            'random_quantile_intervals': 'variation over 20 random gene sets, not biological confidence intervals'},
        'scope': report.get('scope', 'original'),
        'control_role': 'For genetic_only, is_NTC is a storage field: selected Cas9 uses intergenic cutting controls; consult source_metadata.control_role.',
        'genetic_gate': 'training target variance zero is rejected; one donor disables tuning and routing; construct aggregate variance unresolved stays unknown',
        'all_inferred_classes_are_ground_truth': False})
    # Run tracking is linked in tracking.json. Do not ship W&B's mutable symlinks,
    # transport cache, credentials, or duplicate latest-run traversal.
    sources = [p for p in output.iterdir() if p.is_file() and p.suffix in ['.json', '.html', '.parquet', '.yaml', '.log']]
    for directory in ['collection', 'evaluation', 'audit', 'posthoc-zero-variance', 'scope']:
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
        'command': 'experiments/exp002-response-transfer-validation/reproduce.sh --run-id independent-reproduction' + (' --scope genetic_only' if report.get('scope') == 'genetic_only' else ''),
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
