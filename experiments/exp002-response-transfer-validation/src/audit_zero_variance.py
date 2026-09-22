"""Post-hoc diagnostic of zero-observation variance in operational nonzero labels."""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'scripts/dossier'))
import h5py
import numpy as np
import pandas as pd
from profile_responses import write_json
from rna import hash_file


def run(run_folder):
    output = run_folder / 'posthoc-zero-variance'
    output.mkdir(exist_ok=False)
    initial = run_folder / 'report.json'
    shutil.copy2(initial, output / 'preregistered-experiment-report.json')
    rows, summaries = [], []
    for report_path in sorted((run_folder / 'evaluation').glob('*/report.json')):
        report = json.loads(report_path.read_text())
        if report['supplemental']:
            continue
        axes = json.loads((report_path.parent / 'axes.json').read_text())
        counts = pd.read_parquet(report_path.parent / 'classification-counts.parquet')
        with ExitStack() as stack:
            contexts = {}
            for context in report['contexts']:
                folder = run_folder / 'collection' / context['context_id']
                hf = stack.enter_context(h5py.File(folder / 'moments.h5'))
                tasks = pd.read_parquet(folder / 'tasks.parquet')
                assert not tasks.canonical_target.duplicated().any()
                support = pd.read_parquet(folder / 'task-support.parquet')
                full = support.loc[support.variant.eq('full')].set_index('task_uid')
                halves = support.loc[~support.variant.eq('full')].assign(ok=lambda d: d.status.eq('completed')).groupby('task_uid').ok.all()
                symbols = {g: i for i, g in enumerate(hf['safe_symbol'].asstr()[:]) if g}
                contexts[context['source_metadata']['cell_line']] = {
                    'hf': hf, 'task_index': {p: i for i, p in enumerate(tasks.canonical_target)},
                    'task_uids': tasks.task_uid.tolist(), 'genes': np.array([symbols[g] for g in axes['genes']]),
                    'n': full.n_target.to_dict(), 'halves': halves.to_dict()}
            classified = stack.enter_context(h5py.File(report_path.parent / 'classifications.h5'))
            for si, scale in enumerate(['control_only', 'source_matched']):
                selected = counts.loc[counts.scale.eq(scale) & counts.variant.eq(0) & counts.tolerance.eq(.1)].set_index('canonical_target')
                group_rows = []
                for pi, target in enumerate(axes['targets']):
                    indices = np.flatnonzero(classified['conserved_nonzero'][pi, si, 0, 1])
                    if not len(indices):
                        continue
                    lines = json.loads(selected.loc[target, 'context_set'])
                    means, variances, nt, half_ok = [], [], [], []
                    for line in lines:
                        c = contexts[line]; ti = c['task_index'][target]; uid = c['task_uids'][ti]
                        gene_indices = c['genes'][indices]
                        means.append(c['hf']['target'][ti, 0][gene_indices])
                        variances.append(c['hf']['target_variance_mean'][ti, 0][gene_indices])
                        nt.append(c['n'][uid]); half_ok.append(c['halves'][uid])
                    means, variances = np.array(means), np.array(variances)
                    all_zero = (means == 0).all(0)
                    any_zero = (means == 0).any(0)
                    zero_variance = (variances == 0).any(0)
                    assert not np.any((means == 0) & (variances != 0))
                    for j, gi in enumerate(indices):
                        group_rows.append({'group': report['group'], 'scale': scale, 'canonical_target': target,
                                           'readout_gene': axes['genes'][gi], 'contexts': len(lines),
                                           'all_target_observations_zero_in_every_context': bool(all_zero[j]),
                                           'all_target_observations_zero_in_any_context': bool(any_zero[j]),
                                           'zero_target_variance_in_any_context': bool(zero_variance[j]),
                                           'minimum_target_cells_in_context_set': min(nt),
                                           'all_six_halves_supported_in_all_contexts': all(half_ok)})
                expected = int(selected.conserved_nonzero.sum())
                assert len(group_rows) == expected
                rows.extend(group_rows)
                summaries.append({'group': report['group'], 'family': report['group'].split('_')[0], 'scale': scale,
                                  'operational_conserved_nonzero_labels': expected,
                                  'all_target_zero_every_context': sum(r['all_target_observations_zero_in_every_context'] for r in group_rows),
                                  'target_zero_any_context': sum(r['all_target_observations_zero_in_any_context'] for r in group_rows),
                                  'zero_target_variance_any_context': sum(r['zero_target_variance_in_any_context'] for r in group_rows),
                                  'all_contexts_six_halves_supported': sum(r['all_six_halves_supported_in_all_contexts'] for r in group_rows)})
        print(json.dumps({'posthoc_group': report['group'], 'labels': len(rows)}), flush=True)
    pd.DataFrame(rows).to_parquet(output / 'operational-nonzero-label-diagnostics.parquet', index=False, compression='zstd')
    frame = pd.DataFrame(summaries)
    frame.to_parquet(output / 'by-condition.parquet', index=False)
    family = frame.groupby(['family', 'scale']).sum(numeric_only=True).reset_index().to_dict('records')
    write_json(output / 'report.json', {
        'status': 'completed', 'posthoc': True,
        'registration': 'https://github.com/yjcyxky/virtual-cell-challenge/issues/28#issuecomment-5771574109',
        'initial_report_sha256': hash_file(output / 'preregistered-experiment-report.json'),
        'code_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'family_summary': family, 'model_predictions_or_classifications_changed': False,
        'interpretation': 'Zero observed target expression implies zero plug-in target variance, not zero population uncertainty. These counts identify an unreliable interval scenario; they do not label biological false positives.'})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-folder', type=Path, required=True)
    run(p.parse_args().run_folder.resolve())
