"""Audit record independence, frozen identities, and exact split-support estimands."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'scripts/dossier'))
import h5py
import numpy as np
import pandas as pd
from heterogeneity import compare_vectors
from profile_responses import write_json
from rna import hash_file

FROZEN = ROOT / 'data/assessments/response-heterogeneity-dossier-20260919'
SEEDS = [20260921, 20260922, 20260923]


def run(collection, output):
    output.mkdir(parents=True, exist_ok=False)
    parent_path = FROZEN / 'report.json'
    assert hash_file(parent_path) == '17b2676d6fc42609c0bd176a113b430e2439ffe1bf428aee6d580e15bbc71fc9'
    parent = json.loads(parent_path.read_text())
    frozen_hashes = {item['file']: item['sha256'] for item in parent['artifacts']}
    report = json.loads((collection / 'report.json').read_text())
    counts, diagnostics, reproductions, support_effects, independence = [], [], [], [], []
    shared_H1 = None
    consumed = {}
    for context in report['contexts']:
        folder = collection / context['context_id']
        for name, digest in context['artifacts'].items():
            assert hash_file(folder / name) == digest, 'collection artifact changed'
        cells = pd.read_parquet(folder / 'cell-splits.parquet')
        tasks = pd.read_parquet(folder / 'tasks.parquet')
        support = pd.read_parquet(folder / 'task-support.parquet')
        counts.append(support)
        diagnostics.append(pd.read_parquet(folder / 'diagnostics.parquet'))
        reproductions.append(pd.read_parquet(folder / 'source-reproduction.parquet'))
        assert not cells.record_id.duplicated().any()
        assert cells.loc[cells.is_NTC, 'task_uid'].isna().all()
        selected = cells.is_NTC | cells.task_uid.notna()
        for seed in SEEDS:
            labels = cells[f'half_{seed}']
            assert labels[selected].isin([0, 1]).all()
            assert not set(cells.loc[selected & labels.eq(0), 'record_id']) & set(cells.loc[selected & labels.eq(1), 'record_id'])
            for _, group in cells.loc[cells.is_NTC & cells.split_NTC_support_eligible].groupby('source_batch'):
                assert group[f'half_{seed}'].value_counts().reindex([0, 1], fill_value=0).min() >= 2
        if context['baseline_id'] == 'H1_shared_exact_NTC':
            part = cells.loc[cells.is_NTC, ['source_barcode'] + [f'half_{seed}' for seed in SEEDS]].sort_values('source_barcode').reset_index(drop=True)
            if shared_H1 is None:
                shared_H1 = part
            else:
                pd.testing.assert_frame_equal(part, shared_H1)
        independence.append({'context_id': context['context_id'], 'panel_id': context['panel_id'],
                             'cells': len(cells), 'NTC_cells': int(cells.is_NTC.sum()),
                             'target_NTC_overlap': 0, 'split_cell_overlap_per_seed': 0,
                             'shared_H1_assignment_verified': context['baseline_id'] == 'H1_shared_exact_NTC'})
        old_name = 'endpoint/' + context['context_id'] + '/baseline-by-source-stratum.npz'
        old_path = FROZEN / old_name
        assert hash_file(old_path) == frozen_hashes[old_name]
        consumed[old_name] = frozen_hashes[old_name]
        original = np.load(old_path)
        cm = original['mean_logCP10K']
        lookup = {b: i for i, b in enumerate(original['source_batch'])}
        ns = support.pivot(index='task_uid', columns='variant', values='n_target')
        cell_groups = {key: indices for key, indices in cells.loc[cells.task_uid.notna() & cells.split_NTC_support_eligible].groupby('task_uid').groups.items()}
        with h5py.File(folder / 'moments.h5') as hf:
            assert np.array_equal(hf['source_gene'].asstr()[:], original['source_gene'])
            symbols = hf['safe_symbol'].asstr()[:]
            for ti, task in enumerate(tasks.to_dict('records')):
                uid = task['task_uid']
                ids = cell_groups.get(uid, [])
                if not len(ids):
                    support_effects.append({'task_uid': uid, 'status': 'not_estimable', 'reason': 'no_NTC_4_supported_target_cells'})
                    continue
                batch_counts = cells.loc[ids, 'source_batch'].value_counts()
                original_supported_control = (batch_counts.to_numpy() / batch_counts.sum()) @ cm[[lookup[b] for b in batch_counts.index]]
                reconstructed = []
                for si, seed in enumerate(SEEDS):
                    a, b = 1 + 2 * si, 2 + 2 * si
                    na, nb = ns.loc[uid, f's{seed}h0'], ns.loc[uid, f's{seed}h1']
                    terms = []
                    if na: terms.append(na * hf['target'][ti, a].astype(float))
                    if nb: terms.append(nb * hf['target'][ti, b].astype(float))
                    reconstructed.append(sum(terms) / (na + nb))
                maximum = max(float(np.max(np.abs(a - reconstructed[0]))) for a in reconstructed)
                assert maximum < 2e-6, 'disjoint halves do not reconstruct fixed supported target mean'
                valid = (symbols != '') & ~hf['target_excluded'][ti]
                supported = reconstructed[0] - original_supported_control
                full = hf['target'][ti, 0].astype(float) - hf['matched_control'][ti, 0].astype(float)
                support_effects.append({'task_uid': uid, 'status': 'completed', 'supported_target_cells': len(ids),
                                       'original_target_cells': int(ns.loc[uid, 'full']),
                                       'maximum_cross_seed_target_reconstruction_error': maximum,
                                       **compare_vectors(supported[valid], full[valid])})
        print(json.dumps({'audit_context': context['panel_id'], 'tasks': len(tasks)}), flush=True)
    counts = pd.concat(counts, ignore_index=True)
    diagnostics = pd.concat(diagnostics, ignore_index=True)
    reproduction = pd.concat(reproductions, ignore_index=True)
    assert counts.task_uid.nunique() == len(reproduction) == 37845
    assert int(reproduction.maximum_absolute_error.notna().sum()) == 37633
    assert not reproduction.task_uid.duplicated().any()
    for name, frame in [('all-task-support', counts), ('all-task-diagnostics', diagnostics),
                        ('all-source-reproduction', reproduction), ('exact-support-effects', pd.DataFrame(support_effects)),
                        ('independence', pd.DataFrame(independence))]:
        frame.to_parquet(output / (name + '.parquet'), index=False, compression='zstd')
    write_json(output / 'report.json', {'status': 'completed', 'registered_tasks': len(reproduction),
                                      'original_means_reproduced': int(reproduction.maximum_absolute_error.notna().sum()),
                                      'maximum_original_mean_error': float(reproduction.maximum_absolute_error.max()),
                                      'all_seeds_target_and_NTC_disjoint': True,
                                      'H1_shared_NTC_same_assignments': True,
                                      'additional_frozen_inputs': consumed,
                                      'all_collection_artifact_hashes_verified': True})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--collection', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    run(a.collection, a.output)
