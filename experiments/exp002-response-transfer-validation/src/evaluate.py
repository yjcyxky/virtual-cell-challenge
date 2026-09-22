"""Retrospective leave-cell-line-out experiments on fixed native gene axes."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import subprocess
import sys
import time
import warnings

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'scripts/dossier'))
import h5py
import numpy as np
import pandas as pd
from profile_responses import write_json
from rna import hash_file
from estimators import classify, response_mean, choose_hyperparameters, ntc_distances, score, TEMPERATURES

SEEDS = [20260921, 20260922, 20260923]
TOLERANCES = [.05, .1, .2]
SCALES = ['control_only', 'source_matched']
REFERENCE = ROOT / 'data/assessments/jiang-dossier-20260919-v2/references/gene_sets.json'


def context_groups(reports):
    groups = defaultdict(list)
    other = []
    for r in reports:
        pid, meta = r['panel_id'], r['source_metadata']
        if pid.startswith('Jiang__'):
            group = 'Jiang_' + meta['stimulation']
        elif pid.startswith('GxE2:'):
            group = 'GxE2_' + meta['drug'] + '_' + str(meta['dose_uM'])
        elif pid.startswith(('H1:', 'replogle:', 'nadig:')):
            group = 'cross_study_sensitivity'
        else:
            other.append({'context_id': r['context_id'], 'panel_id': pid,
                          'reason': 'outside_registered_multi_cell_line_prediction_groups',
                          'geometry_and_cell_split_diagnostics_included': True})
            continue
        groups[group].append(r)
    return dict(groups), other


def load_group(collection, reports, supplemental=False):
    mappings, task_tables = [], []
    for r in reports:
        folder = collection / r['context_id']
        with h5py.File(folder / 'moments.h5') as hf:
            symbols = hf['safe_symbol'].asstr()[:]
            mappings.append({g: i for i, g in enumerate(symbols) if g})
        task_tables.append(pd.read_parquet(folder / 'tasks.parquet'))
    genes = sorted(set.intersection(*(set(m) for m in mappings)))
    targets = sorted(set.union(*(set(t.canonical_target) for t in task_tables)))
    target_index = {p: i for i, p in enumerate(targets)}
    # Main groups have one source panel per biological background. Supplementary
    # cell-line fold identity groups K562 libraries and all H1 source splits.
    lines = sorted({r['source_metadata']['cell_line'] for r in reports})
    line_index = {line: i for i, line in enumerate(lines)}
    variants = 1 if supplemental else 7
    shape = (len(lines), len(targets), variants, len(genes))
    arrays = {name: np.full(shape, np.nan, np.float32) for name in
              ['target', 'matched_control', 'target_variance_mean', 'matched_control_variance_mean']}
    control = np.full((len(lines), variants, len(genes)), np.nan, np.float32)
    control_var = np.full_like(control, np.nan)
    available = np.zeros(shape[:3], bool)
    excluded = np.zeros((len(targets), len(genes)), bool)
    # Aggregate equal source panels, and within source equal constructs. Source
    # standard errors are not combined as independent biological replication.
    per_line = defaultdict(list)
    task_ledger = []
    for r, mapping, tasks in zip(reports, mappings, task_tables):
        folder = collection / r['context_id']
        indices = np.array([mapping[g] for g in genes])
        support = pd.read_parquet(folder / 'task-support.parquet')
        records = support.pivot(index='task_uid', columns='variant', values='status')
        order = ['full'] + [f's{seed}h{half}' for seed in SEEDS for half in range(2)]
        local = {key: np.full((len(targets), variants, len(genes)), np.nan, np.float32) for key in arrays}
        local_available = np.zeros((len(targets), variants), bool)
        with h5py.File(folder / 'moments.h5') as hf:
            for p, group in tasks.groupby('canonical_target', sort=True):
                pi = target_index[p]
                row_ids = group.index.to_numpy()
                for row in row_ids:
                    excluded[pi] |= hf['target_excluded'][row][indices]
                for v in range(variants):
                    keep = [row for row in row_ids if records.loc[tasks.iloc[row].task_uid, order[v]] == 'completed']
                    if not keep:
                        continue
                    local_available[pi, v] = True
                    for key in arrays:
                        values = np.array([hf[key][row, v][indices] for row in keep])
                        local[key][pi, v] = values.mean(0)
                        if supplemental and 'variance' in key and len(keep) > 1:
                            local[key][pi, v] = np.nan  # no invented construct independence
                for row in row_ids:
                    task_ledger.append({'task_uid': tasks.iloc[row].task_uid, 'canonical_target': p,
                                        'panel_id': r['panel_id'], 'context_id': r['context_id'],
                                        'cell_line': r['source_metadata']['cell_line'],
                                        'full_estimable': bool(records.loc[tasks.iloc[row].task_uid, 'full'] == 'completed'),
                                        'aggregation': 'equal_construct_within_panel_then_equal_panel_within_cell_line'})
            baseline = np.load(folder / 'control-moments.npz')
            per_line[r['source_metadata']['cell_line']].append(
                (local, local_available, baseline['mean'][:variants, indices], baseline['variance_mean'][:variants, indices]))
    for line, entries in per_line.items():
        ci = line_index[line]
        for key in arrays:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                arrays[key][ci] = np.nanmean([item[0][key] for item in entries], axis=0)
        available[ci] = np.any([item[1] for item in entries], axis=0)
        control[ci] = np.mean([item[2] for item in entries], axis=0)
        control_var[ci] = entries[0][3] if len(entries) == 1 else np.nan
        if line == 'H1':
            # Every source split contains exact copies of the same NTC. Consensus
            # safe native mapping already removed any cross-split conflicting ID.
            assert all(np.allclose(item[2], entries[0][2], atol=2e-6) for item in entries)
            control[ci], control_var[ci] = entries[0][2], entries[0][3]
    for pi, p in enumerate(targets):
        excluded[pi] |= np.array(genes) == p
    return {**arrays, 'control': control, 'control_var': control_var, 'available': available,
            'excluded': excluded, 'genes': genes, 'targets': targets, 'lines': lines,
            'task_ledger': task_ledger}


def responses(data, scale):
    if scale == 'control_only':
        baseline = np.broadcast_to(data['control'][:, None], data['target'].shape)
        variance = np.broadcast_to(data['control_var'][:, None], data['target'].shape)
    else:
        baseline, variance = data['matched_control'], data['matched_control_variance_mean']
    return baseline, variance, data['target'] - baseline, data['target_variance_mean'] + variance


def confusion(a, b, valid, categories=5):
    result = np.bincount((categories * a[valid] + b[valid]).astype(int), minlength=categories ** 2).reshape(categories, categories)
    n = int(result.sum())
    agreement = np.trace(result) / n if n else None
    expected = (result.sum(0) @ result.sum(1)) / n ** 2 if n else None
    return {'matrix': result.tolist(), 'genes': n, 'agreement_including_unknown': agreement,
            'known_in_both': int(result[1:, 1:].sum()),
            'known_only_agreement': float(np.trace(result[1:, 1:]) / result[1:, 1:].sum()) if result[1:, 1:].sum() else None,
            'kappa': float((agreement - expected) / (1 - expected)) if n and expected < 1 else None}


def quadrant_experiment(data, output):
    rows, concordance = [], []
    shape = (len(data['targets']), 2, 7, 3, len(data['genes']))
    with h5py.File(output / 'classifications.h5', 'w') as hf:
        for name in ['baseline_state', 'response_state', 'quadrant', 'activity', 'conserved_nonzero']:
            hf.create_dataset(name, shape=shape, dtype='uint8', chunks=(1, 1, 1, 1, min(shape[-1], 8192)), compression='gzip', compression_opts=1)
        for si, scale in enumerate(SCALES):
            baseline, bv, delta, dv = responses(data, scale)
            for pi, p in enumerate(data['targets']):
                context_set = np.flatnonzero(data['available'][:, pi, 0])
                valid = ~data['excluded'][pi]
                labels = {}
                for v in range(7):
                    for di, tolerance in enumerate(TOLERANCES):
                        classification = classify(baseline[context_set, pi, v], bv[context_set, pi, v],
                                                  delta[context_set, pi, v], dv[context_set, pi, v], tolerance)
                        if len(context_set) < 2 or not data['available'][context_set, pi, v].all():
                            for name in ['baseline_state', 'response_state', 'quadrant', 'activity', 'conserved_nonzero']:
                                classification[name][:] = 0
                        for name in ['baseline_state', 'response_state', 'quadrant', 'activity', 'conserved_nonzero']:
                            classification[name][~valid] = 0
                            hf[name][pi, si, v, di] = classification[name]
                        labels[v, di] = classification['quadrant']
                        q = classification['quadrant'][valid]
                        all_observed_zero = ((baseline[context_set, pi, v] == 0) &
                                             (data['target'][context_set, pi, v] == 0)).all(0) if len(context_set) else np.zeros(len(valid), bool)
                        rows.append({'canonical_target': p, 'scale': scale, 'variant': v, 'tolerance': tolerance,
                                     'context_set': json.dumps([data['lines'][i] for i in context_set]),
                                     'contexts': len(context_set), 'genes': int(valid.sum()),
                                     'all_contexts_supported': bool(len(context_set) >= 2 and data['available'][context_set, pi, v].all()),
                                     **{f'quadrant_{k}': int((q == k).sum()) for k in range(5)},
                                     'near_zero': int((classification['activity'][valid] == 1).sum()),
                                     'near_zero_with_all_observed_means_zero': int(((classification['activity'] == 1) & all_observed_zero & valid).sum()),
                                     'active_anywhere': int((classification['activity'][valid] == 2).sum()),
                                     'conserved_nonzero': int(classification['conserved_nonzero'][valid].sum())})
                for seed_index, seed in enumerate(SEEDS):
                    for di, tolerance in enumerate(TOLERANCES):
                        concordance.append({'canonical_target': p, 'scale': scale, 'seed': seed, 'tolerance': tolerance,
                                            'both_halves_contexts_supported': bool(len(context_set) >= 2 and
                                                data['available'][context_set, pi, 1 + 2 * seed_index:3 + 2 * seed_index].all()),
                                            **confusion(labels[1 + 2 * seed_index, di], labels[2 + 2 * seed_index, di], valid)})
    pd.DataFrame(rows).to_parquet(output / 'classification-counts.parquet', index=False)
    pd.DataFrame(concordance).to_parquet(output / 'split-classification-agreement.parquet', index=False)


def module_sets(genes):
    references = json.loads(REFERENCE.read_text())['states']
    lookup = {g: i for i, g in enumerate(genes)}
    rng = np.random.default_rng(20260921)
    modules = []
    for name, ref in references.items():
        indices = [lookup[g] for g in ref['genes'] if g in lookup]
        eligible = len(indices) >= 5 and len(indices) / len(ref['genes']) >= .3
        modules.append({'name': name, 'parent': name, 'kind': 'reference', 'indices': indices,
                        'reference_members': len(ref['genes']), 'eligible': eligible})
        if eligible:
            for repeat in range(20):
                modules.append({'name': f'{name}__random_{repeat}', 'parent': name, 'kind': 'random_size_matched',
                                'indices': rng.choice(len(genes), min(len(genes), len(indices) + 16), replace=False).tolist(),
                                'use_prefix_after_common_gene_exclusions': True,
                                'reference_members': len(ref['genes']), 'eligible': True})
    return modules


def prediction_experiment(data, output, supplemental=False):
    metrics, exclusions, hyperparameters, region_metrics, module_metrics = [], [], [], [], []
    modules = module_sets(data['genes'])
    write_json(output / 'module-membership.json', modules)
    distance = ntc_distances(data['control'][:, 0].astype(float))
    with h5py.File(output / 'predictions.h5', 'w') as hf:
        for scale in SCALES:
            baseline, bv, delta, dv = responses(data, scale)
            response = delta[:, :, 0].astype(float)
            response[:, data['excluded']] = np.nan
            available = data['available'][:, :, 0]
            for held, line in enumerate(data['lines']):
                train = np.arange(len(data['lines'])) != held
                train_lines = [c for c in data['lines'] if c != line]
                assert line not in train_lines
                eligible = available[held] & (available[train].sum(0) >= 2)
                for pi, p in enumerate(data['targets']):
                    exclusions.append({'scale': scale, 'held_cell_line': line, 'canonical_target': p,
                                       'test_available': bool(available[held, pi]), 'training_contexts': int(available[train, pi].sum()),
                                       'eligible': bool(eligible[pi]), 'reason': None if eligible[pi] else
                                       ('test_not_estimable' if not available[held, pi] else 'fewer_than_2_training_backgrounds')})
                if not eligible.any() or len(data['genes']) < 100:
                    continue
                lam, temp, tuning = choose_hyperparameters(response[train], available[train], data['control'][train, 0])
                hyperparameters.append({'scale': scale, 'held_cell_line': line, 'training_cell_lines': train_lines,
                                        'lambda': lam, 'temperature': str(temp), 'details': tuning,
                                        'test_response_used': False, 'test_target_count_used': False})
                shared = response_mean(response[train], available[train])
                weighted = response_mean(response[train], available[train], distance[held, train], .1)
                models = {'zero': np.zeros_like(shared), 'shared': shared, 'shared_shrunk': lam * shared,
                          'NTC_weighted_fixed_0.1': weighted,
                          'NTC_weighted_tuned': response_mean(response[train], available[train], distance[held, train], temp)}
                for t in TEMPERATURES:
                    models['NTC_temperature_' + str(t)] = response_mean(response[train], available[train], distance[held, train], t)
                # Perturbation-agnostic null: equal target means within each training
                # background, then equal backgrounds. Query target omitted.
                agnostic = np.full_like(shared, np.nan)
                wrong = np.full_like(shared, np.nan)
                training_sum = np.nansum(response[train], axis=1)
                training_count = np.isfinite(response[train]).sum(axis=1)
                donors = np.flatnonzero(available[train].sum(0) >= 2)
                donor_order = np.random.default_rng(20260921).permutation(donors)
                donor_mapping = dict(zip(donor_order, np.roll(donor_order, -1)))
                wrong_donor = {}
                for pi in np.flatnonzero(eligible):
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore', RuntimeWarning)
                        denominator = training_count - np.isfinite(response[train, pi])
                        numerator = training_sum - np.nan_to_num(response[train, pi])
                        background_mean = np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 0)
                        agnostic[pi] = np.nanmean(background_mean, axis=0)
                    donor = int(donor_mapping[pi])
                    if donor == pi:
                        raise ValueError('no independent wrong-target donor')
                    wrong[pi] = shared[donor]
                    wrong_donor[pi] = data['targets'][donor]
                models['perturbation_agnostic'], models['wrong_target'] = agnostic, wrong
                gate = lam * shared
                classification_by_target = {}
                for pi in np.flatnonzero(eligible):
                    training_contexts = np.flatnonzero(train & available[:, pi])
                    labels = classify(baseline[training_contexts, pi, 0], bv[training_contexts, pi, 0],
                                      delta[training_contexts, pi, 0], dv[training_contexts, pi, 0], .1)
                    if supplemental:
                        # Aggregate constructs/sources do not supply independent
                        # variance; this sensitivity does not claim quadrant labels.
                        for key in ['quadrant', 'conserved_nonzero', 'activity']:
                            labels[key][:] = 0
                    for key in ['baseline_state', 'response_state', 'quadrant', 'conserved_nonzero', 'activity']:
                        labels[key][data['excluded'][pi]] = 0
                    shared_region = np.isin(labels['quadrant'], [1, 3]) & labels['conserved_nonzero']
                    context_region = np.isin(labels['quadrant'], [2, 4]) & (labels['activity'] == 2)
                    gate[pi, shared_region] = shared[pi, shared_region]
                    gate[pi, context_region] = weighted[pi, context_region]
                    gate[pi, labels['activity'] == 1] = 0
                    classification_by_target[pi] = labels
                models['quadrant_gated'] = gate
                root = hf.create_group(scale + '/' + line)
                target_ids = np.flatnonzero(eligible)
                root.create_dataset('target_index', data=target_ids)
                root.create_dataset('truth', data=response[held, eligible].astype('float32'), compression='gzip', compression_opts=1)
                root.create_dataset('train_quadrant', data=np.array([classification_by_target[pi]['quadrant'] for pi in target_ids], dtype='uint8'), compression='gzip', compression_opts=1)
                root.create_dataset('train_conserved_nonzero', data=np.array([classification_by_target[pi]['conserved_nonzero'] for pi in target_ids], dtype='uint8'), compression='gzip', compression_opts=1)
                for model, prediction in models.items():
                    # Wrong-target's own intervention readout is absent; never let
                    # that silently improve one model by changing evaluation genes.
                    root.create_dataset(model, data=prediction[eligible].astype('float32'), compression='gzip', compression_opts=1)
                for pi in target_ids:
                    all_finite = np.isfinite(response[held, pi])
                    for prediction in models.values():
                        all_finite &= np.isfinite(prediction[pi])
                    if all_finite.sum() < 100:
                        continue
                    labels = classification_by_target[pi]
                    base = {'scale': scale, 'held_cell_line': line, 'canonical_target': data['targets'][pi],
                            'training_contexts': int(available[train, pi].sum()), 'wrong_target_donor': wrong_donor[pi],
                            'all_6_test_halves_supported': bool(data['available'][held, pi].all()),
                            'all_training_6_halves_supported': bool(data['available'][train & available[:, pi], pi].all()),
                            'test_used_for_model_selection': False}
                    truth = response[held, pi]
                    actual_reference_size = {m['name']: int(all_finite[np.array(m['indices'], dtype=int)].sum())
                                             for m in modules if m['kind'] == 'reference'}
                    for model, prediction in models.items():
                        metrics.append({**base, 'model': model, **score(truth[all_finite], prediction[pi, all_finite])})
                        if model in ['zero', 'shared', 'shared_shrunk', 'NTC_weighted_fixed_0.1', 'quadrant_gated'] and not supplemental:
                            regions = {f'quadrant_{q}': labels['quadrant'] == q for q in range(5)}
                            regions['conserved_nonzero'] = labels['conserved_nonzero']
                            regions['near_zero'] = labels['activity'] == 1
                            for region, mask in regions.items():
                                mask = mask & all_finite
                                if mask.any():
                                    region_metrics.append({**base, 'model': model, 'training_region': region,
                                                           **score(truth[mask], prediction[pi, mask])})
                        if model not in ['zero', 'shared', 'shared_shrunk', 'NTC_weighted_fixed_0.1', 'wrong_target', 'perturbation_agnostic']:
                            continue
                        for module in modules:
                            if not module['eligible']:
                                continue
                            ids = np.array(module['indices'])
                            ids = ids[all_finite[ids]]
                            if module['kind'] == 'random_size_matched':
                                required_size = actual_reference_size[module['parent']]
                                if len(ids) < required_size:
                                    raise ValueError('random module candidate pool cannot retain size matching')
                                ids = ids[:required_size]
                            if len(ids) < 5 or len(ids) / module['reference_members'] < .3:
                                continue
                            observed = float(truth[ids].mean())
                            predicted = float(prediction[pi, ids].mean())
                            module_metrics.append({**base, 'model': model, 'module': module['name'],
                                                   'reference_parent': module['parent'], 'kind': module['kind'],
                                                   'genes': len(ids), 'truth': observed, 'prediction': predicted,
                                                   'squared_error': (observed - predicted) ** 2, 'zero_squared_error': observed ** 2})
                print(json.dumps({'fold': line, 'scale': scale, 'targets': int(eligible.sum()), 'lambda': lam}), flush=True)
    for name, rows in [('metrics', metrics), ('fold-eligibility', exclusions), ('region-metrics', region_metrics), ('module-metrics', module_metrics)]:
        pd.DataFrame(rows).to_parquet(output / (name + '.parquet'), index=False, compression='zstd')
    write_json(output / 'hyperparameters.json', hyperparameters)


def run(collection, output, only=None):
    output.mkdir(parents=True, exist_ok=True)
    if (collection / 'report.json').exists():
        report = json.loads((collection / 'report.json').read_text())
        reports = report['contexts']
    else:
        reports = [json.loads(p.read_text()) for p in collection.glob('*/report.json')]
    groups, other = context_groups(reports)
    write_json(output / 'excluded-contexts.json', other)
    identity = {'code_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                'code_sha256': {name: hash_file(Path(__file__).with_name(name)) for name in ['evaluate.py', 'estimators.py']},
                'reference_sha256': hash_file(REFERENCE), 'collection_identity_sha256': hash_file(collection / 'identity.json'),
                'tolerances': TOLERANCES, 'seeds': SEEDS, 'scales': SCALES}
    for group, members in sorted(groups.items()):
        if only and group != only:
            continue
        supplemental = group == 'cross_study_sensitivity'
        expected = 8 if supplemental else (6 if group.startswith('Jiang') else 3)
        assert len(members) == expected, f'incomplete registered group {group}'
        folder = output / group
        if (folder / 'report.json').exists():
            old = json.loads((folder / 'report.json').read_text())
            assert old['identity']['code_sha256'] == identity['code_sha256']
            continue
        folder.mkdir(parents=True, exist_ok=False)
        started = time.monotonic()
        data = load_group(collection, members, supplemental)
        write_json(folder / 'axes.json', {k: data[k] for k in ['genes', 'targets', 'lines']})
        pd.DataFrame(data['task_ledger']).to_parquet(folder / 'task-ledger.parquet', index=False)
        if not supplemental:
            quadrant_experiment(data, folder)
        prediction_experiment(data, folder, supplemental)
        summary = {'status': 'completed', 'group': group, 'identity': identity, 'genes': len(data['genes']),
                   'targets': len(data['targets']), 'cell_lines': data['lines'], 'contexts': members,
                   'seconds': time.monotonic() - started, 'supplemental': supplemental}
        summary['artifacts'] = {p.name: hash_file(p) for p in folder.iterdir() if p.is_file()}
        write_json(folder / 'report.json', summary)
        print(json.dumps({k: v for k, v in summary.items() if k not in ['contexts', 'artifacts', 'identity']}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--collection', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--only')
    args = p.parse_args()
    run(args.collection.resolve(), args.output.resolve(), args.only)
