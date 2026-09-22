"""Issue #29: verified genetic-only cohorts, immutable statistics reuse, new folds."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import h5py
import numpy as np
import pandas as pd

from collect import collect_context, SEEDS, SOURCE
from evaluate import load_group, quadrant_experiment, prediction_experiment, responses, REFERENCE
from heterogeneity import compare_vectors
from heterogeneity_cells import CellContexts
from profile_responses import write_json
from rna import hash_file, value_hash

ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = Path(__file__).resolve().parents[1]
PRIOR = EXPERIMENT / 'outputs/20260922-b/collection'
SC = ROOT / 'data/assessments/scperturb-dossier-20260919'
ISSUE = 'https://github.com/yjcyxky/virtual-cell-challenge/issues/29'
CORE = ['H1:Training', 'H1:Validation', 'H1:Test', 'replogle:K562_essential',
        'replogle:K562_gwps', 'replogle:rpe1', 'nadig:hepg2', 'nadig:jurkat']
CAS9 = ['DixitRegev2016_K562_TFs_7_days', 'DixitRegev2016_K562_TFs_13_days', 'FrangiehIzar2021_RNA']


def guide_identity(source, guide, declared, approved):
    """Parse every literal guide; never trust a collapsed single-gene display label."""
    if pd.isna(guide):
        return None, 'unassigned'
    tokens = re.split(r'[;,|]', str(guide))
    dixit = source.startswith('Dixit')
    control = r'p_INTERGENIC[0-9]+' if dixit else r'ONE_NON-GENE_SITE_[0-9]+'
    if all(re.fullmatch(control, t) for t in tokens):
        return None, 'intergenic_control'
    if not dixit and all(re.fullmatch(r'NO_SITE_[0-9]+', t) for t in tokens):
        return None, 'non_targeting_control_not_used'
    pattern = r'p_sg(.+)_[0-9]+' if dixit else r'(.+)_[0-9]+'
    matches = [re.fullmatch(pattern, t) for t in tokens]
    if not all(matches):
        return None, 'unresolved_or_mixed_guide_roles'
    genes = {m.group(1) for m in matches}
    if len(genes) != 1:
        return None, 'multiple_targets_or_mixed_roles'
    gene = next(iter(genes))
    if gene not in approved:
        return None, 'not_exact_approved_HGNC_symbol'
    if str(declared) != gene:
        return None, 'guide_source_target_disagreement'
    return gene, 'single_gene_target'


def group_specs():
    h1 = CORE[:3]
    common = h1 + ['replogle:rpe1', 'nadig:hepg2', 'nadig:jurkat']
    c = ['scPerturb:' + s for s in CAS9]
    return {
        'CRISPRi_cross_study_K562_day6': (common + [CORE[3]], 2, False),
        'CRISPRi_cross_study_K562_day8': (common + [CORE[4]], 2, False),
        'CRISPRi_Replogle_day6_RPE1': ([CORE[3], CORE[5]], 1, False),
        'CRISPRi_Replogle_day8_RPE1': ([CORE[4], CORE[5]], 1, False),
        'CRISPRi_Nadig_HepG2_Jurkat': (CORE[6:8], 1, False),
        'CRISPRi_K562_time_library': (CORE[3:5], 1, True),
        'Cas9_cross_study_Dixit_day7': ([c[0], c[2]], 1, False),
        'Cas9_cross_study_Dixit_day13': ([c[1], c[2]], 1, False),
        'Cas9_K562_time': (c[:2], 1, True),
    }


def cas9_context(provider, source, audit_out):
    pid = 'scPerturb:' + source
    cells = provider.frozen.parquet(SC, f'cells/{source}/cells.parquet')
    design = provider.frozen.parquet(SC, f'design/{source}/row-applicability.parquet')
    assert np.array_equal(cells.row_index, design.row_index)
    source_audit = provider.frozen.json(SC, f'source-audit/{source}.h5ad.json')
    assert source_audit['intervention_source_table'] == ['CRISPR-cas9']
    hgnc_path = ROOT / 'data/raw/networks/hgnc_complete_set.txt'
    provider.verified_path(hgnc_path, hash_file(hgnc_path))
    hgnc = pd.read_csv(hgnc_path, sep='\t', low_memory=False)
    approved = set(hgnc.loc[hgnc.status.eq('Approved'), 'symbol'])
    declared = cells['source_obs__target' if source.startswith('Dixit') else 'source_obs__perturbation']
    assignments = [guide_identity(source, g, t, approved) for g, t in zip(cells.source_obs__guide_id, declared)]
    targets = pd.Series([a[0] for a in assignments], index=cells.index)
    roles = pd.Series([a[1] for a in assignments], index=cells.index)
    condition = (cells.source_obs__perturbation_2.eq('Control') & design.biological_background_index.eq(1)
                 if source.startswith('Frangieh') else design.biological_background_index.eq(0))
    positive = cells.computed_numeric_valid & cells.computed_total_expression.gt(0)
    assert design.loc[roles.eq('intergenic_control'), 'control_eligible'].all()
    included = condition & positive & roles.isin(['single_gene_target', 'intergenic_control'])
    reason = roles.where(condition, 'excluded_extra_stimulation_or_coculture').where(positive, 'invalid_or_empty_counts')
    ledger = cells[['record_id', 'input_sha256', 'row_index', 'source_barcode', 'source_obs__guide_id']].copy()
    ledger['declared_target'] = declared
    ledger['canonical_target'], ledger['role'], ledger['included'], ledger['reason'] = targets, roles, included, reason
    ledger['source_condition'] = cells.source_obs__perturbation_2 if source.startswith('Frangieh') else 'basal'
    ledger.to_parquet(audit_out / (source + '-scope.parquet'), index=False)
    part = cells.loc[included].copy().reset_index(drop=True)
    d = design.loc[included].reset_index(drop=True)
    part['source_batch'] = d.response_background  # no invented replicate or barcode-derived batch
    target = targets.loc[included].reset_index(drop=True)
    control = roles.loc[included].eq('intergenic_control').to_numpy()
    rows = [{'task_uid': value_hash([pid, 'pure_genetic', g]), 'panel_id': pid,
             'canonical_target': g, 'source_task': g, 'effect_status': 'new_estimand',
             'source_folder': str(SC.relative_to(ROOT)), 'source_gene_result_file': None,
             'source_gene_result_sha256': None, 'aggregation': 'equal_cells_same_gene_within_source_condition'}
            for g in sorted(target.dropna().unique())]
    tasks = pd.DataFrame(rows)
    labels = target.map(dict(zip(tasks.canonical_target, tasks.task_uid)))
    matrix, genes, barcodes = provider.raw_matrix(source_audit['path'], source_audit['input_sha256'])
    log = provider.normalized_selection(matrix, part.row_index, part, barcodes)
    context = provider.make(pid, 'pure_genetic', part, log, genes, tasks, control, labels)
    context['source_metadata'] = {**context['source_metadata'], 'intervention_mechanism': 'Cas9',
        'cell_line': 'K562' if source.startswith('Dixit') else 'Frangieh_melanoma_source_model',
        'model_identity_resolution': 'source_reported_K562' if source.startswith('Dixit') else 'patient_derived_melanoma_screen_not_independent_line_inference',
        'condition': 'basal' if source.startswith('Dixit') else 'Control',
        'control_role': 'intergenic_cutting_control', 'additional_experimental_treatment': False,
        'days_post_transduction': 7 if '_7_days' in source else 13 if '_13_days' in source else None,
        'batch_resolution': 'source_response_background_only_no_independent_replicate_identified',
        'target_aggregation': 'equal_cells_same_gene', 'cell_type_inference_used': False}
    context['mapping'] = context['mapping'].reset_index(drop=True)
    assert context['mapping'].source_gene.astype(str).tolist() == genes
    write_json(audit_out / (source + '-scope.json'), {'source': source, 'source_audit': source_audit,
        'roles': roles.value_counts().to_dict(), 'reasons': reason.value_counts().to_dict(),
        'included_records': int(included.sum()), 'targets': len(tasks), 'control_cells': int(control.sum()),
        'selected_condition': context['source_metadata']['condition'], 'original_numeric_and_barcode_identity_verified': True})
    return context


def collect(output):
    folder = output / 'collection'
    if (folder / 'report.json').exists():
        return
    folder.mkdir(exist_ok=True)
    scope = output / 'scope'
    scope.mkdir(exist_ok=True)
    prior = json.loads((PRIOR / 'report.json').read_text())
    contexts, references = [], []
    for old in prior['contexts']:
        if old['panel_id'] not in CORE:
            continue
        source = PRIOR / old['context_id']
        dest = folder / old['context_id']
        dest.mkdir(exist_ok=True)
        for name, digest in old['artifacts'].items():
            assert hash_file(source / name) == digest, 'immutable prior artifact mismatch'
            if not (dest / name).exists():
                os.link(source / name, dest / name)
        r = deepcopy(old)
        r['source_metadata'].update(intervention_mechanism='CRISPRi', additional_experimental_treatment=False,
                                    control_role='non_targeting', scope_source='verified_core_source_panel')
        write_json(dest / 'report.json', r)
        contexts.append(r)
        references.append({'panel_id': old['panel_id'], 'context_id': old['context_id'],
                           'path': str(source.relative_to(ROOT)), 'report_sha256': hash_file(source / 'report.json'),
                           'artifacts': old['artifacts']})
        print(json.dumps({'verified_reused_panel': old['panel_id']}), flush=True)
    assert len(contexts) == 8 and sum(r['tasks'] for r in contexts) == 20790
    provider = CellContexts(SOURCE)
    excluded = []
    for pid, p in provider.panels.items():
        if pid in CORE or pid in ['scPerturb:' + s for s in CAS9]:
            continue
        reason = 'outside_explicitly_verified_genetic_only_cohort'
        if p.get('confirmed_collection_copy'):
            reason = 'confirmed_collection_duplicate'
        elif pid.startswith(('Jiang', 'GxE', 'chemical')):
            reason = 'drug_or_additional_stimulation_source_excluded'
        elif 'Datlinger' in pid:
            reason = 'serum_starvation_2017_or_unresolved_unstimulated_protocol_2021'
        elif 'Papalexi' in pid:
            reason = 'untreated_arm_no_supported_matched_control'
        excluded.append({'panel_id': pid, 'reason': reason, 'source_metadata': p['source_metadata']})
    write_json(scope / 'excluded-source-panels.json', excluded)
    for name in CAS9:
        context = cas9_context(provider, name, scope)
        dest = folder / context['context_id']
        r = json.loads((dest / 'report.json').read_text()) if (dest / 'report.json').exists() else collect_context(context, dest)
        contexts.append(r)
        print(json.dumps({'collected_Cas9_panel': r['panel_id'], 'tasks': r['tasks']}), flush=True)
    identity = {'issue': ISSUE, 'prior_run': '20260922-b', 'prior_report_sha256': hash_file(PRIOR / 'report.json'),
                'prior_release': 'https://github.com/yjcyxky/virtual-cell-challenge/releases/tag/experiment-response-transfer-20260922',
                'reused_artifacts': references, 'new_cas9_inputs': provider.verify_unchanged(),
                'seeds': SEEDS, 'control_semantics': 'NTC only for CRISPRi; intergenic cutting for selected Cas9',
                'original_input_mutations': 0}
    write_json(folder / 'identity.json', identity)
    write_json(folder / 'report.json', {'status': 'completed', 'identity': identity, 'contexts': contexts,
                                      'tasks': sum(r['tasks'] for r in contexts)})


def audit_collection(output):
    folder = output / 'audit'
    folder.mkdir(exist_ok=True)
    reports = json.loads((output / 'collection/report.json').read_text())['contexts']
    diagnostics, support, rows, seen = [], [], [], set()
    h1 = None
    for r in reports:
        path = output / 'collection' / r['context_id']
        assert not r['source_metadata']['additional_experimental_treatment']
        assert all(hash_file(path / n) == h for n, h in r['artifacts'].items())
        cells = pd.read_parquet(path / 'cell-splits.parquet')
        assert not cells.record_id.duplicated().any()
        assert cells.loc[cells.is_NTC, 'task_uid'].isna().all()
        selected = cells.loc[cells.is_NTC | cells.task_uid.notna()]
        source_ids = set(zip(selected.input_sha256, selected.row_index))
        overlap = seen & source_ids
        assert not overlap or r['baseline_id'] == 'H1_shared_exact_NTC'
        seen |= source_ids
        for seed in SEEDS:
            half = selected[f'half_{seed}']
            assert half.isin([0, 1]).all()
            assert not set(selected.loc[half.eq(0), 'record_id']) & set(selected.loc[half.eq(1), 'record_id'])
        if r['baseline_id'] == 'H1_shared_exact_NTC':
            ctrl = selected.loc[selected.is_NTC, ['source_barcode'] + [f'half_{s}' for s in SEEDS]].sort_values('source_barcode').reset_index(drop=True)
            if h1 is not None:
                pd.testing.assert_frame_equal(h1, ctrl)
            h1 = ctrl
        sup = pd.read_parquet(path / 'task-support.parquet')
        support.append(sup)
        diagnostics.append(pd.read_parquet(path / 'diagnostics.parquet'))
        rows.append({'panel_id': r['panel_id'], 'tasks': r['tasks'], 'selected_cells': len(selected),
                     'control_cells': r['NTC_cells'], 'control_role': r['source_metadata']['control_role'],
                     'source_mean_max_error': r['source_mean_max_error'], 'within_seed_cell_overlap': 0})
    pd.concat(diagnostics, ignore_index=True).to_parquet(folder / 'all-task-diagnostics.parquet', index=False)
    pd.concat(support, ignore_index=True).to_parquet(folder / 'all-task-support.parquet', index=False)
    pd.DataFrame(rows).to_parquet(folder / 'independence.parquet', index=False)
    write_json(folder / 'report.json', {'status': 'completed', 'registered_tasks': sum(r['tasks'] for r in reports),
        'all_collection_artifact_hashes_verified': True, 'all_seeds_target_and_control_disjoint': True,
        'H1_shared_NTC_same_assignments': True, 'scope_audit_files': '../scope/', 'source_panels': rows,
        'unique_source_row_identities': len(seen), 'inferred_labels_used_as_ground_truth': False})


def evaluate(output, only):
    reports = json.loads((output / 'collection/report.json').read_text())['contexts']
    panels, minimum, time_comparison = group_specs()[only]
    members = deepcopy([r for r in reports if r['panel_id'] in panels])
    assert set(panels) == {r['panel_id'] for r in members}
    assert len({r['source_metadata']['intervention_mechanism'] for r in members}) == 1
    if time_comparison:
        for r in members:
            r['source_metadata']['biological_cell_line'] = r['source_metadata']['cell_line']
            r['source_metadata']['cell_line'] = r['panel_id'].replace(':', '_')
    folder = output / 'evaluation' / only
    if (folder / 'report.json').exists():
        return
    folder.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    data = load_group(output / 'collection', members, genetic=True)
    write_json(folder / 'axes.json', {k: data[k] for k in ['genes', 'targets', 'lines']})
    pd.DataFrame(data['task_ledger']).to_parquet(folder / 'task-ledger.parquet', index=False)
    pd.DataFrame(data['target_scope']).to_parquet(folder / 'target-scope.parquet', index=False)
    geometry = []
    for scale in ['control_only', 'source_matched']:
        base, _, delta, _ = responses(data, scale)
        for pi, target in enumerate(data['targets']):
            ids = np.flatnonzero(data['available'][:, pi, 0])
            mask = ~data['excluded'][pi]
            for ii, a in enumerate(ids):
                for b in ids[ii + 1:]:
                    for metric, values in [('baseline', base), ('absolute_target', data['target']), ('response', delta)]:
                        geometry.append({'scale': scale, 'canonical_target': target, 'background_a': data['lines'][a],
                                         'background_b': data['lines'][b], 'metric': metric,
                                         **compare_vectors(values[a, pi, 0, mask], values[b, pi, 0, mask])})
    pd.DataFrame(geometry).to_parquet(folder / 'cross-background-geometry.parquet', index=False)
    if len(data['targets']) >= 2 and len(data['genes']) >= 100:
        quadrant_experiment(data, folder)
        prediction_experiment(data, folder, minimum_training=minimum, genetic=True)
        applicability = 'estimated'
    else:
        applicability = 'not_estimable_insufficient_shared_targets_or_genes'
    result = {'status': 'completed', 'group': only, 'applicability': applicability,
              'genes': len(data['genes']), 'targets': len(data['targets']), 'cell_lines': data['lines'],
              'contexts': members, 'seconds': time.monotonic() - start, 'minimum_training_backgrounds': minimum,
              'unit': 'source_time_library_not_distinct_cell_types' if time_comparison else 'cell_background',
              'identity': {'reference_sha256': hash_file(REFERENCE), 'collection_identity_sha256': hash_file(output / 'collection/identity.json'),
                           'code_sha256': {n: hash_file(Path(__file__).with_name(n)) for n in ['genetic.py', 'evaluate.py', 'estimators.py']}},
              'artifacts': {p.name: hash_file(p) for p in folder.iterdir() if p.is_file()}}
    write_json(folder / 'report.json', result)
    print(json.dumps({k: result[k] for k in ['group', 'applicability', 'targets', 'genes', 'seconds']}), flush=True)


def run(run_id):
    from run import execute, tracking
    output = EXPERIMENT / 'outputs' / run_id
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'report.json').exists():
        return
    config = {'experiment_id': EXPERIMENT.name, 'run_id': run_id, 'scope': 'genetic_only', 'issue': ISSUE,
              'pipeline_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              'uv_lock_sha256': hash_file(EXPERIMENT / 'uv.lock'), 'seeds': SEEDS, 'groups': group_specs(),
              'prior_run': '20260922-b', 'base_counts_read_only': True,
              'preregistration': ISSUE + '#issuecomment-5776794168', 'control_amendment': ISSUE + '#issuecomment-5776825517'}
    if (output / 'config.yaml').exists():
        assert json.loads((output / 'config.yaml').read_text()) == json.loads(json.dumps(config))
    write_json(output / 'config.yaml', config)
    tracked, info = tracking(output, config)
    write_json(output / 'tracking.json', info)
    try:
        with (output / 'tests.log').open('w') as stream:
            subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', str(EXPERIMENT / 'tests'), '-v'], stdout=stream, stderr=subprocess.STDOUT, check=True)
        execute('genetic.py', ['--output', output, '--stage', 'collect'], output / 'collect.log')
        execute('genetic.py', ['--output', output, '--stage', 'audit'], output / 'audit.log')
        for name in group_specs():
            execute('genetic.py', ['--output', output, '--stage', 'evaluate', '--only', name], output / 'evaluate.log')
            tracked.log({'progress/completed_prediction_groups': len(list((output / 'evaluation').glob('*/report.json')))})
        execute('summarize.py', ['--output', output, '--genetic'], output / 'summarize.log')
        report = json.loads((output / 'report.json').read_text())
        for row in report['prediction_summary']:
            if row['scale'] == 'control_only' and row['support_subset'] == 'all':
                tracked.summary[row['family'] + '/' + row['model'] + '/relative_MSE_improvement'] = row['relative_MSE_improvement']
        import wandb
        artifact = wandb.Artifact(EXPERIMENT.name + '-' + run_id, type='evaluation')
        for name in ['report.json', 'report.html', 'prediction-summary.parquet', 'config.yaml']:
            artifact.add_file(str(output / name), name=name)
        for p in (output / 'evaluation').glob('*/hyperparameters.json'):
            artifact.add_file(str(p), name=str(p.relative_to(output)))
        tracked.log_artifact(artifact)
        tracked.finish()
        print(json.dumps({'status': 'completed', 'run_id': run_id, 'output': str(output)}), flush=True)
    except Exception as error:
        write_json(output / 'pipeline-failure.json', {'error': repr(error), 'config': config})
        tracked.finish(exit_code=1)
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--stage', choices=['collect', 'audit', 'evaluate'], required=True)
    p.add_argument('--only')
    args = p.parse_args()
    if args.stage == 'evaluate':
        evaluate(args.output, args.only)
    else:
        {'collect': collect, 'audit': audit_collection}[args.stage](args.output)
