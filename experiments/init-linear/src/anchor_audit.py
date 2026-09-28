"""Registered 2x2 scoring-baseline diagnostic of an already completed fit."""
import argparse
import fcntl
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

EXPERIMENT = Path(__file__).resolve().parents[1]
ROOT = EXPERIMENT.parents[1]
CONFIG = EXPERIMENT / 'configs/anchor-audit-s01.json'
sys.path.insert(0, str(ROOT / 'scripts'))
from research import git, ref_error


def gate(root=ROOT, config_path=CONFIG):
    config = json.loads(config_path.read_text())
    dag = json.loads((root / 'docs/research/experiment_dag.json').read_text())
    ledger = json.loads((root / 'docs/research/evidence_ledger.json').read_text())
    node = dag['nodes'][config['run_id']]
    stage = node['anchor_audit']
    comparison = dag['comparisons'][config['comparison_id']]
    if (node['status'] != 'completed' or stage['status'] not in {'registered', 'completed'}
            or stage['comparison_id'] != config['comparison_id']
            or (root / stage['config_ref']['path']).resolve() != config_path.resolve()):
        raise ValueError('audit requires completed fit and registered evaluation stage')
    for eid in stage['requires_evidence']:
        if ledger['evidence'][eid]['state'] not in {'supported', 'not_supported', 'inconclusive'}:
            raise ValueError('audit requires closed prerequisite evidence')
    if not comparison.get('decision_rule') or not config.get('source_refs') or not config.get('conditions'):
        raise ValueError('audit requires frozen design, decision rule and sources')
    committed = [stage['config_ref'], *stage['code_refs'], node['config_ref'], *node['code_refs']]
    for item in committed + config['source_refs'] + [node['metrics_ref']]:
        if problem := ref_error(root, item):
            raise ValueError(f'{item["path"]}: {problem}')
    for item in committed:
        if hashlib.sha256(git(root, 'show', 'HEAD:' + item['path']).stdout).hexdigest() != item['sha256']:
            raise ValueError('audit code/config must be committed: ' + item['path'])
    frozen = json.loads(git(root, 'show', 'HEAD:docs/research/experiment_dag.json').stdout)
    frozen_stage = frozen['nodes'][config['run_id']]['anchor_audit']
    if any(stage[k] != frozen_stage[k] for k in ('config_ref', 'code_refs', 'comparison_id', 'requires_evidence')):
        raise ValueError('audit registration must be committed')
    if comparison != frozen['comparisons'][config['comparison_id']]:
        raise ValueError('audit comparison must be committed')
    return config, node, git(root, 'rev-parse', 'HEAD').stdout.decode().strip()


def scores(aggregate, meta, bundle):
    import numpy as np
    from cell_eval2 import score_metrics
    from cell_eval2.competition import competition_members
    frame = score_metrics(aggregate, real_bundle=str(bundle), user_meta=meta)
    values = {m: frame.filter(frame['metric'] == m)['from_replicate'].item()
              for m in competition_members()}
    overall = frame.filter(frame['metric'] == 'avg_score')['from_replicate'].item()
    if not all(v is not None and np.isfinite(v) for v in [overall, *values.values()]):
        raise ValueError('official six scores must be finite')
    np.testing.assert_allclose(overall, np.mean(list(values.values())), rtol=0, atol=1e-12)
    return {'Overall': overall, 'normalized': values}, frame


def verify_resume(path, binding):
    if path.exists() and json.loads(path.read_text())['binding'] != binding:
        raise ValueError('audit identity changed on resume')


def effects(results):
    """Within-reference factorial contrasts, descriptive and not biological inference."""
    contrasts = {}
    for panel, conditions in results.items():
        contrasts[panel] = {}
        for arm in conditions['dispersed-exclude']['arms']:
            values = {c: {'Overall': r['arms'][arm]['Overall'], **r['arms'][arm]['normalized']}
                      for c, r in conditions.items()}
            contrasts[panel][arm] = {
                metric: {'emission_at_exclude': values['tile-exclude'][metric] - values['dispersed-exclude'][metric],
                         'include_at_dispersed': values['dispersed-include'][metric] - values['dispersed-exclude'][metric],
                         'interaction': values['tile-include'][metric] - values['tile-exclude'][metric]
                                        - values['dispersed-include'][metric] + values['dispersed-exclude'][metric]}
                for metric in values['dispersed-exclude']}
    return contrasts


def evaluate(config, node, directory, tracked):
    import gc
    import anndata as ad
    import numpy as np
    import polars as pl
    from cell_eval2.baseline import generic_response_profile, build_baseline_prediction
    from cell_eval2.real_bundle import build_real_bundle
    from data import NTC, ref, write_json
    from challenge import scorer_config, scorer_contract
    training = node['expected_config']
    if scorer_contract() != json.loads((ROOT / training['benchmark']['scorer']['path']).read_text()):
        raise ValueError('installed official scorer changed')
    original = json.loads((ROOT / node['metrics_ref']['path']).read_text())
    real_all = ad.read_h5ad(ROOT / config['reference_ref']['path'])
    results = {}
    for panel, specification in config['panels'].items():
        real = real_all[real_all.obs.target_gene.isin([NTC] + specification['targets'])].copy()
        panel_dir = directory / panel
        panel_dir.mkdir(exist_ok=True)
        cfg = scorer_config(**config['runtime'], cache_real=str(panel_dir / 'real-cache'))
        old_bundle = ROOT / specification['original_bundle']
        old_manifest = json.loads((old_bundle / 'manifest.json').read_text())
        results[panel] = {}
        for condition in config['conditions']:
            label = condition['id']
            destination = panel_dir / label
            destination.mkdir(exist_ok=True)
            done = destination / 'result.json'
            if done.exists():
                record = json.loads(done.read_text())
                for item in record['output_refs']:
                    if ref(ROOT / item['path']) != item:
                        raise ValueError('completed audit output changed')
                results[panel][label] = record
                continue
            bundle = destination / 'bundle'
            if not (bundle / 'manifest.json').exists():
                print(f'Audit baseline/anchors: {panel}/{label}, real={real.shape}', flush=True)
                profile = generic_response_profile(real, pert_col=cfg.pert_col, control=cfg.control,
                                                   exclude_target_gene=condition['exclude_target_gene'])
                baseline = build_baseline_prediction(profile, real, pert_col=cfg.pert_col,
                                                     control=cfg.control, emit=condition['emit'], seed=config['baseline_seed'])
                write_json(destination / 'construction.json', {
                    **condition, 'seed': config['baseline_seed'], 'n_excluded': profile.n_excluded,
                    'n_profile_perturbations': profile.n_perturbations,
                    'baseline_emission': baseline.uns['baseline_emission']})
                build_real_bundle(real, baseline, config=cfg, outdir=str(bundle),
                                  bundle_id=f'{config["run_id"]}-audit-{panel}-{label}',
                                  base_seed=config['anchor_seed'], n_splits=config['anchor_splits'])
                del baseline, profile
                gc.collect()
            manifest = json.loads((bundle / 'manifest.json').read_text())
            for field in ['real_fingerprint', 'anchor_semantic_identity', 'derived_seeds', 'rule_digest']:
                if manifest[field] != old_manifest[field]:
                    raise ValueError('audit changed fixed reference/anchor semantics: ' + field)
            anchor = pl.read_parquet(bundle / 'anchor_agg.parquet').sort('metric')
            old_anchor = pl.read_parquet(old_bundle / 'anchor_agg.parquet').sort('metric')
            for field in ['replicate', 'replicate_sd']:
                if field in anchor.columns:
                    np.testing.assert_allclose(anchor[field].to_numpy(), old_anchor[field].to_numpy(),
                                               rtol=0, atol=config['replay_atol'], equal_nan=True)
            record = {'condition': condition, 'reference_cells': real.n_obs, 'genes': real.n_vars,
                      'targets': len(specification['targets']), 'arms': {},
                      'baseline_raw': pl.read_csv(bundle / 'baseline_agg.csv').filter(pl.col('statistic') == 'mean').to_dicts()[0]}
            for arm in config['arms']:
                source = ROOT / original['evaluation'][panel][arm]['result_directory']
                values, frame = scores(source / 'aggregate.csv', json.loads((source / 'run_meta.json').read_text()), bundle)
                if label == 'dispersed-exclude':
                    expected = original['evaluation'][panel][arm]
                    for key in values['normalized']:
                        np.testing.assert_allclose(values['normalized'][key], expected['normalized'][key],
                                                   rtol=0, atol=config['replay_atol'])
                    np.testing.assert_allclose(values['Overall'], expected['Overall'], rtol=0, atol=config['replay_atol'])
                values['raw_aggregate_ref'] = ref(source / 'aggregate.csv')
                record['arms'][arm] = values
                frame.write_csv(destination / f'{arm}-scores.csv')
                tracked.log({f'anchor_audit/{panel}/{label}/{arm}/Overall': values['Overall']})
            record['linear_minus_shared'] = record['arms']['linear']['Overall'] - record['arms']['shared']['Overall']
            record['output_refs'] = [ref(p) for p in sorted(destination.rglob('*')) if p.is_file()]
            write_json(done, record)
            results[panel][label] = record
            print(json.dumps({'completed': f'{panel}/{label}', 'Overall': {a: r['Overall'] for a,r in record['arms'].items()}}), flush=True)
        del real
        gc.collect()
    # Frozen source bytes, including original prediction matrices and raw outputs, remain intact.
    for item in config['source_refs']:
        if ref(ROOT / item['path']) != item:
            raise ValueError('audit source changed during evaluation')
    result = {'evaluation_completed': True, 'comparison_id': config['comparison_id'],
              'research': original['research'], 'source_metrics_ref': node['metrics_ref'],
              'raw_scores_unchanged': True, 'original_condition_reproduced': True,
              'results': results, 'contrasts': effects(results),
              'interpretation': 'Fixed H1 reference baseline sensitivity only; not an r4 deployment reconstruction.'}
    write_json(directory / 'results.json', result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gate', action='store_true')
    args = parser.parse_args()
    config, node, commit = gate()
    if args.gate:
        print('Registered anchor audit gate passed', flush=True)
        return
    import wandb
    import yaml
    from data import ref, write_json
    output = EXPERIMENT / 'outputs' / config['run_id']
    training = yaml.safe_load((output / 'config.yaml').read_text())
    runtime = {p: importlib.metadata.version(p) for p in training['runtime_versions']}
    if runtime != training['runtime_versions'] or ref(EXPERIMENT / 'uv.lock') != training['lock_ref']:
        raise ValueError('audit environment differs from completed training')
    directory = output / 'cache/anchor-audit'
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / '.audit.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        identity = directory / 'identity.json'
        binding = {'config_ref': ref(CONFIG), 'code_refs': node['anchor_audit']['code_refs'],
                   'runtime_versions': runtime, 'lock_ref': training['lock_ref']}
        verify_resume(identity, binding)
        if not identity.exists():
            write_json(identity, {'binding': binding, 'git_commit': commit, 'config': config})
        tracked = wandb.init(entity='yjcyxky', project='virtual-cell-challenge', group='init-linear',
                             id=config['run_id'], name=config['run_id'], dir=str(output), resume='must',
                             config={'anchor_audit': json.loads(identity.read_text())})
        try:
            evaluate(config, node, directory, tracked)
            marker = directory / 'artifact.json'
            if not marker.exists():
                artifact = wandb.Artifact(config['run_id'] + '-anchor-audit', type='evaluation', metadata=binding)
                for path in sorted(directory.rglob('*')):
                    if path.is_file() and 'real-cache' not in path.parts and path.suffix in {'.json', '.csv', '.parquet', '.yaml'}:
                        artifact.add_file(str(path), name=str(path.relative_to(directory)))
                logged = tracked.log_artifact(artifact)
                logged.wait()
                write_json(marker, {'artifact': logged.qualified_name})
            tracked.summary['anchor_audit/status'] = 'completed'
            tracked.finish()
        except BaseException:
            tracked.finish(exit_code=1)
            raise


if __name__ == '__main__':
    main()
