"""Summarize the complete registered batch; never select runs by observed score."""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src'))
from vcc_task.common import verified, ref, write_json

COMPARISON = 'C-DEPMAP-RESPONSE-ALIGNMENT'
ARMS = ('shared', 'template', 'knn', 'ridge', 'ridge-shuffled', 'mlp', 'mlp-shuffled')
FOLDS = ('S3-H1', 'S3-K562', 'S3-RPE1', 'S3-HepG2', 'S3-Jurkat', 'S2-H1')
METRICS = ('pds_cosine', 'expr_mse_unbiased_capped_norm', 'de_wilcoxon_lfc_nmae',
           'de_wilcoxon_direction_fidelity_yield_raw', 'de_wilcoxon_direction_reach_raw',
           'de_wilcoxon_sig_jaccard')


def paired(table, candidate, control, metric, folds=FOLDS[:5]):
    effects = []
    for fold in folds:
        a, b = table[(candidate, fold)][metric], table[(control, fold)][metric]
        if a is not None and b is not None and np.isfinite(a) and np.isfinite(b):
            effects.append({'fold': fold, 'candidate': a, 'control': b, 'effect': a - b})
    values = np.asarray([r['effect'] for r in effects])
    return {'pairs': effects, 'n_contexts': len(values),
            'mean_effect': float(values.mean()) if len(values) else None,
            'std_between_contexts': float(values.std(ddof=1)) if len(values) > 1 else None,
            'positive_contexts': int((values > 0).sum()),
            'min_effect': float(values.min()) if len(values) else None,
            'max_effect': float(values.max()) if len(values) else None}


def main():
    dag = json.loads((ROOT / 'docs/research/experiment_dag.json').read_text())
    comparison = dag['comparisons'][COMPARISON]
    ids = comparison['controls'] + comparison['candidates']
    if len(set(ids)) != 42:
        raise ValueError('Require all 42 independently registered fits')
    table, rows, sources, details = {}, [], [], {}
    for node_id in ids:
        node = dag['nodes'][node_id]
        if node['status'] != 'completed':
            raise ValueError(f'Incomplete registered fit: {node_id}')
        metrics_path = verified(node['metrics_ref'])
        metrics = json.loads(metrics_path.read_text())
        if not metrics['evaluation_completed'] or not metrics['diagnostics_completed']:
            raise ValueError(f'Incomplete evaluation: {node_id}')
        config = node['expected_config']
        fold = config['fit_scope']['outer_split']
        arm = config['model']['kind'] + ('-shuffled' if config['model']['shuffled_prior'] else '')
        if (arm, fold) in table or len(metrics['panels']) != 1:
            raise ValueError('Unexpected duplicate fit or panel scope')
        context, panel = next(iter(metrics['panels'].items()))
        scores = panel['scores']
        pre_score = json.loads(verified(panel['pre_score_completed_ref']).read_text())
        if pre_score['status'] != 'completed' or not pre_score['hard_constraints_passed']:
            raise ValueError(f'Incomplete pre-score diagnostics: {node_id}')
        for item in pre_score['files']:
            verified(item)
        diagnostic = json.loads(verified(panel['diagnostic_ref']).read_text())
        null = json.loads(verified(panel['null_ref']).read_text())
        generated = json.loads(verified(panel['generation_ref']).read_text())
        emission_ref = next(r for r in generated['files'] if Path(r['path']).name == 'emission.json')
        emission = json.loads(verified(emission_ref).read_text())
        readout = pd.read_parquet(verified(panel['readout_diagnostics_ref']))
        de_calls = pd.read_parquet(verified(panel['de_diagnostics_ref']))
        row = {'run_id': node_id, 'arm': arm, 'fold': fold, 'context': context,
               'targets': panel['targets'], 'measured_genes': panel['measured_genes'],
               'Overall': scores['Overall'], 'residual_pearson': panel['residual_mean_pearson'],
               'centroid_removed_pearson': panel['evaluator_only_centering']['mean_centroid_removed_pearson'],
               'target_template_residual_pearson': panel['evaluator_only_centering']['mean_target_template_residual_pearson'],
               'residual_rmse': panel['residual_mean_rmse'], **{k: scores['raw'][k] for k in METRICS},
               **{'normalized/' + k: (scores['normalized'] or {}).get(k) for k in METRICS}}
        rows.append(row); table[(arm, fold)] = row; sources.append(ref(metrics_path))
        details[node_id] = {
            'raw': scores['raw'], 'normalized': scores['normalized'], 'Overall': scores['Overall'],
            'valid_targets_per_metric': panel['valid_targets_per_metric'],
            'metric_support': panel['metric_support'],
            'raw_strata': scores['raw_strata'], 'training': metrics['training'],
            'geometry': diagnostic['geometry'], 'decomposition': diagnostic['decomposition'],
            'evaluator_only_centering': panel['evaluator_only_centering'],
            'null_identity': null['zero_identity'], 'null_decoder_identity': null['zero_decoder_identity'],
            'added_null_de': null['added_zero_de'],
            'null_calls': null['de_summary'], 'independent_repeats': emission['independent_repeats'],
            'readout_numeric_means': readout.select_dtypes('number').mean().replace({np.nan: None}).to_dict(),
            'de_call_numeric_sums': de_calls.select_dtypes('number').sum().to_dict(),
            'diagnostic_refs': [panel['pre_score_completed_ref'], panel['diagnostic_ref'], panel['readout_diagnostics_ref'],
                                panel['de_diagnostics_ref'], panel['null_ref'], emission_ref],
            'elapsed_seconds': metrics['elapsed_seconds_this_attempt'], 'wandb_url': metrics['wandb_url']}
    if set(table) != {(arm, fold) for arm in ARMS for fold in FOLDS}:
        raise ValueError('Batch identity differs from the frozen comparison')
    comparisons, decisions = {}, {}
    for candidate in ('knn', 'ridge', 'mlp'):
        controls = ['shared', 'template'] + ([candidate + '-shuffled'] if candidate != 'knn' else [])
        passes_signal, passes_overall = [], []
        for control in controls:
            key = candidate + '_vs_' + control
            result = {m: paired(table, candidate, control, m)
                      for m in (*METRICS, *('normalized/' + m for m in METRICS),
                                'Overall', 'residual_pearson', 'residual_rmse',
                                'centroid_removed_pearson', 'target_template_residual_pearson')}
            comparisons[key] = result
            pds, residual, overall = result['pds_cosine'], result['residual_pearson'], result['Overall']
            passes_signal.append(pds['n_contexts'] == 5 and residual['n_contexts'] == 5
                                 and pds['mean_effect'] >= .02 and pds['positive_contexts'] >= 3
                                 and residual['mean_effect'] >= .02)
            passes_overall.append(overall['n_contexts'] > 0 and overall['mean_effect'] >= .02)
        decisions[candidate] = {'mechanism_signal_threshold_passed': all(passes_signal),
            'additional_defined_Overall_threshold_passed': all(passes_overall),
            'full_task_requires_review': 'Inspect all six metrics and weak/real-n strata; numerical thresholds alone do not establish full-task improvement',
            'matched_prior_shuffle_available': candidate != 'knn'}
    folder = ROOT / 'experiments/depmap-response'
    summary_path = folder / 'outputs/depmap-mlp-shuffled-s2-h1-s930/diagnostics/batch-results.json'
    write_json(summary_path, {'comparison_id': COMPARISON, 'source_refs': sources, 'rows': rows,
                             'paired_S3': comparisons, 'decisions': decisions, 'details': details})
    def fmt(x):
        return '不可定义' if x is None else f'{x:.4f}'
    text = ['# DepMap 监督对齐完整批次结果', '',
            f'Comparison: `{COMPARISON}`。七方案 × 六独立外层 fit，seed 930，42 fits 全部完成。', '',
            '## 逐背景结果', '',
            '|划分|方案|raw PDS|Overall|残差相关|表达误差↓|LFC误差↓|DE方向↑|DE reach↑|DE重叠↑|',
            '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for fold in FOLDS:
        for arm in ARMS:
            r = table[(arm, fold)]
            text.append('|' + '|'.join([fold, arm, fmt(r['pds_cosine']), fmt(r['Overall']),
                fmt(r['residual_pearson']), *[fmt(r[m]) for m in METRICS[1:]]]) + '|')
    text += ['', '## 五背景 S3 配对效应', '',
             '|候选−对照|PDS均差|PDS正向背景|残差相关均差|Overall均差|Overall有效背景|',
             '|---|---:|---:|---:|---:|---:|']
    for name, values in comparisons.items():
        text.append('|' + '|'.join([name, fmt(values['pds_cosine']['mean_effect']),
            str(values['pds_cosine']['positive_contexts']) + '/5', fmt(values['residual_pearson']['mean_effect']),
            fmt(values['Overall']['mean_effect']), str(values['Overall']['n_contexts'])]) + '|')
    text += ['', '## 阈值与证据范围', '']
    for arm, decision in decisions.items():
        text.append(f'- {arm}: 预登记的扰动特异信号门槛'
                    + ('通过' if decision['mechanism_signal_threshold_passed'] else '未通过')
                    + '；配对可定义 Overall 数值门槛'
                    + ('通过。' if decision['additional_defined_Overall_threshold_passed'] else '未通过。'))
    text += ['', '原始六项、归一化六项、有效靶点数、全部配对效应和背景间波动、分层、计数与 null、'
             '源监督覆盖、训练收敛和通路诊断引用见 '
             '[机器汇总](outputs/depmap-mlp-shuffled-s2-h1-s930/diagnostics/batch-results.json)。'
             '最终采用/停止决定须结合这些诊断写入 Evidence Ledger。', '',
             '全部本地背景均为开发证据；单 seed、五背景来自三项研究，不能把靶点或生成细胞当成独立生物背景。'
             '真实 n、测量轴和 assay 深度不同；剂量只观察未校正。固定自身下降及继承 NTC 分布不是模型学习证据。'
             '原始近邻没有等架构置乱对照，因此不能单独证明功能先验的特异贡献。无 leaderboard 提交或 SOTA 主张。', '']
    (folder / 'REPORT.md').write_text('\n'.join(text))
    print(json.dumps({'summary_ref': ref(summary_path), 'decisions': decisions}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
