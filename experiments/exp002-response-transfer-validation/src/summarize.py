"""Summarize fixed experiments without selecting models on held-out outcomes."""
import argparse
from datetime import datetime, timezone
import html
import json
from pathlib import Path
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'scripts/dossier'))
import numpy as np
import pandas as pd
from profile_responses import write_json
from rna import hash_file


def family(group):
    return group.split('_')[0] if group.startswith(('Jiang_', 'GxE2_')) else group


def quantiles(values):
    values = pd.Series(values).dropna().to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    return {'n': len(values), **{label: float(np.quantile(values, q)) if len(values) else None
                              for label, q in [('q05', .05), ('q25', .25), ('median', .5), ('q75', .75), ('q95', .95)]}}


def balanced_metrics(frame, bootstraps=1000):
    """Equal target/background observations within condition, then equal conditions."""
    groups = frame.groupby('group')[['MSE', 'zero_MSE']].mean()
    mse, zero = groups.mean()
    result = {'target_backgrounds': len(frame), 'targets': frame.canonical_target.nunique(), 'conditions': len(groups),
              'balanced_MSE': float(mse), 'balanced_zero_MSE': float(zero),
              'relative_MSE_improvement': float(1 - mse / zero) if zero else None,
              'fraction_target_backgrounds_beating_zero': float((frame.MSE < frame.zero_MSE).mean()),
              'median_correlation': float(frame.correlation.median()) if frame.correlation.notna().any() else None}
    if 'MAE' in frame:
        result['balanced_MAE'] = float(frame.groupby('group').MAE.mean().mean())
    if 'zero_MAE' in frame:
        result['balanced_zero_MAE'] = float(frame.groupby('group').zero_MAE.mean().mean())
        result['relative_MAE_improvement'] = 1 - result['balanced_MAE'] / result['balanced_zero_MAE'] if result['balanced_zero_MAE'] else None
    if bootstraps:
        ids = sorted(frame.canonical_target.unique())
        grouped = frame.groupby(['canonical_target', 'group']).agg(MSE=('MSE', 'sum'), zero_MSE=('zero_MSE', 'sum'), n=('MSE', 'size'))
        arrays = [grouped[key].unstack(fill_value=0).reindex(ids).to_numpy() for key in ['MSE', 'zero_MSE', 'n']]
        rng = np.random.default_rng(20260921)
        weights = rng.multinomial(len(ids), np.full(len(ids), 1 / len(ids)), size=bootstraps)
        denominator = weights @ arrays[2]
        mean_error = np.divide(weights @ arrays[0], denominator, out=np.full_like(denominator, np.nan, dtype=float), where=denominator > 0)
        mean_zero = np.divide(weights @ arrays[1], denominator, out=np.full_like(denominator, np.nan, dtype=float), where=denominator > 0)
        improvements = 1 - np.nanmean(mean_error, axis=1) / np.nanmean(mean_zero, axis=1)
        result['descriptive_target_bootstrap_95_interval'] = np.quantile(improvements, [.025, .975]).tolist()
    return result


def render(report, output):
    def table(rows, columns):
        if not rows:
            return '<p>不可估计</p>'
        def cell(value):
            if isinstance(value, (dict, list)):
                value = json.dumps(value, ensure_ascii=False)
            if isinstance(value, float):
                value = f'{value:.5g}'
            return html.escape(str(value if value is not None else '—'))
        return '<div class="scroll"><table><thead><tr>' + ''.join('<th>' + cell(c) + '</th>' for c in columns) + \
            '</tr></thead><tbody>' + ''.join('<tr>' + ''.join('<td>' + cell(row.get(c)) + '</td>' for c in columns) + '</tr>' for row in rows) + '</tbody></table></div>'
    primary = [r for r in report['prediction_summary'] if r['scale'] == 'control_only' and r['support_subset'] == 'all' and
               not (r['family'] == 'cross_study_sensitivity' and r['model'] == 'quadrant_gated') and
               r['model'] in ['zero', 'shared', 'shared_shrunk', 'NTC_weighted_fixed_0.1', 'quadrant_gated', 'wrong_target', 'perturbation_agnostic']]
    content = '<!doctype html><html lang="zh"><meta charset="utf-8"><title>共享扰动响应：实验检验</title><style>' \
        'body{font:16px system-ui;line-height:1.6;max-width:1250px;margin:32px auto;padding:0 24px;color:#182330}' \
        'table{border-collapse:collapse;font-size:13px}th,td{border:1px solid #ccd3dc;padding:7px;text-align:left}' \
        'th{background:#edf3f8}.scroll{overflow:auto;margin:18px 0}h2{margin-top:40px}code{font-size:13px}' \
        '</style><h1>共享扰动响应与四象限方案：实验检验</h1>' \
        '<p>计划、判定和结论：<a href="https://github.com/yjcyxky/virtual-cell-challenge/issues/28">GitHub Issue #28</a>。' \
        '本页由固定机器结果生成；回顾性留出实验，不是新培养复现。正值表示优于零响应，负值表示更差。</p>'
    content += '<h2>留出细胞背景：主要结果</h2>' + table(primary, ['family', 'model', 'target_backgrounds', 'targets',
                   'relative_MSE_improvement', 'relative_MAE_improvement', 'descriptive_target_bootstrap_95_interval', 'fraction_target_backgrounds_beating_zero', 'median_correlation'])
    content += '<h2>绝对表达与扰动效应</h2>' + table(report['geometry_summary'], ['family', 'scale', 'metric', 'n', 'median', 'q25', 'q75'])
    content += '<h2>独立细胞与 NTC 半样本</h2>' + table(report['split_response_summary'], ['family', 'scale', 'seed', 'registered_tasks', 'completed', 'correlation'])
    content += '<h2>四象限与非零信号</h2>' + table([r for r in report['classification_summary'] if r['variant'] == 0 and r['tolerance'] == .1],
                    ['family', 'scale', 'tolerance', 'gene_target_sets', 'known_fraction', 'near_zero_fraction', 'conserved_nonzero_fraction', 'zero_observed_share_of_near_zero'])
    content += '<h2>分类一致性</h2>' + table([r for r in report['classification_agreement'] if r['tolerance'] == .1],
                    ['family', 'scale', 'seed', 'support_subset', 'gene_target_sets', 'known_in_both_fraction', 'known_only_agreement', 'agreement_including_unknown', 'kappa'])
    content += '<h2>固定模块与随机平均对照</h2>' + table(report['module_summary'], ['family', 'scale', 'model', 'reference_module',
                    'reference_relative_MSE_improvement', 'random_relative_improvement_median', 'random_relative_improvement_q05', 'random_relative_improvement_q95'])
    if 'posthoc_zero_variance' in report:
        content += '<h2>标签可靠性：事后零方差检查</h2><p>此检查在预登记实验完成后追加，不改变原分类或预测。' \
                   '参与估计的目标细胞全部未检出某读出时，样本方差为零；这不等于总体不确定性为零。' \
                   '下表所列标签需要进一步校准，不能直接视为可靠的生物学共享信号。</p>'
        content += table(report['posthoc_zero_variance']['family_summary'], ['family', 'scale',
                         'operational_conserved_nonzero_labels', 'all_target_zero_every_context',
                         'target_zero_any_context', 'all_contexts_six_halves_supported'])
    content += '<h2>解释范围</h2><ul>' + ''.join('<li>' + html.escape(x) + '</li>' for x in report['limitations']) + '</ul>'
    content += '<h2>下载与复现</h2><p>完整结果：<a href="report.json">report.json</a>；' \
               '<a href="prediction-summary.parquet">预测汇总</a>；<a href="classification-summary.parquet">分类汇总</a>。' \
               '逐任务预测、基因轴、模块成员、分类及细胞划分位于 evaluation/ 与 collection/。' \
               '复现入口见代码提交 <code>' + report['code_commit'] + '</code> 的 experiments/exp002-response-transfer-validation/reproduce.sh。</p></html>'
    (output / 'report.html').write_text(content)


def run(output):
    collection = json.loads((output / 'collection/report.json').read_text())
    audit = json.loads((output / 'audit/report.json').read_text())
    assert audit['status'] == collection['status'] == 'completed'
    groups = sorted((output / 'evaluation').glob('*/report.json'))
    assert len(groups) == 15, 'all 14 main groups and cross-study sensitivity required'
    metrics, classifications, agreements, modules, region_frames = [], [], [], [], []
    for path in groups:
        r = json.loads(path.read_text())
        assert r['status'] == 'completed'
        group = r['group']
        for name, destination in [('metrics', metrics), ('classification-counts', classifications), ('split-classification-agreement', agreements), ('region-metrics', region_frames)]:
            p = path.parent / (name + '.parquet')
            if p.exists():
                frame = pd.read_parquet(p)
                if len(frame):
                    frame['group'], frame['family'] = group, family(group)
                    destination.append(frame)
        module = pd.read_parquet(path.parent / 'module-metrics.parquet', columns=['scale', 'model', 'module', 'reference_parent', 'kind', 'squared_error', 'zero_squared_error'])
        module = module.groupby(['scale', 'model', 'module', 'reference_parent', 'kind']).agg(
            MSE=('squared_error', 'mean'), zero_MSE=('zero_squared_error', 'mean'), n=('squared_error', 'size')).reset_index()
        module['group'], module['family'] = group, family(group)
        modules.append(module)
    metrics = pd.concat(metrics, ignore_index=True)
    metric_keys = ['family', 'group', 'held_cell_line', 'scale', 'canonical_target']
    zero_mae = metrics.loc[metrics.model.eq('zero'), metric_keys + ['MAE']].rename(columns={'MAE': 'zero_MAE'})
    metrics = metrics.merge(zero_mae, on=metric_keys, validate='many_to_one')
    metrics.to_parquet(output / 'all-prediction-metrics.parquet', index=False)
    predictions = []
    for subset, selected in [('all', metrics), ('all_test_and_training_halves_supported', metrics.loc[
            metrics.all_6_test_halves_supported & metrics.all_training_6_halves_supported])]:
        for (fam, scale, model), frame in selected.groupby(['family', 'scale', 'model']):
            predictions.append({'family': fam, 'scale': scale, 'model': model, 'support_subset': subset,
                                'gate_status': 'disabled_fallback_to_shrunk_mean' if fam == 'cross_study_sensitivity' and model == 'quadrant_gated' else 'as_registered',
                                **balanced_metrics(frame)})
    pd.DataFrame(predictions).to_parquet(output / 'prediction-summary.parquet', index=False)
    paired = []
    comparisons = [('shared', 'wrong_target'), ('shared', 'perturbation_agnostic'),
                   ('shared_shrunk', 'zero'), ('quadrant_gated', 'shared_shrunk'),
                   ('NTC_weighted_fixed_0.1', 'shared')]
    pair_keys = ['group', 'canonical_target', 'held_cell_line']
    for (fam, scale), frame in metrics.groupby(['family', 'scale']):
        for model, comparator in comparisons:
            if fam == 'cross_study_sensitivity' and model == 'quadrant_gated':
                continue
            selected = frame.loc[frame.model.eq(model)].copy()
            reference = frame.loc[frame.model.eq(comparator), pair_keys + ['MSE', 'MAE']].rename(columns={'MSE': 'reference_MSE', 'MAE': 'reference_MAE'})
            selected = selected.merge(reference, on=pair_keys, validate='one_to_one')
            selected['zero_MSE'] = selected.reference_MSE
            selected['zero_MAE'] = selected.reference_MAE
            result = balanced_metrics(selected)
            result['relative_MSE_improvement_vs_comparator'] = result.pop('relative_MSE_improvement')
            result['balanced_comparator_MSE'] = result.pop('balanced_zero_MSE')
            result['fraction_beating_comparator'] = result.pop('fraction_target_backgrounds_beating_zero')
            result['relative_MAE_improvement_vs_comparator'] = result.pop('relative_MAE_improvement')
            result['balanced_comparator_MAE'] = result.pop('balanced_zero_MAE')
            paired.append({'family': fam, 'scale': scale, 'model': model, 'comparator': comparator, **result})
    pd.DataFrame(paired).to_parquet(output / 'paired-model-comparisons.parquet', index=False)
    by_background = []
    for keys, frame in metrics.groupby(['family', 'group', 'held_cell_line', 'scale', 'model']):
        by_background.append(dict(zip(['family', 'group', 'held_cell_line', 'scale', 'model'], keys)) | balanced_metrics(frame, 0))
    pd.DataFrame(by_background).to_parquet(output / 'prediction-by-background.parquet', index=False)
    classification = pd.concat(classifications, ignore_index=True)
    counts = []
    for (fam, scale, variant, tolerance), frame in classification.groupby(['family', 'scale', 'variant', 'tolerance']):
        total = int(frame.genes.sum())
        counts.append({'family': fam, 'scale': scale, 'variant': int(variant), 'tolerance': tolerance,
                       'gene_target_sets': total, 'target_condition_sets': len(frame),
                       'known_fraction': float(1 - frame.quadrant_0.sum() / total),
                       'quadrant_fractions': {str(k): float(frame[f'quadrant_{k}'].sum() / total) for k in range(5)},
                       'near_zero_fraction': float(frame.near_zero.sum() / total),
                       'conserved_nonzero_fraction': float(frame.conserved_nonzero.sum() / total),
                       'zero_observed_share_of_near_zero': float(frame.near_zero_with_all_observed_means_zero.sum() / frame.near_zero.sum()) if frame.near_zero.sum() else None})
    pd.DataFrame(counts).to_parquet(output / 'classification-summary.parquet', index=False)
    agreement_rows = []
    for keys, frame in pd.concat(agreements, ignore_index=True).groupby(['family', 'scale', 'seed', 'tolerance']):
        for subset, selected in [('all', frame), ('both_halves_supported', frame.loc[frame.both_halves_contexts_supported])]:
            if not len(selected): continue
            matrix = sum(np.array([np.asarray(row, dtype=np.int64) for row in value]) for value in selected.matrix)
            n = int(matrix.sum()); agreement = np.trace(matrix) / n
            expected = float(matrix.sum(0) @ matrix.sum(1) / n ** 2)
            known = int(matrix[1:, 1:].sum())
            agreement_rows.append(dict(zip(['family', 'scale', 'seed', 'tolerance'], keys)) |
                                  {'support_subset': subset, 'gene_target_sets': n, 'matrix': matrix.tolist(),
                                   'agreement_including_unknown': float(agreement), 'known_in_both_fraction': known / n,
                                   'known_only_agreement': float(np.trace(matrix[1:, 1:]) / known) if known else None,
                                   'kappa': float((agreement - expected) / (1 - expected)) if expected < 1 else None})
    pd.DataFrame(agreement_rows).to_parquet(output / 'classification-agreement.parquet', index=False)
    module = pd.concat(modules, ignore_index=True)
    module.to_parquet(output / 'module-by-condition.parquet', index=False)
    module_summary = []
    for keys, frame in module.groupby(['family', 'scale', 'model', 'reference_parent']):
        losses = frame.groupby(['kind', 'module'])[['MSE', 'zero_MSE']].mean()
        improvement = 1 - losses.MSE / losses.zero_MSE
        reference = improvement.loc['reference']
        random = improvement.loc['random_size_matched'] if 'random_size_matched' in improvement.index.get_level_values(0) else pd.Series(dtype=float)
        module_summary.append(dict(zip(['family', 'scale', 'model', 'reference_module'], keys)) |
                              {'reference_relative_MSE_improvement': float(reference.iloc[0]),
                               'random_relative_improvement_median': float(random.median()) if len(random) else None,
                               'random_relative_improvement_q05': float(random.quantile(.05)) if len(random) else None,
                               'random_relative_improvement_q95': float(random.quantile(.95)) if len(random) else None,
                               'random_sets': len(random)})
    pd.DataFrame(module_summary).to_parquet(output / 'module-summary.parquet', index=False)
    regions = pd.concat(region_frames, ignore_index=True)
    regions.to_parquet(output / 'all-training-region-metrics.parquet', index=False)
    region_summary = []
    for keys, frame in regions.groupby(['family', 'scale', 'model', 'training_region']):
        region_summary.append(dict(zip(['family', 'scale', 'model', 'training_region'], keys)) | balanced_metrics(frame, 0))
    pd.DataFrame(region_summary).to_parquet(output / 'training-region-summary.parquet', index=False)
    panel_family = {}
    for c in collection['contexts']:
        pid = c['panel_id']
        panel_family[pid] = 'Jiang' if pid.startswith('Jiang') else 'GxE2' if pid.startswith('GxE2') else pid.split(':')[0]
    diagnostics = pd.read_parquet(output / 'audit/all-task-diagnostics.parquet')
    diagnostics['family'] = diagnostics.panel_id.map(panel_family)
    geometry_rows, split_rows = [], []
    for keys, frame in diagnostics.loc[diagnostics.experiment.eq('geometry')].groupby(['family', 'scale']):
        for metric in ['NTC_target_correlation', 'response_RMS_over_baseline_SD', 'fraction_abs_delta_gt_0.05',
                       'fraction_abs_delta_gt_0.1', 'fraction_abs_delta_gt_0.2', 'fraction_abs_delta_gt_0.5']:
            geometry_rows.append(dict(zip(['family', 'scale'], keys)) | {'metric': metric, **quantiles(frame[metric])})
    for keys, frame in diagnostics.loc[diagnostics.experiment.eq('split_reproducibility')].groupby(['family', 'scale', 'seed']):
        complete = frame.loc[frame.status.eq('completed')]
        split_rows.append(dict(zip(['family', 'scale', 'seed'], keys)) | {'registered_tasks': len(frame), 'completed': len(complete),
                          'correlation': quantiles(complete.correlation), 'difference_RMS': quantiles(complete.difference_RMS)})
    pd.DataFrame(geometry_rows).to_parquet(output / 'geometry-summary.parquet', index=False)
    pd.DataFrame(split_rows).to_parquet(output / 'split-response-summary.parquet', index=False)
    limitations = [
        '公开响应此前已被探索；本次为预先固定分析规则的回顾性留出验证，不是全新未接触数据或独立培养复现。',
        '细胞拆分每个种子内目标细胞与 NTC 不交叉；三个种子重复使用同批细胞，不增加独立样本量。',
        'Pearson 高相似不等于每个基因不变；零响应分类与保守非零响应分别统计。',
        '等价界限是 mean log1p(CP10K) 单位的操作化阈值；0.05/0.1/0.2 并非普适生物学边界。',
        '正态区间使用条件于当前细胞样本和技术权重的均值方差；不覆盖培养/批次/构件间生物变异。',
        '区间同时性限定于每个靶点/基因的所有背景对，不是全基因组 FDR；全零观测的零方差区间可能过度自信，单独列明。',
        '分类一致性不是准确率；包含 unknown 的高一致性和全部未检出基因的等价不能成为迁移证据。',
        '主要预测只使用训练响应及测试背景 NTC。测试目标细胞数、终点状态、DE 或真实响应不参与模型选择。',
        'control-only 基线用于可获得的 C+delta；来源匹配对照的目标细胞层权重仅定义标签，作为平行敏感性报告。',
        '每个条件内先按靶点/背景等权，条件再等权；bootstrap 按靶点聚类，区间是观察数据上的描述性不确定性。',
        '跨研究敏感性把 K562 两来源一起留出、H1 三划分合并为一个背景；研究条件混杂，不能归因为纯细胞类型差异。',
        '跨研究合并构件/来源没有可识别的独立重复方差，所以不在该敏感性中宣称可靠象限标签。',
        '固定 RNA 模块不是机制真值；随机集合仅大小匹配，没有同时匹配表达量或基因间协方差。',
        '共享均值或当前路由失败只能否定被检验的实现，不能证明所有模型无法迁移；官方 A/B/C 没有公开扰动真值，未编造其响应验证。']
    summary = {'status': 'completed', 'bundle_id': 'response-transfer-validation-' + uuid.uuid4().hex,
               'experiment_id': 'exp002-response-transfer-validation', 'run_id': output.name,
               'code_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
               'completed_at': datetime.now(timezone.utc).isoformat(), 'audit': audit,
               'prediction_summary': predictions, 'paired_model_comparisons': paired,
               'geometry_summary': geometry_rows, 'split_response_summary': split_rows,
               'classification_summary': counts, 'classification_agreement': agreement_rows,
               'module_summary': module_summary, 'training_region_summary': region_summary, 'limitations': limitations,
               'issue': 'https://github.com/yjcyxky/virtual-cell-challenge/issues/28'}
    write_json(output / 'report.json', summary)
    render(summary, output)
    print(json.dumps({'status': 'completed', 'bundle_id': summary['bundle_id'], 'predictions': len(metrics)}), flush=True)


def attach_posthoc(output):
    report_path = output / 'report.json'
    report = json.loads(report_path.read_text())
    diagnostic = json.loads((output / 'posthoc-zero-variance/report.json').read_text())
    if 'posthoc_zero_variance' in report:
        assert report['posthoc_zero_variance'] == diagnostic
        return
    assert hash_file(report_path) == diagnostic['initial_report_sha256']
    report['predefined_experiment_report_sha256'] = hash_file(report_path)
    report['predefined_experiment_bundle_id'] = report['bundle_id']
    report['bundle_id'] = 'response-transfer-validation-' + uuid.uuid4().hex
    report['posthoc_zero_variance'] = diagnostic
    report['code_commit'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    report['completed_at'] = datetime.now(timezone.utc).isoformat()
    report['limitations'].append('事后审计单列目标细胞全部未检出读出、导致 plug-in 目标方差为零的非零等价标签；它们不能直接解释为可靠的生物学共享响应，审计没有将其宣布为生物学假阳性。')
    write_json(report_path, report)
    render(report, output)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--attach-posthoc', action='store_true')
    args = p.parse_args()
    (attach_posthoc if args.attach_posthoc else run)(args.output.resolve())
