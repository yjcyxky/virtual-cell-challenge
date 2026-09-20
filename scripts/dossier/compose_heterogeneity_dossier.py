#!/usr/bin/env python
"""Publishable decision views from completed and verified heterogeneity phases."""
import argparse
from collections import Counter
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
import uuid
import numpy as np
import pandas as pd
from compose_cross_source_dossier import copy_component
from profile_cross_source_coverage import ROOT
from profile_responses import write_json
from render import render
from render_heterogeneity import explorer_page
from rna import hash_file,value_hash,quantiles


def joint_summary(frame,a,b,label):
    mask=np.isfinite(pd.to_numeric(frame[a],errors='coerce'))&np.isfinite(pd.to_numeric(frame[b],errors='coerce'))
    d=frame.loc[mask]
    return {'comparison':label,'all_registered_rows':len(frame),'joint_finite_rows':len(d),'missing_at_least_one_metric':int((~mask).sum()),
            'metric_A':a,'metric_B':b,'A_quantiles_on_same_rows':quantiles(d[a].to_numpy()),'B_quantiles_on_same_rows':quantiles(d[b].to_numpy()),
            'biological_replication_inferred':False}


def comparison_design(a,b):
    if a.startswith('GxE2:') and b.startswith('GxE2:'):
        aa=json.loads(a.split(':',1)[1]);bb=json.loads(b.split(':',1)[1])
        if aa[0]==bb[0] and aa[1:]!=bb[1:]:return 'GxE2_same_source_line_different_drug_or_dose'
        if aa[0]!=bb[0] and aa[1:]==bb[1:]:return 'GxE2_different_source_line_same_drug_and_dose'
        return 'GxE2_joint_line_and_exposure_difference'
    if a.startswith('Jiang__') and b.startswith('Jiang__'):
        aa=a.split('__');bb=b.split('__')
        if aa[1]==bb[1]:return 'Jiang_same_source_line_different_24h_stimulus'
        if aa[2]==bb[2]:return 'Jiang_different_source_line_same_24h_stimulus'
        return 'Jiang_joint_line_and_stimulus_difference'
    if a==b=='GxE1':return 'GxE1_same_line_different_source_exposure_or_construct'
    if {a,b}=={'replogle:K562_essential','replogle:K562_gwps'}:return 'Replogle_K562_day6_essential_vs_day8_GWPS_time_library_capture_confounded'
    if a.startswith('nadig:') and b.startswith('nadig:') and a!=b:return 'Nadig_day7_line_effector_medium_confounded'
    if a==b:return 'same_source_panel_construct_or_recorded_biological_context_difference'
    return 'other_source_or_background_difference_not_factor_isolated'


def compose(source,comparisons,endpoint,supplements,baselines,extra,output):
    paths={k:v.resolve() for k,v in {'source':source,'comparisons':comparisons,'endpoint':endpoint,'supplements':supplements,'baselines':baselines,'extra':extra}.items()}
    output=output.resolve();output.mkdir(parents=True,exist_ok=False)
    reports={}
    for k in ['comparisons','endpoint','supplements','baselines','extra']:
        print('verify and copy '+k,flush=True);reports[k]=copy_component(paths[k],output/k)
    primary=reports['comparisons'];ep=reports['endpoint'];supp=reports['supplements'];base=reports['baselines']
    if primary['all_same_target_pairs']!=249071 or base['endpoint_task_partitions']!=75690 or base['source_effects_reproduced']!=37633:raise ValueError('incomplete_registered_analysis_denominators')
    for r in [supp,base]:
        if r['source_bundle_id']!=primary['source_bundle_id']:raise ValueError('phase_parent_identity_mismatch')
    pairs=pd.read_parquet(output/'supplements/pair-sensitivities.parquet');tasks=pd.read_parquet(output/'comparisons/selected-task-index.parquet')
    endpoint_rows=pd.read_parquet(output/'baselines/all-endpoint-task-partitions.parquet');baseline_pairs=pd.read_parquet(output/'baselines/all-baseline-pairs.parquet')
    if tasks.task_uid.duplicated().any() or len(tasks)!=37845 or tasks.confirmed_collection_copy.any():raise ValueError('task_duplicate_or_confirmed_copy_in_primary')
    expected=sum(n*(n-1)//2 for n in tasks.canonical_target.value_counts())
    if len(pairs)!=expected or pairs[['task_uid_A','task_uid_B']].duplicated().any():raise ValueError('pair_enumeration_incomplete_or_duplicated')
    target_map=tasks.set_index('task_uid').canonical_target
    if not pairs.canonical_target.eq(pairs.task_uid_A.map(target_map)).all() or not pairs.canonical_target.eq(pairs.task_uid_B.map(target_map)).all():raise ValueError('pair_intervention_identity_mismatch')
    pairs['design_interpretation']=[comparison_design(a,b) for a,b in zip(pairs.panel_A,pairs.panel_B)]
    pairs.to_parquet(output/'all-pair-designs.parquet',index=False,compression='zstd')
    # Validate the exact source evidence after all computations. Hash large
    # raw/count inputs once in their cell adapter; recheck their frozen snapshot
    # identities here without pretending report status validates file contents.
    consumed={};raw={}
    def merge(m):
        for name,digest in m.items():
            if name in consumed and consumed[name]!=digest:raise ValueError('one_input_two_recorded_identities:'+name)
            consumed[name]=digest
    original=json.loads((paths['comparisons']/'consumed-inputs.json').read_text());merge(original['frozen_artifacts']);merge(original['response_gene_files'])
    cell_inputs=json.loads((paths['endpoint']/'consumed-inputs.json').read_text());merge(cell_inputs['frozen_artifacts']);raw.update(cell_inputs['raw_or_normalized_view_sha256'])
    extra_inputs=json.loads((paths['extra']/'consumed-inputs.json').read_text());merge(extra_inputs['frozen_artifacts'])
    for name,digest in extra_inputs['raw_count_files'].items():
        if name in raw and raw[name]!=digest:raise ValueError('one_count_input_two_recorded_identities:'+name)
        raw[name]=digest
    merge(json.loads((paths['supplements']/'consumed-inputs.json').read_text()));merge(json.loads((paths['baselines']/'consumed-inputs.json').read_text()))
    for i,(name,digest) in enumerate(consumed.items()):
        if hash_file(ROOT/name)!=digest:raise ValueError('source_input_changed_after_calculation:'+name)
        if (i+1)%5000==0:print('frozen evidence reverified '+str(i+1)+'/'+str(len(consumed)),flush=True)
    for i,(name,digest) in enumerate(raw.items()):
        if hash_file(ROOT/name)!=digest:raise ValueError('raw_or_normalized_view_changed_after_calculation:'+name)
        print('source count/cache reverified '+str(i+1)+'/'+str(len(raw)),flush=True)
    write_json(output/'final-input-verification.json',{'status':'completed','frozen_evidence_files':consumed,'source_count_or_log_views':raw,'source_input_mutations':0})
    evidence=json.loads((output/'supplements/source-evidence-index.json').read_text());source_protocol=json.loads((paths['source']/'source-protocol-index.json').read_text())
    releases={r['source_dossier']:r['release'] for r in source_protocol}
    def release_for(folder):
        name=Path(folder).name
        if name.startswith('h1-response'):return 'https://github.com/yjcyxky/virtual-cell-challenge/releases/tag/assessment-h1-response-20260919'
        if name.startswith('replogle-response'):name='replogle-dossier-20260919'
        if name.startswith('nadig-response'):name='nadig-dossier-20260919'
        return releases.get(name,'https://github.com/yjcyxky/virtual-cell-challenge/issues/20')
    task_view=pd.read_parquet(output/'supplements/task-sensitivities.parquet').merge(tasks[['task_uid','source_folder','source_task_file','source_task_file_sha256','source_gene_result_file','source_gene_result_sha256']],on='task_uid',validate='one_to_one')
    task_view['source_release']=task_view.source_folder.map(release_for)
    task_view['source_summary_file']=['comparisons/source-task-summaries/'+value_hash([a,b])+'.json' for a,b in zip(task_view.source_folder,task_view.source_task_file)]
    if not all((output/p).is_file() for p in task_view.source_summary_file):raise ValueError('source_task_drilldown_not_packaged')
    task_view.to_parquet(output/'task-evidence-index.parquet',index=False,compression='zstd')
    targets=tasks.groupby('canonical_target').agg(registered_tasks=('task_uid','size'),source_panels=('panel_id','nunique'),completed_source_effects=('effect_status',lambda s:int(s.eq('completed').sum()))).reset_index()
    targets['all_same_target_pairs']=targets.registered_tasks*(targets.registered_tasks-1)//2
    targets.to_parquet(output/'all-canonical-target-denominators.parquet',index=False)
    if len(targets)!=10198 or int(targets.all_same_target_pairs.sum())!=249071 or int(targets.all_same_target_pairs.gt(0).sum())!=4163:raise ValueError('target_denominator_changed')
    repeat_path=next(r['file'] for r in evidence if r['source_file']=='plate6-plate14-repeat-comparisons.parquet')
    repeat=pd.read_parquet(output/'supplements'/repeat_path)
    joint=[joint_summary(pairs,'native_absolute_target_correlation','native_effect_correlation','all registered same-target pairs: absolute versus NTC-relative expression'),
           joint_summary(pairs,'native_effect_correlation','official_effect_correlation','native versus official common gene intersections'),
           joint_summary(pairs,'native_effect_correlation','native_thinned_effect_correlation','native versus existing expected-depth 10k sensitivity'),
           joint_summary(repeat,'mean_logRNA_correlation','response_correlation','all author repeat pairs: same jointly estimable rows only')]
    design_summaries=[]
    for design,d in pairs.groupby('design_interpretation',sort=True):
        design_summaries.append({'design':design,'registered_pairs':len(d),'canonical_targets':d.canonical_target.nunique(),'status_counts':d.status.value_counts().to_dict(),
            'native_effect_correlation':quantiles(d.native_effect_correlation.dropna().to_numpy()),'native_effect_correlation_missing':int(d.native_effect_correlation.isna().sum()),
            'native_effect_difference_RMS':quantiles(d.native_effect_difference_RMS.dropna().to_numpy()),'factor_causal_effect':'not_identifiable'})
    write_json(output/'same-row-metric-comparisons.json',joint);write_json(output/'experimental-design-summaries.json',design_summaries)
    # Every row and numeric coordinate is in the offline payload; only display
    # columns are shortened. Missing values remain null through JSON transport.
    pair_columns=['canonical_target','panel_A','panel_B','comparison_role','design_interpretation','status','native_common_genes','native_baseline_difference_RMS','native_absolute_target_correlation','native_effect_correlation','native_effect_difference_RMS','official_effect_correlation','native_thinned_effect_correlation']
    pair_metrics=['native_absolute_target_correlation','native_effect_correlation','native_baseline_difference_RMS','native_effect_difference_RMS','official_effect_correlation','native_thinned_effect_correlation','native_common_genes','target_RNA_ratio_absolute_difference','library_size_median_absolute_difference','matched_target_cells_absolute_difference']
    (output/'responses.html').write_text(explorer_page('同靶点跨背景响应：全部 249,071 对',pairs,pair_columns,pair_metrics,
        '所有身份合格 CRISPRi 任务的同 canonical target 无序对。干预基因从比较集合排除；不同构件保留；原生尺度与归一化分母未改变。共同基因文件包含精确原生索引，任务特异排除还包括来源 is_target 标记。','all-pair-designs.parquet',{'common_gene_file':'comparisons/'}))
    endpoint_display=['canonical_target','panel_id','source_context','partition','status','supported_target_cells','supported_target_fraction','target_unknown_partition_fraction','composition_RMS_safe_downstream','within_RMS_safe_downstream','difference_from_full_matched_effect_RMS_safe_downstream']
    endpoint_metrics=['composition_RMS_safe_downstream','within_RMS_safe_downstream','inferred_partition_total_variation','supported_target_fraction','target_unknown_partition_fraction','NTC_common_state_support_fraction_target_batch_weighted','difference_from_full_matched_effect_RMS_safe_downstream','common_support_vs_full_effect_correlation','target_RNA_ratio','matched_target_cells']
    (output/'endpoint.html').write_text(explorer_page('终点组成与状态内表达：全部 75,690 个任务分层',endpoint_rows,endpoint_display,endpoint_metrics,
        '两种预定分层分别计算。共同支持集至少 10 个目标细胞，每个目标出现的状态×技术层至少 2 个 NTC。未知类型独立保留。原始全匹配效应与共同支持效应不同，其差异单独报告；分量不是校正后数据。','baselines/all-endpoint-task-partitions.parquet',{'component_file':''}))
    (output/'baselines.html').write_text(explorer_page('登记来源与官方 A/B/C 的 NTC 基线',baseline_pairs,
        ['panel_A','panel_B','NTC_A','NTC_B','status','native_genes','native_correlation','native_difference_RMS','official_correlation','official_difference_RMS'],
        ['native_correlation','native_difference_RMS','official_correlation','official_difference_RMS','NTC_A','NTC_B','native_genes'],
        '登记 CRISPRi 背景、官方 A/B/C 及其他已验证人 RNA negative-guide 背景；后者不增加主 CRISPRi 响应任务。H1 完全相同的 NTC 只计一次。intergenic cutting、vehicle、untreated 和 scBase 混合终点不冒充 NTC；NTC 差异不是纯技术噪声。','baselines/all-baseline-pairs.parquet',{'common_gene_file':'baselines/'}))
    (output/'tasks.html').write_text(explorer_page('全部 37,845 个任务的干预与测量敏感性',task_view,
        ['canonical_target','panel_id','source_condition','effect_status','matched_target_cells','target_RNA_ratio','downstream_RMS','library_size_median','guide_median_correlation','depth_effect_correlation','resampling_median_correlation'],
        ['target_RNA_ratio','downstream_RMS','matched_target_cells','library_size_median','detected_genes_median','guide_median_correlation','technical_stratum_median_correlation','depth_effect_correlation','resampling_median_correlation'],
        'RNA 比值是原生 CP10K 分母上的转录代理，不能当作蛋白活性或每个细胞的有效剂量。guide/批次差异保留为可观察关联；细胞重采样不是生物重复置信区间。完整原生响应的路径、哈希与来源发布包见记录。','task-evidence-index.parquet',{'source_summary_file':'','source_release':''}))
    (output/'tahoe-repeats.html').write_text(explorer_page('Tahoe plate 6 / 14：全部 4,738 对作者声明重复',repeat,
        ['cell_line_id','complete_drug_dose_unit_key','is_vehicle','status','plate6_cells','plate14_cells','mean_logRNA_correlation','response_correlation','response_difference_RMS'],
        ['mean_logRNA_correlation','response_correlation','mean_logRNA_difference_RMS','response_difference_RMS','plate6_cells','plate14_cells','plate6_DMSO_cells','plate14_DMSO_cells'],
        '作者声明两个培养重复；并未独立核实实验随机化。保留全部药物和 vehicle 比较。方差为零时相关不可估计，不能补零；绝对表达与响应的汇总使用双方都有值的同一组记录。','supplements/'+repeat_path))
    assoc=pd.read_parquet(output/'supplements/descriptive-factor-associations.parquet')
    (output/'associations.html').write_text(explorer_page('来源内描述性关联及不可估计项',assoc,
        ['scope','factor','outcome','observations','complete_observations','missing_observations','status','reason','Spearman_rho'],
        ['complete_observations','Spearman_rho','observations','missing_observations'],
        '每个来源任务背景或来源面板对单独计算。至少 10 个双有效观察且两变量都有变化；不提供生物学 p 值，不把来源相关性当作独立因素的因果效应。scope 可在同包 source-context-summaries / pair-group-summaries 中查到原始背景。','supplements/descriptive-factor-associations.parquet'))
    factors=[
      {'factor':'遗传、谱系与细胞身份','observed':'来源细胞系、研究／capture 身份；保守 RNA inferred_type','unresolved':'逐细胞突变、CNV、倍性、亚克隆、传代和染色质状态未由本评估测得','interpretation':'同名细胞系不是同一个响应系统；类型推断不是 ground truth'},
      {'factor':'分子与生理状态','observed':'来源 NTC 基线、12 个冻结 RNA 代理及终点周期分层','unresolved':'干预前状态、蛋白定位／活性、真实周期阶段及代谢通量','interpretation':'同层描述不构成因果中介；终点 score 不能作预干预输入'},
      {'factor':'靶点与系统依赖','observed':'靶基因 RNA、全基因响应、独立记录的静态先验与 DepMap fitness','unresolved':'有效蛋白剩余量、功能冗余与机制在当前单细胞中的真值','interpretation':'fitness、网络和表达代理不等价于 CRISPRi RNA 响应'},
      {'factor':'干预实现与强度','observed':'来源 guide／构件／转录本、效应器、RNA 比值、全部 guide 阈值敏感性','unresolved':'逐细胞有效蛋白剂量、脱靶贡献、构件间共同随机化','interpretation':'保留构件与 gene 身份；guide 差异不预设为噪声'},
      {'factor':'时间与历史','observed':'Replogle 6/7/8 天、Nadig 7 天及来源刺激／药物窗口；完整时间条件附录','unresolved':'有效敲低起点、蛋白周转、传代／恢复／既往处理历史','interpretation':'时间与 library、细胞系、效应器及培养完全混杂时不可辨识'},
      {'factor':'环境与细胞相互作用','observed':'Jiang line×stimulus、GxE line×drug×dose、来源培养和 mixed-culture 声明','unresolved':'逐细胞局部营养、密度、旁分泌、培养批次的实际变化','interpretation':'各自来源内参考成立的条件描述；不是独立环境因果效应'},
      {'factor':'存活、捕获与组成选择','observed':'观察细胞数、原始 barcode/CDS 支持、类型比例、共同状态支持损失','unresolved':'未捕获／死亡细胞和干预前群体组成，缺失机制','interpretation':'观察终点组成变化不能区分状态转换、增殖、死亡或捕获'},
      {'factor':'测量与计算处理','observed':'原生基因 panel、深度、检测、thinning、已证实 capture 的重处理差异','unresolved':'纯技术方差、跨平台绝对 RNA 校准、完全交叉重复设计','interpretation':'安全共同 panel 不消除原生 CP10K 分母差异；不补零或校正'}]
    decisions=[
      {'decision':'按干预身份、来源背景与构件保留证据','basis':'全部 37,845 任务和 249,071 对；已确认合集副本不增加实验分母；H1 NTC 基线合并仅依据完全相同的源记录证明','next_use':'后续使用时沿用 task_uid、源身份、capture 及原生基因轴；未证明独立的记录不自动作重复'},
      {'decision':'分别检查基线、绝对表达、相对对照响应','basis':joint[0],'next_use':'优先按具体 target×背景查看效应与支持；全局中位数包含弱响应和依赖记录，不作来源质量排行榜'},
      {'decision':'把组成／状态内表达作为条件描述','basis':{'status_counts':base['endpoint_status_counts'],'source_effects_reproduced':base['source_effects_reproduced'],'maximum_identity_residual':base['maximum_mixture_identity_residual']},'next_use':'同时查看 unknown 与目标／NTC 支持损失、共同支持总差和原始全匹配效应；不把分量当成校正后的监督目标'},
      {'decision':'对测量与实际干预强度保留敏感性','basis':{'guide_threshold_rows':supp['GxE2_threshold_rows'],'associations':supp['association_status_counts'],'source_evidence_files':supp['source_evidence_files']},'next_use':'保留全部阈值与未校正源计数；RNA 比值和深度关联不能分离实际敲低、总 RNA、存活与捕获'},
      {'decision':'化学与作者重复保持各自参考设计','basis':joint[3],'next_use':'Tahoe 两个作者声明重复可以说明绝对表达相似并不保证响应相似；不能跨药物／遗传／时间设计混成同一种重复'},
      {'decision':'无法辨识的因素保留未知并补来源证据','basis':'八类候选因素的观测边界；完全混杂无随机化／重复设计支持','next_use':'未解决的具体来源证据继续跟踪 #22–#27；#21 和父 #1 的最终综合验收独立管理'}]
    write_json(output/'factor-observability.json',factors);write_json(output/'decisions.json',decisions)
    code_dir=output/'reproduction';code_dir.mkdir()
    code_names=['profile_response_heterogeneity.py','heterogeneity.py','heterogeneity_cells.py','profile_endpoint_decomposition.py','profile_heterogeneity_supplements.py','profile_extra_ntc_baselines.py','profile_heterogeneity_baselines.py','compose_heterogeneity_dossier.py','render_heterogeneity.py','reproduce_heterogeneity.py','pyproject.toml','uv.lock']
    for name in code_names:shutil.copy2(Path(__file__).with_name(name),code_dir/name)
    source_identity={name:hash_file(path/'report.json') for name,path in paths.items()}
    methods={'registration':'https://github.com/yjcyxky/virtual-cell-challenge/issues/20#issuecomment-5746896397',
        'comparison':primary.get('reproduce'),'endpoint_parameters':ep['identity']['parameters'],
        'scale':'native mean cell log1p(CP10K); no common-panel renormalization, missing-zero filling, regression, or correction',
        'gene_identity':'unique conflict-free canonical intersections; exclude canonical intervention and source is_target; at least 100 common finite genes; zero variance correlation null',
        'source_effect_reproduction':'every one of 37633 completed native source target means, matched NTC means and effect vectors was reproduced from verified counts or its full-hash normalized cache',
        'composition_identity':'within each source technical stratum, retained-target-weighted symmetric composition + within = common-support total; source full effect difference retained',
        'scope':'all 62675 source tasks have applicability states; 37845 qualified primary tasks; 10198 canonical targets and 4163 with pairs; two endpoint partitions per task',
        'analysis_exposure':'all registered source RNA counts, source intervention labels, source H1 Training/Validation/Test response labels, official A/B/C control RNA, frozen type/state references and previous source results; not a blinded evaluation',
        'future_availability':'endpoint inferred type and cycle are post-measurement; matched NTC requires available same-context control. No claim these covariates are available before a future perturbation',
        'reference_weights':'original source-specific type/state weights and input identities remain in pinned source releases; new per-cell partition labels, NTC cycle boundaries and exact source cells in endpoint subdirectories',
        'inference':'source labels, observations and inferred annotations are separate; probability_correct stays null, uncalibrated; no mechanism ground truth',
        'reproduce':['micromamba','run','-n','virtual-cell','uv','run','--project','scripts/dossier','--locked','python','scripts/dossier/reproduce_heterogeneity.py','--source',str(paths['source'].relative_to(ROOT)),'--work','data/assessments/heterogeneity-reproduction-FRESH','--output','data/assessments/heterogeneity-reproduced-FRESH'],
        'prerequisites':'original repository at recorded commits and exact pinned source dossiers/count caches; the source paths and full hashes are in final-input-verification.json; locked uv environment, source files read-only'}
    report={'schema_version':2,'bundle_id':'response-heterogeneity-'+uuid.uuid4().hex,'title':'跨背景响应异质性与测量敏感性：只读完整评估','status':'completed',
        'summary':{'all_source_tasks':62675,'identity_qualified_tasks':len(tasks),'same_target_pairs':len(pairs),'pair_status_counts':pairs.status.value_counts().to_dict(),
            'canonical_targets':len(targets),'targets_with_pairs':int(targets.all_same_target_pairs.gt(0).sum()),'endpoint_task_partitions':len(endpoint_rows),
            'endpoint_status_counts':endpoint_rows.status.value_counts().to_dict(),'baseline_pairs':len(baseline_pairs),'source_effects_reproduced':37633,'unestimable_source_tasks_retained':212,
            'frozen_inputs_reverified':len(consumed),'raw_or_count_views_reverified':len(raw),'source_input_mutations':0,'new_unified_training_corpus':False,'prediction_model_trained':False},
        'methods':methods,'source_report_identities':source_identity,'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'uv_lock_sha256':hash_file(Path(__file__).with_name('uv.lock')),'completed_at':datetime.now(timezone.utc).isoformat(),
        'limitations':['终点类型／RNA 状态未校准；unknown 和不可估计保留，不作为 ground truth。','全部比较为描述性；组成与状态内分量不是因果中介或可加的方差占比。',
            '来源时间、平台、效应器、培养和细胞背景缺少完全交叉重复设计，无法独立分解因果贡献。','NTC 差异并非纯测量噪声；观测终点不能识别死亡、增殖、状态转换或捕获的独立贡献。',
            '相同 canonical target 保留不同构件；共同基因交集不改变源 CP10K 分母；对照及 target 复用使比较对相互依赖。',
            '未运行或失败的中间目录不属于本结果包。全部公开标签已用于评估，不声称训练／测试未暴露。'],
        'tables':[{'title':'完整范围与状态','rows':[{'measure':k,'value':v} for k,v in {'primary':{k:primary[k] for k in ['identity_qualified_tasks','all_same_target_pairs','pair_status_counts']},'endpoint':{k:base[k] for k in ['endpoint_task_partitions','endpoint_status_counts','maximum_source_mean_reproduction_error','maximum_mixture_identity_residual']},'baseline':{k:base[k] for k in ['baseline_scope','baseline_context_rows','distinct_source_NTC_baselines','baseline_pairs','baseline_status_counts']}}.items()]},
            {'title':'数据使用决策与依据','rows':decisions},{'title':'八类异质性来源的观测边界','rows':factors},
            {'title':'同一组有效记录上的尺度比较','rows':joint},{'title':'来源实验设计分组','rows':design_summaries},
            {'title':'各 endpoint 背景的组成／状态内支持与差异','rows':json.loads((output/'baselines/endpoint-context-summaries.json').read_text())},
            {'title':'全部来源内任务敏感性汇总','rows':json.loads((output/'supplements/source-context-summaries.json').read_text())},
            {'title':'全部来源附录与证据','rows':[{**r,'file':'supplements/'+r['file']} for r in evidence]}],
        'reproduce':sys.argv,'external_behavior_verification':'see linked Issue #20 closure evidence; final offline viewers verified on this bundle after composition'}
    views=['responses.html','endpoint.html','baselines.html','tasks.html','tahoe-repeats.html','associations.html']
    artifacts=[{'file':str(p.relative_to(output)),'sha256':hash_file(p)} for p in sorted(output.rglob('*')) if p.is_file()]
    report['artifacts']=sorted(artifacts,key=lambda r:(0 if r['file'] in views else 1,r['file']))
    write_json(output/'report.json',report);(output/'report.html').write_text(render(report))
    (output/'SHA256SUMS').write_text(''.join(hash_file(p)+'  '+str(p.relative_to(output))+'\n' for p in sorted(output.rglob('*')) if p.is_file() and p.name!='SHA256SUMS'))
    print(json.dumps({'bundle_id':report['bundle_id'],'status':'completed','summary':report['summary']}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['source','comparisons','endpoint','supplements','baselines','extra','output']:p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();compose(a.source,a.comparisons,a.endpoint,a.supplements,a.baselines,a.extra,a.output)
