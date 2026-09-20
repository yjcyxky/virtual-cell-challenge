#!/usr/bin/env python
"""Freeze all scPerturb RNA inference and response outcomes without sampling files."""
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
import pyarrow.parquet as pq
from profile_responses import write_json,serial
from publish_assessment import verified_files
from rna import hash_file
from render import render


def checked_copy(source,destination,artifacts):
    destination.mkdir(parents=True,exist_ok=True)
    if isinstance(artifacts,list):artifacts={r['file']:r['sha256'] for r in artifacts}
    for name,digest in artifacts.items():
        if Path(name).is_absolute() or '..' in Path(name).parts or (source/name).is_symlink():raise ValueError('unsafe_component_path')
        if hash_file(source/name)!=digest:raise ValueError('component_artifact_changed:'+name)
        target=destination/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source/name,target)
        if hash_file(target)!=digest:raise ValueError('copied_component_changed')


def annotation_contract(folder,summary):
    names=pq.read_schema(folder/'cells.parquet').names
    columns=['row_index','record_id','input_sha256','probability_correct','confidence_calibration','inferred_type']
    cells=pd.read_parquet(folder/'cells.parquet',columns=columns)
    if len(cells)!=summary['cells'] or not np.array_equal(cells.row_index,np.arange(len(cells))) or cells.record_id.duplicated().any():raise ValueError('cell_identity_scope_mismatch')
    if not (cells.input_sha256==summary['input_sha256']).all() or cells.probability_correct.notna().any() or not (cells.confidence_calibration=='uncalibrated').all():raise ValueError('cell_inference_contract_changed')
    if cells.inferred_type.value_counts().to_dict()!=summary['pooled_types'] or len([n for n in names if n.startswith('state__')])!=12:raise ValueError('cell_annotation_coverage_mismatch')
    if not any(n.startswith('source_obs__') for n in names):raise ValueError('source_metadata_missing')
    return {'source_file':summary['file'],'records':len(cells),'all_source_rows_indexed':True,'unique_record_ids':True,
        'source_obs_fields_separate':True,'all_12_state_columns':True,'uncalibrated_probability_null':True}


def task_row(file,task):
    return {'source_file':file,'task':task.get('task'),'target':task.get('target',task.get('target_gene')),
        'source_context':task.get('context'),'source_transcript':task.get('source_transcript'),'source_guide':task.get('source_guide_id'),
        'biological_background_index':task.get('biological_background_index'),'target_kind':task.get('target_kind','original_source_construct'),
        'status':task['status'],'reason':task.get('reason'),'DE_status':task.get('DE',{}).get('status','not_estimable'),
        'source_candidate_cells':task.get('n_source_candidate_cells',task.get('n_cells')),'matched_cells':task.get('n_matched_target_cells',task.get('n_cells')),
        'downstream_RMS':task.get('downstream_RMS'),'DEG_BH':task.get('DEG_BH_excluding_target'),'DEG_BY':task.get('DEG_BY_excluding_target'),
        'target_RNA':task.get('target_RNA'),'stability':task.get('stability'),'consistency':task.get('consistency'),
        'gene_results':task.get('gene_results'),'external_response_evidence':task.get('external_response_evidence')}


def compose(cells,design,response,audit,adamson,human_reference,mouse_reference,evidence,output):
    c=json.loads((cells/'report.json').read_text());d=json.loads((design/'report.json').read_text());r=json.loads((response/'report.json').read_text())
    if any(x['status']!='completed' for x in [c,d,r]) or r['actual_RNA_files']!=51 or r['actual_RNA_records']!=8802191:raise ValueError('all_51_RNA_files_required')
    if r['identity']['design_report_sha256']!=hash_file(design/'report.json') or r['identity']['cell_phase_report_sha256']!=hash_file(cells/'report.json'):raise ValueError('response_input_identity_changed')
    if c['identity']['manifest_sha256']!=hash_file(audit/'analysis_manifest.json') or d['identity']['manifest_sha256']!=c['identity']['manifest_sha256']:raise ValueError('modality_manifest_changed')
    if d['identity']['Adamson_identity_report_sha256']!=hash_file(adamson/'report.json'):raise ValueError('original_Adamson_identity_changed')
    for prefix,ref in [('human',human_reference),('mouse',mouse_reference)]:
        if c['identity'][prefix+'_reference_sha256']!=hash_file(ref/'gene_sets.json'):raise ValueError('inference_reference_changed')
    output.mkdir(parents=True,exist_ok=False)
    for phase,source in [('cells',cells),('design',design),('response',response)]:
        (output/phase).mkdir();shutil.copy2(source/'report.json',output/phase/'report.json')
        if (source/'identity.json').exists():shutil.copy2(source/'identity.json',output/phase/'identity.json')
    checked_copy(audit,output/'source-audit',{n:hash_file(audit/n) for n in verified_files(audit)})
    ar=json.loads((adamson/'report.json').read_text());checked_copy(adamson,output/'Adamson-original-identities',ar['artifacts']);shutil.copy2(adamson/'report.json',output/'Adamson-original-identities/report.json')
    for prefix,ref in [('human',human_reference),('mouse',mouse_reference)]:
        hashes={name:digest for digest,name in [line.split('  ',1) for line in (ref/'SHA256SUMS').read_text().splitlines()]}
        checked_copy(ref,output/(prefix+'-references'),hashes);shutil.copy2(ref/'SHA256SUMS',output/(prefix+'-references')/'SHA256SUMS')
    for index,folder in enumerate(evidence):
        included={};omitted={}
        for p in folder.rglob('*'):
            if p.is_file():
                (omitted if p.name.endswith('_matrix.mtx.txt.gz') else included)[str(p.relative_to(folder))]=hash_file(p)
        target=output/'primary-evidence'/str(index);checked_copy(folder,target,included)
        if omitted:write_json(target/'omitted-original-matrices.json',{'reason':'Original downloaded matrix inputs omitted from result archive; retrieval URLs and hashes retained; all original count comparisons included.','artifacts':omitted})
    cell_index={v['file']:v for v in c['files']};design_index={v['file']:v for v in d['files']}
    overview=[];all_tasks=[];contracts=[];applicability=[];description_counts=Counter();task_counts=Counter();de_counts=Counter();total_conditions=0;copied_records=0
    resampling={name:Counter() for name in ['condition_descriptions','new_deep_responses','reused_original_deep_responses']}
    for record in r['files']:
        file=record['file'];name=file.removesuffix('.h5ad');dr=design_index[file]
        if record['status']=='not_applicable':overview.append({**record,'inference':'not_applicable_protein_not_RNA'});continue
        summary=cell_index[file];source=cells/name;cr=json.loads((source/'report.json').read_text())
        if cr['identity']!=c['identity'] or cr['summary']!=summary or not summary['inputs_unchanged']:raise ValueError('cell_component_scope_changed')
        if record['source_cell_report_sha256']!=hash_file(source/'report.json') or record['source_design_report_sha256']!=hash_file(design/name/'report.json'):raise ValueError('file_response_inputs_changed')
        checked_copy(source,output/'cells'/name,cr['artifact_hashes'])
        for report_name in ['report.json','report.html']:shutil.copy2(source/report_name,output/'cells'/name/report_name)
        contracts.append(annotation_contract(source,summary))
        checked_copy(design/name,output/'design'/name,dr['artifacts']);shutil.copy2(design/name/'report.json',output/'design'/name/'report.json')
        rr=json.loads((response/name/'report.json').read_text())
        if rr!=record:raise ValueError('response_component_changed')
        expected=r['identity'];old=expected.get('reused_completed_components',{})
        if name in old.get('completed_file_reports',{}):
            if hash_file(response/name/'report.json')!=old['completed_file_reports'][name]:raise ValueError('reused_file_report_changed')
            expected=old['source_identity']
        if rr['identity']!=expected:raise ValueError('response_component_code_identity_changed')
        checked_copy(response/name,output/'response'/name,rr['artifacts']);shutil.copy2(response/name/'report.json',output/'response'/name/'report.json')
        descriptions=pd.read_parquet(response/name/'descriptions/condition-contrasts.parquet')
        if len(descriptions)!=dr['all_source_conditions'] or len(descriptions)!=record['description']['all_conditions'] or descriptions.n_cells.sum()!=summary['cells']:raise ValueError('all_condition_scope_mismatch')
        if descriptions.status.value_counts().to_dict()!=record['description']['contrast_status_counts']:raise ValueError('condition_status_mismatch')
        if record['description']['null_reference_intersections'] or record['deep_response'].get('null_reference_intersections',0):raise ValueError('overlapping_target_reference')
        counts=descriptions.groupby(['status','reason'],dropna=False).agg(conditions=('condition_id','size'),source_records=('n_cells','sum')).reset_index().to_dict('records')
        applicability.extend([{'source_file':file,**row} for row in counts]);description_counts.update(record['description']['contrast_status_counts']);total_conditions+=len(descriptions)
        path=response/name/'deep-response/tasks.json';tasks=json.loads(path.read_text()) if path.exists() else []
        if len(tasks)!=record['deep_response']['actual_tasks']:raise ValueError('deep_task_scope_mismatch')
        task_counts.update(t['status'] for t in tasks);de_counts.update(t.get('DE',{}).get('status','not_estimable') for t in tasks)
        rows=[task_row(file,t) for t in tasks];all_tasks.extend(rows)
        copied=record['deep_response'].get('execution_mode') is not None
        if copied:copied_records+=summary['cells']
        deep_sampling=json.loads((response/name/'deep-response/original-response-report.json').read_text()) if copied else record['deep_response']
        for kind,values in [('condition_descriptions',record['description']),('reused_original_deep_responses' if copied else 'new_deep_responses',deep_sampling)]:
            for key in ['resampling_rows','null_reference_intersections','null_size_shortfall_rows']:resampling[kind][key]+=values.get(key,0)
            if values.get('null_reference_intersections',0):raise ValueError('reused_or_new_null_reference_overlap')
        methods=json.loads((design/name/'methods.json').read_text())
        row={'file':file,'status':'completed','species':summary['species'],'source_records':summary['cells'],'native_features':summary['genes'],
            'expression_scale':summary['expression_scale'],'all_condition_strata':len(descriptions),'eligible_reference_records':dr['eligible_controls'],
            'unsupported_condition_records':dr['unsupported_condition_cells'],'condition_status_counts':record['description']['contrast_status_counts'],
            'deep_tasks':len(tasks),'deep_task_status_counts':record['deep_response'].get('task_status_counts',{}),'DE_status_counts':record['deep_response'].get('DE_status_counts',{}),
            'exact_source_copy_verified':copied,'unknown_type_records':summary['pooled_types'].get('unknown',0),'source_input_sha256':summary['input_sha256'],
            'file_profile':'profiles/'+name+'.html','cell_records':'cells/'+name+'/cells.parquet','all_condition_results':'response/'+name+'/descriptions/condition-contrasts.parquet',
            'observed_distributions':'response/'+name+'/descriptions/all-condition-observed-distributions.parquet'}
        overview.append(row)
        # All eligible detailed contrasts are visible; the complete millions-row
        # applicability universe is losslessly retained in the linked sidecars.
        profile={'bundle_id':'scPerturb-profile-'+name,'title':file+' 独立 RNA 来源档案','status':'completed','completed_at':record['completed_at'],
            'methods':{'inference':cr['methods'],'design':methods,'response_identity':rr['identity'],'response':record['deep_response']},
            'limitations':cr['limitations']+['状态背景来自同来源生物条件的全部已观测终点，不是干预前状态；对照资格由单独的冻结设计决定。',
                '条件×匹配层不是独立实验。所有不可估计条件保留原数量；完整观测分位数、组成和状态见关联Parquet。'],
            'tables':[{'title':'文件范围','rows':[row]},{'title':'全部条件资格与原因','rows':counts},
                {'title':'全部达到门槛的描述性对照比较','rows':descriptions.loc[descriptions.status=='completed'].to_dict('records')},
                {'title':'全部单目标CRISPRi任务','rows':rows}],
            'artifacts':[{'file':'../'+row[k],'description':k} for k in ['cell_records','all_condition_results','observed_distributions']]+
                [{'file':'../cells/'+name+'/report.html','description':'完整逐细胞推断方法、状态与来源字段'},
                 {'file':'../response/'+name+'/report.json','description':'固定响应结果与完整 sidecar 哈希'}]}
        (output/'profiles').mkdir(exist_ok=True);(output/'profiles'/(name+'.html')).write_text(render(serial(profile)))
        print('scPerturb dossier files '+str(len(contracts))+'/51 '+file,flush=True)
    if len(contracts)!=51 or sum(v['records'] for v in contracts)!=8802191 or len(all_tasks)!=r['actual_deep_response_tasks']:raise ValueError('incomplete_final_scope')
    for name,rows in [('file-index',overview),('all-deep-tasks',all_tasks),('all-condition-status-denominators',applicability),('annotation-contract',contracts)]:
        write_json(output/(name+'.json'),rows)
    pd.DataFrame(all_tasks).to_parquet(output/'all-deep-tasks.parquet',index=False)
    decisions=[{'use':'CRISPRi响应','recommendation':'仅在来源干预、对照和匹配层证据支持的范围使用；原生测量panel保留，不把缺测基因补零。'},
        {'use':'其他RNA来源','recommendation':'分研究使用全部观测分布、状态和组成；CRISPRa/KO/Cas13/化学刺激不是CRISPRi。'},
        {'use':'细胞标签与状态','recommendation':'人/鼠固定参考分开；unknown、低覆盖、冲突和未验证连续尺度保留；推断不作ground truth。'},
        {'use':'Replogle/Nadig合集副本','recommendation':'逐记录计数、基因轴、barcode、construct、guide、target及batch完全一致后复用原始任务；不增加独立重复。'},
        {'use':'缺失或冲突证据','recommendation':'Adamson原始GEM/guide不一致、无验证阴性、Xie GFP、McFarland混合时间等见#27；Joung numeric ORF及Gehring剂量见#23。'}]
    report={'schema_version':2,'bundle_id':'scperturb-dossier-'+uuid.uuid4().hex,'title':'scPerturb 全51个RNA文件独立来源评估','status':'completed',
        'completed_at':datetime.now(timezone.utc).isoformat(),'RNA_files':51,'protein_files_not_applicable':3,'RNA_records':8802191,
        'all_source_condition_strata':total_conditions,'condition_status_counts':dict(description_counts),'deep_tasks':len(all_tasks),
        'deep_task_status_counts':dict(task_counts),'DE_status_counts':dict(de_counts),'verified_original_copy_records':copied_records,
        'resampling_by_evidence_role':{k:dict(v) for k,v in resampling.items()},
        'pooled_inferred_types':c['pooled_types'],'all_inputs_unchanged':True,'all_probability_correct_null':True,
        'component_report_sha256':{name:hash_file(folder/'report.json') for name,folder in [('cells',cells),('design',design),('response',response),('source-audit',audit),('Adamson-original-identities',adamson)]},
        'methods':{'inference':c['identity'],'response':r['identity'],'analysis_unit':'Source file, biological context and separately registered technical matching stratum; no cross-study control pooling',
            'source_count_semantics':'47 numerically count-compatible RNA matrices; 4 continuous matrices retain source transforms, no invented UMI meaning',
            'description':'Every observed source condition x matching stratum; all 12 state distributions, type proportions and expression totals; detailed same-background contrasts require 20 supported targets and 2 controls',
            'deep':'Every frozen single-target CRISPRi candidate; matched controls, full native gene axis, conditional cell-level BH/BY, target RNA, 20 attempted splits, depth thinning and guide/stratum consistency. Regulatory loci have no own-target RNA.',
            'reuse':'Exact original Replogle/Nadig records verified across count and intervention identity; full original construct tasks linked with release paths and hashes, including unmapped promoter constructs',
            'display':'All file and task outcomes; all qualified descriptive contrasts per-file. Complete condition eligibility and distributions in lossless Parquet; HTML pagination/filtering do not subsample evidence.'},
        'limitations':['8,802,191个RNA记录包含已证实的原始数据副本，不代表独立细胞或独立生物重复。',
            '来源标签、直接观测与推断字段分开；类型和12项状态代理不作真实活性、干预前特征或ground truth。',
            'NTC及其他参考变异不等于纯测量噪声；source batch/GEM/guide/replicate标签不自动证明独立培养。',
            '不可靠对照、来源标签不一致、未知连续尺度及原生panel缺测显式降级；不可估计不等于零响应。',
            'Adamson原始RNA计数完全匹配，但GEM后缀被移除后的部分guide注释不一致；保留合集原标签、排除相关不可靠任务记录，不回写纠正。',
            '全部数据已探索；没有保留未接触验证集，也没有生成校正后的训练矩阵。'],
        'exposure':'Every stored RNA record, metadata field, native feature and frozen candidate task examined; linked protein files remain modality-inapplicable for RNA inference.',
        'evidence_issues':['https://github.com/yjcyxky/virtual-cell-challenge/issues/23','https://github.com/yjcyxky/virtual-cell-challenge/issues/27'],
        'full_reproduction':{'entrypoint':'scripts/dossier/reproduce_scperturb.py','entrypoint_sha256':hash_file(Path(__file__).with_name('reproduce_scperturb.py')),
            'invocation':'OPENBLAS_NUM_THREADS=1 micromamba run -n virtual-cell uv run --project scripts/dossier --locked python scripts/dossier/reproduce_scperturb.py --audit AUDIT --human-reference HUMAN --mouse-reference MOUSE --adamson-evidence ORIGINAL_ADAMSON_EVIDENCE --additional-evidence OTHER_FROZEN_EVIDENCE --output FRESH_DIRECTORY',
            'requirements':'Original scPerturb H5AD files in data/raw/scperturb; full original Adamson matrices matching frozen retrieval manifests; frozen species references and #13 audit; #7/#8 original response and annotation components at their registered assessment paths.',
            'equivalence':'Full rebuild runs every file with the current equivalent eligibility implementation. Published interrupted-run components retain their original code identities; timestamps and serialization hashes need not match a rebuild.'},
        'tables':[{'title':'全部54个候选文件','columns':list(next(x for x in overview if x['status']=='completed'))+['inference','reason'],'rows':overview},
            {'title':'全部条件资格分母与原因','rows':applicability},{'title':'全部深度响应任务','rows':all_tasks},{'title':'数据使用建议','rows':decisions}],
        'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'code_sha256':hash_file(Path(__file__)),'reproduce':sys.argv}
    report['artifacts']=[{'file':str(p.relative_to(output)),'sha256':hash_file(p)} for p in sorted(output.rglob('*')) if p.is_file()]
    # Make each independent study profile directly reachable from the root page.
    report['artifacts'].sort(key=lambda x:(not x['file'].startswith('profiles/'),x['file']))
    write_json(output/'report.json',report);(output/'report.html').write_text(render(serial(report)))
    (output/'SHA256SUMS').write_text(''.join(hash_file(p)+'  '+str(p.relative_to(output))+'\n' for p in sorted(output.rglob('*')) if p.is_file() and p.name!='SHA256SUMS'))
    print(json.dumps({k:report[k] for k in ['bundle_id','status','RNA_files','RNA_records','deep_tasks','all_source_condition_strata','condition_status_counts','deep_task_status_counts']}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['cells','design','response','audit','adamson','human-reference','mouse-reference','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--evidence',type=Path,nargs='+',required=True);a=p.parse_args()
    compose(a.cells,a.design,a.response,a.audit,a.adamson,a.human_reference,a.mouse_reference,a.evidence,a.output)
