#!/usr/bin/env python
"""Freeze the complete GxE2 barcode, capture, genetic and chemical assessment."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
import uuid
import pandas as pd
from rna import hash_file,quantiles
from profile_responses import write_json,serial
from render import render


def copy_phase(source,destination,report_name='report.json',selected=None):
    report=json.loads((source/report_name).read_text())
    if report['status']!='completed':raise ValueError('completed_component_required')
    artifacts=report.get('artifacts')
    if artifacts is None and report.get('phase')=='all_fixed_source_genotype_chemical_exposure_descriptions':
        artifacts={'identity.json':hash_file(source/'identity.json')}
        for row in report['conditions']:
            prefix=row['directory'];checkpoint=json.loads((source/prefix/'identity.json').read_text())
            if checkpoint['identity']!=report['identity'] or checkpoint['summary']!=row:raise ValueError('chemical_checkpoint_scope_changed')
            artifacts[prefix+'/identity.json']=hash_file(source/prefix/'identity.json')
            artifacts.update({prefix+'/'+name:digest for name,digest in checkpoint['artifacts'].items()})
    if artifacts is None:raise ValueError('component_artifact_manifest_required')
    if isinstance(artifacts,list):artifacts={a['file']:a['sha256'] for a in artifacts}
    destination.mkdir(parents=True)
    included={};omitted={}
    for name,digest in artifacts.items():
        if Path(name).is_absolute() or '..' in Path(name).parts:raise ValueError('unsafe_component_path')
        if selected is not None and not selected(name):omitted[name]=digest;continue
        if hash_file(source/name)!=digest:raise ValueError('source_component_changed:'+name)
        target=destination/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source/name,target)
        if hash_file(target)!=digest:raise ValueError('component_copy_changed')
        included[name]=digest
    shutil.copy2(source/report_name,destination/report_name)
    if omitted:write_json(destination/'omitted-rebuildable-cache.json',{'interpretation':'Adapter working counts and text exports can be regenerated from immutable original inputs; all omitted hashes remain in the original adapter identity.',
        'artifacts':omitted,'included':included})
    return report


def compose(adapter,records,capture,candidates,genetic,chemical,references,evidence,output):
    if output.exists():raise ValueError('fresh_dossier_required')
    output.mkdir(parents=True)
    ar=copy_phase(adapter,output/'adapter','identity.json',lambda n:n.endswith('.parquet') or n.endswith('.log'))
    rr=copy_phase(records,output/'records');cr=copy_phase(capture,output/'capture');nr=copy_phase(candidates,output/'candidate-annotations')
    gr=copy_phase(genetic,output/'genetic');er=copy_phase(chemical,output/'chemical')
    if rr['source_identity_sha256']!=hash_file(adapter/'identity.json') or cr['adapter_identity_sha256']!=hash_file(adapter/'identity.json'):raise ValueError('adapter_identity_not_shared')
    if gr['identity']['records_report_sha256']!=hash_file(records/'report.json') or gr['identity']['capture_report_sha256']!=hash_file(capture/'report.json'):raise ValueError('genetic_phase_identity_not_shared')
    if er['identity']['genetic_report_sha256']!=hash_file(genetic/'report.json') or nr['raw_source_record_report_sha256']!=hash_file(records/'report.json'):raise ValueError('downstream_component_identity_not_shared')
    if gr['actual_tasks']!=14121 or er['genotype_tasks']!=12576 or gr['n_cells']!=989299 or nr['n_candidates']!=62906:raise ValueError('incomplete_declared_GxE2_scope')
    if gr['null_reference_intersections'] or er['null_reference_intersections']:raise ValueError('overlapping_null_reference')
    for line in (references/'SHA256SUMS').read_text().splitlines():
        digest,name=line.split('  ',1)
        if hash_file(references/name)!=digest:raise ValueError('reference_changed')
    shutil.copytree(references,output/'references')
    for i,folder in enumerate(evidence):shutil.copytree(folder,output/'primary-evidence'/str(i))
    contexts=[];tasks=[];assignment_counts={};chemical_rows=[]
    for row in gr['contexts']:
        folder=output/'genetic'/row['directory'];line,drug,dose=json.loads(row['context']);all_tasks=json.loads((folder/'tasks.json').read_text())
        contexts.append({'cell_line':line,'drug':drug,'dose_uM':dose,'hours':72,'source_CDS_cells':row['n_cells'],'primary_cells':row['n_primary_cells'],'NTC_cells':row['n_NTC'],
            'declared_targets':row['tasks'],'effects':row['task_status_counts'],'DE':row['DE_status_counts'],'resampling_rows':row['resampling_rows'],
            'unknown_types':row['pooled_types'].get('unknown',0),'RNA_ratio':quantiles([t['target_RNA']['RNA_ratio'] for t in all_tasks if t.get('target_RNA',{}).get('status')=='completed']),
            'response_RMS':quantiles([t['downstream_RMS'] for t in all_tasks if t.get('downstream_RMS') is not None]),
            'source_assignment_status':row['source_assignment_status'],'per_cell_sidecar':'genetic/'+row['directory']+'/cells.parquet'})
        for key,value in row['source_assignment_status'].items():assignment_counts[key]=assignment_counts.get(key,0)+value
        for t in all_tasks:
            tasks.append({'context':row['context'],'target':t['target'],'status':t['status'],'observed_primary_target_cells':t['observed_primary_target_cells'],
                'matched_primary_target_cells':t['matched_primary_target_cells'],'unmatched_primary_target_cells':t['unmatched_primary_target_cells'],
                'DE_status':t.get('DE',{}).get('status','not_estimable'),'downstream_RMS':t.get('downstream_RMS'),'DEG_BH':t.get('DEG_BH_excluding_target'),
                'DEG_BY':t.get('DEG_BY_excluding_target'),'target_RNA':t.get('target_RNA'),'stability':t.get('stability'),'guide_and_hash_well_consistency':t.get('consistency'),
                'read_threshold_sensitivity':t.get('source_assignment_sensitivity'),'gene_effect_file':'genetic/'+row['directory']+'/'+t['gene_results'] if t.get('gene_results') else None})
    for row in er['conditions']:
        folder=output/'chemical'/row['directory'];part=pd.read_parquet(folder/'diagnostics.parquet')
        part['gene_effect_file']='chemical/'+row['directory']+'/gene-effects.h5';chemical_rows.extend(part.to_dict('records'))
    pd.DataFrame(tasks).to_parquet(output/'all-genetic-tasks.parquet',index=False);pd.DataFrame(chemical_rows).to_parquet(output/'all-chemical-tasks.parquet',index=False)
    mapping=pd.read_parquet(output/'genetic/gene-mapping.parquet');mapping_rows=[{'status':str(k),'native_features':len(v)} for k,v in mapping.groupby('mapping_status',dropna=False)]
    records_rows=[{'record_role':k,'records':v,'interpretation':'Source applicability categories; not a new filter or evidence of established cells'} for k,v in rr['coverage'].items()]
    capture_rows=[{'capture':'guide','metric':k,'value':v} for k,v in cr['guide_capture'].items()]+[{'capture':'hash','metric':k,'value':v} for k,v in cr['hash_capture'].items()]
    capture_rows += [{'capture':'guide_source_reproduction','metric':k,'value':v} for k,v in cr['guide_check_disagreements'].items()]
    capture_rows += [{'capture':'hash_source_reproduction','metric':k,'value':v} for k,v in cr['hash_check_disagreements'].items()]
    observability=[
        {'factor':'genetic_and_lineage_background','observed':'A172, T98G, U87MG source hash identity and conservative RNA marker inference','not_identified':'Per-cell genome, copy-number, clone, chromatin and passage; cell line is not a calibrated type label'},
        {'factor':'intervention','observed':'522 gene targets plus random genomic regions, original guide calls and exact capture reproduction checks; primary maximum guide reads >=10 and thresholds 1..10','not_identified':'Guide reads and target RNA are not effective protein dose; five PRKCZ guide names each have two source sequences (#26)'},
        {'factor':'culture','observed':'DMEM, 10% FBS, 1% penicillin/streptomycin; dCas9-BFP-KRAB enrichment, MOI 0.1, 72 h then puromycin 1 ug/mL; 10–14 d expansion','not_identified':'Independent culture units, passage, batch history, per-cell environment'},
        {'factor':'time_and_chemical_environment','observed':'25,000 cells/well in 100 uL, 24 h attachment; 72 h lapatinib/nintedanib/trametinib/zstk474 at 1 or 10 uM, vehicle 0.1% DMSO','not_identified':'Longitudinal trajectory or isolated time modifier; source replicate labels are not independently verified biological replication'},
        {'factor':'molecular_state','observed':'12 RNA proxies, type candidates, confidence conflicts and unknown; all CDS and non-CDS >=500 UMI candidates','not_identified':'Actual pathway/protein activity, effective knockdown or pre-intervention states of the same cells'},
        {'factor':'composition_and_selection','observed':'Endpoint inferred type proportions, state distributions and within-type differences; eligibility of every raw barcode record','not_identified':'Death, proliferation, transition and capture mechanisms cannot be causally separated'},
        {'factor':'measurement_and_assignment','observed':'All 43,209,765 barcode count records and all guide/hash capture records; source CDS counts exactly reproduced; fixed depth/read-threshold diagnostics','not_identified':'NTC variation is not pure measurement noise; 51,026 source-unassigned guide discrepancies remain unassigned (#26)'},
        {'factor':'cross_drug_comparison','observed':'12,576 fixed-source-genotype drug-versus-vehicle descriptions; shared vehicle-NTC state weights and guide/replicate matched state sensitivity','not_identified':'Guide mixture, culture well and endpoint selection still confounded; pooled RNA difference is not a pure drug causal effect'}]
    decisions=[{'use':'Conditional genetic RNA responses','recommendation':'Candidate within source line/drug/dose, guide/hash-well eligibility and native measured panel; keep not_estimable tasks and all exclusions indexed'},
        {'use':'Drug response at fixed source genotype','recommendation':'Use as explicitly conditional descriptive evidence; inspect guide composition and matched-state sensitivity; do not call it a pure drug effect'},
        {'use':'Cell-state or type labels','recommendation':'Keep methods/references, conflicts, unknown and probability=null; not ground truth or pre-intervention features'},
        {'use':'Raw barcode universe','recommendation':'43 million count rows are not 43 million established cells; distinguish 989,299 source CDS cells, 62,906 extra source-threshold candidates and remaining barcode records'},
        {'use':'Guide source ambiguities','recommendation':'Track #26; retain author unassigned labels and sequence-name ambiguity without rewriting source data'}]
    write_json(output/'observability.json',observability);write_json(output/'decisions.json',decisions)
    report={'schema_version':2,'bundle_id':'gxe2-dossier-'+uuid.uuid4().hex,'title':'McFaline GxE2 激酶组全量只读评估','status':'completed','completed_at':datetime.now(timezone.utc).isoformat(),
        'source_id':'McFalineGxE2:GSM7056149','input_sha256':ar['input_sha256'],'inputs_unchanged':gr['inputs_unchanged'] and rr['inputs_unchanged'] and cr['inputs_unchanged'],
        'coverage':rr['coverage'],'source_assignment_counts':assignment_counts,'source_CDS_type_counts':gr['pooled_types'],'additional_candidate_type_counts':nr['pooled_types'],
        'genetic_tasks':gr['actual_tasks'],'genetic_task_status_counts':gr['task_status_counts'],'genetic_DE_status_counts':gr['DE_status_counts'],
        'genetic_resampling_rows':gr['resampling_rows'],'genetic_null_size_shortfall_rows':gr['null_size_shortfall_rows'],
        'chemical_tasks':er['genotype_tasks'],'chemical_status_counts':er['task_status_counts'],'chemical_resampling_rows':er['resampling_rows'],
        'chemical_null_size_shortfall_rows':er['null_size_shortfall_rows'],'null_reference_intersections':0,
        'component_reports':{name:hash_file(output/name/file) for name,file in [('adapter','identity.json'),('records','report.json'),('capture','report.json'),('candidate-annotations','report.json'),('genetic','report.json'),('chemical','report.json')]},
        'methods':{'protocol_issue':'https://github.com/yjcyxky/virtual-cell-challenge/issues/11','evidence_issue':'https://github.com/yjcyxky/virtual-cell-challenge/issues/26',
            'genetic':'All 27 x 523 tasks; source-supported single target with maximum guide reads >=10; same hash well/source replicate NTC, target-composition weighting; complete native genes, conditional cell-level DE BH/BY, target RNA, 20 resamples, expected-depth 10k thinning and guide/stratum consistency',
            'chemical':'All 24 x 524 genotype tasks; same cell line and source genotype vehicle; pooled full-gene difference explicitly retains guide/well/selection confounding; shared vehicle NTC proxy weights; guide/replicate matched state sensitivity',
            'annotations':'Frozen conservative rank/expression marker agreement, unknown and target-marker sensitivity; all probability_correct null; 12 expression-matched RNA proxies',
            'adapter':'Read-only base-R S4 slot export plus all-coordinate audit and exact per-CDS count equality; rebuildable adapter working matrices omitted from archive and identified by original hashes'},
        'references':['https://pmc.ncbi.nlm.nih.gov/articles/PMC10879025/','https://doi.org/10.1016/j.xgen.2023.100487','https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE225775'],
        'exposure':'All downloaded RNA/capture records and all declared tasks examined. No untouched validation, corrected training matrix, or trained predictive model is produced.',
        'limitations':['43,209,765个原始barcode记录不是同等数量的已建立细胞；所有记录保留身份、数值QC和适用性。',
            '所有类型/状态均为未校准RNA推断，unknown保留，概率为空，不作ground truth。',
            '细胞、guide、hash well和source replicate标签不自动等于独立生物重复；DE与重采样仅为条件细胞抽样诊断。',
            'NTC池不足的空参照保留实际臂大小及短缺；不能据此声称等样本量精确显著性。',
            '51,026个作者未分配guide的细胞与公开调用规则不同；不补标签。PRKCZ名称/序列歧义另见#26。',
            '药物比较固定的是来源推定genotype；guide组成、培养well、存活/捕获选择仍可能混杂。'],
        'tables':[{'title':'全部原始记录的适用范围','rows':records_rows},{'title':'全部27个遗传背景','rows':contexts},
            {'title':'全部14,121个遗传任务','rows':tasks},{'title':'全部12,576个固定genotype药物任务','rows':chemical_rows},
            {'title':'全量guide和hash核对','rows':capture_rows},{'title':'原生基因身份映射','rows':mapping_rows},
            {'title':'异质性可观测性','rows':observability},{'title':'数据使用建议','rows':decisions}],
        'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'code_sha256':hash_file(Path(__file__)),'reproduce':sys.argv}
    report['artifacts']=[{'file':str(p.relative_to(output)),'sha256':hash_file(p)} for p in sorted(output.rglob('*')) if p.is_file()]
    write_json(output/'report.json',report);(output/'report.html').write_text(render(serial(report)))
    (output/'SHA256SUMS').write_text(''.join(hash_file(p)+'  '+str(p.relative_to(output))+'\n' for p in sorted(output.rglob('*')) if p.is_file() and p.name!='SHA256SUMS'))
    print(json.dumps({k:v for k,v in report.items() if k in ['bundle_id','status','genetic_tasks','chemical_tasks','genetic_task_status_counts','genetic_DE_status_counts','chemical_status_counts']}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['adapter','records','capture','candidates','genetic','chemical','references','output']:p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--evidence',type=Path,nargs='+',required=True);a=p.parse_args();compose(a.adapter,a.records,a.capture,a.candidates,a.genetic,a.chemical,a.references,a.evidence,a.output)
