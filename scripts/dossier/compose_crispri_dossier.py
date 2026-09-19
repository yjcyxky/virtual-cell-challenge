#!/usr/bin/env python
"""Freeze a source dossier from verified complete structure/response/annotation bundles."""
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

SOURCES={
 'replogle2022':{
    'name':'Replogle K562/RPE1', 'issue':7,
    'references':['https://doi.org/10.1016/j.cell.2022.05.013','https://plus.figshare.com/articles/dataset/20029387','https://plus.figshare.com/articles/dataset/20022944'],
    'culture':'K562: RPMI-1640 with HEPES, FBS and glutamine; hTERT-RPE1: DMEM:F12 with FBS and hygromycin. Culture and effector covary with cell line.',
    'selection':'Published single-cell artifact follows source guide assignment and cell QC; >0.01 mean UMI gene panel. All 100 bulk-only groups are non-targeting and have missing filtered cell counts.',
    'time':'K562 essential day 6, K562 GWPS day 8, RPE1 essential day 7; time/library/cell line are not factorially separated.',
    'design':'K562 dCas9-BFP-KRAB versus RPE1 ZIM3-KRAB-dCas9. Two guides form one construct; promoter/transcript identities preserved.'},
 'nadig2025':{
    'name':'Nadig HepG2/Jurkat', 'issue':8,
    'references':['https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE264667','https://doi.org/10.1038/s41588-025-02169-3','https://pmc.ncbi.nlm.nih.gov/articles/PMC11244993/'],
    'culture':'HepG2: EMEM, FBS, Accutase and harvest FACS; Jurkat: RPMI-1640, FBS and glutamine. Both 37 C, 5% CO2. Culture/harvest covary with cell line.',
    'selection':'Local released matrices are already processed. Source preprint QC sentence interchanges percent and UMI units; source thresholds are not reapplied or silently repaired.',
    'time':'Both sampled day 7 post-transduction, with day 3 FACS selection; no time series within either background.',
    'design':'HepG2 KOX1-derived dCas9-BFP-KRAB versus Jurkat ZIM3-dCas9-P2A-mCherry; dJR092 dual-guide library enriched for essential genes. Growth screen replicate claims do not establish Perturb-seq biological replicates.'}
}


def verified_copy(source,destination):
    report=json.loads((source/'report.json').read_text())
    if report['status']!='completed' or report.get('inputs_unchanged') is not True:raise ValueError('incomplete_or_changed_component')
    files=[]
    for line in (source/'SHA256SUMS').read_text().splitlines():
        digest,name=line.split('  ',1)
        if Path(name).is_absolute() or '..' in Path(name).parts:raise ValueError('unsafe_artifact_path')
        if hash_file(source/name)!=digest:raise ValueError('component_artifact_hash_mismatch: '+name)
        files.append(name)
    if 'report.json' not in files:raise ValueError('unverified_component_report')
    destination.mkdir(parents=True)
    for name in files+['SHA256SUMS']:
        target=destination/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source/name,target)
    return report


def compose(source_id,structure,responses,annotations,output):
    if output.exists():raise ValueError('fresh_dossier_required')
    spec=SOURCES[source_id];output.mkdir(parents=True)
    structural=verified_copy(structure,output/'structure')
    contexts={r['context']:r for r in structural['tables'][0]['rows']}
    response_reports={};annotation_reports={}
    for kind,paths,records in [('response',responses,response_reports),('annotation',annotations,annotation_reports)]:
        for path in paths:
            header=json.loads((path/'report.json').read_text());context=header['context']
            if context in records or context not in contexts:raise ValueError('unexpected_or_duplicate_component_context')
            records[context]=verified_copy(path,output/kind/context)
    if set(response_reports)!=set(contexts) or set(annotation_reports)!=set(contexts):raise ValueError('missing_context_component')
    rows=[]
    for context,facts in contexts.items():
        response=response_reports[context];annotation=annotation_reports[context]
        if response['identity']['structure_sha256']!=hash_file(structure/'report.json') or annotation['identity']['structure_sha256']!=hash_file(structure/'report.json'):
            raise ValueError('component_structure_identity_mismatch')
        if annotation['identity']['response_sha256']!=hash_file(output/'response'/context/'report.json'):
            raise ValueError('annotation_response_identity_mismatch')
        if response['actual_tasks']!=facts['perturbation_construct_tasks'] or annotation['record_count']!=facts['n_cells']:
            raise ValueError('component_coverage_mismatch')
        tasks=json.loads((output/'response'/context/'tasks.json').read_text())
        resamples=pd.read_parquet(output/'response'/context/'resampling.parquet')
        if len(resamples) and (resamples.null_reference_intersection!=0).any():raise ValueError('non_disjoint_NTC_reference')
        rows.append({'context':context,'cell_line':facts['cell_line'],'days_post_transduction':facts['days_post_transduction'],
            'n_cells':facts['n_cells'],'n_genes':facts['n_genes'],'n_NTC':facts['n_NTC_cells'],
            'target_genes':facts['target_genes'],'construct_tasks':response['actual_tasks'],
            'effects':response['task_status_counts'],'DE':response['DE_status_counts'],
            'downstream_RMS':quantiles([t.get('downstream_RMS',float('nan')) for t in tasks]),
            'DEG_BH_excluding_target':quantiles([t.get('DEG_BH_excluding_target') if t.get('DEG_BH_excluding_target') is not None else float('nan') for t in tasks]),
            'target_RNA_ratio':quantiles([t['target_RNA']['RNA_ratio'] for t in tasks if t.get('target_RNA',{}).get('status')=='completed']),
            'resampling_rows':len(resamples),'NTC_null_arm_overlap':int(resamples.null_reference_intersection.sum()),
            'resamples_with_control_size_shortfall':int((resamples.null_target_cells_not_matched>0).sum()),
            'inferred_type_counts':annotation['pooled_types'],'unknown_type_fraction':annotation['pooled_types'].get('unknown',0)/facts['n_cells'],
            'uncalibrated_records':annotation['uncalibrated_records'],'source_context_marker_mismatch':annotation['source_context_mismatches'],
            'probability_correct':None,'inputs_unchanged':True,'status':'completed'})
    observability=[
        {'factor':'genetic_and_lineage_background','observed':'Source cell line and engineered effector; counts and markers',
         'missing':'Per-cell genome/epigenome, clone and passage','identification':'Source description plus uncalibrated coarse RNA inference; not type ground truth'},
        {'factor':'molecular_state','observed':'12 RNA expression proxies; NTC baseline distribution and endpoint distributions',
         'missing':'True phase, protein activity and pre-intervention state of the same cell','identification':'Descriptive expression association'},
        {'factor':'target_system_context','observed':'Measured target RNA, native panel, mapped pathway marker coverage',
         'missing':'Unmeasured official genes and protein dose','identification':'Target RNA association is separate from downstream response'},
        {'factor':'intervention_implementation','observed':spec['design'],'missing':'Per-cell effective perturbation and confirmed biological replicate identity',
         'identification':'Construct and GEM comparisons describe differences; no automatic biological replication'},
        {'factor':'time_history','observed':spec['time'],'missing':'Longitudinal observations and independently crossed time/culture design',
         'identification':'No isolated causal time modifier'},
        {'factor':'culture_environment','observed':spec['culture'],'missing':'Independent cultures and per-GEM physiological history',
         'identification':'Context/effector/culture confounding remains'},
        {'factor':'population_composition_selection','observed':'All per-cell inferred types; all-task proportions, state quantiles and within-type state differences',
         'missing':'Lineage tracing, death/proliferation/capture mechanism separation',
         'identification':'Endpoint inferred strata may introduce selection bias; no causal composition decomposition'},
        {'factor':'measurement_processing','observed':spec['selection'],'missing':'Pre-QC cells and omitted native features',
         'identification':'Raw library depth, detection, fixed thinning, numeric audit and provenance retained; NTC variability is not pure technical noise'}]
    decisions=[{'candidate_use':'Conditional CRISPRi response supervision','recommendation':'Retain as candidate, stratified by experiment, source construct/promoter, time and measured gene panel',
        'conditions':'Use matched context NTC; do not treat GEM/cells as biological replicates; preserve count and mapping provenance'},
        {'candidate_use':'Background/type/state representation','recommendation':'Use source background description and explicitly uncertain RNA proxies',
         'conditions':'Do not turn inferred labels or endpoint states into truth or pre-intervention features'},
        {'candidate_use':'Cross-context transfer evaluation','recommendation':'Only on confirmed shared measurable genes/targets with explicit source and processing overlap',
         'conditions':'All responses here have been explored; subsequent independent validation must be redesigned'},
        {'candidate_use':'Duplicate evidence','recommendation':'Do not add scPerturb/scBase copies as independent observations',
         'conditions':'Study overlap is known; exact cell-level count/identity reconciliation belongs to #19; barcode alone is insufficient'}]
    write_json(output/'observability.json',observability);write_json(output/'decisions.json',decisions)
    pd.DataFrame([{k:v for k,v in r.items() if not isinstance(v,dict)} for r in rows]).to_parquet(output/'coverage.parquet',index=False)
    report={'schema_version':2,'bundle_id':source_id+'-dossier-'+uuid.uuid4().hex,'title':spec['name']+' 全量科学评估',
        'status':'completed','completed_at':datetime.now(timezone.utc).isoformat(),'source_id':source_id,
        'component_reports':{'structure':hash_file(output/'structure/report.json'),
            **{kind+'/'+c:hash_file(output/kind/c/'report.json') for c in contexts for kind in ['response','annotation']}},
        'input_sha256':structural['input_sha256'],'inputs_unchanged':True,
        'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'code_sha256':hash_file(Path(__file__)),'references':spec['references'],
        'methods':{'issue':f'https://github.com/yjcyxky/virtual-cell-challenge/issues/{spec["issue"]}',
            'component_methods':'Exact input/code/runtime/reference hashes, parameters and commands are frozen in each component report',
            'identity':'Original input SHA256 + row identity, native feature axis, source study/experiment/GEM/barcode/construct retained',
            'status':'Data constraints and statistical non-estimability remain separate from completed processing'},
        'exposure':'All local endpoint counts, all applicable responses, DE and inferred annotations explored; no untouched validation claim',
        'limitations':['全部类型和状态均为未校准 RNA 推断，probability_correct 为空，不能当作 ground truth。',
            '源细胞系不是每细胞细类型标签；marker 变化、参考不匹配和 unknown 均保留，不能据此断言细胞身份转换。',
            '缺少已确认独立培养重复；DE/CI 和重采样只描述条件细胞抽样。',
            'NTC 池不足的重采样保留实际臂大小与短缺数，其零效应 RMS 不能当作等样本量精确显著性检验。',
            '相同基因、barcode 或来源论文不自动表示独立观测；scPerturb/scBase 的处理副本候选继续交由 #19 核实。'],
        'tables':[{'title':'全部实验与任务覆盖','rows':rows},{'title':'异质性可观测性','rows':observability},
            {'title':'使用建议与条件','rows':decisions},*structural['tables'][1:]],'reproduce':sys.argv}
    files=[p for p in output.rglob('*') if p.is_file()]
    files.sort(key=lambda p:(p.name!='report.html',str(p)))
    report['artifacts']=[{'file':str(p.relative_to(output)),'sha256':hash_file(p)} for p in files]
    write_json(output/'report.json',report);(output/'report.html').write_text(render(serial(report)))
    (output/'SHA256SUMS').write_text(''.join(f'{hash_file(p)}  {p.relative_to(output)}\n' for p in sorted(output.rglob('*')) if p.is_file() and p.name!='SHA256SUMS'))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-id',choices=list(SOURCES),required=True)
    for name in ['structure','output']:parser.add_argument('--'+name,type=Path,required=True)
    for name in ['responses','annotations']:parser.add_argument('--'+name,type=Path,nargs='+',required=True)
    args=parser.parse_args();r=compose(args.source_id,args.structure,args.responses,args.annotations,args.output)
    print(json.dumps({'status':r['status'],'bundle_id':r['bundle_id']}))
