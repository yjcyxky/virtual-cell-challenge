#!/usr/bin/env python
"""Run every declared GxE2 single-gene task in bounded condition views."""
import argparse
from collections import Counter
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import numpy as np
import pandas as pd
from annotation import state_model
from gxe2_context import assess_context
from profile_responses import write_json,PARAMETERS
from rna import RNAFile,hash_file,mapping_audit,value_hash

ROOT=Path(__file__).resolve().parents[2]
CODE=['profile_gxe2.py','gxe2_context.py','conditioned.py','profile_responses.py','response.py','rna.py','annotation.py','sparse_annotation.py','profile_crispri.py']


def context_key(line,drug,dose):return json.dumps([line,drug,float(dose)],separators=(',',':'))


def source_design(cache,capture,records):
    metadata=pd.read_parquet(cache/'all-CDS-source-metadata.parquet')
    qc=pd.read_parquet(records/'CDS-records.parquet').drop(columns=['CDS_metadata_row','source_reported_UMI','source_UMI_matches'])
    cells=qc.merge(metadata.add_prefix('source_CDS_'),left_on='source_barcode',right_on='source_CDS_source_barcode',validate='1:1',sort=False)
    guide=pd.read_parquet(capture/'guide-source-reproduction.parquet');hashes=pd.read_parquet(capture/'hash-source-reproduction.parquet')
    checks=guide[['source_barcode','source_guide_call_matches','source_gene_labels_match_whitelist','source_maxCount_matches','source_topRatio_matches','source_call_has_ambiguous_sequence_name']].merge(hashes[['source_barcode','source_UMI_matches_total','RT_replicate_matches','source_condition_matches_whitelist']],on='source_barcode',validate='1:1')
    cells=cells.merge(checks,on='source_barcode',validate='1:1',sort=False)
    sets=[sorted(set(str(g).split(','))) if pd.notna(g) else [] for g in cells.source_CDS_gene_id]
    cells['analysis_target']=[s[0] if len(s)==1 else None for s in sets]
    source_label_checks=cells.source_guide_call_matches&cells.source_gene_labels_match_whitelist&cells.source_maxCount_matches&cells.source_topRatio_matches
    hash_supported=cells.source_UMI_matches_total&cells.RT_replicate_matches&cells.source_condition_matches_whitelist&(cells.source_CDS_hash_umis>=5)&(cells.source_CDS_top_to_second_best_ratio>=2.5)
    cells['assignment_limitation']=np.select([~hash_supported,np.array([len(s)==0 for s in sets]),np.array([len(s)>1 for s in sets]),~source_label_checks],
        ['hash_design_or_source_support_inconsistent','source_guide_unassigned','multiple_source_target_genes','source_guide_count_or_call_disagreement'],default='supported')
    cells['source_assignment_supported']=cells.assignment_limitation=='supported'
    cells['source_context']=[context_key(line,drug,dose) for line,drug,dose in zip(cells.source_CDS_cell_line,cells.source_CDS_treatment,cells.source_CDS_dose)]
    cells['source_batch']=[json.dumps([rep,h],separators=(',',':')) for rep,h in zip(cells.source_CDS_replicate,cells.source_CDS_top_oligo)]
    cells['source_guide_id']=cells.source_CDS_gRNA_id.map(lambda x:','.join(sorted(str(x).split(','))) if pd.notna(x) else 'unassigned')
    cells['source_target_gene']=cells.analysis_target.replace({'NTC':'non-targeting'});cells['source_task']=cells.analysis_target.fillna('unassigned_or_multiple_genes')
    cells['source_label_inconsistent']=~hash_supported|(~source_label_checks&cells.source_CDS_gene_id.notna())
    views=[]
    for folder in sorted(cache.glob('CDS-*')):
        part=pd.read_parquet(folder/'raw-count-comparison.parquet')[['source_barcode','CDS_row']].rename(columns={'CDS_row':'CDS_view_row'})
        part['count_view_file']=str(folder/'counts.h5ad');views.append(part)
    cells=cells.merge(pd.concat(views,ignore_index=True),on='source_barcode',validate='1:1',sort=False)
    if cells.record_id.duplicated().any() or cells.CDS_view_row.isna().any():raise ValueError('invalid_GxE2_source_cell_identity')
    return cells


def vehicle_background(path,cells,resolved,states):
    desired=set(cells.loc[cells.source_assignment_supported&(cells.source_CDS_gRNA_maxCount>=10)&(cells.analysis_target=='NTC')&(cells.source_CDS_treatment=='vehicle'),'CDS_view_row'])
    with RNAFile(path) as source:
        mean=np.zeros(source.shape[1]);n=0
        for start,matrix in source.blocks(2048):
            selected=[i for i in range(matrix.shape[0]) if start+i in desired]
            if not selected:continue
            x=matrix[selected];totals=np.asarray(x.sum(axis=1)).ravel();x.data=np.log1p(x.data*np.repeat(10000/totals,np.diff(x.indptr)))
            mean+=np.asarray(x.sum(axis=0)).ravel();n+=len(selected)
    if n!=len(desired) or not n:raise ValueError('missing_or_incomplete_shared_vehicle_NTC')
    mean/=n;weights,coverage=state_model(resolved,states,mean)
    return {'mean':mean,'weights':weights,'coverage':coverage,'NTC_cells':n}


def run(cache,capture,records,references,output,workers=4,resume=False):
    started=time.monotonic();commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    reports={name:json.loads((directory/file).read_text()) for name,directory,file in [('adapter',cache,'identity.json'),('capture',capture,'report.json'),('records',records,'report.json')]}
    for name,report in reports.items():
        if report['status']!='completed':raise ValueError('completed_upstream_'+name+'_required')
    for directory,report in [(capture,reports['capture']),(records,reports['records'])]:
        for name,digest in report['artifacts'].items():
            if hash_file(directory/name)!=digest:raise ValueError('upstream_phase_artifact_changed')
    for file,digest in reports['adapter']['input_sha256'].items():
        if hash_file(ROOT/file)!=digest:raise ValueError('raw_input_changed')
    for line in (references/'SHA256SUMS').read_text().splitlines():
        digest,name=line.split('  ',1)
        if hash_file(references/name)!=digest:raise ValueError('reference_changed')
    identity={'adapter_report_sha256':hash_file(cache/'identity.json'),'capture_report_sha256':hash_file(capture/'report.json'),'records_report_sha256':hash_file(records/'report.json'),
        'reference_sha256':hash_file(references/'gene_sets.json'),'code':{n:hash_file(Path(__file__).with_name(n)) for n in CODE},'parameters':PARAMETERS,'minimum_primary_guide_reads':10,
        'uv_lock_sha256':hash_file(Path(__file__).with_name('uv.lock'))}
    if output.exists():
        if not resume or json.loads((output/'identity.json').read_text())!=identity or (output/'report.json').exists():raise ValueError('GxE2_run_resume_identity_changed_or_completed')
    else:output.mkdir(parents=True);write_json(output/'identity.json',identity)
    genes=pd.read_parquet(records/'genes.parquet');hgnc=pd.read_csv(ROOT/'data/raw/networks/hgnc_complete_set.txt',sep='\t',low_memory=False)
    mapping=mapping_audit(genes.gene_name.tolist(),hgnc,pd.read_csv(ROOT/'data/raw/arc_vcc2026_controls/gene_names.csv').gene_name.tolist(),genes.source_gene.tolist()).rename(columns={'source_gene':'source_symbol'})
    mapping.insert(0,'source_gene',genes.source_gene);mapping.to_parquet(output/'gene-mapping.parquet',index=False);genes.to_parquet(output/'genes.parquet',index=False)
    safe=mapping.mapped_symbol.where(~mapping.many_to_one_mapping&(mapping.symbol_vs_ensembl!='conflict'));resolved=[str(x) if pd.notna(x) else None for x in safe]
    cells=source_design(cache,capture,records);cells.to_parquet(output/'source-design.parquet',index=False)
    whitelist=pd.read_parquet(capture/'guide-whitelist.parquet');targets=sorted(set(whitelist.gene)-{'NTC'})
    if len(targets)!=523 or 'random' not in targets:raise ValueError('declared_522_gene_plus_random_library_scope_changed')
    declared=pd.read_parquet(capture/'hash-whitelist.parquet')[['cell_line','treatment','dose']].drop_duplicates().sort_values(['cell_line','treatment','dose'])
    declared['context']=[context_key(r.cell_line,r.treatment,r.dose) for r in declared.itertuples()]
    if len(declared)!=27 or set(cells.source_context)!=set(declared.context):raise ValueError('declared_and_observed_27_context_scope_mismatch')
    declared.to_parquet(output/'declared-contexts.parquet',index=False);reference=json.loads((references/'gene_sets.json').read_text());results=[]
    backgrounds=output/'shared-vehicle-backgrounds';backgrounds.mkdir(exist_ok=True)
    for line,group in cells.groupby('source_CDS_cell_line',sort=True):
        paths=group.count_view_file.unique()
        if len(paths)!=1:raise ValueError('cell_line_count_view_ambiguity')
        relative=str(Path(paths[0]).relative_to(cache))
        if hash_file(Path(paths[0]))!=reports['adapter']['artifacts'][relative]:raise ValueError('source_CDS_count_cache_changed')
        with RNAFile(Path(paths[0])) as source:
            if source.var.index.astype(str).tolist()!=genes.source_gene.tolist():raise ValueError('CDS_count_view_gene_axis_order_mismatch')
        vehicle=vehicle_background(Path(paths[0]),group,resolved,reference['states'])
        np.savez_compressed(backgrounds/(line+'.npz'),mean_logCP10K=vehicle['mean'],state_weights=vehicle['weights'],n_NTC=vehicle['NTC_cells'])
        write_json(backgrounds/(line+'.json'),{'coverage':vehicle['coverage'],'NTC_cells':vehicle['NTC_cells'],'scope':'same source cell line and vehicle NTC; fixed shared weights for drug exposure descriptions'})
        for row in declared.loc[declared.cell_line==line].itertuples():
            chosen=group.loc[group.source_context==row.context];destination=output/'contexts'/value_hash(row.context)[:20]
            result=assess_context(Path(paths[0]),chosen,genes,mapping,reference,targets,vehicle,destination,row.context,identity,workers)
            results.append({**{k:v for k,v in result.items() if k not in ['identity','artifacts','view_identity']},'directory':str(destination.relative_to(output))})
            print('GxE2 completed contexts '+str(len(results))+'/27 '+row.context,flush=True)
    for file,digest in reports['adapter']['input_sha256'].items():
        if hash_file(ROOT/file)!=digest:raise ValueError('raw_input_changed_during_assessment')
    report={'status':'completed','phase':'all_27_genetic_contexts_and_all_989299_source_CDS_cells','code_commit':commit,'identity':identity,
        'contexts':results,'n_cells':len(cells),'declared_tasks':27*len(targets),'actual_tasks':sum(r['tasks'] for r in results),
        'task_status_counts':dict(sum((Counter(r['task_status_counts']) for r in results),Counter())),
        'DE_status_counts':dict(sum((Counter(r['DE_status_counts']) for r in results),Counter())),
        'pooled_types':dict(sum((Counter(r['pooled_types']) for r in results),Counter())),
        'resampling_rows':sum(r['resampling_rows'] for r in results),'null_reference_intersections':sum(r['null_reference_intersections'] for r in results),'null_size_shortfall_rows':sum(r['null_size_shortfall_rows'] for r in results),
        'input_sha256':reports['adapter']['input_sha256'],'inputs_unchanged':True,'duration_seconds':time.monotonic()-started,'completed_at':datetime.now(timezone.utc).isoformat(),'reproduce':sys.argv}
    report['artifacts']={str(p.relative_to(output)):hash_file(p) for p in output.rglob('*') if p.is_file() and p.name!='report.json'};write_json(output/'report.json',report);return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['cache','capture','records','references','output']:p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--workers',type=int,default=4);p.add_argument('--resume',action='store_true');a=p.parse_args()
    r=run(a.cache,a.capture,a.records,a.references,a.output,a.workers,a.resume);print(json.dumps({k:v for k,v in r.items() if k not in ['artifacts','contexts','input_sha256','identity']}))
