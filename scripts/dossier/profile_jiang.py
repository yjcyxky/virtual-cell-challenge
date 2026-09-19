#!/usr/bin/env python
"""Assess all Jiang source counts and every observed cell-line/stimulus/target condition."""
import argparse
from collections import Counter
from datetime import datetime,timezone
import gc
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid
import numpy as np
import pandas as pd
from conditioned import assess_context
from convert_seurat_cache import RROOT
from rna import RNAFile,hash_file,mapping_audit,scan,quantiles,value_hash
from profile_responses import write_json,serial
from render import render
ROOT=Path(__file__).resolve().parents[2]
STIMULI=['IFNB','IFNG','TGFB','TNFA','INS']
LINES={'A549','MCF7','HT29','HAP1','BXPC3','K562'}
PROTOCOL='https://github.com/yjcyxky/virtual-cell-challenge/issues/9#issuecomment-5744934419'


def source_design(cells,stimulus):
    expected='TGFB1' if stimulus=='TGFB' else stimulus
    for name in ['source_cell_type','source_pathway','source_Batch_info','source_bc1_well','source_sample_ID','source_guide','source_gene']:
        if name not in cells or cells[name].isna().any() or (cells[name].astype(str)=='').any():raise ValueError('required_condition_label_missing: '+name)
    if set(cells.source_pathway)!={expected}:raise ValueError('stimulus_file_vs_source_label_conflict')
    if not set(cells.source_cell_type)<=LINES:raise ValueError('unexpected_source_cell_line')
    cells=cells.copy()
    cells['source_context']='Jiang__'+cells.source_cell_type.astype(str)+'__'+cells.source_pathway.astype(str)
    cells['source_batch']=[json.dumps([str(r),str(w),str(s)],separators=(',',':')) for r,w,s in zip(cells.source_Batch_info,cells.source_bc1_well,cells.source_sample_ID)]
    cells['source_task']=cells.source_gene.astype(str)
    cells['source_target_gene']=cells.source_gene.where(cells.source_gene!='NT','non-targeting')
    cells['source_guide_id']=cells.source_guide
    cells['source_label_inconsistent']=cells.source_sample.astype(str)!=(cells.source_cell_type.astype(str)+'_'+cells.source_pathway.astype(str))
    cells['source_cell_line_truth_status']='author_supplied_context_label_not_per_cell_groundtruth'
    cells['stimulation_hours_from_source_protocol']=24
    cells['source_stimulation_dose']=None
    cells['replicate_interpretation']='source fixed-cell split for scRNA-seq; not independent culture'
    return cells


def gene_lists(output,hgnc,official):
    directory=output/'author-derived-signatures';directory.mkdir()
    rcode='''args<-commandArgs(trailingOnly=TRUE);files<-Sys.glob(file.path(args[[1]],"*genelist.rds"));out<-args[[2]]
for (f in files) {x<-readRDS(f);stopifnot(is.list(x),!is.null(names(x)),all(vapply(x,is.character,logical(1))));rows<-do.call(rbind,lapply(names(x),function(n) data.frame(source_set=n,source_gene=x[[n]],source_position=seq_along(x[[n]]))));write.csv(rows,file.path(out,paste0(basename(f),".csv")),row.names=FALSE)}'''
    env=os.environ.copy();env['R_HOME']=str(RROOT);env['LD_LIBRARY_PATH']=str(RROOT/'lib')+':'+str(RROOT.parent)
    subprocess.run(['/lib/ld-linux-aarch64.so.1',str(RROOT/'bin/exec/R'),'--vanilla','--slave','-e',rcode,'--args',str(ROOT/'data/raw/jiang2025'),str(directory.resolve())],env=env,check=True)
    rows=[]
    for path in directory.glob('*.csv'):
        frame=pd.read_csv(path);mapping=mapping_audit(frame.source_gene.tolist(),hgnc,official)
        frame['mapped_symbol']=mapping.mapped_symbol;frame['mapping_status']=mapping.mapping_status
        frame['derived_from_this_endpoint_dataset']=True;frame['independent_annotation_reference']=False
        frame.to_parquet(path.with_suffix('.parquet'),index=False)
        rows.append({'file':path.name,'sets':frame.source_set.nunique(),'memberships':len(frame),'unique_source_genes':frame.source_gene.nunique(),
            'official_axis_mapped_genes':len(set(frame.mapped_symbol.dropna())&set(official)),
            'role':'Author endpoint-derived signature, not independent prior, activity ground truth, or untouched validation'})
    return rows


def run(cache,references,evidence,inventory,output,workers=2,resume=False):
    started=time.monotonic();commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    inputs={};conversions={}
    inv=json.loads((inventory/'report.json').read_text());expected={x['file']:x['sha256'] for x in inv['file_results'] if x['source_id']=='jiang2025'}
    for source,digest in expected.items():
        if hash_file(ROOT/source)!=digest:raise ValueError('original_source_hash_mismatch')
        inputs[source]=digest
    for stimulus in STIMULI:
        path=cache/stimulus/'counts.h5ad';record=json.loads((cache/stimulus/'identity.json').read_text())
        if record['status']!='completed' or record['source_sha256']!=expected[f'data/raw/jiang2025/Seurat_object_{stimulus}_Perturb_seq.rds'] or hash_file(path)!=record['output_sha256']:raise ValueError('RDS_conversion_identity_mismatch')
        conversions[stimulus]=record;inputs[str(path.relative_to(ROOT)) if path.is_absolute() else str(path)]=record['output_sha256']
    hgnc_path=ROOT/'data/raw/networks/hgnc_complete_set.txt';axis=ROOT/'data/raw/arc_vcc2026_controls/gene_names.csv'
    for p in [hgnc_path,axis]:inputs[str(p.relative_to(ROOT))]=hash_file(p)
    identity={'input_sha256':inputs,'reference_sha256':hash_file(references/'gene_sets.json'),'driver_sha256':hash_file(Path(__file__)),
        'engine_sha256':hash_file(Path(__file__).with_name('conditioned.py'))}
    if output.exists():
        if not resume or json.loads((output/'identity.json').read_text())!=identity:raise ValueError('resume_input_or_method_mismatch')
        if (output/'report.json').exists():raise ValueError('complete_outputs_immutable')
    else:output.mkdir(parents=True);write_json(output/'identity.json',identity);shutil.copytree(references,output/'references');shutil.copytree(evidence,output/'evidence')
    hgnc=pd.read_csv(hgnc_path,sep='\t',low_memory=False);official=pd.read_csv(axis).gene_name.tolist()
    summaries=[];contexts=[];coverage=[];design_rows=[]
    for stimulus in STIMULI:
        path=cache/stimulus/'counts.h5ad';directory=output/('source-'+stimulus)
        if (directory/'structure.json').exists():
            checkpoint=json.loads((directory/'structure.json').read_text())
            for name,digest in checkpoint['artifacts'].items():
                if hash_file(directory/name)!=digest:raise ValueError('structure_checkpoint_changed')
            facts=checkpoint['facts'];cells=pd.read_parquet(directory/'cells.parquet');genes=pd.read_parquet(directory/'genes.parquet');mapping=pd.read_parquet(directory/'gene-mapping.parquet')
        else:
            directory.mkdir();facts,cells,genes=scan(path,conversions[stimulus]['source_sha256'],'Jiang2025_'+stimulus)
            cells=source_design(cells,stimulus);mapping=mapping_audit(genes.source_gene.tolist(),hgnc,official)
            if not facts['all_counts_finite_nonnegative_integer']:raise ValueError('invalid_source_counts_require_diagnostic_review')
            facts.update(stimulus_file=stimulus,source_RDS_sha256=conversions[stimulus]['source_sha256'],converted_cache_sha256=conversions[stimulus]['output_sha256'],
                nCount_RNA_disagreements=int((cells.source_nCount_RNA!=cells.computed_total_counts).sum()),
                nFeature_RNA_disagreements=int((cells.source_nFeature_RNA!=cells.computed_detected_genes).sum()),
                source_label_inconsistent_records=int(cells.source_label_inconsistent.sum()),conversion=conversions[stimulus])
            cells.to_parquet(directory/'cells.parquet',index=False,compression='zstd');genes.to_parquet(directory/'genes.parquet',index=False);mapping.to_parquet(directory/'gene-mapping.parquet',index=False)
            write_json(directory/'structure.json',{'facts':facts,'artifacts':{n:hash_file(directory/n) for n in ['cells.parquet','genes.parquet','gene-mapping.parquet']}})
        summaries.append(facts)
        mapped=set(mapping.loc[~mapping.many_to_one_mapping,'mapped_symbol'].dropna());literal=set(genes.source_gene)
        coverage.extend({'stimulus_file':stimulus,'official_gene':g,'literal_present':g in literal,'unambiguous_mapped_present':g in mapped,'missing_means':'not_measured_or_not_mapped_never_zero'} for g in official)
        design=cells.groupby(['source_context','source_cell_type','source_pathway','source_Batch_info','source_bc1_well','source_sample_ID','source_task','source_target_gene'],observed=True).agg(
            cells=('record_id','size'),guides=('source_guide_id',lambda x:'|'.join(sorted(set(x)))),label_inconsistent=('source_label_inconsistent','sum')).reset_index()
        design.to_parquet(directory/'task-stratum-coverage.parquet',index=False);design_rows.append(design)
        for context,group in cells.groupby('source_context',observed=True,sort=True):
            target_dir=output/context
            if (target_dir/'report.json').exists():
                for line in (target_dir/'SHA256SUMS').read_text().splitlines():
                    digest,name=line.split('  ',1)
                    if hash_file(target_dir/name)!=digest:raise ValueError('completed_context_artifact_changed')
                report=json.loads((target_dir/'report.json').read_text())
                if report['status']!='completed':raise ValueError('incomplete_context_report')
            else:
                line=group.source_cell_type.iloc[0];expected_lineages=['hematopoietic'] if line in ['HAP1','K562'] else ['epithelial']
                report=assess_context(path,group,genes,mapping,references,target_dir,context,expected_lineages,PROTOCOL,
                    {'original_RDS_sha256':conversions[stimulus]['source_sha256'],'count_cache_sha256':conversions[stimulus]['output_sha256'],
                        'structure_sha256':hash_file(directory/'structure.json'),'row_selection_sha256':value_hash(group.row_index.tolist())},workers)
            contexts.append({'context':context,**{k:report[k] for k in ['n_cells','n_NTC','actual_tasks','task_status_counts','DE_status_counts','pooled_types','uncalibrated_records','source_label_inconsistent_records']},
                'report':str((target_dir/'report.html').relative_to(output)),'status':'completed'})
            print(context+' completed: '+str(report['actual_tasks'])+' targets, '+str(report['n_cells'])+' cells',flush=True);gc.collect()
    if len(contexts)!=30:raise ValueError('expected_all_30_source_background_stimulus_contexts')
    signatures=gene_lists(output,hgnc,official) if not (output/'author-derived-signatures').exists() else json.loads((output/'signature-summary.json').read_text())
    write_json(output/'signature-summary.json',signatures)
    pd.DataFrame(coverage).to_parquet(output/'official-axis-coverage.parquet',index=False);pd.concat(design_rows).to_parquet(output/'all-task-stratum-design.parquet',index=False)
    pd.DataFrame(contexts).to_parquet(output/'context-coverage.parquet',index=False)
    for source,digest in expected.items():
        if hash_file(ROOT/source)!=digest:raise ValueError('original_source_changed_during_assessment')
    design_facts=[{'factor':'background','observed':'Six source cancer cell-line labels, monoclonal CRISPRi engineering; no per-cell genome/epigenome'},
        {'factor':'molecular_state','observed':'12 explicitly uncertain RNA proxies and inferred coarse type/unknown per cell'},
        {'factor':'target_system','observed':'Target RNA and all native genes; independent guide consistency; no measured protein dose'},
        {'factor':'implementation','observed':'Dolcetto three sgRNAs/target, source NT guides; monoclonal dCas9-KRAB-MeCP2; source Mixscale remains inferred endpoint score'},
        {'factor':'time_history','observed':'3–5 day puromycin selection, 7–10 day expansion, 24-hour stimulation; no isolated time-series effect'},
        {'factor':'environment','observed':'Source stimulus and cell-line-specific media; dose unresolved in accessible source metadata; no matched unstimulated cohort in these files'},
        {'factor':'composition_selection','observed':'Source filtering precedes release; endpoint inferred composition/state and label inconsistency sensitivity cannot identify death/capture/transition mechanisms'},
        {'factor':'measurement','observed':'Two fixed-cell split measurements; Parse v1 WT/WT Mega, Ultima and subset Illumina per source; exact per-cell instrument unknown; technical strata retained'}]
    report={'schema_version':2,'bundle_id':'jiang-dossier-'+uuid.uuid4().hex,'title':'Jiang 五刺激、六细胞背景全量条件评估','status':'completed',
        'input_sha256':inputs,'inputs_unchanged':True,'code_commit':commit,'identity':identity,
        'completed_at':datetime.now(timezone.utc).isoformat(),'duration_seconds':time.monotonic()-started,
        'n_cells':sum(r['n_cells'] for r in contexts),'contexts':len(contexts),'actual_tasks':sum(r['actual_tasks'] for r in contexts),
        'methods':{'protocol':PROTOCOL,'condition_keys':['source_cell_type','source_pathway'],
            'control_strata':['source_Batch_info','source_bc1_well','source_sample_ID'],'NTC':'Exact source gene == NT; NTC still cytokine-stimulated',
            'replication':'GEO says split after fixation; no independent-culture replication claim',
            'RNA_layer':'RNA/counts only, exact all-value CSC-to-CSR transpose cache verified; data/scale.data audited independently',
            'source_label_inconsistency':'Keep source cell_type/pathway/sample, report mismatch and subset sensitivity; no metadata repair'},
        'references':['https://doi.org/10.1038/s41556-025-01622-z','https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE281048','https://zenodo.org/records/14518762'],
        'limitations':['五文件是刺激而非五个背景；本包包括全部六细胞系×五刺激。',
            '源 sample 与 cell_type/pathway 不一致时保留两者，作者标签不是物理身份 ground truth。',
            'Rep1/Rep2 是固定后测量拆分，不能替代独立培养重复；NTC 有对应刺激，不能估计刺激相对未刺激效应。',
            '刺激剂量在已取得的 GEO 元数据中缺失；产品编号不是剂量；不能声称跨刺激等剂量或比较剂量效应。',
            '源 Mixscale 和作者通路基因集来自本数据终点；不是实测敲低剂量、独立验证或类型参考。',
            '所有推断未校准，probability_correct 为空；缺失的基因/重复/统计量不填零。'],
        'exposure':'All local endpoint counts, source-derived Mixscale and signature lists examined; no untouched validation claim',
        'tables':[{'title':'全部背景×刺激覆盖','rows':contexts},{'title':'异质性可观测性','rows':design_facts},{'title':'作者派生通路签名','rows':signatures},
            {'title':'全部来源计数与转换核验','rows':[{k:v for k,v in r.items() if k!='conversion'} for r in summaries]}],'reproduce':sys.argv}
    files=sorted((p for p in output.rglob('*') if p.is_file()),key=lambda p:(p.name!='report.html',str(p)))
    report['artifacts']=[{'file':str(p.relative_to(output)),'sha256':hash_file(p)} for p in files]
    write_json(output/'report.json',report);(output/'report.html').write_text(render(serial(report)))
    (output/'SHA256SUMS').write_text(''.join(f'{hash_file(p)}  {p.relative_to(output)}\n' for p in sorted(output.rglob('*')) if p.is_file() and p.name!='SHA256SUMS'))
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['cache','references','evidence','inventory','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--workers',type=int,default=2);p.add_argument('--resume',action='store_true');a=p.parse_args()
    try:r=run(a.cache,a.references,a.evidence,a.inventory,a.output,a.workers,a.resume)
    except Exception as error:
        if a.output.exists() and not (a.output/'report.json').exists():write_json(a.output/'failure.json',{'status':'failed','error':str(error),'type':type(error).__name__})
        raise
    print(json.dumps({'status':r['status'],'bundle_id':r['bundle_id']}))
