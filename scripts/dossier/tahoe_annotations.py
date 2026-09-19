#!/usr/bin/env python
"""All-record Tahoe marker inference with local plate/line vehicle backgrounds."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,as_completed
from datetime import datetime,timezone
import json
import multiprocessing as mp
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import uuid
import h5py
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from annotation import RULES,marker_model,state_model
from sparse_annotation import annotate_sparse
from tahoe import decode
from rna import hash_file
from profile_responses import write_json
ROOT=Path(__file__).resolve().parents[2]
CODE=['tahoe_annotations.py','sparse_annotation.py','annotation.py','tahoe.py','rna.py','profile_responses.py']


def prepare_models(counts,references,output):
    mapping=pd.read_parquet(counts/'gene-mapping.parquet')
    safe=mapping.mapped_symbol.where(~mapping.many_to_one_mapping&(mapping.symbol_vs_ensembl!='conflict'))
    resolved=[str(x) if pd.notna(x) else None for x in safe]
    reference=json.loads((references/'gene_sets.json').read_text());model=marker_model(resolved,reference['profiles'])
    write_json(output/'type-coverage.json',model['coverage']);write_json(output/'resolved-symbols.json',resolved)
    conditions=pd.read_parquet(counts/'conditions.parquet');conditions['background_index']=-1;groups=[];coverage=[]
    grouped=list(conditions.groupby(['plate','cell_line_id'],sort=True,observed=True))
    weights=np.lib.format.open_memmap(output/'state-weights.npy',mode='w+',dtype='float32',shape=(len(grouped),len(mapping),len(reference['states'])))
    baselines=np.zeros((len(grouped),len(mapping)),np.float32)
    with h5py.File(counts/'condition-profiles.h5','r') as handle:
        for i,((plate,line),group) in enumerate(grouped):
            controls=group.loc[(group.source_drug=='DMSO_TF')&(group.eligible_profile_cells>0)]
            chosen=controls if len(controls) else group.loc[group.eligible_profile_cells>0]
            baseline=np.zeros(len(mapping));n=int(chosen.eligible_profile_cells.sum())
            for row in chosen.itertuples():baseline+=handle['mean_logCP10K'][row.condition_index].astype(float)*row.eligible_profile_cells/max(1,n)
            w,c=state_model(resolved,reference['states'],baseline);weights[i]=w;baselines[i]=baseline
            if not n:weights[i]=0
            conditions.loc[group.index,'background_index']=i
            groups.append({'background_index':i,'plate':plate,'cell_line_id':line,'n_DMSO':int(controls.eligible_profile_cells.sum()),
                'DMSO_sample_ids':controls['sample'].tolist(),'baseline_cells':n,
                'background_method':'same_plate_same_cell_line_DMSO' if len(controls) else 'same_plate_line_pooled_endpoint_fallback_no_local_DMSO',
                'source_condition_indices':chosen.condition_index.tolist(),'not_independent_biological_replicates':True})
            coverage.extend([{'background_index':i,**r,'status':r['status'] if n else 'not_estimable'} for r in c])
            if (i+1)%50==0:print('Tahoe background models '+str(i+1)+'/'+str(len(grouped)),flush=True)
    weights.flush();del weights
    np.savez_compressed(output/'background-means.npz',mean_logCP10K=baselines)
    write_json(output/'backgrounds.json',groups);write_json(output/'state-coverage.json',coverage)
    conditions.to_parquet(output/'conditions.parquet',index=False)
    return groups


def annotate_shard(record,counts_string,output_string,references_string,identity):
    counts,output,references=Path(counts_string),Path(output_string),Path(references_string)
    key=str(record['shard_index']).zfill(3);directory=output/'shards'/key
    if (directory/'identity.json').exists():
        saved=json.loads((directory/'identity.json').read_text())
        if saved['run_identity']!=identity:raise ValueError('annotation_checkpoint_identity_changed')
        for name,digest in saved['artifacts'].items():
            if hash_file(directory/name)!=digest:raise ValueError('annotation_checkpoint_artifact_changed')
        return saved['summary']
    started=time.monotonic();source=ROOT/record['file'];stat=source.stat()
    if hash_file(source)!=record['input_sha256']:raise ValueError('raw_expression_changed')
    frame=pd.read_parquet(counts/'shards'/key/'cells.parquet');genes=pd.read_parquet(counts/'genes.parquet')
    tokens=genes.token_id.to_numpy();lookup=np.full(tokens.max()+1,-1,dtype=np.int32);lookup[tokens]=np.arange(len(genes))
    conditions=pd.read_parquet(output/'conditions.parquet');ci={(r.sample,r.cell_line_id):r.condition_index for r in conditions.itertuples()}
    frame['condition_index']=[ci[(s,l)] for s,l in zip(frame['sample'],frame.cell_line_id)]
    frame['state_background_index']=conditions.background_index.to_numpy()[frame.condition_index]
    backgrounds=json.loads((output/'backgrounds.json').read_text());reference=json.loads((references/'gene_sets.json').read_text())
    resolved=json.loads((output/'resolved-symbols.json').read_text());model=marker_model(resolved,reference['profiles'])
    weights=np.load(output/'state-weights.npy',mmap_mode='r');coverage=json.loads((output/'state-coverage.json').read_text())
    applicable=np.array([r['status']=='completed' for r in coverage]).reshape(len(backgrounds),len(reference['states']))
    parts=[];position=0
    for batch in pq.ParquetFile(source).iter_batches(batch_size=512):
        matrix,valid,_=decode(batch,lookup,len(genes));stop=position+len(batch)
        if batch.column(batch.schema.get_field_index('BARCODE_SUB_LIB_ID')).to_pylist()!=frame.BARCODE_SUB_LIB_ID.iloc[position:stop].tolist():raise ValueError('annotation_row_identity_mismatch')
        sizes=np.asarray(matrix.sum(axis=1)).ravel();allowed=valid&(sizes>0)
        if not np.array_equal(valid,frame.computed_numeric_valid.iloc[position:stop]) or not np.array_equal(sizes,frame.computed_total_counts.iloc[position:stop]):raise ValueError('count_phase_disagrees')
        log=matrix.copy();log.data=np.log1p(log.data*np.repeat(np.divide(10000,sizes,out=np.zeros_like(sizes),where=sizes>0),np.diff(log.indptr)))
        if not valid.all():log[~valid]=0
        labels,scores=annotate_sparse(log,['not_genetic_target']*len(batch),model)
        for name,values in scores.items():labels[name]=values
        state=np.full((len(batch),len(reference['states'])),np.nan);bg=frame.state_background_index.iloc[position:stop].to_numpy()
        for index in np.unique(bg):
            ids=np.flatnonzero((bg==index)&allowed);values=log[ids]@weights[index];values[:,~applicable[index]]=np.nan;state[ids]=values
        for j,name in enumerate(reference['states']):labels['state__'+name]=state[:,j]
        for field in ['inferred_lineage','inferred_type','inferred_subtype']:labels.loc[~allowed,field]='unknown'
        labels.loc[~allowed,'inference_status']='not_estimable';labels.loc[~allowed,'inference_reason']='invalid_or_zero_library'
        parts.append(labels);position=stop
    if position!=len(frame):raise ValueError('incomplete_annotation_scan')
    combined=pd.concat([frame,pd.concat(parts,ignore_index=True)],axis=1)
    combined['state_background_method']=[backgrounds[i]['background_method'] for i in combined.state_background_index]
    combined['state_background_control_cells']=[backgrounds[i]['n_DMSO'] for i in combined.state_background_index]
    combined['state_inference_method']='fixed_expression_matched_gene_background_24_bins_10_controls_seed_20260919'
    combined['truth_label']=False;combined['available_before_endpoint']=False
    if combined.record_id.duplicated().any() or not combined.probability_correct.isna().all():raise ValueError('invalid_annotation_identity_or_calibration')
    if (stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns)!=(source.stat().st_size,source.stat().st_mtime_ns,source.stat().st_ctime_ns):raise ValueError('source_changed_during_annotation')
    summary={'shard_index':record['shard_index'],'status':'completed','n_cells':len(combined),'pooled_types':combined.inferred_type.value_counts().to_dict(),
        'pooled_lineages':combined.inferred_lineage.value_counts().to_dict(),'method_conflicts':int(combined.method_conflict.sum()),
        'fallback_background_cells':int((combined.state_background_control_cells==0).sum()),'uncalibrated_records':len(combined),'inputs_unchanged':True,'duration_seconds':time.monotonic()-started}
    with tempfile.TemporaryDirectory(prefix='annotation-'+key+'-',dir=output) as temporary:
        temporary=Path(temporary);combined.to_parquet(temporary/'cells.parquet',index=False,compression='zstd')
        write_json(temporary/'identity.json',{'run_identity':identity,'summary':summary,'artifacts':{'cells.parquet':hash_file(temporary/'cells.parquet')}})
        os.rename(temporary,directory)
    return summary


def run(counts,references,output,workers=4,resume=False):
    started=time.monotonic();commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip();source=json.loads((counts/'report.json').read_text())
    if source['status']!='completed' or source['actual_cells']!=8467330:raise ValueError('complete_registered_count_scope_required')
    for name,digest in source['artifacts'].items():
        if hash_file(counts/name)!=digest:raise ValueError('count_evidence_changed')
    for line in (references/'SHA256SUMS').read_text().splitlines():
        digest,name=line.split('  ',1)
        if hash_file(references/name)!=digest:raise ValueError('annotation_reference_changed')
    identity={'count_report_sha256':hash_file(counts/'report.json'),'reference_sha256':hash_file(references/'gene_sets.json'),
        'code':{n:hash_file(Path(__file__).with_name(n)) for n in CODE},'rules':RULES,'uv_lock_sha256':hash_file(Path(__file__).with_name('uv.lock'))}
    if output.exists():
        if not resume or json.loads((output/'identity.json').read_text())!=identity or (output/'report.json').exists():raise ValueError('resume_identity_mismatch_or_completed_output')
        saved=json.loads((output/'model-artifacts.json').read_text())
        for name,digest in saved.items():
            if hash_file(output/name)!=digest:raise ValueError('model_checkpoint_changed')
    else:
        output.mkdir(parents=True);(output/'shards').mkdir();write_json(output/'identity.json',identity)
        prepare_models(counts,references,output)
        write_json(output/'model-artifacts.json',{p.name:hash_file(p) for p in output.iterdir() if p.is_file()})
    records=json.loads((counts/'shard-results.json').read_text());summaries=[];failures=[]
    with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn')) as pool:
        futures={pool.submit(annotate_shard,r,str(counts),str(output),str(references),identity):r for r in records}
        for future in as_completed(futures):
            row=futures[future]
            try:summaries.append(future.result())
            except Exception as e:failures.append({'shard_index':row['shard_index'],'status':'failed','type':type(e).__name__,'error':str(e)})
            if (len(summaries)+len(failures))%5==0:print(f'Tahoe annotations {len(summaries)}/300 completed, {len(failures)} failed',flush=True)
    if failures:
        write_json(output/('failures-'+str(time.time_ns())+'.json'),failures);raise RuntimeError('incomplete_Tahoe_annotations')
    summaries.sort(key=lambda r:r['shard_index']);write_json(output/'shard-results.json',summaries)
    totals=Counter();lineages=Counter()
    for row in summaries:totals.update(row['pooled_types']);lineages.update(row['pooled_lineages'])
    report={'status':'completed','phase':'all_record_type_and_state_inference_only','bundle_id':'tahoe-annotations-'+uuid.uuid4().hex,'identity':identity,
        'code_commit':commit,'actual_cells':sum(r['n_cells'] for r in summaries),'completed_shards':len(summaries),'failed_shards':0,
        'pooled_types':dict(totals),'pooled_lineages':dict(lineages),'method_conflicts':sum(r['method_conflicts'] for r in summaries),
        'fallback_background_cells':sum(r['fallback_background_cells'] for r in summaries),'uncalibrated_records':sum(r['uncalibrated_records'] for r in summaries),
        'inputs_unchanged':True,'source_count_report_sha256':hash_file(counts/'report.json'),'reference_sha256':identity['reference_sha256'],
        'methods':{'type':'Same frozen two-score broad marker agreement; exact implicit-zero tied ranks; stable float64 variance; no atlas calibration',
            'states':'Plate and source cell-line matched local DMSO mean expression bins; explicit same-plate/line pooled endpoint fallback where no DMSO',
            'source_labels':'Source cell line, phase, scores and drug annotations retained independently; none asserted ground truth',
            'exposure':'All inferred annotations use observed endpoint RNA; unavailable as pretreatment input',
            'chemical_target':'No genetic target marker exclusion applies to drug exposures'},
        'completed_at':datetime.now(timezone.utc).isoformat(),'duration_seconds':time.monotonic()-started,'reproduce':sys.argv}
    report['artifacts']={str(p.relative_to(output)):hash_file(p) for p in sorted(output.rglob('*')) if p.is_file()};write_json(output/'report.json',report);return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['counts','references','output']:p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--workers',type=int,default=4);p.add_argument('--resume',action='store_true');a=p.parse_args()
    r=run(a.counts,a.references,a.output,a.workers,a.resume);print(json.dumps({k:v for k,v in r.items() if k not in ['artifacts','identity','reproduce']}))
