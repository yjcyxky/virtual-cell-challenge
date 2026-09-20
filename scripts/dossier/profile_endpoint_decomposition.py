#!/usr/bin/env python
"""Full native-gene descriptive mixture identities on registered endpoint strata."""
import argparse
from collections import Counter
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import sys
import time
import h5py
import numpy as np
import pandas as pd
from scipy import sparse
from heterogeneity_cells import CellContexts
from heterogeneity import safe_symbols,endpoint_cycle_partition,mixture_identity,compare_vectors
from profile_cross_source_coverage import ROOT
from profile_responses import write_json
from rna import hash_file,quantiles

PARAMETERS={'partitions':['endpoint_inferred_type','endpoint_cycle_RNA_tercile'],'minimum_supported_target_cells':10,
            'minimum_NTC_per_state_technical_stratum':2,'normalization_total':10000,'source_mean_verification_atol':2e-6,
            'mixture_identity_maximum_error':1e-9,'unknown_type_is_biological_truth':False,'causal_interpretation':False,
            'preregistered_issue_comment':'https://github.com/yjcyxky/virtual-cell-challenge/issues/20#issuecomment-5746896397'}
CODE=['profile_endpoint_decomposition.py','heterogeneity_cells.py','heterogeneity.py','profile_response_heterogeneity.py','rna.py']


def grouped_sums(log,indices,codes):
    indices=np.asarray(indices,dtype=int);codes=np.asarray(codes,dtype=int)
    if not len(indices):return np.array([],int),np.array([],int),np.empty((0,log.shape[1]),float)
    unique,inverse=np.unique(codes,return_inverse=True);counts=np.bincount(inverse)
    assignment=sparse.csr_matrix((np.ones(len(indices)),(inverse,np.arange(len(indices)))),shape=(len(unique),len(indices)))
    sums=assignment@log[indices]
    return unique,counts,sums.toarray() if sparse.issparse(sums) else np.asarray(sums,dtype=float)


def control_model(log,control,codes,total_groups):
    ids=np.flatnonzero(control);groups,counts,sums=grouped_sums(log,ids,codes[ids])
    lookup=np.full(total_groups,-1,dtype=int);lookup[groups]=np.arange(len(groups))
    return {'groups':groups,'counts':counts,'means':sums/counts[:,None] if len(counts) else sums,'lookup':lookup}


def assess_context(context,output,identity):
    output.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    cells=context['cells'];log=context['log'];genes=context['genes'];tasks=context['tasks'];control=context['control'];labels=context['task_labels']
    if len(cells.source_batch) and cells.source_batch.isna().any():raise ValueError('missing_frozen_technical_stratum')
    batch_codes,batch_names=pd.factorize(cells.source_batch.to_numpy(),sort=True);batch_codes=batch_codes.astype(int)
    baseline_model=control_model(log,control,batch_codes,len(batch_names))
    baseline=np.sum(baseline_model['means']*baseline_model['counts'][:,None],axis=0)/int(control.sum()) if control.any() else np.full(len(genes),np.nan)
    np.savez_compressed(output/'baseline.npz',source_gene=np.asarray(genes,dtype=str),mean_logCP10K=baseline,n_NTC=int(control.sum()),baseline_id=context['baseline_id'])
    np.savez_compressed(output/'baseline-by-source-stratum.npz',source_gene=np.asarray(genes,dtype=str),source_batch=np.asarray(batch_names[baseline_model['groups']],dtype=str),mean_logCP10K=baseline_model['means'],n_NTC=baseline_model['counts'])
    safe=np.asarray(safe_symbols(context['mapping']),dtype=object);safe_valid=np.array([pd.notna(x) for x in safe])
    if len(safe)!=len(genes):raise ValueError('native_gene_mapping_length_changed')
    partition_labels={'endpoint_inferred_type':cells.inferred_type.fillna('unknown').to_numpy()}
    cycle,cycle_meta=endpoint_cycle_partition(cells.state__cycle_s,cells.state__cycle_g2m,control)
    partition_labels['endpoint_cycle_RNA_tercile']=cycle
    partition_metadata={'endpoint_inferred_type':{'status':'completed','truth_label':False,'unknown_is_unresolved_annotation':True},'endpoint_cycle_RNA_tercile':cycle_meta}
    partition_models={};sidecar=cells[['record_id','input_sha256','row_index','source_barcode','source_batch','inferred_type','probability_correct','confidence_calibration']].copy()
    sidecar['comparison_task_uid']=labels;sidecar['registered_reference_eligible']=control;sidecar['available_before_endpoint']=False;sidecar['truth_label']=False
    for partition,state_labels in partition_labels.items():
        state_codes,state_names=pd.factorize(np.asarray(state_labels),sort=True);combined=batch_codes*len(state_names)+state_codes
        model=control_model(log,control,combined,len(batch_names)*len(state_names))
        partition_models[partition]={**model,'codes':combined,'state_names':state_names,'states_per_batch':len(state_names)}
        sidecar[partition]=state_labels
        partition_metadata[partition].update(state_names=state_names.tolist(),NTC_state_counts=pd.Series(state_labels[control]).value_counts().to_dict(),
            all_source_analysis_rows=len(cells),source_NTC_rows=int(control.sum()),probability_correct=None)
    sidecar.to_parquet(output/'cell-partitions.parquet',index=False,compression='zstd');write_json(output/'partition-methods.json',partition_metadata)
    task_groups={key:np.asarray(indices,dtype=int) for key,indices in pd.Series(labels).groupby(pd.Series(labels),dropna=True,sort=False).groups.items()}
    summary_rows=[];composition_rows=[];support_rows=[];verification=[]
    positions={uid:i for i,uid in enumerate(tasks.task_uid)}
    with h5py.File(output/'native-gene-components.h5','w') as hf:
        hf.create_dataset('source_gene',data=np.asarray(genes,dtype=h5py.string_dtype()))
        hf.create_dataset('task_uid',data=np.asarray(tasks.task_uid,dtype=h5py.string_dtype()))
        for partition in PARAMETERS['partitions']:
            group=hf.create_group(partition)
            for name in ['total_common_support','composition','within','difference_from_full_matched_effect']:
                group.create_dataset(name,shape=(len(tasks),len(genes)),dtype='float32',chunks=(1,min(len(genes),4096)),compression='gzip',compression_opts=1,fillvalue=np.nan)
        for ti,task in enumerate(tasks.to_dict('records')):
            uid=task['task_uid'];ids=task_groups.get(uid,np.array([],dtype=int));native_row=positions[uid]
            groups,counts,sums=grouped_sums(log,ids,batch_codes[ids]);lookup=baseline_model['lookup'][groups] if len(groups) else np.array([],int);matched=lookup>=0
            n_matched=int(counts[matched].sum());full_effect=None
            if n_matched:
                mean_target=sums[matched].sum(axis=0)/n_matched
                mean_control=np.sum(baseline_model['means'][lookup[matched]]*(counts[matched]/n_matched)[:,None],axis=0)
                full_effect=mean_target-mean_control
            original_path=task['source_gene_result_file'];maximum_error=None;source_is_target=np.zeros(len(genes),dtype=bool)
            if task['effect_status']=='completed':
                path=ROOT/task['source_folder']/original_path
                if hash_file(path)!=task['source_gene_result_sha256']:raise ValueError('source_gene_evidence_changed')
                original=pd.read_parquet(path,columns=['source_gene','is_target','effect_all_matched_cells','mean_logCP10K_target','mean_logCP10K_matched_control'])
                source_is_target=original.is_target.to_numpy(dtype=bool)
                if original.source_gene.astype(str).tolist()!=genes or full_effect is None:raise ValueError('native_source_effect_or_gene_axis_not_reproduced')
                diffs=[]
                for a,b in [(full_effect,original.effect_all_matched_cells.to_numpy()),(mean_target,original.mean_logCP10K_target.to_numpy()),(mean_control,original.mean_logCP10K_matched_control.to_numpy())]:
                    diffs.append(float(np.max(np.abs(a-b))))
                    if not np.allclose(a,b,rtol=2e-5,atol=PARAMETERS['source_mean_verification_atol']):raise ValueError('native_source_mean_or_effect_not_reproduced:'+uid+':'+str(diffs[-1]))
                maximum_error=max(diffs)
            elif n_matched:raise ValueError('source_not_estimable_but_new_matched_targets_exist:'+uid)
            verification.append({'task_uid':uid,'source_status':task['effect_status'],'source_mean_comparison_status':'completed' if maximum_error is not None else 'not_applicable',
                                 'maximum_absolute_mean_or_effect_error':maximum_error,'matched_target_cells_reconstructed':n_matched,'observed_target_cells':len(ids)})
            downstream=safe_valid & (safe!=task['canonical_target']) & ~source_is_target
            for partition,model in partition_models.items():
                base={'task_uid':uid,'panel_id':context['panel_id'],'context_id':context['context_id'],'source_context':context['source_context'],
                      'source_task':task['source_task'],'source_target':task['source_target'],'canonical_target':task['canonical_target'],'partition':partition,
                      'source_effect_status':task['effect_status'],'native_gene_file':'native-gene-components.h5','native_gene_row':native_row,
                      'safe_downstream_genes':int(downstream.sum()),'partition_truth_label':False,'causal_interpretation':False}
                state_n=model['states_per_batch'];tc,nt,tsum=grouped_sums(log,ids,model['codes'][ids]);tm=tsum/nt[:,None] if len(nt) else tsum
                match=model['lookup'][tc] if len(tc) else np.array([],int);available=match>=0
                nc=np.zeros(len(tc));cm=np.full_like(tm,np.nan)
                nc[available]=model['counts'][match[available]];cm[available]=model['means'][match[available]]
                supported=(nc>=2)&(nt>0);batches=tc//state_n if state_n else np.array([],dtype=int)
                # Proportions use every original matched stratum, including
                # observed zero state counts; expression for absent states is
                # never imputed by this proportion calculation.
                matched_counts={int(b):int(n) for b,n in zip(groups[matched],counts[matched])}
                p_target=np.zeros(state_n);p_control=np.zeros(state_n)
                if n_matched:
                    for code,n in zip(tc,nt):
                        if int(code//state_n) in matched_counts:p_target[code%state_n]+=n/n_matched
                    for code,n in zip(model['groups'],model['counts']):
                        b=int(code//state_n)
                        if b in matched_counts:
                            ctrl_n=baseline_model['counts'][baseline_model['lookup'][b]]
                            p_control[code%state_n]+=matched_counts[b]/n_matched*n/ctrl_n
                for j,label in enumerate(model['state_names']):
                    composition_rows.append({**{k:base[k] for k in ['task_uid','context_id','partition']},'endpoint_partition_label':label,
                        'matched_target_fraction':float(p_target[j]) if n_matched else None,'matched_NTC_fraction':float(p_control[j]) if n_matched else None,
                        'fraction_difference':float(p_target[j]-p_control[j]) if n_matched else None,'status':'completed' if n_matched else 'not_estimable',
                        'truth_label':False,'unknown_state_label':'unknown' in str(label)})
                for j,code in enumerate(tc):
                    support_rows.append({'task_uid':uid,'partition':partition,'source_batch':str(batch_names[code//state_n]),'endpoint_partition_label':str(model['state_names'][code%state_n]),
                        'target_cells':int(nt[j]),'NTC_cells':int(nc[j]),'common_support':bool(supported[j]),'reason':None if supported[j] else 'fewer_than_2_same_state_NTC'})
                if partition_metadata[partition]['status']!='completed':
                    result=None;meta={'status':'not_estimable','reason':partition_metadata[partition]['reason'],'observed_target_cells':len(ids),
                                      'supported_target_cells':None,'supported_target_fraction':None,'causal_interpretation':False}
                else:result,meta=mixture_identity(tm,cm,nt,nc,batches,PARAMETERS['minimum_supported_target_cells'])
                row={**base,**meta,'full_matched_target_cells':n_matched,'inferred_partition_total_variation':float(np.abs(p_target-p_control).sum()/2) if n_matched else None}
                unknown=np.array(['unknown' in str(s) for s in model['state_names']])
                row['target_unknown_partition_fraction']=float(p_target[unknown].sum()) if n_matched else None
                row['NTC_unknown_partition_fraction']=float(p_control[unknown].sum()) if n_matched else None
                if result is not None:
                    for name,vector in [('total_common_support',result['total']),('composition',result['composition']),('within',result['within']),('difference_from_full_matched_effect',result['total']-full_effect)]:
                        hf[partition][name][native_row]=vector.astype(np.float32)
                        row[name+'_RMS_safe_downstream']=float(np.sqrt(np.mean(vector[downstream]**2))) if downstream.any() else None
                    row['common_support_vs_full_effect_correlation']=compare_vectors(result['total'][downstream],full_effect[downstream])['correlation']
                    retained=int(nt[supported].sum());control_support=0.
                    for batch in np.unique(batches[supported]):
                        local=supported & (batches==batch);total_control=baseline_model['counts'][baseline_model['lookup'][batch]]
                        control_support+=nt[local].sum()/retained*nc[local].sum()/total_control
                    row['NTC_common_state_support_fraction_target_batch_weighted']=float(control_support)
                summary_rows.append(row)
            if (ti+1)%100==0:print(context['panel_id']+' endpoint identities '+str(ti+1)+'/'+str(len(tasks)),flush=True)
    pd.DataFrame(summary_rows).to_parquet(output/'task-partition-diagnostics.parquet',index=False,compression='zstd')
    pd.DataFrame(composition_rows).to_parquet(output/'all-task-partition-compositions.parquet',index=False,compression='zstd')
    pd.DataFrame(support_rows).to_parquet(output/'all-task-state-stratum-support.parquet',index=False,compression='zstd')
    pd.DataFrame(verification).to_parquet(output/'all-source-effect-reproduction.parquet',index=False)
    context['mapping'].to_parquet(output/'native-gene-mapping.parquet',index=False)
    report={'status':'completed','identity':identity,'context_id':context['context_id'],'panel_id':context['panel_id'],'source_context':context['source_context'],
            'source_metadata':context['source_metadata'],'baseline_id':context['baseline_id'],'source_analysis_cells':len(cells),'NTC_cells':int(control.sum()),
            'registered_tasks':len(tasks),'partition_tasks':len(summary_rows),'partition_status_counts':dict(Counter(r['status'] for r in summary_rows)),
            'source_effect_reproduction_status':dict(Counter(r['source_mean_comparison_status'] for r in verification)),
            'maximum_source_mean_reproduction_error':max([r['maximum_absolute_mean_or_effect_error'] for r in verification if r['maximum_absolute_mean_or_effect_error'] is not None],default=None),
            'maximum_mixture_identity_residual':max([r.get('maximum_identity_residual',0) for r in summary_rows],default=None),
            'partition_methods':partition_metadata,'independent_biological_replicates':None,'duration_seconds':time.monotonic()-started,
            'completed_at':datetime.now(timezone.utc).isoformat()}
    report['artifacts']={str(p.relative_to(output)):hash_file(p) for p in output.iterdir() if p.is_file()};write_json(output/'report.json',report)
    return report


def run(source,output):
    output=output.resolve();output.mkdir(parents=True,exist_ok=False);start=time.monotonic();contexts=CellContexts(source)
    identity={'source_report_sha256':hash_file(source/'report.json'),'parameters':PARAMETERS,'code_sha256':{n:hash_file(Path(__file__).with_name(n)) for n in CODE},'uv_lock_sha256':hash_file(Path(__file__).with_name('uv.lock'))}
    write_json(output/'identity.json',identity);results=[]
    for context in contexts.iter_contexts():
        report=assess_context(context,output/context['context_id'],identity);results.append(report)
        print(json.dumps({k:report[k] for k in ['panel_id','source_context','registered_tasks','partition_status_counts','maximum_source_mean_reproduction_error']}),flush=True)
    if sum(r['registered_tasks'] for r in results)!=37845:raise ValueError('incomplete_registered_endpoint_task_universe')
    inputs=contexts.verify_unchanged();write_json(output/'consumed-inputs.json',inputs)
    summary={'status':'completed','phase':'full_native_gene_endpoint_partition_identities','identity':identity,'contexts':results,
             'registered_tasks':sum(r['registered_tasks'] for r in results),'partition_tasks':sum(r['partition_tasks'] for r in results),
             'partition_status_counts':dict(sum((Counter(r['partition_status_counts']) for r in results),Counter())),
             'all_consumed_count_views_unchanged':True,'original_input_mutations':0,'independent_biological_replicates':None,
             'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'completed_at':datetime.now(timezone.utc).isoformat(),'duration_seconds':time.monotonic()-start,'reproduce':sys.argv}
    summary['artifacts']={str(p.relative_to(output)):hash_file(p) for p in sorted(output.rglob('*')) if p.is_file()};write_json(output/'report.json',summary)
    (output/'SHA256SUMS').write_text(''.join(hash_file(p)+'  '+str(p.relative_to(output))+'\n' for p in sorted(output.rglob('*')) if p.is_file() and p.name!='SHA256SUMS'))
    print(json.dumps({k:v for k,v in summary.items() if k not in ['artifacts','contexts','identity']}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    try:run(a.source,a.output)
    except Exception as error:
        if a.output.exists():write_json(a.output/'failure.json',{'status':'failed','error':repr(error),'code_sha256':hash_file(Path(__file__))})
        raise
