"""All-condition descriptions and complete eligible CRISPRi response diagnostics."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor,as_completed
import json
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
from chemical_diagnostics import distribution_contrast,PARAMETERS as DESCRIPTION_PARAMETERS
from profile_background import matrix_in_memory
from profile_responses import PARAMETERS,task_result,write_json
from response import grouped_moments,control_half_means
from rna import RNAFile,hash_file,value_hash


def describe_conditions(cells,design,directory):
    """Observe every condition; detailed contrasts require 20 targets/2 controls."""
    directory.mkdir();frame=design.copy()
    names=[c for c in cells if c.startswith('state__')]+['log1p_expression_total','log1p_detected_genes']
    for name in names[:-2]:frame[name]=cells[name].to_numpy()
    frame[names[-2]]=np.log1p(cells.computed_total_expression.to_numpy());frame[names[-1]]=np.log1p(cells.computed_detected_genes.to_numpy())
    frame['inferred_type']=cells.inferred_type.to_numpy()
    frame['condition_id']=[value_hash([a,b]) for a,b in zip(frame.response_background,frame.analysis_condition)]
    grouped=frame.groupby('condition_id',sort=True)
    means=grouped[names].mean().add_suffix('__mean');medians=grouped[names].median().add_suffix('__median')
    q10=grouped[names].quantile(.1).add_suffix('__q10');q90=grouped[names].quantile(.9).add_suffix('__q90')
    observed=pd.concat([grouped.size().rename('n_cells'),means,medians,q10,q90],axis=1).reset_index()
    observed.to_parquet(directory/'all-condition-observed-distributions.parquet',index=False)
    types=frame.groupby(['condition_id','inferred_type'],sort=True).size().rename('n_cells').reset_index();types['truth_label']=False
    types.to_parquet(directory/'all-condition-type-composition.parquet',index=False)
    controls={key:g.index.to_numpy() for key,g in frame.loc[frame.control_eligible].groupby('response_background',sort=False)}
    rows=[];states=[];composition=[];within=[];draws=[]
    for condition,group in grouped:
        bg=group.response_background.iloc[0];ids=controls.get(bg,np.array([],int));ref=frame.loc[ids]
        row={'condition_id':condition,'response_background':bg,'analysis_condition':group.analysis_condition.iloc[0],
             'n_cells':len(group),'eligible_reference_cells':len(ref),'observation_status':'completed',
             'proxy_scale':'Fixed species-specific endpoint RNA scores from per-cell phase; not cross-background calibrated'}
        if group.control_eligible.all():row.update(status='not_applicable',reason='reference_condition_no_self_comparison',stability_status='not_applicable')
        elif not group.condition_identity_supported.all():row.update(status='not_estimable',reason='source_condition_identity_not_supported',stability_status='not_estimable')
        elif len(group)<20 or len(ref)<2:row.update(status='not_estimable',reason='requires_20_targets_2_eligible_same_background_controls_for_detailed_contrast',stability_status='not_estimable')
        else:
            if np.intersect1d(group.index,ref.index).size:raise ValueError('target_reference_cells_overlap')
            result,st,co,wi,dr=distribution_contrast(group[names].to_numpy(float),ref[names].to_numpy(float),group.inferred_type.to_numpy(),ref.inferred_type.to_numpy(),names,
                                                  DESCRIPTION_PARAMETERS['seed']^int(condition[:8],16))
            row.update(result)
            for out,values in [(states,st),(composition,co),(within,wi),(draws,dr)]:out.extend([{'condition_id':condition,**v} for v in values])
        rows.append(row)
    for name,records in [('condition-contrasts',rows),('state-contrasts',states),('type-contrasts',composition),('within-type-contrasts',within),('resampling',draws)]:
        pd.DataFrame(records).to_parquet(directory/(name+'.parquet'),index=False)
    return {'all_conditions':len(rows),'observed_distribution_conditions':len(observed),'contrast_status_counts':dict(Counter(r['status'] for r in rows)),
            'resampling_rows':len(draws),'null_reference_intersections':sum(r['null_reference_intersection'] for r in draws),
            'null_size_shortfall_rows':sum(r['null_target_cells_not_matched']>0 for r in draws)}


def count_views(path,rows,directory,seed):
    rows=np.asarray(rows,int)
    if not len(rows) or (np.diff(rows)<=0).any():raise ValueError('selected_rows_must_be_ordered_unique')
    directory.mkdir();rng=np.random.default_rng(seed)
    with RNAFile(path) as source:
        selection=np.full(source.shape[0],-1,int);selection[rows]=np.arange(len(rows));g=source.shape[1]
        log=np.lib.format.open_memmap(directory/'log.npy',mode='w+',dtype='float32',shape=(len(rows),g));thin=np.lib.format.open_memmap(directory/'thin.npy',mode='w+',dtype='float32',shape=log.shape)
        memory=matrix_in_memory(source) if source.encoding=='csc_matrix' else None
        def blocks():
            if memory is None:yield from source.blocks(1024)
            else:
                for start in range(0,source.shape[0],1024):yield start,memory[start:start+1024].astype(float)
        for start,matrix in blocks():
            destinations=selection[start:start+matrix.shape[0]];keep=destinations>=0
            if not keep.any():continue
            x=matrix[keep].astype(float);destinations=destinations[keep];totals=np.asarray(x.sum(axis=1)).ravel()
            if not np.isfinite(x.data).all() or (x.data<0).any() or (x.data!=np.floor(x.data)).any() or (totals<=0).any():raise ValueError('invalid_count_diagnostic_input')
            t=x.copy();t.data=rng.binomial(x.data.astype(np.int64),np.repeat(np.minimum(1,10000/totals),np.diff(x.indptr))).astype(float);t.eliminate_zeros()
            ts=np.asarray(t.sum(axis=1)).ravel();scale=np.divide(10000.,ts,out=np.zeros_like(ts),where=ts>0)
            t.data=np.log1p(t.data*np.repeat(scale,np.diff(t.indptr)));x.data=np.log1p(x.data*np.repeat(10000/totals,np.diff(x.indptr)))
            log[destinations]=x.toarray();thin[destinations]=t.toarray()
        log.flush();thin.flush();del log,thin,memory
    identity={'rows_sha256':value_hash(rows.tolist()),'seed':seed,'log_sha256':hash_file(directory/'log.npy'),'thin_sha256':hash_file(directory/'thin.npy'),
              'interpretation':'Binomial thinning of source count-compatible X; no claim that harmonized X is original UMI molecules'}
    return np.load(directory/'log.npy',mmap_mode='r'),np.load(directory/'thin.npy',mmap_mode='r'),identity


def deep_responses(file,path,cells,design,mapping,directory,workers=2):
    directory.mkdir();positive=cells.computed_numeric_valid&(cells.computed_total_expression>0)
    eligible=(design.deep_response_candidate|design.control_eligible)&positive
    selected=design.loc[eligible].copy();selected['computed_total_counts']=cells.computed_total_expression.loc[eligible].to_numpy();selected['computed_detected_genes']=cells.computed_detected_genes.loc[eligible].to_numpy()
    selected['source_batch']=selected.response_background;selected['source_guide_id']=selected.source_guide_for_consistency
    declared=design.loc[design.deep_response_candidate].groupby(['biological_background_index','single_target','target_kind'],sort=True).size().rename('all_candidate_cells').reset_index()
    genes=mapping.source_gene.tolist();safe=mapping.mapped_symbol.where(~mapping.many_to_one_mapping&(mapping.symbol_vs_ensembl!='conflict'))
    native={symbol:gene for gene,symbol in zip(genes,safe) if pd.notna(symbol)}
    conflicts=set(mapping.loc[safe.isna(),'source_gene']);summaries=[];draws=[];views=[]
    # A separate bounded view per biological context prevents stimulation mixing.
    for bg,tasks in declared.groupby('biological_background_index',sort=True):
        local=selected.loc[selected.biological_background_index==bg].copy().reset_index(drop=True)
        ntc=np.flatnonzero(local.control_eligible);has_controls=len(ntc)>0
        context=str(bg);temp=directory/('temporary-'+context)
        if has_controls:
            log,thin,identity=count_views(path,local.row_index.to_numpy(),temp,PARAMETERS['seed']^int(value_hash([file,context])[:8],16));views.append({'context':context,**identity})
            batches=sorted(local.source_batch.unique());cb=local.source_batch.iloc[ntc].to_numpy();control=log[ntc];control_thin=thin[ntc]
            moments=grouped_moments(control,cb,batches);thin_moments=grouped_moments(control_thin,cb,batches)
            halves=control_half_means(control,cb,batches,PARAMETERS['resampling_repetitions'],PARAMETERS['seed']+10);pools={b:np.flatnonzero(cb==b) for b in batches}
        def assess(item):
            target=item.single_target;kind=item.target_kind;task_key=value_hash([file,int(bg),target]);destination=directory/('task-'+task_key[:20]);destination.mkdir()
            ids=np.flatnonzero((local.single_target==target)&local.deep_response_candidate)
            if not has_controls or not len(ids):result={'status':'not_estimable','reason':'no_eligible_controls_or_positive_target_counts'};features=None;samples=[];used=np.array([],int)
            else:
                used=ids[local.source_batch.iloc[ids].isin(set(cb)).to_numpy()]
                if not len(used):result={'status':'not_estimable','reason':'no_same_source_background_control'};features=None;samples=[]
                else:
                    # The own-target RNA mean and every downstream statistic use
                    # the same matched target rows; unmatched records stay indexed.
                    result,features,samples=task_result(native.get(target,'__unresolved_target__'+target),log[used],thin[used],local.iloc[used],control,cb,batches,moments,thin_moments,halves,genes,conflicts,
                        PARAMETERS['seed']^int(task_key[:8],16),control_pools=pools)
            result.update(task=task_key,source_file=file,biological_background_index=int(bg),target=target,target_kind=kind,
                n_source_candidate_cells=int(item.all_candidate_cells),n_positive_target_cells=len(ids),n_matched_target_cells=len(used),
                n_unmatched_or_nonpositive_source_candidate_cells=int(item.all_candidate_cells)-len(used),independent_biological_replicates=None,
                measurement_interpretation='source count-compatible X RNA association; UMI semantics conditional on source processing')
            if kind=='single_source_regulatory_locus':result['target_RNA']={'status':'not_applicable','reason':'regulatory_locus_is_not_a_measured_RNA_gene'}
            if features is not None:
                features.to_parquet(destination/'genes.parquet',index=False);result['gene_results']={'file':str((destination/'genes.parquet').relative_to(directory)),'sha256':hash_file(destination/'genes.parquet')}
            write_json(destination/'result.json',result);return result,[{'task':task_key,**r} for r in samples]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures=[pool.submit(assess,item) for item in tasks.itertuples()]
            for future in as_completed(futures):
                result,samples=future.result();summaries.append(result);draws.extend(samples)
                if len(summaries)%100==0:print(file+' deep tasks '+str(len(summaries))+'/'+str(len(declared)),flush=True)
        if has_controls:
            del log,thin,control,control_thin,moments,thin_moments,halves,pools;shutil.rmtree(temp)
    summaries.sort(key=lambda r:r['task']);write_json(directory/'tasks.json',summaries);write_json(directory/'analysis-views.json',views)
    pd.DataFrame(draws).to_parquet(directory/'resampling.parquet',index=False)
    return {'declared_tasks':len(declared),'actual_tasks':len(summaries),'task_status_counts':dict(Counter(r['status'] for r in summaries)),
        'DE_status_counts':dict(Counter(r.get('DE',{}).get('status','not_estimable') for r in summaries)),'resampling_rows':len(draws),
        'null_reference_intersections':sum(r['null_reference_intersection'] for r in draws),'null_size_shortfall_rows':sum(r['null_target_cells_not_matched']>0 for r in draws),
        'zero_expression_records_not_used':int((~positive).sum()),'parameters':PARAMETERS}
