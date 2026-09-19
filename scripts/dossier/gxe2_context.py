"""All declared GxE2 single-gene tasks, with source assignments and thresholds separate."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor,as_completed
import json
from pathlib import Path
import shutil
import time
import h5py
import numpy as np
import pandas as pd
from scipy import sparse
from annotation import RULES,marker_model,state_model
from sparse_annotation import annotate_sparse
from conditioned import normalize_selection
from profile_crispri import state_composition
from profile_responses import PARAMETERS,task_result,write_json
from response import grouped_moments,matched_effect,control_half_means,correlation
from rna import hash_file,value_hash,quantiles


def matched_rows(cells,target,threshold):
    eligible=cells.source_assignment_supported&(cells.source_CDS_gRNA_maxCount>=threshold)
    controls=np.flatnonzero(eligible&(cells.analysis_target=='NTC'))
    targets=np.flatnonzero(eligible&(cells.analysis_target==target))
    present=set(cells.source_batch.iloc[controls]);used=targets[cells.source_batch.iloc[targets].isin(present).to_numpy()]
    return used,controls,len(targets)


def threshold_RNA(cells,log,target,feature):
    rows=[]
    for threshold in range(1,11):
        used,control,observed=matched_rows(cells,target,threshold)
        result={'target':target,'minimum_guide_reads':threshold,'source_target_cells':observed,'matched_target_cells':len(used),'NTC_cells':len(control)}
        if feature is None:result.update(status='not_applicable' if target=='random' else 'not_estimable',reason='random_control_has_no_single_target' if target=='random' else 'target_gene_not_uniquely_mapped')
        elif not len(used):result.update(status='not_estimable',reason='no_target_and_same_hash_well_NTC')
        else:
            target_mean=float(np.expm1(log[used,feature].astype(float)).mean());control_mean=0.
            for batch,count in cells.source_batch.iloc[used].value_counts().items():
                ids=control[cells.source_batch.iloc[control].eq(batch).to_numpy()]
                control_mean+=count/len(used)*float(np.expm1(log[ids,feature].astype(float)).mean())
            result.update(status='completed' if control_mean>0 else 'not_estimable',reason=None if control_mean>0 else 'zero_target_RNA_in_NTC',mean_CP10K_target=target_mean,mean_CP10K_control=control_mean,RNA_ratio=target_mean/control_mean if control_mean>0 else None)
        rows.append(result)
    return rows


def assess_context(path,cells,genes,mapping,reference,targets,vehicle_weights,output,context,identity,workers=4):
    started=time.monotonic()
    if (output/'report.json').exists():
        old=json.loads((output/'report.json').read_text())
        if old['identity']!=identity:raise ValueError('GxE2_context_resume_identity_changed')
        for name,digest in old['artifacts'].items():
            if hash_file(output/name)!=digest:raise ValueError('GxE2_completed_context_changed')
        return old
    output.mkdir(parents=True,exist_ok=False);cells=cells.sort_values('CDS_view_row').reset_index(drop=True)
    log,thin,view=normalize_selection(path,cells.CDS_view_row.to_numpy(),output/'temporary-view',PARAMETERS['seed']^int(value_hash(context)[:8],16))
    safe=mapping.mapped_symbol.where(~mapping.many_to_one_mapping&(mapping.symbol_vs_ensembl!='conflict'))
    resolved=[str(x) if pd.notna(x) else None for x in safe];native={symbol:gene for symbol,gene in zip(resolved,genes.source_gene) if symbol is not None}
    positions={symbol:i for i,symbol in enumerate(resolved) if symbol is not None};conflicts=set(mapping.loc[safe.isna(),'source_gene'])
    primary=cells.source_assignment_supported&(cells.source_CDS_gRNA_maxCount>=10)
    ntc=np.flatnonzero(primary&(cells.analysis_target=='NTC'));batches=sorted(cells.source_batch.unique());cb=cells.source_batch.iloc[ntc].to_numpy()
    control=log[ntc];control_thin=thin[ntc];controls=grouped_moments(control,cb,batches);thin_controls=grouped_moments(control_thin,cb,batches)
    halves=control_half_means(control,cb,batches,20,PARAMETERS['seed']+10);pools={b:np.flatnonzero(cb==b) for b in batches}
    broad_ntc=np.flatnonzero(cells.source_assignment_supported&(cells.analysis_target=='NTC')&(cells.source_CDS_gRNA_maxCount>=1))
    broad_controls=grouped_moments(log[broad_ntc],cells.source_batch.iloc[broad_ntc].to_numpy(),batches)
    base=control.mean(axis=0,dtype=float) if len(ntc) else np.asarray(log).mean(axis=0,dtype=float)
    np.savez_compressed(output/'baseline.npz',gene_ids=genes.source_gene.to_numpy(dtype=str),mean_logCP10K=base,n_NTC=len(ntc))
    np.savez_compressed(output/'baseline-by-hash-well.npz',hash_wells=np.asarray(batches),mean_logCP10K=np.stack([r['mean'] for r in controls]),n_NTC=np.asarray([r['n'] for r in controls]))

    def one_task(target):
        used,_,observed=matched_rows(cells,target,10);feature=native.get(target,'__NO_UNIQUE_TARGET_FEATURE__')
        seed=PARAMETERS['seed']^int(value_hash([context,target])[:8],16)
        result,per_gene,draws=task_result(feature,log[used],thin[used],cells.iloc[used],control,cb,batches,controls,thin_controls,halves,genes.source_gene.tolist(),conflicts,seed,control_pools=pools)
        result.update(target=target,task=target,context=context,source_target_feature=native.get(target),observed_primary_target_cells=observed,matched_primary_target_cells=len(used),unmatched_primary_target_cells=observed-len(used))
        if target=='random':result['target_RNA']={'status':'not_applicable','reason':'random_genomic_region_control_has_no_single_RNA_target'}
        if target=='PRKCZ' and 'consistency' in result:
            result['consistency']['guide']={'status':'not_estimable','reason':'five_source_guide_names_each_map_to_two_distinct_sequences'}
        broad,_,broad_observed=matched_rows(cells,target,1)
        sensitivity={'all_source_supported_target_cells':broad_observed,'matched_source_supported_target_cells':len(broad),'primary_read_threshold':10,'source_supported_threshold':1}
        if len(broad) and result['status']=='completed':
            effect,match=matched_effect(grouped_moments(log[broad],cells.source_batch.iloc[broad].to_numpy(),batches),broad_controls)
            mask=genes.source_gene.to_numpy()!=native.get(target,'__NO_TARGET__')
            sensitivity.update(status='completed',correlation_with_primary=correlation(effect[mask],per_gene.effect_all_matched_cells.to_numpy()[mask]),source_supported_RMS=float(np.sqrt(np.mean(effect[mask]**2))),interpretation='Low-guide-read source assignments retained as sensitivity; not proof of incorrect or correct assignments')
        else:sensitivity.update(status='not_estimable',reason='primary_or_broad_matched_effect_unavailable')
        directory=output/'tasks'/value_hash(target)[:20];directory.mkdir(parents=True)
        if per_gene is not None:
            per_gene.to_parquet(directory/'genes.parquet',index=False,compression='zstd');result['gene_results']=str((directory/'genes.parquet').relative_to(output))
        result['source_assignment_sensitivity']=sensitivity
        write_json(directory/'result.json',result)
        return result,[{'target':target,**r} for r in draws],threshold_RNA(cells,log,target,positions.get(target))

    results=[];draws=[];thresholds=[]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(one_task,t) for t in targets]):
            result,draw,threshold=future.result();results.append(result);draws.extend(draw);thresholds.extend(threshold)
            if len(results)%100==0:print(context+' genetic tasks '+str(len(results))+'/'+str(len(targets)),flush=True)
    results.sort(key=lambda r:r['target']);write_json(output/'tasks.json',results)
    pd.DataFrame(draws).to_parquet(output/'resampling.parquet',index=False);pd.DataFrame(thresholds).to_parquet(output/'guide-read-threshold-sensitivity.parquet',index=False)
    model=marker_model(resolved,reference['profiles']);weights,coverage=state_model(resolved,reference['states'],base)
    parts=[]
    for start in range(0,len(cells),512):
        stop=min(start+512,len(cells));block=np.asarray(log[start:stop]);target_labels=cells.analysis_target.iloc[start:stop].where(cells.source_assignment_supported.iloc[start:stop],None).tolist()
        labels,scores=annotate_sparse(sparse.csr_matrix(block),target_labels,model)
        for name,values in scores.items():labels[name]=values
        states=block@weights;vehicle_states=block@vehicle_weights['weights']
        for j,(name,cov) in enumerate(zip(reference['states'],coverage)):
            labels['state__'+name]=states[:,j] if cov['status']=='completed' else np.nan
            labels['vehicle_state__'+name]=vehicle_states[:,j] if vehicle_weights['coverage'][j]['status']=='completed' else np.nan
        labels['source_context_mismatch']=~labels.inferred_lineage.isin(['neural','unknown']);parts.append(labels)
    combined=pd.concat([cells,pd.concat(parts,ignore_index=True)],axis=1)
    combined['primary_genetic_response_eligible']=primary;combined['state_background']='same_drug_dose_line_primary_NTC' if len(ntc) else 'pooled_endpoint_fallback_no_NTC'
    combined['vehicle_state_background']='same_cell_line_vehicle_primary_NTC_shared_across_drugs';combined['available_before_endpoint']=False;combined['truth_label']=False
    combined.to_parquet(output/'cells.parquet',index=False,compression='zstd')
    genetic=combined.loc[primary].copy();genetic['source_target_gene']=genetic.analysis_target.replace({'NTC':'non-targeting'});genetic['source_task']=genetic.analysis_target
    composition,states,within,type_tasks=state_composition(genetic,list(reference['states']))
    within=within.rename(columns={'shared_GEM_groups':'shared_hash_wells'})
    for table in [composition,states,within]:
        if 'reason' in table:table['reason']=table.reason.str.replace('GEM','hash_well',regex=False)
    for name,table in [('composition',composition),('states',states),('within-type-states',within)]:table.to_parquet(output/(name+'.parquet'),index=False)
    write_json(output/'type-tasks.json',type_tasks);write_json(output/'type-coverage.json',model['coverage']);write_json(output/'state-coverage.json',coverage)
    np.savez_compressed(output/'scoring-weights.npz',type_weights=model['weights'],state_weights=weights,vehicle_state_weights=vehicle_weights['weights'],valid_gene_axis=model['valid_axis'])
    # Unmatched primary records remain in these condition profiles for later chemical descriptions.
    genotype_rows=[]
    with h5py.File(output/'condition-genotype-profiles.h5','w') as h:
        profile=h.create_dataset('mean_logCP10K',shape=(len(targets)+1,len(genes)),dtype='float32',compression='gzip',compression_opts=1,chunks=(1,len(genes)),fillvalue=np.nan)
        for j,target in enumerate(['NTC',*targets]):
            ids=np.flatnonzero(primary&(cells.analysis_target==target))
            if len(ids):profile[j]=np.asarray(log[ids]).mean(axis=0,dtype=float)
            genotype_rows.append({'profile_row':j,'target':target,'primary_cells':len(ids),'source_guide_combinations':cells.source_guide_id.iloc[ids].nunique(),'source_hash_wells':cells.source_batch.iloc[ids].nunique(),'status':'completed' if len(ids) else 'not_estimable'})
    pd.DataFrame(genotype_rows).to_parquet(output/'condition-genotype-profiles.parquet',index=False)
    del log,thin,control,control_thin;shutil.rmtree(output/'temporary-view')
    report={'status':'completed','phase':'genetic_response_and_all_source_cell_annotation','context':context,'identity':identity,'view_identity':view,'n_cells':len(cells),'n_primary_cells':int(primary.sum()),'n_NTC':len(ntc),
        'tasks':len(results),'task_status_counts':dict(Counter(r['status'] for r in results)),
        'DE_status_counts':dict(Counter(r.get('DE',{}).get('status','not_estimable') for r in results)),
        'resampling_rows':len(draws),'null_reference_intersections':sum(r['null_reference_intersection'] for r in draws),'null_size_shortfall_rows':sum(r['null_target_cells_not_matched']>0 for r in draws),
        'pooled_types':combined.inferred_type.value_counts().to_dict(),'uncalibrated_records':len(combined),'source_assignment_status':cells.assignment_limitation.fillna('supported').value_counts().to_dict(),
        'duration_seconds':time.monotonic()-started}
    if combined.probability_correct.notna().any() or combined.truth_label.any() or len(combined)!=len(cells):raise ValueError('annotation_truth_or_coverage_violation')
    report['artifacts']={str(p.relative_to(output)):hash_file(p) for p in output.rglob('*') if p.is_file()};write_json(output/'report.json',report);return report
