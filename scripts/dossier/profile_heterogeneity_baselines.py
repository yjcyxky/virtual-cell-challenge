#!/usr/bin/env python
"""Compare every registered source NTC baseline and validate endpoint denominators."""
import argparse
from collections import Counter
from datetime import datetime,timezone
from itertools import combinations
import json
from pathlib import Path
import subprocess
import sys
import h5py
import numpy as np
import pandas as pd
from heterogeneity import safe_symbols,compare_vectors
from profile_cross_source_coverage import Frozen,ROOT,BASE
from profile_heterogeneity_supplements import association
from profile_responses import write_json
from rna import hash_file,value_hash,quantiles


def shared_baseline_consensus(original,other):
    if not original.source_gene.equals(other.source_gene) or not np.allclose(original.mean_logCP10K,other.mean_logCP10K,rtol=0,atol=1e-12):
        raise ValueError('shared_baseline_native_axis_or_means_changed')
    a=original.safe_canonical_symbol;b=other.safe_canonical_symbol
    same=a.eq(b)|(a.isna()&b.isna());common=a.notna()&b.notna()&a.eq(b)
    changes=pd.DataFrame({'source_gene':original.source_gene,'prior_safe_symbol':a,'additional_copy_safe_symbol':b}).loc[~same].copy()
    merged=original.copy();merged['safe_canonical_symbol']=a.where(common)
    return merged,changes


def verify_archived_components(path,diagnostics):
    rows=[]
    with h5py.File(path,'r') as h:
        uids=h['task_uid'].asstr()[:]
        for partition,d in diagnostics.groupby('partition',sort=True):
            d=d.sort_values('native_gene_row')
            if not np.array_equal(d.native_gene_row,np.arange(len(uids))) or not np.array_equal(d.task_uid,uids):raise ValueError('archived_component_task_axis_mismatch')
            expected=d.status.eq('completed').to_numpy();maximum=0.
            for start in range(0,len(uids),64):
                stop=min(start+64,len(uids));valid=expected[start:stop];vectors={}
                for name in ['total_common_support','composition','within','difference_from_full_matched_effect']:
                    a=h[partition][name][start:stop].astype(np.float64)
                    if not np.isfinite(a[valid]).all() or not np.isnan(a[~valid]).all():raise ValueError('archived_component_missingness_or_numeric_status_changed')
                    vectors[name]=a
                if valid.any():maximum=max(maximum,float(np.max(np.abs(vectors['total_common_support'][valid]-vectors['composition'][valid]-vectors['within'][valid]))))
            if maximum>2e-6:raise ValueError('archived_float32_identity_error_exceeds_rounding_bound')
            rows.append({'partition':partition,'archived_tasks':len(uids),'completed_vectors':int(expected.sum()),'unestimable_vectors_all_NaN':int((~expected).sum()),
                         'stored_dtype':'float32','maximum_archived_identity_residual':maximum,'absolute_float32_rounding_bound':2e-6,'all_native_genes_checked':True})
    return rows


def run(source,endpoint,comparisons,extra,output):
    source=source.resolve();endpoint=endpoint.resolve();comparisons=comparisons.resolve();extra=extra.resolve();output=output.resolve();output.mkdir(parents=True,exist_ok=False)
    f=Frozen();er=f.json(endpoint,'report.json');pr=f.json(comparisons,'report.json');sr=f.json(source,'report.json')
    if er['status']!='completed' or er['registered_tasks']!=37845 or pr['source_bundle_id']!=sr['bundle_id']:raise ValueError('incomplete_registered_input')
    official=f.parquet(source,'coverage/official-identifiers.parquet');official_symbols=set(official.mapped_symbol.dropna())
    panels={p['panel_id']:p for p in f.json(source,'coverage/panels.json')};baselines={};baseline_rows=[];endpoint_rows=[];reproductions=[];task_context=[];archive_checks=[];copy_mapping_disagreements=[]
    (output/'baselines').mkdir();(output/'common-genes').mkdir()
    def add_baseline(id,pid,context,genes,means,n,mapping,evidence):
        symbols=safe_symbols(mapping)
        if len(symbols)!=len(means) or len(genes)!=len(means):raise ValueError('baseline_gene_axis_mismatch')
        status='completed' if n and np.isfinite(means).all() else 'not_estimable'
        row={'baseline_id':id,'panel_id':pid,'source_context':context,'source_NTC_cells':n,'status':status,
            'reason':None if status=='completed' else 'no_verified_source_NTC_mean','source_evidence':evidence,'baseline_file':'baselines/'+value_hash(id)+'.parquet',
            'native_features':len(genes),'safe_canonical_genes':sum(pd.notna(x) for x in symbols),'independent_biological_replicates':None}
        frame=pd.DataFrame({'source_gene':genes,'safe_canonical_symbol':symbols,'mean_logCP10K':means})
        if id in baselines:
            old=baselines[id]
            if id!='H1_shared_exact_NTC' or old['n']!=n:raise ValueError('unproven_or_changed_shared_baseline')
            merged,changes=shared_baseline_consensus(old['frame'],frame)
            copy_mapping_disagreements.extend({'baseline_id':id,'additional_copy_panel':pid,**r} for r in changes.to_dict('records'))
            old['frame']=merged;merged.to_parquet(output/row['baseline_file'],index=False)
            row['comparison_representative']=False;row['copy_proof']='source #19 exact-copy-proofs/H1-NTC-copy-groups.parquet'
            row['mapping_policy']='intersection of safe canonical assignments across proven copies; absence of Ensembl metadata cannot rescue a conflicted mapping'
        else:
            baselines[id]={'frame':frame,'n':n,'row':row};frame.to_parquet(output/row['baseline_file'],index=False);row['comparison_representative']=True
        baseline_rows.append(row)
    for c in er['contexts']:
        folder=endpoint/c['context_id'];r=f.json(endpoint,c['context_id']+'/report.json')
        with np.load(f.path(endpoint,c['context_id']+'/baseline.npz')) as b:
            mapping=f.parquet(endpoint,c['context_id']+'/native-gene-mapping.parquet')
            add_baseline(r['baseline_id'],r['panel_id'],r['source_context'],b['source_gene'].astype(str).tolist(),b['mean_logCP10K'],int(b['n_NTC']),mapping,
                {'endpoint_context':c['context_id'],'baseline_sha256':hash_file(folder/'baseline.npz')})
        d=f.parquet(endpoint,c['context_id']+'/task-partition-diagnostics.parquet');d['component_file']='endpoint/'+c['context_id']+'/native-gene-components.h5';endpoint_rows.append(d)
        for check in verify_archived_components(f.path(endpoint,c['context_id']+'/native-gene-components.h5'),d):archive_checks.append({'context_id':c['context_id'],'panel_id':c['panel_id'],**check})
        v=f.parquet(endpoint,c['context_id']+'/all-source-effect-reproduction.parquet');v['context_id']=c['context_id'];reproductions.append(v)
        task_context.extend({'task_uid':uid,'context_id':c['context_id']} for uid in v.task_uid)
    for context in ['A','B','C']:
        pid='official:'+context;p=panels[pid];folder=BASE/'official-controls-20260919';name='baseline_'+context+'.parquet'
        d=f.parquet(folder,name);mapping=f.parquet(source,'coverage/'+p['native_mapping_file'])
        if d.source_gene.astype(str).tolist()!=mapping.source_gene.astype(str).tolist():raise ValueError('official_baseline_axis_mismatch')
        add_baseline(pid,pid,context,d.source_gene.astype(str).tolist(),d.mean_logCP10K.to_numpy(),18400,mapping,{'source_file':str((folder/name).relative_to(ROOT)),'sha256':hash_file(folder/name)})
    additional=f.json(extra,'report.json')
    if additional['status']!='completed' or additional['source_bundle_id']!=sr['bundle_id']:raise ValueError('additional_baseline_input_incomplete_or_changed')
    for c in additional['contexts']:
        prefix=c['context_id']+'/'
        with np.load(f.path(extra,prefix+'baseline.npz')) as b:
            mapping=f.parquet(extra,prefix+'native-gene-mapping.parquet')
            add_baseline(c['baseline_id'],c['panel_id'],c['source_context'],b['source_gene'].astype(str).tolist(),b['mean_logCP10K'],int(b['n_NTC']),mapping,
                {'extra_context':c['context_id'],'baseline_sha256':hash_file(extra/prefix/'baseline.npz'),'source_control_rule':c['source_control_rule'],'reference_role':c['reference_role']})
    # H1 shared controls are a documented content+label copy; equal expression
    # alone is never used to declare any other baselines the same observation.
    proof=f.parquet(source,'exact-copy-proofs/H1-NTC-copy-groups.parquet')
    if len(proof)!=38176 or not proof.records.eq(3).all() or not proof.classification.eq('content_and_label_identical').all():raise ValueError('H1_copy_proof_changed')
    for r in baseline_rows:r['safe_canonical_genes_in_merged_comparison']=int(baselines[r['baseline_id']]['frame'].safe_canonical_symbol.notna().sum())
    pd.DataFrame(copy_mapping_disagreements).to_parquet(output/'shared-NTC-source-mapping-disagreements.parquet',index=False)
    diagnostics=pd.concat(endpoint_rows,ignore_index=True);verify=pd.concat(reproductions,ignore_index=True)
    tasks=f.parquet(comparisons,'selected-task-index.parquet')
    if len(verify)!=37845 or verify.task_uid.duplicated().any() or set(verify.task_uid)!=set(tasks.task_uid):raise ValueError('endpoint_tasks_not_exact_primary_universe')
    if len(diagnostics)!=2*len(tasks) or diagnostics[['task_uid','partition']].duplicated().any():raise ValueError('endpoint_partition_denominator_mismatch')
    expected=tasks.set_index('task_uid').effect_status
    if not verify.source_status.eq(verify.task_uid.map(expected)).all():raise ValueError('source_effect_status_changed')
    reproduced=verify.source_mean_comparison_status.eq('completed')
    if int(reproduced.sum())!=37633:raise ValueError('not_all_completed_effects_reproduced')
    if (diagnostics.loc[diagnostics.status.eq('completed'),'maximum_identity_residual']>1e-9).any():raise ValueError('mixture_identity_residual_too_large')
    # The gene identity relation depends on native axes, not on a particular
    # baseline's expression values. Reuse that verified alignment for every
    # pair sharing the same axes, retaining all baseline-specific means.
    axes={};aligned={}
    for b in baselines.values():
        frame=b['frame'];axis=value_hash([frame.source_gene.tolist(),frame.safe_canonical_symbol.fillna('').tolist()]);b['axis']=axis
        b['mean']=frame.mean_logCP10K.to_numpy(dtype=float)
        if axis not in axes:
            a=frame[['source_gene','safe_canonical_symbol']].copy();a['native_index']=np.arange(len(a));axes[axis]=a.dropna(subset=['safe_canonical_symbol'])
    rows=[]
    for a,b in combinations(sorted(baselines),2):
        da,db=baselines[a],baselines[b];key=(da['axis'],db['axis'])
        if key not in aligned:
            joined=axes[key[0]].merge(axes[key[1]],on='safe_canonical_symbol',suffixes=('_A','_B'),validate='one_to_one').sort_values('safe_canonical_symbol').reset_index(drop=True)
            joined['in_official_canonical_axis']=joined.safe_canonical_symbol.isin(official_symbols);aligned[key]=joined
        joined=aligned[key].copy();joined['mean_logCP10K_A']=da['mean'][joined.native_index_A.to_numpy()];joined['mean_logCP10K_B']=db['mean'][joined.native_index_B.to_numpy()]
        name='common-genes/'+value_hash([a,b])+'.parquet';joined.to_parquet(output/name,index=False)
        row={'baseline_A':a,'baseline_B':b,'panel_A':da['row']['panel_id'],'panel_B':db['row']['panel_id'],
            'NTC_A':da['n'],'NTC_B':db['n'],'common_gene_file':name,'independent_biological_replicates':None,'pure_measurement_noise':False}
        for scope,mask in [('native',np.ones(len(joined),dtype=bool)),('official',joined.in_official_canonical_axis.to_numpy())]:
            comparison=compare_vectors(joined.loc[mask,'mean_logCP10K_A'],joined.loc[mask,'mean_logCP10K_B'])
            row.update({scope+'_'+k:v for k,v in comparison.items()})
        row['status']=row['native_status'];rows.append(row)
        if len(rows)%1000==0:print('baseline comparisons '+str(len(rows))+'/'+str(len(baselines)*(len(baselines)-1)//2),flush=True)
    diagnostics=diagnostics.merge(tasks[['task_uid','downstream_RMS','target_RNA_ratio','matched_target_cells']],on='task_uid',validate='many_to_one')
    associations=[];summaries=[]
    metrics=['inferred_partition_total_variation','target_unknown_partition_fraction','supported_target_fraction','NTC_common_state_support_fraction_target_batch_weighted',
             'total_common_support_RMS_safe_downstream','composition_RMS_safe_downstream','within_RMS_safe_downstream','difference_from_full_matched_effect_RMS_safe_downstream','common_support_vs_full_effect_correlation']
    for (context,partition),d in diagnostics.groupby(['context_id','partition'],sort=True):
        summaries.append({'context_id':context,'panel_id':d.panel_id.iloc[0],'source_context':d.source_context.iloc[0],'partition':partition,'tasks':len(d),
            'status_counts':d.status.value_counts().to_dict(),**{v:quantiles(pd.to_numeric(d[v],errors='raise').dropna().to_numpy(dtype=float)) for v in metrics},**{v+'_missing':int(d[v].isna().sum()) for v in metrics}})
        for x in ['target_RNA_ratio','matched_target_cells','target_unknown_partition_fraction','supported_target_fraction']:
            for y in ['composition_RMS_safe_downstream','within_RMS_safe_downstream','difference_from_full_matched_effect_RMS_safe_downstream']:
                associations.append(association(d,x,y,context+':'+partition))
    pd.DataFrame(rows).to_parquet(output/'all-baseline-pairs.parquet',index=False,compression='zstd');write_json(output/'baseline-index.json',baseline_rows)
    diagnostics.to_parquet(output/'all-endpoint-task-partitions.parquet',index=False,compression='zstd');verify.to_parquet(output/'all-source-effect-reproduction.parquet',index=False)
    pd.DataFrame(archive_checks).to_parquet(output/'all-native-archive-verification.parquet',index=False)
    pd.DataFrame(associations).to_parquet(output/'endpoint-factor-associations.parquet',index=False);write_json(output/'endpoint-context-summaries.json',summaries)
    # All other source panels have an explicit scope, rather than masquerading
    # as zero baselines or shared NTCs (scBase is not a reference population).
    covered={r['panel_id'] for r in baseline_rows};scope=[]
    for p in panels.values():
        scope.append({'panel_id':p['panel_id'],'source_family':p['family'],'status':'completed' if p['panel_id'] in covered else 'not_applicable',
            'reason':None if p['panel_id'] in covered else 'no_additional_verified_count_compatible_human_negative_guide_NTC_background; other references or modalities remain in source-specific designs',
            'human_RNA_applicable':p['human_RNA_applicable'],'confirmed_collection_copy':p['confirmed_collection_copy']})
    pd.DataFrame(scope).to_parquet(output/'all-source-baseline-scope.parquet',index=False)
    write_json(output/'consumed-inputs.json',f.used)
    report={'status':'completed','phase':'registered_NTC_baselines_and_complete_endpoint_denominators','source_bundle_id':sr['bundle_id'],
        'baseline_scope':'all primary CRISPRi contexts plus official A/B/C and additional verified human count-compatible negative-guide RNA references; intergenic cutting, vehicle, untreated and source copies are separate',
        'baseline_context_rows':len(baseline_rows),'distinct_registered_reference_contexts':len(baselines),'distinct_source_NTC_baselines':sum(b['n']>0 for b in baselines.values()),
        'registered_contexts_without_NTC':sum(b['n']==0 for b in baselines.values()),'baseline_pairs':len(rows),'baseline_status_counts':dict(Counter(r['status'] for r in rows)),
        'exact_H1_baseline_copies_not_recounted':2,'endpoint_task_partitions':len(diagnostics),'endpoint_status_counts':diagnostics.status.value_counts().to_dict(),
        'shared_NTC_mapping_disagreements':len(copy_mapping_disagreements),'shared_NTC_mapping_policy':'safe canonical intersection across proven exact native-count copies; preserve conflicting or missing identity evidence',
        'source_effects_reproduced':int(reproduced.sum()),'source_unestimable_tasks_retained':int((~reproduced).sum()),
        'maximum_source_mean_reproduction_error':float(verify.maximum_absolute_mean_or_effect_error.max()),'maximum_mixture_identity_residual':float(diagnostics.maximum_identity_residual.max()),
        'maximum_archived_float32_identity_residual':max(r['maximum_archived_identity_residual'] for r in archive_checks),
        'all_archived_native_genes_and_missing_vectors_checked':True,'archived_float32_identity_absolute_rounding_bound':2e-6,
        'component_RMS_values_are_additive':False,'causal_mediation_or_variance_fraction':False,'endpoint_strata_available_before_measurement':False,
        'source_input_mutations':0,'completed_at':datetime.now(timezone.utc).isoformat(),'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'code_sha256':hash_file(Path(__file__)),'uv_lock_sha256':hash_file(Path(__file__).with_name('uv.lock')),'reproduce':sys.argv}
    report['artifacts']={str(p.relative_to(output)):hash_file(p) for p in sorted(output.rglob('*')) if p.is_file()};write_json(output/'report.json',report)
    (output/'SHA256SUMS').write_text(''.join(hash_file(p)+'  '+str(p.relative_to(output))+'\n' for p in sorted(output.rglob('*')) if p.is_file() and p.name!='SHA256SUMS'))
    print(json.dumps({k:v for k,v in report.items() if k!='artifacts'}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['source','endpoint','comparisons','extra','output']:p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args()
    try:run(a.source,a.endpoint,a.comparisons,a.extra,a.output)
    except Exception as error:
        if a.output.exists():write_json(a.output/'failure.json',{'status':'failed','error':repr(error),'code_sha256':hash_file(Path(__file__))})
        raise
