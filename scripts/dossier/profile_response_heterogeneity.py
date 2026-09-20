#!/usr/bin/env python
"""All registered same-target comparisons on frozen native response statistics."""
import argparse
from collections import Counter,defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
from itertools import combinations
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from heterogeneity import safe_symbols,compare_vectors
from profile_cross_source_coverage import Frozen,ROOT
from profile_responses import write_json
from rna import hash_file,value_hash,quantiles

PARAMETERS={'minimum_common_genes':100,'own_target_excluded':True,'primary_scale':'difference of mean cell log1p(CP10K), source-native normalization panel',
            'gene_sets':['safe_native_intersection','safe_official_canonical_intersection'],'thinning':'existing source seeded expected-depth 10000 sensitivity',
            'independent_biological_replicates':None,'preregistered_issue_comment':'https://github.com/yjcyxky/virtual-cell-challenge/issues/20#issuecomment-5746896397'}
FIELDS=['effect_all_matched_cells','mean_logCP10K_matched_control','mean_logCP10K_target','effect_thinned']


def task_eligibility(row,panel):
    reasons=[]
    if row['confirmed_collection_copy']:reasons.append('confirmed_collection_copy_no_additional_evidence')
    if row['modality']!='CRISPRi':reasons.append('not_CRISPRi')
    if not panel['human_RNA_applicable']:reasons.append('human_RNA_not_verified_applicable')
    if pd.isna(row['canonical_target']):reasons.append('unresolved_or_conflicting_intervention_gene')
    if row['target_kind']=='single_source_regulatory_locus':reasons.append('regulatory_locus_not_single_gene')
    return not reasons,'|'.join(reasons) if reasons else None


def source_task_details(folder,name):
    data=json.loads((folder/name).read_text())
    return {str(t.get('task',t.get('task_id',i))):t for i,t in enumerate(data)}


def task_diagnostics(task,original,panel):
    library=original.get('library_size') or {};detected=original.get('detected_genes') or {}
    depth=original.get('depth_sensitivity') or {};consistency=original.get('consistency') or {};stability=original.get('stability') or {}
    result={k:task[k] for k in ['task_index','task_uid','panel_id','source_task','source_target','canonical_target','source_condition','source_background_index','effect_status','DE_status','matched_target_cells','source_target_cells','target_identity_status','source_intervention_Ensembl','target_RNA_ratio','downstream_RMS']}
    result.update(source_family=panel['family'],library_size_median=library.get('median'),detected_genes_median=detected.get('median'),
                  depth_sensitivity_status=depth.get('status','not_estimable'),depth_effect_correlation=depth.get('correlation'),
                  guide_consistency=consistency.get('guide'),technical_stratum_consistency=consistency.get('batch'),
                  resampling_stability=stability,source_assignment_sensitivity=original.get('source_assignment_sensitivity'),
                  target_RNA_status=(original.get('target_RNA') or {}).get('status','not_estimable'),
                  target_RNA_is_effective_protein_dose=False,independent_biological_replicates=None)
    return result


def run(source,output,workers):
    source=source.resolve();output=output.resolve();output.mkdir(parents=True,exist_ok=False);start=time.monotonic();frozen=Frozen()
    parent=frozen.json(source,'report.json')
    if parent['bundle_id']!='cross-source-dossier-cba1296d48884511b1477b5f9417314b' or hash_file(source/'report.json')!='5c87d7cc87a210cdb23e012a0d91a03b4cffe5be58e65a66f16e09888873368a':raise ValueError('preregistered_source_bundle_changed')
    panels=frozen.json(source,'coverage/panels.json');panel_index={p['panel_id']:p for p in panels}
    tasks=frozen.parquet(source,'coverage/all-source-tasks.parquet');ledger=[]
    for row in tasks.to_dict('records'):
        eligible,reason=task_eligibility(row,panel_index[row['panel_id']]);row.update(identity_eligible=eligible,identity_ineligibility_reason=reason)
        row['task_uid']=value_hash([row['panel_id'],row['source_task'],row['source_task_file']]);ledger.append(row)
    ledger=pd.DataFrame(ledger);ledger['task_index']=np.arange(len(ledger));selected=ledger.loc[ledger.identity_eligible].copy()
    if len(selected)!=37845 or selected.panel_id.nunique()!=79 or selected.canonical_target.nunique()!=10198:raise ValueError('preregistered_task_identity_universe_changed')
    if ledger.task_uid.duplicated().any():raise ValueError('task_uid_collision')
    ledger.to_parquet(output/'all-task-applicability.parquet',index=False,compression='zstd')
    selected.to_parquet(output/'selected-task-index.parquet',index=False,compression='zstd')
    official=frozen.parquet(source,'coverage/official-identifiers.parquet');official_symbols=set(official.mapped_symbol.dropna())
    arrays={};gene_axes={};axis_key={};metadata=[];used_gene_files={};diagnostics=[];task_jsons={};row_positions={}
    (output/'gene-sets').mkdir();(output/'source-task-summaries').mkdir()
    for pi,(pid,group) in enumerate(selected.groupby('panel_id',sort=True)):
        panel=panel_index[pid];mapping=frozen.parquet(source,'coverage/'+panel['native_mapping_file']);symbols=safe_symbols(mapping)
        # Source identity columns retain native gene keys; no alias replacement
        # is made in the result vectors.
        native_column=next((c for c in ['source_gene_id','source_feature_id','source_ensembl_id'] if c in mapping),'source_gene')
        native=mapping[native_column].astype(str).tolist();axis=value_hash([native,symbols]);axis_key[pid]=axis
        if axis not in gene_axes:
            axis_frame=pd.DataFrame({'native_index':np.arange(len(native)),'source_gene':native,'safe_canonical_symbol':symbols})
            axis_frame.to_parquet(output/'gene-sets'/(axis+'.parquet'),index=False);gene_axes[axis]=axis_frame
        records=group.to_dict('records');position={r['task_uid']:i for i,r in enumerate(records)};row_positions[pid]=position
        # These arrays exist only in the analysis process. The delivered native
        # effects remain the already-published source statistics, not a unified
        # cell matrix or a newly assembled training corpus.
        values=np.full((len(records),len(native),len(FIELDS)),np.nan,dtype=np.float32)
        target_masks=np.zeros((len(records),len(native)),dtype=bool)
        def load_one(row):
            if row['effect_status']!='completed':return row,None,None,None
            folder=ROOT/row['source_folder'];name=row['source_gene_result_file'];expected=row['source_gene_result_sha256']
            if not name or not expected or hash_file(folder/name)!=expected:raise ValueError('response_gene_file_missing_or_changed:'+row['task_uid'])
            df=pd.read_parquet(folder/name,columns=['source_gene','is_target']+FIELDS)
            if df.source_gene.astype(str).tolist()!=native:raise ValueError('native_response_gene_axis_changed:'+pid)
            a=df[FIELDS].to_numpy(dtype=np.float32)
            if not np.isfinite(a[:,:3]).all():raise ValueError('completed_response_has_nonfinite_primary_values')
            # Main source effect must equal its reported target and matched NTC
            # means; validate before using the two levels in separate panels.
            if not np.allclose(a[:,0],a[:,2]-a[:,1],rtol=2e-5,atol=2e-6):raise ValueError('source_effect_reference_inconsistency')
            return row,a,df.is_target.to_numpy(dtype=bool),(str((folder/name).relative_to(ROOT)),expected)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for row,a,mask,used in pool.map(load_one,records):
                if a is not None:values[position[row['task_uid']]]=a;target_masks[position[row['task_uid']]]=mask;used_gene_files[used[0]]=used[1]
        arrays[pid]=(values,target_masks)
        for row in records:
            key=(row['source_folder'],row['source_task_file'])
            if key not in task_jsons:
                folder=ROOT/key[0];path=frozen.path(folder,key[1]);task_jsons[key]=source_task_details(folder,key[1])
                copy_name=value_hash(list(key))+'.json';(output/'source-task-summaries'/copy_name).write_bytes(path.read_bytes())
            original=task_jsons[key][str(row['source_task'])]
            diagnostics.append(task_diagnostics(row,original,panel))
        metadata.append({'panel_id':pid,'family':panel['family'],'source_context':panel['source_context'],'source_metadata':panel['source_metadata'],
                         'native_axis':axis,'native_features':len(native),'safe_canonical_features':int(pd.Series(symbols).notna().sum()),
                         'selected_tasks':len(group),'source_effect_status_counts':group.effect_status.value_counts().to_dict()})
        print('native response panels '+str(pi+1)+'/79 '+pid,flush=True)
    write_json(output/'panel-design.json',metadata);pd.DataFrame(diagnostics).to_parquet(output/'all-task-diagnostics.parquet',index=False,compression='zstd')
    write_json(output/'all-task-diagnostics.json',diagnostics)
    pair_gene_cache={}
    def shared_genes(pa,pb):
        aa,bb=axis_key[pa],axis_key[pb];key=(aa,bb)
        if key not in pair_gene_cache:
            a=gene_axes[aa].dropna(subset=['safe_canonical_symbol']);b=gene_axes[bb].dropna(subset=['safe_canonical_symbol'])
            merged=a.merge(b,on='safe_canonical_symbol',suffixes=('_A','_B'),validate='one_to_one').sort_values('safe_canonical_symbol').reset_index(drop=True)
            merged['in_official_canonical_axis']=merged.safe_canonical_symbol.isin(official_symbols);name='common-'+value_hash(list(key))+'.parquet'
            merged.to_parquet(output/'gene-sets'/name,index=False)
            pair_gene_cache[key]=(merged.native_index_A.to_numpy(),merged.native_index_B.to_numpy(),merged.safe_canonical_symbol.to_numpy(),merged.in_official_canonical_axis.to_numpy(),name)
        return pair_gene_cache[key]
    pairs=[];source_lookup={r['task_uid']:r for r in selected.to_dict('records')}
    for ti,(target,group) in enumerate(selected.groupby('canonical_target',sort=True)):
        records=group.sort_values(['panel_id','task_uid']).to_dict('records')
        for a,b in combinations(records,2):
            pa,pb=a['panel_id'],b['panel_id']
            same_condition=(pd.isna(a['source_condition']) and pd.isna(b['source_condition'])) or a['source_condition']==b['source_condition']
            same_background=(pd.isna(a['source_background_index']) and pd.isna(b['source_background_index'])) or a['source_background_index']==b['source_background_index']
            same_context=pa==pb and same_condition and same_background
            row={'canonical_target':target,'task_index_A':a['task_index'],'task_index_B':b['task_index'],'task_uid_A':a['task_uid'],'task_uid_B':b['task_uid'],
                 'panel_A':pa,'panel_B':pb,'source_family_A':panel_index[pa]['family'],'source_family_B':panel_index[pb]['family'],
                 'comparison_role':'different_construct_same_source_context' if same_context else 'different_source_or_biological_condition',
                 'status':'completed','reason':None,'independent_biological_replicates':None}
            if a['effect_status']!='completed' or b['effect_status']!='completed':
                row.update(status='not_estimable',reason='one_or_both_source_effects_not_estimable');pairs.append(row);continue
            ia,ib,genes,official_mask,gene_file=shared_genes(pa,pb);va,ma=arrays[pa];vb,mb=arrays[pb];ai=row_positions[pa][a['task_uid']];bi=row_positions[pb][b['task_uid']]
            keep=(genes!=target)&~ma[ai,ia]&~mb[bi,ib];ia=ia[keep];ib=ib[keep];official_mask=official_mask[keep]
            av=va[ai,ia];bv=vb[bi,ib];row.update(common_gene_file='gene-sets/'+gene_file,excluded_canonical_target=target,
                native_common_genes=len(ia),official_common_genes=int(official_mask.sum()))
            for tag,mask in [('native',np.ones(len(ia),dtype=bool)),('official',official_mask)]:
                for label,column in [('effect',0),('baseline',1),('absolute_target',2),('thinned_effect',3)]:
                    metric=compare_vectors(av[mask,column],bv[mask,column],PARAMETERS['minimum_common_genes'])
                    row[tag+'_'+label+'_status']=metric['status'];row[tag+'_'+label+'_reason']=metric['reason']
                    row[tag+'_'+label+'_genes']=metric['genes'];row[tag+'_'+label+'_correlation']=metric['correlation'];row[tag+'_'+label+'_difference_RMS']=metric['difference_RMS']
                    row[tag+'_'+label+'_correlation_status']=metric.get('correlation_status')
                    if label=='effect':row[tag+'_effect_RMS_A']=metric.get('RMS_A');row[tag+'_effect_RMS_B']=metric.get('RMS_B')
            if row['native_effect_status']!='completed':row.update(status='not_estimable',reason='insufficient_common_safe_native_genes')
            pairs.append(row)
        if (ti+1)%500==0:print('all canonical targets compared '+str(ti+1)+'/10198; pairs '+str(len(pairs)),flush=True)
    if len(pairs)!=249071:raise ValueError('preregistered_pair_universe_changed')
    frame=pd.DataFrame(pairs);frame.to_parquet(output/'all-task-pairs.parquet',index=False,compression='zstd')
    pair_summaries=[]
    for (pa,pb,role),group in frame.groupby(['panel_A','panel_B','comparison_role'],dropna=False,sort=True):
        valid=group.loc[group.status.eq('completed')]
        pair_summaries.append({'panel_A':pa,'panel_B':pb,'comparison_role':role,'all_pairs':len(group),'completed_pairs':len(valid),
            'not_estimable_pairs':len(group)-len(valid),'targets':group.canonical_target.nunique(),
            'native_response_correlation':quantiles(valid.get('native_effect_correlation',[])),
            'native_response_difference_RMS':quantiles(valid.get('native_effect_difference_RMS',[])),
            'native_baseline_difference_RMS':quantiles(valid.get('native_baseline_difference_RMS',[])),
            'absolute_target_correlation':quantiles(valid.get('native_absolute_target_correlation',[])),
            'official_response_correlation':quantiles(valid.get('official_effect_correlation',[])),
            'thinned_response_correlation':quantiles(valid.get('native_thinned_effect_correlation',[])),
            'interpretation':'All registered tasks; descriptive comparisons, no independent-culture inference or causal assignment to time/platform/state.'})
    write_json(output/'all-panel-pair-summaries.json',pair_summaries)
    write_json(output/'consumed-inputs.json',{'frozen_artifacts':frozen.used,'response_gene_files':used_gene_files})
    summary={'status':'completed','phase':'all_same_canonical_target_response_comparisons','source_bundle_id':parent['bundle_id'],
        'source_report_sha256':hash_file(source/'report.json'),'all_source_tasks':len(ledger),'identity_qualified_tasks':len(selected),
        'source_effect_status_counts':selected.effect_status.value_counts().to_dict(),'canonical_targets':selected.canonical_target.nunique(),
        'source_panels':len(metadata),'native_gene_axes':len(gene_axes),'all_same_target_pairs':len(frame),'pair_status_counts':frame.status.value_counts().to_dict(),
        'pair_role_counts':frame.comparison_role.value_counts().to_dict(),'panel_pair_groups':len(pair_summaries),
        'parameters':PARAMETERS,'independent_biological_replicates':None,'source_input_mutations':0,
        'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'code_sha256':{n:hash_file(Path(__file__).with_name(n)) for n in ['profile_response_heterogeneity.py','heterogeneity.py','profile_cross_source_coverage.py','rna.py']},
        'uv_lock_sha256':hash_file(Path(__file__).with_name('uv.lock')),'duration_seconds':time.monotonic()-start,'completed_at':datetime.now(timezone.utc).isoformat(),'reproduce':sys.argv}
    summary['artifacts']={str(p.relative_to(output)):hash_file(p) for p in sorted(output.rglob('*')) if p.is_file()};write_json(output/'report.json',summary)
    (output/'SHA256SUMS').write_text(''.join(hash_file(p)+'  '+str(p.relative_to(output))+'\n' for p in sorted(output.rglob('*')) if p.is_file() and p.name!='SHA256SUMS'))
    print(json.dumps({k:v for k,v in summary.items() if k not in ['artifacts','code_sha256','parameters']}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=4);a=p.parse_args()
    try:run(a.source,a.output,a.workers)
    except Exception as error:
        if a.output.exists():write_json(a.output/'failure.json',{'status':'failed','error':repr(error),'code_sha256':hash_file(Path(__file__))})
        raise
