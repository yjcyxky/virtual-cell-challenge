#!/usr/bin/env python
"""Complete source-scoped sensitivity evidence and descriptive associations."""
import argparse
from collections import Counter
from datetime import datetime,timezone
from itertools import combinations
import json
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np
import pandas as pd
from profile_cross_source_coverage import Frozen,ROOT,BASE
from profile_responses import write_json
from rna import hash_file,quantiles,value_hash


def association(frame,x,y,scope):
    a=pd.to_numeric(frame[x],errors='coerce').to_numpy(dtype=float);b=pd.to_numeric(frame[y],errors='coerce').to_numpy(dtype=float)
    valid=np.isfinite(a)&np.isfinite(b);n=int(valid.sum());reason=None
    if n<10:reason='fewer_than_10_complete_observations'
    elif np.ptp(a[valid])==0 or np.ptp(b[valid])==0:reason='constant_observed_factor_or_outcome'
    rho=None if reason else float(pd.Series(a[valid]).rank().corr(pd.Series(b[valid]).rank()))
    return {'scope':scope,'factor':x,'outcome':y,'observations':len(frame),'complete_observations':n,'missing_observations':len(frame)-n,
            'status':'not_estimable' if reason else 'completed','reason':reason,'Spearman_rho':rho,'p_value':None,
            'unit':'source task or same-target task pair; shared controls and repeated targets remain dependent',
            'causal_effect':False,'independent_biological_replicates':None}


def numeric_nested(value,*keys):
    for k in keys:
        if not isinstance(value,dict):return None
        value=value.get(k)
    return value if isinstance(value,(int,float)) and np.isfinite(value) else None


def run(source,comparisons,output):
    source=source.resolve();comparisons=comparisons.resolve();output=output.resolve();output.mkdir(parents=True,exist_ok=False);f=Frozen();catalog=[]
    parent=f.json(source,'report.json');primary=f.json(comparisons,'report.json')
    if parent['bundle_id']!=primary['source_bundle_id'] or primary['status']!='completed':raise ValueError('comparison_parent_not_complete_or_shared')
    tasks=f.parquet(comparisons,'all-task-diagnostics.parquet');pairs=f.parquet(comparisons,'all-task-pairs.parquet')
    for out,field,keys in [
        ('guide_median_correlation','guide_consistency',('median_correlation',)),
        ('technical_stratum_median_correlation','technical_stratum_consistency',('median_correlation',)),
        ('resampling_median_correlation','resampling_stability',('correlations','median')),
        ('resampling_null_RMS_median','resampling_stability',('null_RMS','median'))]:
        tasks[out]=tasks[field].map(lambda v:numeric_nested(v,*keys))
    tasks['source_analysis_context']=[value_hash([p,c if pd.notna(c) else None,b if pd.notna(b) else None]) for p,c,b in zip(tasks.panel_id,tasks.source_condition,tasks.source_background_index)]
    factors=['target_RNA_ratio','matched_target_cells','library_size_median','detected_genes_median','guide_median_correlation','technical_stratum_median_correlation']
    outcomes=['downstream_RMS','depth_effect_correlation','resampling_median_correlation']
    associations=[];context_summaries=[]
    for context,d in tasks.groupby('source_analysis_context',sort=True):
        for x in factors:
            for y in outcomes:associations.append(association(d,x,y,context))
        context_summaries.append({'source_analysis_context':context,'panel_id':d.panel_id.iloc[0],'source_condition':d.source_condition.iloc[0],
            'source_background_index':d.source_background_index.iloc[0],'tasks':len(d),'canonical_targets':d.canonical_target.nunique(),
            'effect_status_counts':d.effect_status.value_counts().to_dict(),
            **{v:quantiles(pd.to_numeric(d[v],errors='coerce').dropna().to_numpy()) for v in factors+outcomes},
            **{v+'_missing':int(pd.to_numeric(d[v],errors='coerce').isna().sum()) for v in factors+outcomes}})
    index=tasks.set_index('task_uid')
    for v in ['target_RNA_ratio','library_size_median','detected_genes_median','matched_target_cells']:
        for side in ['A','B']:pairs[v+'_'+side]=pairs['task_uid_'+side].map(index[v])
        pairs[v+'_absolute_difference']=(pairs[v+'_A']-pairs[v+'_B']).abs()
    pairs['native_vs_official_effect_correlation_difference']=pairs.native_effect_correlation-pairs.official_effect_correlation
    pairs['native_vs_thinned_effect_correlation_difference']=pairs.native_effect_correlation-pairs.native_thinned_effect_correlation
    pairs['absolute_vs_response_correlation_difference']=pairs.native_absolute_target_correlation-pairs.native_effect_correlation
    pair_groups=[]
    metrics=['native_baseline_difference_RMS','native_absolute_target_correlation','native_effect_correlation','native_effect_difference_RMS','official_effect_correlation',
             'native_thinned_effect_correlation','native_vs_official_effect_correlation_difference','native_vs_thinned_effect_correlation_difference','absolute_vs_response_correlation_difference']
    for keys,d in pairs.groupby(['panel_A','panel_B','comparison_role'],sort=True):
        scope=value_hash(list(keys));pair_groups.append({'comparison_group':scope,'panel_A':keys[0],'panel_B':keys[1],'comparison_role':keys[2],
            'all_pairs':len(d),'canonical_targets':d.canonical_target.nunique(),'status_counts':d.status.value_counts().to_dict(),
            **{v:quantiles(d[v].dropna().to_numpy()) for v in metrics},**{v+'_missing':int(d[v].isna().sum()) for v in metrics}})
        for x in ['native_baseline_difference_RMS','native_common_genes','target_RNA_ratio_absolute_difference','library_size_median_absolute_difference','detected_genes_median_absolute_difference','matched_target_cells_absolute_difference']:
            for y in ['native_effect_correlation','native_effect_difference_RMS']:associations.append(association(d,x,y,scope))
    tasks.to_parquet(output/'task-sensitivities.parquet',index=False,compression='zstd')
    pairs.to_parquet(output/'pair-sensitivities.parquet',index=False,compression='zstd')
    pd.DataFrame(associations).to_parquet(output/'descriptive-factor-associations.parquet',index=False,compression='zstd')
    write_json(output/'source-context-summaries.json',context_summaries);write_json(output/'pair-group-summaries.json',pair_groups)

    def evidence(folder,name,role):
        original=f.path(folder,name);relative=str(folder.relative_to(BASE));dest=output/'source-evidence'/relative/name
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(original,dest)
        row={'source_folder':str(folder.relative_to(ROOT)),'source_file':name,'file':str(dest.relative_to(output)),'sha256':hash_file(original),'role':role,
             'pooled_into_CRISPRi_response_pairs':False,'biological_replication_assumed':False}
        if hash_file(dest)!=row['sha256']:raise ValueError('evidence_copy_hash_mismatch')
        if name.endswith('.parquet'):
            d=pd.read_parquet(original);row['rows']=len(d)
            for col in ['status','stability_status','source_status']:
                if col in d:row[col+'_counts']=d[col].fillna('missing').value_counts().to_dict()
        catalog.append(row);return dest

    evidence(source,'source-protocol-index.json','source protocol provenance and source releases')
    for p in sorted((source/'source-protocols').glob('*.json')):evidence(source,str(p.relative_to(source)),'frozen platform/time/effector/culture/source observability')
    evidence(source,'identities/supervised-library-overlaps.parquet','all 48 author capture associations; reprocessing or modality, not independent experiments')
    evidence(source,'record-denominators.json','confirmed copies and unproven record/cell/replicate equivalence')
    gxe2=BASE/'mcfaline-gxe2-dossier-20260919'
    genetic=f.json(gxe2,'genetic/report.json');thresholds=[]
    for c in genetic['contexts']:
        prefix='genetic/'+c['directory']+'/'
        p=evidence(gxe2,prefix+'guide-read-threshold-sensitivity.parquet','all guide read thresholds 1–10, including unsupported tasks')
        d=pd.read_parquet(p);d['source_context']=c['context'];thresholds.append(d)
        for n in ['states.parquet','composition.parquet','within-type-states.parquet']:
            evidence(gxe2,prefix+n,'within registered genetic context endpoint RNA proxies and conservative inferred types')
    all_thresholds=pd.concat(thresholds,ignore_index=True)
    if len(all_thresholds)!=27*523*10:raise ValueError('incomplete_GxE2_guide_threshold_universe')
    all_thresholds.to_parquet(output/'all-GxE2-guide-read-thresholds.parquet',index=False,compression='zstd')
    chem=f.json(gxe2,'chemical/report.json')
    for c in chem['conditions']:
        for name in ['diagnostics.parquet','within-type-states.parquet','state-distributions.parquet','type-composition.parquet','guide-matched-state-differences.parquet']:
            evidence(gxe2,'chemical/'+c['directory']+'/'+name,'fixed source genotype drug versus matched vehicle; shared weights and all guide-matched sensitivity')
    for name in ['mcfaline-chemical3-dossier-20260919-v2','mcfaline-chemical4-dossier-20260919-v2']:
        folder=BASE/name
        for file in ['condition-diagnostics.parquet','within-type-states.parquet','state-distributions.parquet','composition-contrasts.parquet']:
            evidence(folder,file,'all cohort-specific drug/vehicle conditions; endpoint proxies, no CRISPRi or causal mechanism equivalence')
        if 'chemical4' in name:evidence(folder,'combination-differences.parquet','all matched double differences; not a synergy claim')
    tahoe=BASE/'tahoe-dossier-20260919-v2'
    repeat=evidence(tahoe,'plate6-plate14-repeat-comparisons.parquet','all 4738 author declared two-culture repeats, not independently verified')
    repeats=pd.read_parquet(repeat)
    if len(repeats)!=4738:raise ValueError('incomplete_author_repeat_universe')
    evidence(tahoe,'plate6-plate14-baseline-comparisons.parquet','all repeat plate/line DMSO baselines')
    evidence(tahoe,'all-condition-diagnostics.parquet','all 65576 sample × cell-line conditions, including absent/insufficient references')
    for folder in sorted((tahoe/'conditions').iterdir()):
        if not folder.is_dir():continue
        for name in ['within-type-states.parquet','state-distributions.parquet','composition-contrasts.parquet']:
            evidence(tahoe,str(folder.relative_to(tahoe))+'/'+name,'all Tahoe endpoint proxy and composition results within the source plate × cell-line background')
    # Complete source condition and endpoint state results remain in their own
    # scales/designs; no ad hoc choice of successful or strongly responding cells.
    sc=BASE/'scperturb-dossier-20260919';file_index=f.json(sc,'file-index.json')
    for r in file_index:
        for field in ['all_condition_results','observed_distributions']:
            if r.get(field):evidence(sc,r[field],'complete source condition/timecourse denominator; source transformation and control qualifications preserved')
    evidence(sc,'all-condition-status-denominators.json','all source condition statuses, including non-RNA not applicable')
    for folder,prefix,names in [(BASE/'h1-annotation-20260919','',['states.parquet','composition.parquet'])]:
        for name in names:evidence(folder,prefix+name,'all frozen endpoint RNA proxies, not preintervention truth')
    for family,contexts in [('replogle',['K562_essential','K562_gwps','rpe1']),('nadig',['hepg2','jurkat'])]:
        for context in contexts:
            for name in ['states.parquet','composition.parquet','within_type_states.parquet']:
                evidence(BASE/(family+'-annotation-'+context+'-20260919'),name,'all source-matched endpoint RNA proxy and type strata')
    jiang=BASE/'jiang-dossier-20260919-v2'
    for p in sorted(jiang.glob('Jiang__*/states.parquet')):
        for name in ['states.parquet','composition.parquet','within-type-states.parquet']:
            evidence(jiang,str(p.parent.relative_to(jiang))+'/'+name,'all source line × stimulus endpoint descriptions; 24 h is not knockdown onset')
    for name in ['states.parquet','composition.parquet','within-type-states.parquet']:
        evidence(BASE/'mcfaline-gxe1-dossier-20260919-v2',name,'all GxE1 genetic and fixed-genotype chemical endpoint contrasts by original role')
    write_json(output/'source-evidence-index.json',catalog)
    # Validation checks the actual consumed source files after all copies.
    for path,digest in f.used.items():
        if hash_file(ROOT/path)!=digest:raise ValueError('consumed_source_evidence_changed:'+path)
    write_json(output/'consumed-inputs.json',f.used)
    report={'status':'completed','phase':'complete_source_scoped_sensitivity_evidence','source_bundle_id':parent['bundle_id'],'tasks':len(tasks),'pairs':len(pairs),
        'source_analysis_contexts':len(context_summaries),'pair_groups':len(pair_groups),'descriptive_associations':len(associations),
        'association_status_counts':dict(Counter(r['status'] for r in associations)),'GxE2_threshold_rows':len(all_thresholds),
        'Tahoe_repeat_pairs':len(repeats),'Tahoe_repeat_status_counts':repeats.status.value_counts().to_dict(),
        'Tahoe_repeat_absolute_correlation':quantiles(repeats.mean_logRNA_correlation.dropna().to_numpy()),
        'Tahoe_repeat_response_correlation':quantiles(repeats.response_correlation.dropna().to_numpy()),
        'source_evidence_files':len(catalog),'source_input_mutations':0,'biological_p_values_computed':0,
        'causal_factor_decomposition':'not_identifiable; source, platform, culture, time, and effector lack a crossed replicated design',
        'RNA_target_ratio':'relative transcript proxy on native CP10K denominator; not protein dose or absolute RNA',
        'completed_at':datetime.now(timezone.utc).isoformat(),'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'code_sha256':hash_file(Path(__file__)),'uv_lock_sha256':hash_file(Path(__file__).with_name('uv.lock')),'reproduce':sys.argv}
    report['artifacts']={str(p.relative_to(output)):hash_file(p) for p in sorted(output.rglob('*')) if p.is_file()};write_json(output/'report.json',report)
    (output/'SHA256SUMS').write_text(''.join(hash_file(p)+'  '+str(p.relative_to(output))+'\n' for p in sorted(output.rglob('*')) if p.is_file() and p.name!='SHA256SUMS'))
    print(json.dumps({k:v for k,v in report.items() if k!='artifacts'}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--comparisons',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    try:run(a.source,a.comparisons,a.output)
    except Exception as error:
        if a.output.exists():write_json(a.output/'failure.json',{'status':'failed','error':repr(error),'code_sha256':hash_file(Path(__file__))})
        raise
