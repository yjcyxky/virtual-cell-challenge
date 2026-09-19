#!/usr/bin/env python
"""Describe every GxE2 drug/dose at fixed source genotype and show guide confounding."""
import argparse
from collections import Counter
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import time
import h5py
import numpy as np
import pandas as pd
from chemical_diagnostics import distribution_contrast,PARAMETERS
from profile_responses import write_json
from rna import hash_file,value_hash


def guide_comparison(target,control,names,genotype):
    if genotype=='PRKCZ':return {'status':'not_estimable','reason':'source_guide_names_do_not_uniquely_identify_sequences'},[]
    def keys(frame):return pd.Series([json.dumps([g,r],separators=(',',':')) for g,r in zip(frame.source_guide_id,frame.source_CDS_replicate)],index=frame.index)
    tk,ck=keys(target),keys(control);common=set(tk)&set(ck);ids=tk.isin(common)
    n=int(ids.sum());nc=int(ck.isin(common).sum())
    row={'source_guide_and_replicate_matched_target_cells':n,'source_guide_and_replicate_matched_control_cells':nc,'shared_source_guide_replicate_groups':len(common),
        'matching_unit':'source guide combination + source replicate label; not independently verified biological replicate'}
    allkeys=set(tk)|set(ck);tp=tk.value_counts(normalize=True).to_dict();cp=ck.value_counts(normalize=True).to_dict()
    row['source_guide_replicate_total_variation']=.5*sum(abs(tp.get(k,0)-cp.get(k,0)) for k in allkeys) if len(target) and len(control) else None
    if not n:row.update(status='not_estimable',reason='no_matching_source_guide_replicate_cells');return row,[]
    results=[]
    for name in names:
        total=0.;used=0
        for key,count in tk.loc[ids].value_counts().items():
            a=target.loc[tk==key,name].to_numpy(float);b=control.loc[ck==key,name].to_numpy(float)
            if not np.isfinite(a).all() or not np.isfinite(b).all():continue
            total+=len(a)*(a.mean()-b.mean());used+=len(a)
        results.append({'variable':name,'matched_state_mean_difference':total/used if used else None,'finite_matched_target_cells':used,'status':'completed' if used else 'not_estimable'})
    row.update(status='completed',reason=None,interpretation='Target guide-composition weighted state/QC comparison; exact source guide labels and replicate matched, culture well and endpoint selection still confounded')
    return row,results


def run(genetic,output,resume=False):
    started=time.monotonic();source=json.loads((genetic/'report.json').read_text())
    if source['status']!='completed' or source['actual_tasks']!=14121:raise ValueError('all_GxE2_genetic_contexts_required')
    identity={'genetic_report_sha256':hash_file(genetic/'report.json'),'code':{name:hash_file(Path(__file__).with_name(name)) for name in ['gxe2_exposure.py','chemical_diagnostics.py','rna.py','profile_responses.py']},'parameters':PARAMETERS}
    if output.exists():
        if not resume or json.loads((output/'identity.json').read_text())!=identity or (output/'report.json').exists():raise ValueError('exposure_resume_identity_changed_or_completed')
    else:output.mkdir(parents=True);write_json(output/'identity.json',identity)
    contexts={tuple(json.loads(r['context'])):r for r in source['contexts']};results=[]
    for (line,drug,dose),record in sorted(contexts.items()):
        if drug=='vehicle':continue
        key=value_hash([line,drug,dose])[:20];destination=output/key
        if (destination/'identity.json').exists():
            old=json.loads((destination/'identity.json').read_text())
            if old['identity']!=identity:raise ValueError('exposure_condition_identity_changed')
            for name,digest in old['artifacts'].items():
                if hash_file(destination/name)!=digest:raise ValueError('completed_exposure_artifact_changed')
            results.append(old['summary']);continue
        destination.mkdir();a_dir=genetic/record['directory'];b_dir=genetic/contexts[(line,'vehicle',0.0)]['directory']
        for folder in [a_dir,b_dir]:
            upstream=json.loads((folder/'report.json').read_text())
            for name in ['cells.parquet','condition-genotype-profiles.parquet','condition-genotype-profiles.h5','scoring-weights.npz']:
                if hash_file(folder/name)!=upstream['artifacts'][name]:raise ValueError('genetic_context_source_changed')
        a=pd.read_parquet(a_dir/'cells.parquet');b=pd.read_parquet(b_dir/'cells.parquet')
        a=a.loc[a.primary_genetic_response_eligible].copy();b=b.loc[b.primary_genetic_response_eligible].copy()
        aw=np.load(a_dir/'scoring-weights.npz')['vehicle_state_weights'];bw=np.load(b_dir/'scoring-weights.npz')['vehicle_state_weights']
        if not np.array_equal(aw,bw):raise ValueError('cross_drug_state_weights_not_identical')
        names=[c for c in a if c.startswith('vehicle_state__')]+['log1p_total_counts','log1p_detected_genes']
        for frame in [a,b]:frame['log1p_total_counts']=np.log1p(frame.computed_total_counts);frame['log1p_detected_genes']=np.log1p(frame.computed_detected_genes)
        profiles_a=pd.read_parquet(a_dir/'condition-genotype-profiles.parquet');profiles_b=pd.read_parquet(b_dir/'condition-genotype-profiles.parquet').set_index('target')
        rows=[];state=[];composition=[];within=[];draws=[];matched=[]
        with h5py.File(a_dir/'condition-genotype-profiles.h5','r') as ah,h5py.File(b_dir/'condition-genotype-profiles.h5','r') as bh,h5py.File(destination/'gene-effects.h5','w') as out:
            g=ah['mean_logCP10K'].shape[1];effects=out.create_dataset('drug_minus_vehicle_mean_logCP10K',shape=(len(profiles_a),g),dtype='float32',compression='gzip',compression_opts=1,chunks=(1,g),fillvalue=np.nan)
            out.attrs['interpretation']='Pooled fixed source-genotype contrast; source guide distribution, culture wells, and endpoint selection can differ. Not a pure drug causal effect or a genetic knockdown ratio.'
            for record_a in profiles_a.itertuples():
                target=record_a.target;at=a.loc[a.analysis_target==target];bt=b.loc[b.analysis_target==target];record_b=profiles_b.loc[target]
                if len(at)!=record_a.primary_cells or len(bt)!=record_b.primary_cells:raise ValueError('genotype_profile_cell_scope_mismatch')
                result,st,co,wi,dr=distribution_contrast(at[names].to_numpy(float),bt[names].to_numpy(float),at.inferred_type.to_numpy(),bt.inferred_type.to_numpy(),names,PARAMETERS['seed']^int(value_hash([line,drug,dose,target])[:8],16))
                guide,gs=guide_comparison(at,bt,names,target)
                if len(at) and len(bt):
                    effect=ah['mean_logCP10K'][record_a.profile_row].astype(float)-bh['mean_logCP10K'][int(record_b.profile_row)].astype(float);effects[record_a.profile_row]=effect
                    result['pooled_full_gene_RMS']=float(np.sqrt(np.mean(effect**2)))
                else:result['pooled_full_gene_RMS']=None
                rows.append({'cell_line':line,'drug':drug,'dose_uM':dose,'hours':72,'source_genotype':target,**result,'guide_composition_sensitivity':guide,'gene_effect_row':record_a.profile_row,
                    'comparison_role':'chemical_exposure_at_fixed_source_genotype_vs_vehicle','target_RNA_status':'not_applicable_chemical_contrast',
                    'interpretation':'Genotype held by author guide inference; pooled RNA and state/composition changes are descriptive, with guide/culture/selection confounding'})
                for collection,values in [(state,st),(composition,co),(within,wi),(draws,dr),(matched,gs)]:collection.extend([{'source_genotype':target,**r} for r in values])
        for name,values in [('diagnostics',rows),('state-distributions',state),('type-composition',composition),('within-type-states',within),('resampling',draws),('guide-matched-state-differences',matched)]:pd.DataFrame(values).to_parquet(destination/(name+'.parquet'),index=False)
        summary={'cell_line':line,'drug':drug,'dose_uM':dose,'directory':key,'genotype_tasks':len(rows),'task_status_counts':dict(Counter(r['status'] for r in rows)),
            'stability_status_counts':dict(Counter(r['stability_status'] for r in rows)),'resampling_rows':len(draws),'null_reference_intersections':sum(r['null_reference_intersection'] for r in draws),
            'null_size_shortfall_rows':sum(r['null_target_cells_not_matched']>0 for r in draws)}
        write_json(destination/'identity.json',{'identity':identity,'summary':summary,'artifacts':{p.name:hash_file(p) for p in destination.iterdir() if p.is_file()}});results.append(summary)
        print('GxE2 fixed-genotype chemical conditions '+str(len(results))+'/24',flush=True)
    report={'status':'completed','phase':'all_fixed_source_genotype_chemical_exposure_descriptions','identity':identity,'conditions':results,
        'genotype_tasks':sum(r['genotype_tasks'] for r in results),'task_status_counts':dict(sum((Counter(r['task_status_counts']) for r in results),Counter())),
        'resampling_rows':sum(r['resampling_rows'] for r in results),'null_reference_intersections':sum(r['null_reference_intersections'] for r in results),'null_size_shortfall_rows':sum(r['null_size_shortfall_rows'] for r in results),
        'duration_seconds':time.monotonic()-started,'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'completed_at':datetime.now(timezone.utc).isoformat()}
    write_json(output/'report.json',report);return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['genetic','output']:p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--resume',action='store_true');a=p.parse_args();r=run(a.genetic,a.output,a.resume);print(json.dumps({k:v for k,v in r.items() if k not in ['identity','conditions']}))
