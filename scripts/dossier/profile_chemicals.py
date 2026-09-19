#!/usr/bin/env python
"""Full-source sciPlex3/4 chemical, composition and RNA-state assessments."""
import argparse
from collections import Counter
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid
import h5py
import numpy as np
import pandas as pd
from scipy import sparse
from annotation import RULES,marker_model,state_model,validate_records
from sparse_annotation import annotate_sparse
from chemical_design import parse_hash,design,read_hash_capture
from chemical_diagnostics import PARAMETERS,distribution_contrast
from rna import RNAFile,hash_file,scan,mapping_audit,value_hash,quantiles
from profile_responses import write_json,serial
from render import render
ROOT=Path(__file__).resolve().parents[2]
CODE=['profile_chemicals.py','chemical_design.py','chemical_diagnostics.py','sparse_annotation.py','annotation.py','rna.py','response.py','profile_responses.py','render.py']
PROTOCOL='https://github.com/yjcyxky/virtual-cell-challenge/issues/12#issuecomment-5745591351'


def normalized(matrix):
    matrix=matrix.copy();matrix.sum_duplicates();matrix.eliminate_zeros();matrix.sort_indices()
    total=np.asarray(matrix.sum(axis=1)).ravel()
    matrix.data=np.log1p(matrix.data*np.repeat(np.divide(10000,total,out=np.zeros_like(total),where=total>0),np.diff(matrix.indptr)))
    return matrix


def run(cache,references,evidence,output,group):
    started=time.monotonic();commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip();adapter=json.loads((cache/'identity.json').read_text())
    if adapter['status']!='completed' or adapter['group']!=group or any(r['differing_values'] for r in adapter['CDS_members']):raise ValueError('verified_matching_collection_required')
    for file,digest in adapter['input_sha256'].items():
        if hash_file(ROOT/file)!=digest:raise ValueError('raw_input_changed')
    for file,digest in adapter['artifacts'].items():
        if hash_file(cache/file)!=digest:raise ValueError('adapter_artifact_changed')
    for line in (references/'SHA256SUMS').read_text().splitlines():
        digest,file=line.split('  ',1)
        if hash_file(references/file)!=digest:raise ValueError('reference_changed')
    output.mkdir(parents=True,exist_ok=False);shutil.copytree(evidence,output/'evidence');shutil.copytree(references,output/'references')
    write_json(output/'adapter-identity.json',adapter)
    for r in adapter['CDS_members']:shutil.copy2(cache/r['directory']/'raw-count-comparison.parquet',output/(r['directory']+'-count-comparison.parquet'))
    prefix='GSM7056150_sciPlex_3_' if group=='chemical3' else 'GSM7056151_sciPlex_4_'
    raw=ROOT/'data/raw/mcfaline_figueroa2024';sheet=pd.read_csv(raw/(prefix+'hash_sample_sheet.txt.gz'),sep='\t',header=None,names=['hash','sequence','axis'])
    if sheet.sequence.duplicated().any():raise ValueError('nonunique_hash_sequence')
    sheet.to_parquet(output/'source-hash-sheet.parquet',index=False)
    metadata=pd.read_parquet(cache/'all-CDS-source-metadata.parquet')
    if metadata.source_barcode.duplicated().any():raise ValueError('ambiguous_CDS_membership')
    assignments=design(metadata,sheet,group);assignments.to_parquet(output/'condition-assignment.parquet',index=False)
    hashes,format_audit=read_hash_capture(raw/(prefix+'hashTable.out.txt.gz'))
    write_json(output/'hash-table-format.json',format_audit)
    if not set(hashes['hash'])<=set(sheet['hash']) or (hashes.umi<0).any() or not np.isfinite(hashes.umi).all() or (hashes.umi!=np.floor(hashes.umi)).any():raise ValueError('invalid_capture_hash_counts')
    hash_totals=hashes.groupby('barcode').umi.sum();by_hash=hashes.groupby(['barcode','hash']).umi.sum();max_hash=by_hash.groupby(level=0).max()
    audit=metadata[['source_barcode','top_oligo_W','hash_umis_W','top_to_second_best_ratio_W']].copy()
    audit['computed_all_hash_UMI']=audit.source_barcode.map(hash_totals)
    audit['assigned_hash_observed']=[(b,h) in by_hash.index for b,h in zip(audit.source_barcode,audit.top_oligo_W)]
    audit['assigned_hash_has_maximum_raw_UMI']=[by_hash.get((b,h))==max_hash.get(b) for b,h in zip(audit.source_barcode,audit.top_oligo_W)]
    audit['source_total_UMI_matches']=audit.hash_umis_W==audit.computed_all_hash_UMI
    audit.to_parquet(output/'hash-assignment-evidence.parquet',index=False)
    # Disagreement invalidates conditional assignment only; it never deletes source RNA.
    bad=set(audit.loc[~audit.source_total_UMI_matches|~audit.assigned_hash_observed,'source_barcode'])
    assignments.loc[assignments.source_barcode.isin(bad),'condition_eligible']=False
    assignments.loc[assignments.source_barcode.isin(bad),'assignment_limitation']='source_hash_count_or_presence_disagreement'
    assignments.to_parquet(output/'condition-assignment.parquet',index=False)
    path=cache/'coordinate-counts.h5ad';facts,cells,genes=scan(path,adapter['coordinate']['coordinate_sha256'],'McFaline2024_'+group)
    if not facts['all_counts_finite_nonnegative_integer']:raise ValueError('invalid_source_counts')
    cells=cells.merge(metadata.add_prefix('source_CDS_').rename(columns={'source_CDS_source_barcode':'source_barcode'}),on='source_barcode',how='left',validate='1:1',sort=False)
    cells=cells.merge(assignments,on='source_barcode',how='left',validate='1:1',sort=False)
    cells['condition_eligible']=cells.condition_eligible.fillna(False).astype(bool);missing=cells.assignment_limitation.isna()&~cells.condition_eligible
    cells.loc[missing,'assignment_limitation']='no_author_CDS_condition_metadata'
    if cells.loc[cells.condition_eligible,'assignment_limitation'].notna().any():raise ValueError('eligible_assignment_has_exclusion_reason')
    cells['genetic_supervision_role']='not_applicable_no_genetic_intervention'
    declared=[]
    for label in sorted(set(sheet['hash'])):
        for line in ([None] if group=='chemical3' else ['A172','T98G','U87MG']):declared.append(parse_hash(label,group,line))
    declared=pd.DataFrame(declared);declared.to_parquet(output/'declared-hash-conditions.parquet',index=False)
    conditions=declared.drop_duplicates('condition_key').drop(columns=['source_hash_label','hash_well']).sort_values('condition_key').reset_index(drop=True)
    conditions.insert(0,'condition_index',np.arange(len(conditions)));lookup=dict(zip(conditions.condition_key,conditions.condition_index))
    cells['condition_index']=cells.condition_key.map(lookup).where(cells.condition_eligible,-1).fillna(-1).astype(int)
    cells['annotation_stratum']=cells.stratum.where(cells.condition_eligible,'unresolved_source_condition')
    conditions['observed_cells']=conditions.condition_index.map(cells.loc[cells.condition_eligible].condition_index.value_counts()).fillna(0).astype(int)
    strata=sorted(set(conditions.stratum)|{'unresolved_source_condition'});stratum_lookup={s:i for i,s in enumerate(strata)};bgid=cells.annotation_stratum.map(stratum_lookup).to_numpy()
    g=len(genes);profiles=np.zeros((len(conditions),g));group_sums=np.zeros((len(strata),g));group_n=np.zeros(len(strata),np.int64)
    with RNAFile(path) as source:
        symbols=source.var.gene_name.tolist();ensembl=source.var.index.str.split('.').str[0].tolist()
        for start,matrix in source.blocks(1024):
            log=normalized(matrix);stop=start+len(log.indptr)-1;ids=cells.condition_index.iloc[start:stop].to_numpy();eligible=ids>=0
            member=sparse.csr_matrix((np.ones(eligible.sum()),(ids[eligible],np.flatnonzero(eligible))),shape=(len(conditions),len(ids)))
            profiles+=np.asarray((member@log).toarray())
            b=bgid[start:stop];member=sparse.csr_matrix((np.ones(len(ids)),(b,np.arange(len(ids)))),shape=(len(strata),len(ids)))
            group_sums+=np.asarray((member@log).toarray());group_n+=np.bincount(b,minlength=len(strata))
    profiles=np.divide(profiles,conditions.observed_cells.to_numpy()[:,None],out=np.full_like(profiles,np.nan),where=conditions.observed_cells.to_numpy()[:,None]>0)
    hgnc=ROOT/'data/raw/networks/hgnc_complete_set.txt';official=ROOT/'data/raw/arc_vcc2026_controls/gene_names.csv'
    mapping=mapping_audit(symbols,pd.read_csv(hgnc,sep='\t',low_memory=False),pd.read_csv(official).gene_name.tolist(),ensembl);mapping.insert(0,'source_feature_id',genes.source_gene)
    mapping.to_parquet(output/'gene-mapping.parquet',index=False);genes.to_parquet(output/'genes.parquet',index=False)
    safe=mapping.mapped_symbol.where(~mapping.many_to_one_mapping&(mapping.symbol_vs_ensembl!='conflict'));resolved=[str(s) if pd.notna(s) else None for s in safe]
    reference=json.loads((references/'gene_sets.json').read_text());model=marker_model(resolved,reference['profiles']);write_json(output/'type-coverage.json',model['coverage'])
    weights=[];covs=[];backgrounds=[];control_ids={};baseline=[]
    for index,stratum in enumerate(strata):
        control=conditions.loc[(conditions.stratum==stratum)&conditions.is_vehicle&(conditions.observed_cells>0)];n=int(control.observed_cells.sum())
        mean=np.sum(profiles[control.condition_index]*control.observed_cells.to_numpy()[:,None],axis=0)/n if n else group_sums[index]/max(1,group_n[index])
        w,c=state_model(resolved,reference['states'],mean);weights.append(w);covs.extend([{'stratum':stratum,**r} for r in c]);baseline.append(mean)
        ids=np.flatnonzero(cells.condition_eligible.to_numpy()&(cells.annotation_stratum.to_numpy()==stratum)&cells.is_vehicle.fillna(False).to_numpy());control_ids[stratum]=ids
        backgrounds.append({'stratum':stratum,'background_index':index,'n_vehicle':n,'n_observed':int(group_n[index]),
            'method':'same_line_plate_replicate_vehicle' if n else 'pooled_observed_stratum_fallback_no_vehicle','source_condition_indices':control.condition_index.tolist()})
    np.savez_compressed(output/'backgrounds.npz',state_weights=np.stack(weights),mean_logCP10K=np.stack(baseline));write_json(output/'backgrounds.json',backgrounds);write_json(output/'state-coverage.json',covs)
    labels=[]
    with RNAFile(path) as source:
        for start,matrix in source.blocks(512):
            log=normalized(matrix);n=log.shape[0];part,scores=annotate_sparse(log,['not_genetic_target']*n,model)
            for name,values in scores.items():part[name]=values
            state=np.full((n,len(reference['states'])),np.nan);b=bgid[start:start+n]
            for index in np.unique(b):
                ids=np.flatnonzero(b==index);v=log[ids]@weights[index];ok=np.array([r['status']=='completed' for r in covs[index*len(reference['states']):(index+1)*len(reference['states'])]])
                v[:,~ok]=np.nan;state[ids]=v
            for j,name in enumerate(reference['states']):part['state__'+name]=state[:,j]
            part['state_background_method']=[backgrounds[i]['method'] for i in b];part['source_context_mismatch']=~part.inferred_lineage.isin(['unknown','neural']);labels.append(part)
    combined=pd.concat([cells,pd.concat(labels,ignore_index=True)],axis=1);combined['available_before_endpoint']=False;combined['truth_label']=False
    validate_records(combined,adapter['coordinate']['coordinate_sha256'],len(cells));combined.to_parquet(output/'cells.parquet',index=False,compression='zstd')
    names=['state__'+n for n in reference['states']]+['log1p_total_counts','log1p_detected_genes']
    vectors=np.column_stack([combined[['state__'+n for n in reference['states']]].to_numpy(),np.log1p(combined.computed_total_counts),np.log1p(combined.computed_detected_genes)])
    types=combined.inferred_type.to_numpy();ids_by_condition=combined.groupby('condition_index').indices
    summaries=[];states=[];compositions=[];within=[];draws=[];interaction=[]
    with h5py.File(output/'expression-profiles.h5','w') as h:
        h.create_dataset('mean_logCP10K',data=profiles.astype(np.float32),compression='gzip',compression_opts=1,shuffle=True)
        effects=h.create_dataset('effect_vs_matched_vehicle',shape=profiles.shape,dtype='float32',chunks=(1,g),compression='gzip',compression_opts=1,shuffle=True,fillvalue=np.nan)
        interactions=h.create_dataset('combination_difference',shape=profiles.shape,dtype='float32',chunks=(1,g),compression='gzip',compression_opts=1,shuffle=True,fillvalue=np.nan)
        h.attrs['interpretation']='All source genes; mean cell log1p(CP10K), not independent biological replicates. Combination difference is scale-dependent and not pharmacologic synergy.'
        h['condition_index']=conditions.condition_index.to_numpy()
        for row in conditions.to_dict('records'):
            i=row['condition_index'];ids=ids_by_condition.get(i,np.array([],int));controls=control_ids[row['stratum']]
            if row['is_vehicle']:
                # Other independently identified wells are not available for every vehicle record;
                # avoid using the identical records on both sides of a descriptive response.
                summaries.append({**row,'status':'not_applicable','reason':'vehicle_baseline_not_drug_response','stability_status':'not_applicable'});continue
            result,st,co,wi,dr=distribution_contrast(vectors[ids],vectors[controls],types[ids],types[controls],names,PARAMETERS['seed']^int(value_hash([group,i])[:8],16))
            effects[i]=profiles[i]-baseline[stratum_lookup[row['stratum']]] if len(ids) and len(controls) else np.nan
            result['full_gene_RMS']=float(np.sqrt(np.nanmean(effects[i].astype(float)**2))) if len(ids) and len(controls) else None
            summaries.append({**row,**result})
            for target,items in [(states,st),(compositions,co),(within,wi),(draws,dr)]:target.extend([{'condition_index':i,**r} for r in items])
            components=json.loads(row['components'])
            if len(components)==2:
                arm_indices=[]
                for component in components:
                    wanted=json.dumps([component],separators=(',',':'));found=conditions.loc[(conditions.stratum==row['stratum'])&(conditions.components==wanted)&(conditions.observed_cells>0)]
                    if len(found)>1:raise ValueError('ambiguous_single_drug_comparator')
                    arm_indices.append(int(found.condition_index.iloc[0]) if len(found) else None)
                ok=len(ids)>0 and len(controls)>0 and all(x is not None for x in arm_indices)
                if ok:
                    delta=profiles[i]-profiles[arm_indices[0]]-profiles[arm_indices[1]]+baseline[stratum_lookup[row['stratum']]];interactions[i]=delta
                    state_delta=vectors[ids].mean(axis=0)-vectors[ids_by_condition[arm_indices[0]]].mean(axis=0)-vectors[ids_by_condition[arm_indices[1]]].mean(axis=0)+vectors[controls].mean(axis=0)
                    composition_delta={label:float(np.mean(types[ids]==label)-np.mean(types[ids_by_condition[arm_indices[0]]]==label)-np.mean(types[ids_by_condition[arm_indices[1]]]==label)+np.mean(types[controls]==label)) for label in sorted(set(types))}
                interaction.append({'condition_index':i,'components':row['components'],'single_arm_indices':arm_indices,'status':'completed' if ok else 'not_estimable',
                    'reason':None if ok else 'missing_same_line_plate_replicate_single_drug_or_vehicle_arm','full_gene_RMS':float(np.sqrt(np.mean(delta**2))) if ok else None,
                    'state_and_QC_difference':dict(zip(names,state_delta)) if ok else None,'inferred_composition_difference':composition_delta if ok else None,'interpretation':'Combination minus each single drug plus vehicle in mean logRNA; not pharmacologic synergy'})
    conditions.to_parquet(output/'conditions.parquet',index=False)
    for name,rows in [('condition-diagnostics',summaries),('state-distributions',states),('composition-contrasts',compositions),('within-type-states',within),('resampling',draws),('combination-differences',interaction)]:
        pd.DataFrame(rows).to_parquet(output/(name+'.parquet'),index=False)
    combined.groupby(['condition_index','inferred_type'],dropna=False,observed=True).size().reset_index(name='cells').to_parquet(output/'all-record-composition.parquet',index=False)
    for file,digest in adapter['input_sha256'].items():
        if hash_file(ROOT/file)!=digest:raise ValueError('raw_input_changed_during_assessment')
    report={'schema_version':2,'bundle_id':'mcfaline-'+group+'-'+uuid.uuid4().hex,'title':'McFaline '+group+' 全量化学条件与细胞注释评估','status':'completed',
        'n_cells':len(cells),'n_genes':len(genes),'CDS_cells':len(metadata),'condition_eligible_cells':int(cells.condition_eligible.sum()),'assignment_limitations':cells.assignment_limitation.value_counts().to_dict(),
        'declared_conditions':len(conditions),'observed_conditions':int((conditions.observed_cells>0).sum()),'diagnostic_status_counts':dict(Counter(r['status'] for r in summaries)),
        'stability_status_counts':dict(Counter(r['stability_status'] for r in summaries)),'combination_status_counts':dict(Counter(r['status'] for r in interaction)),
        'resampling_rows':len(draws),'null_overlap':sum(r['null_reference_intersection'] for r in draws),'null_size_shortfall_rows':sum(r['null_target_cells_not_matched']>0 for r in draws),
        'pooled_types':combined.inferred_type.value_counts().to_dict(),'uncalibrated_records':len(combined),'source_context_mismatch':int(combined.source_context_mismatch.sum()),
        'hash_capture_rows':len(hashes),'hash_total_disagreements':int((~audit.source_total_UMI_matches).sum()),'assigned_hash_not_raw_max':int((~audit.assigned_hash_has_maximum_raw_UMI).sum()),
        'count_facts':facts,'code_commit':commit,'code':{n:hash_file(Path(__file__).with_name(n)) for n in CODE},'input_sha256':adapter['input_sha256'],'inputs_unchanged':True,
        'reference_sha256':hash_file(references/'gene_sets.json'),'gene_reference_sha256':{str(p.relative_to(ROOT)):hash_file(p) for p in [hgnc,official]},
        'runtime':{'python':sys.version,'numpy':np.__version__,'pandas':pd.__version__,'uv_lock_sha256':hash_file(Path(__file__).with_name('uv.lock'))},'parameters':PARAMETERS,'annotation_rules':RULES,
        'methods':{'protocol':PROTOCOL,'condition_keys':['source line','hash plate','source replicate','complete drug components/doses'],
            'response':'All declared conditions, all source genes; per-condition mean logCP10K versus same line/plate/replicate vehicle; 20 state/logQC half-sample and disjoint vehicle draws when estimable',
            'genetic_diagnostics':'Target RNA knockdown, sgRNA consistency and CRISPRi supervision not applicable to these non-genetic collections',
            'annotation':'Frozen broad human marker agreement with unknown, uncalibrated probability=null; 12 RNA proxies with matched vehicle gene background; explicit pooled fallback',
            'selection':'Source CDS filtered using hash support and transcriptome clustering; all raw records preserved, missing source assignment remains unresolved',
            'exposure':'All local endpoint RNA and source inferred annotations explored; unavailable as pretreatment input'},
        'limitations':['药物终点不是 CRISPRi 遗传真值；vehicle 是含 DMSO 的处理条件，不是无处理 NTC。',
            '源 replicate 名称不证明独立培养；细胞抽样稳定性不能替代生物重复。hash plate、药物、RT 与细胞系可能混杂。',
            '同 hash 标签可对应多个捕获寡核苷酸；它们不是独立培养。enrichment ratio 与原始 UMI 最大值不同。',
            '全部类型/状态推断未校准；肿瘤细胞与健康广义谱系参考可能不匹配。unknown 及源标签保留。',
            '来源 CDS 未包含的记录仍完成 RNA 注释，但不猜测其化学处理；条件差异仅使用有证据支持的标签。',
            '组合减单药差分依赖 logRNA 尺度，不代表药理协同、存活效应或机制确认。'],
        'references':['https://pmc.ncbi.nlm.nih.gov/articles/PMC10879025/','https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE225775'],
        'tables':[{'title':'全部化学条件与对照适用性','columns':list(dict.fromkeys(k for row in summaries for k in row)),'rows':summaries},
            {'title':'联合用药描述性差分','columns':list(dict.fromkeys(k for row in interaction for k in row)),'rows':interaction},
            {'title':'全部状态背景','rows':backgrounds}],
        'completed_at':datetime.now(timezone.utc).isoformat(),'duration_seconds':time.monotonic()-started,'reproduce':sys.argv}
    report['artifacts']=[{'file':str(p.relative_to(output)),'sha256':hash_file(p)} for p in sorted(output.rglob('*')) if p.is_file()]
    write_json(output/'report.json',report);(output/'report.html').write_text(render(serial(report)))
    (output/'SHA256SUMS').write_text(''.join(f'{hash_file(p)}  {p.relative_to(output)}\n' for p in sorted(output.rglob('*')) if p.is_file() and p.name!='SHA256SUMS'))
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--group',choices=['chemical3','chemical4'],required=True)
    for n in ['cache','references','evidence','output']:p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args()
    try:r=run(a.cache,a.references,a.evidence,a.output,a.group)
    except Exception as e:
        if a.output.exists():write_json(a.output/'failure.json',{'status':'failed','type':type(e).__name__,'error':str(e)})
        raise
    print(json.dumps({k:v for k,v in r.items() if k not in ['artifacts','tables','count_facts','input_sha256','code','runtime']}))
