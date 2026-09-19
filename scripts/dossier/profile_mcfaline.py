#!/usr/bin/env python
"""Full GxE1 genetic and chemical assessment with original condition identities."""
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
import numpy as np
import pandas as pd
from annotation import RULES,marker_model,state_model,annotate,validate_records
from conditioned import normalize_selection
from mcfaline import parse_gxe1_hash
from profile_responses import PARAMETERS,task_result,write_json,serial
from profile_crispri import state_composition
from response import grouped_moments,control_half_means
from rna import hash_file,scan,mapping_audit,value_hash,quantiles
from render import render
ROOT=Path(__file__).resolve().parents[2]
PROTOCOL='https://github.com/yjcyxky/virtual-cell-challenge/issues/10#issuecomment-5745312439'
CONDITION=['cell_line','effector','guide_library','drug','dose_value']
LIBRARIES={'HPRT1':['HPRT1'],'MMR':['MGMT','MLH1','MSH2','MSH3','MSH6','PMS2']}


def condition_key(row):return json.dumps([row[c] for c in CONDITION],separators=(',',':'))


def contrast(log,thin,cells,target_ids,control_ids,strata,genes,target_feature,conflicts,seed,role):
    """Use only explicit eligible controls; preserve unmatched counts in the result."""
    target_ids=np.asarray(target_ids,dtype=int);control_ids=np.asarray(control_ids,dtype=int)
    original_n=len(target_ids)
    names=sorted(set(strata[control_ids]) & set(strata[target_ids]))
    keep=np.isin(strata[target_ids],names);used=target_ids[keep]
    if not len(used) or not len(control_ids):
        result={'status':'not_estimable','reason':'no_target_or_matching_control','observed_target_cells':original_n,'matched_target_cells':0,'control_cells':len(control_ids),'role':role}
        result['target_RNA']={'status':'not_applicable','reason':'chemical_intervention_has_no_single_genetic_target'} if role=='chemical_response' else {'status':'not_estimable','reason':'no_target_or_matching_control'}
        return result,None,[]
    rows=cells.iloc[used].copy();rows['source_batch']=strata[used]
    control=log[control_ids];control_thin=thin[control_ids];cb=strata[control_ids]
    moments=grouped_moments(control,cb,names);thin_moments=grouped_moments(control_thin,cb,names)
    halves=control_half_means(control,cb,names,PARAMETERS['resampling_repetitions'],seed)
    pools={b:np.flatnonzero(cb==b) for b in names}
    result,features,draws=task_result(target_feature,log[used],thin[used],rows,control,cb,names,moments,thin_moments,halves,genes,conflicts,seed,control_pools=pools)
    result.update(role=role,observed_target_cells=original_n,matched_target_cells=len(used),unmatched_target_cells=original_n-len(used),control_cells=len(control_ids))
    if role=='chemical_response':
        result['target_RNA']={'status':'not_applicable','reason':'chemical_intervention_has_no_single_genetic_target'}
        for draw in draws:draw['interpretation']='Disjoint matched DMSO vehicle sampling; genotype/guide fixed, no independent culture inference'
    return result,features,draws


def run(cache,references,evidence,output):
    started=time.monotonic();commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    identity=json.loads((cache/'identity.json').read_text())
    if identity['status']!='completed' or not identity['inputs_unchanged']:raise ValueError('verified_source_adapter_required')
    for file,digest in identity['input_sha256'].items():
        if hash_file(ROOT/file)!=digest:raise ValueError('source_input_changed')
    for file,digest in identity['artifacts'].items():
        if hash_file(cache/file)!=digest:raise ValueError('adapter_artifact_changed')
    if identity['CDS_differing_values'] or identity['guide_label_check_failures'] or identity['guide_source_rule_disagreements'] or identity['hash_total_UMI_disagreements']:
        raise ValueError('unresolved_adapter_disagreement')
    for line in (references/'SHA256SUMS').read_text().splitlines():
        digest,name=line.split('  ',1)
        if hash_file(references/name)!=digest:raise ValueError('annotation_reference_changed')
    output.mkdir(parents=True,exist_ok=False);shutil.copytree(evidence,output/'evidence');shutil.copytree(references,output/'references')
    (output/'adapter').mkdir();shutil.copy2(cache/'identity.json',output/'adapter/identity.json')
    for path in cache.glob('*.parquet'):shutil.copy2(path,output/'adapter'/path.name)
    path=cache/'coordinate-counts.h5ad';facts,cells,genes=scan(path,identity['coordinate']['coordinate_sha256'],'McFaline2024_GxE1')
    metadata=pd.read_parquet(cache/'CDS-source-metadata.parquet').add_prefix('source_CDS_').rename(columns={'source_CDS_source_barcode':'source_barcode'})
    design=pd.read_parquet(cache/'source-design.parquet')
    cells=cells.merge(metadata,on='source_barcode',how='left',validate='1:1',sort=False).merge(design,on='source_barcode',how='left',validate='1:1',sort=False)
    missing=cells.genetic_response_assignment_status.isna();cells.loc[missing,'genetic_response_assignment_status']='not_estimable';cells.loc[missing,'assignment_limitation']='no_CDS_metadata'
    cells['genetic_response_eligible']=cells.genetic_response_assignment_status=='eligible'
    cells['source_target_gene']=cells.analysis_target.where(cells.analysis_target!='NTC','non-targeting')
    cells['source_guide_id']=cells.source_CDS_protospacer_sequence.map(lambda x:','.join(sorted(str(x).split(','))) if pd.notna(x) else None)
    cells['source_context']=[condition_key(row) if pd.notna(row['effector']) else 'unresolved_hash' for row in cells.to_dict('records')]
    cells['source_batch']=[json.dumps([r.hash_plate,r.hash_well]) if pd.notna(r.hash_plate) else 'unresolved_hash' for r in cells.itertuples()]
    cells['source_task']=[value_hash([c,t])[:20] for c,t in zip(cells.source_context,cells.analysis_target.fillna('unresolved'))]
    hgnc=ROOT/'data/raw/networks/hgnc_complete_set.txt';axis=ROOT/'data/raw/arc_vcc2026_controls/gene_names.csv'
    mapping=mapping_audit(genes.gene_name.tolist(),pd.read_csv(hgnc,sep='\t',low_memory=False),pd.read_csv(axis).gene_name.tolist(),genes.source_gene.tolist())
    mapping=mapping.rename(columns={'source_gene':'source_symbol'});mapping.insert(0,'source_gene',genes.source_gene)
    safe=mapping.mapped_symbol.where(~mapping.many_to_one_mapping&(mapping.symbol_vs_ensembl!='conflict'))
    resolved=[str(x) if pd.notna(x) else None for x in safe];native={s:g for s,g in zip(resolved,genes.source_gene) if s is not None}
    conflicts=set(mapping.loc[safe.isna(),'source_gene'])
    genes.to_parquet(output/'genes.parquet',index=False);mapping.to_parquet(output/'gene-mapping.parquet',index=False)
    log,thin,view=normalize_selection(path,cells.row_index.to_numpy(),output/'temporary-view',PARAMETERS['seed'])
    sheet=pd.read_parquet(cache/'hash-design.parquet');declared=pd.DataFrame([{'source_hash':r.hash,**parse_gxe1_hash(r.hash)} for r in sheet.itertuples()])
    declared['context']=[condition_key(r) for r in declared.to_dict('records')]
    all_contexts=declared.drop_duplicates('context').sort_values('context');eligible=cells.genetic_response_eligible.to_numpy()
    results=[];draws=[];coverage=[];feature_names=genes.source_gene.tolist()
    def save_task(key,kind,target,ids,ntc,strata,details):
        task_id=value_hash([key,kind,target]);seed=PARAMETERS['seed']^int(task_id[:8],16)
        feature=native.get(target,target) if kind=='genetic_response' else '__chemical_intervention__'
        result,per_gene,samples=contrast(log,thin,cells,ids,ntc,strata,feature_names,feature,conflicts,seed,kind)
        result.update(task_id=task_id,condition=key,intervention_target=target,
            composition_task=value_hash([key,target])[:20] if kind=='genetic_response' else None,**details)
        folder=output/'tasks'/task_id;folder.mkdir(parents=True)
        if per_gene is not None:
            per_gene.to_parquet(folder/'genes.parquet',index=False,compression='zstd');result['gene_results']=str((folder/'genes.parquet').relative_to(output))
        write_json(folder/'result.json',result);results.append(result);draws.extend({'task_id':task_id,**s} for s in samples)
    for row in all_contexts.to_dict('records'):
        mask=(cells.source_context==row['context']).to_numpy();control=np.flatnonzero(mask&eligible&(cells.analysis_target=='NTC').to_numpy())
        coverage.append({**{c:row[c] for c in CONDITION},'context':row['context'],'source_hash_wells':int((declared.context==row['context']).sum()),
            'all_assigned_source_cells':int(mask.sum()),'eligible_genetic_cells':int((mask&eligible).sum()),'eligible_NTC':len(control),
            'status':'completed' if mask.any() else 'not_estimable','reason':None if mask.any() else 'no_local_source_cells_in_declared_condition'})
        for target in LIBRARIES[row['guide_library']]:
            ids=np.flatnonzero(mask&eligible&(cells.analysis_target==target).to_numpy())
            save_task(row['context'],'genetic_response',target,ids,control,cells.source_batch.astype(str).to_numpy(),
                {'effector':row['effector'],'control_role':'same drug/dose source-inferred non-targeting guide','matching':'same source experimental hash well'})
        print('GxE1 genetic '+row['context'],flush=True)
    # Vehicle is compared at fixed genetic assignment, with exact protospacer combinations matched.
    for row in all_contexts.loc[all_contexts.drug!='dmso'].to_dict('records'):
        background=(cells.effector==row['effector'])&(cells.guide_library==row['guide_library'])
        for genotype in ['NTC',*LIBRARIES[row['guide_library']]]:
            fixed=background&cells.genetic_response_eligible&(cells.analysis_target==genotype)
            ids=np.flatnonzero((fixed&(cells.source_context==row['context'])).to_numpy());controls=np.flatnonzero((fixed&(cells.drug=='dmso')).to_numpy())
            strata=cells.source_guide_id.fillna('unassigned').astype(str).to_numpy()
            save_task(row['context'],'chemical_response',genotype,ids,controls,strata,
                {'effector':row['effector'],'drug':row['drug'],'dose_uM':row['dose_value'],'fixed_genotype':genotype,
                    'control_role':'0.1% v/v DMSO with same effector/library/source genotype and exact guide combination',
                    'matching':'guide matched across different culture wells; well/treatment confounding remains'})
    # Every original RNA cell is annotated, including assignment-ineligible and CDS-missing rows.
    reference=json.loads((references/'gene_sets.json').read_text());model=marker_model(resolved,reference['profiles'])
    supported=cells.hash_supported.fillna(False)&cells.hash_condition_consistent.fillna(False)
    cells['annotation_background_group']=cells.source_context.where(supported,'unresolved_hash')
    labels=[];state_coverage=[];(output/'annotation-backgrounds').mkdir()
    for context,group in cells.groupby('annotation_background_group',sort=True):
        indices=group.index.to_numpy();ntc=group.index[group.genetic_response_eligible&(group.analysis_target=='NTC')].to_numpy()
        base_ids=ntc if len(ntc) else indices;baseline=np.asarray(log[base_ids]).mean(axis=0,dtype=np.float64)
        weights,cov=state_model(resolved,reference['states'],baseline);background='same_condition_source_NTC' if len(ntc) else 'pooled_observed_RNA_fallback_no_defensible_NTC'
        group_id=value_hash(context)[:20];np.savez_compressed(output/'annotation-backgrounds'/(group_id+'.npz'),mean_logCP10K=baseline,state_weights=weights)
        for item in cov:state_coverage.append({'context':context,'background':background,'baseline_cells':len(base_ids),**item})
        for start in range(0,len(indices),256):
            ids=indices[start:start+256];block=np.asarray(log[ids]);targets=cells.analysis_target.iloc[ids].where(cells.genetic_response_eligible.iloc[ids],None).tolist()
            part,scores=annotate(block,targets,model)
            for name,values in scores.items():part[name]=values
            values=block@weights
            for j,(name,details) in enumerate(zip(reference['states'],cov)):part['state__'+name]=values[:,j] if details['status']=='completed' else np.nan
            part['row_index']=ids;part['state_background']=background;part['source_context_mismatch']=~part.inferred_lineage.isin(['neural','unknown']);labels.append(part)
    annotations=pd.concat(labels,ignore_index=True).sort_values('row_index').reset_index(drop=True).drop(columns='row_index')
    combined=pd.concat([cells,annotations],axis=1);combined['available_before_endpoint']=False;combined['truth_label']=False
    validate_records(combined,identity['coordinate']['coordinate_sha256'],facts['n_cells'])
    combined.to_parquet(output/'cells.parquet',index=False,compression='zstd')
    composition,states,within,type_tasks=state_composition(combined.loc[combined.genetic_response_eligible],list(reference['states']))
    for table in [composition,states,within]:
        if 'reason' in table:table['reason']=table.reason.str.replace('GEM','hash_well',regex=False)
    within=within.rename(columns={'shared_GEM_groups':'shared_hash_wells'})
    for name,table in [('composition',composition),('states',states),('within-type-states',within)]:table.to_parquet(output/(name+'.parquet'),index=False)
    write_json(output/'type-coverage.json',model['coverage']);write_json(output/'state-coverage.json',state_coverage)
    write_json(output/'tasks.json',results);write_json(output/'resampling.json',draws);write_json(output/'condition-coverage.json',coverage)
    assignment_composition=[]
    for (context,status),group in combined.groupby(['source_context','genetic_response_assignment_status'],dropna=False):
        assignment_composition.append({'context':context,'assignment_status':status,'n_cells':len(group),'inferred_types':group.inferred_type.value_counts().to_dict(),
            'unknown_fraction':float((group.inferred_type=='unknown').mean()),'state_distributions':{s:quantiles(group['state__'+s]) for s in reference['states']}})
    write_json(output/'assignment-composition.json',assignment_composition)
    for file,digest in identity['input_sha256'].items():
        if hash_file(ROOT/file)!=digest:raise ValueError('source_changed_during_assessment')
    del log,thin;shutil.rmtree(output/'temporary-view')
    compact=[{k:v for k,v in r.items() if k not in ['matching','gene_results']} for r in results]
    report={'schema_version':2,'bundle_id':'mcfaline-gxe1-'+uuid.uuid4().hex,'title':'McFaline GxE1 全遗传、药物条件与逐细胞评估',
        'status':'completed','completed_at':datetime.now(timezone.utc).isoformat(),'duration_seconds':time.monotonic()-started,
        'code_commit':commit,'input_sha256':identity['input_sha256'],'inputs_unchanged':True,'adapter_identity_sha256':hash_file(cache/'identity.json'),
        'code':{n:hash_file(Path(__file__).with_name(n)) for n in ['profile_mcfaline.py','mcfaline.py','conditioned.py','annotation.py','response.py','profile_responses.py','rna.py','profile_crispri.py','render.py']},
        'reference_sha256':hash_file(references/'gene_sets.json'),'gene_reference_sha256':{str(p.relative_to(ROOT)):hash_file(p) for p in [hgnc,axis]},
        'runtime':{'python':sys.version,'uv_lock_sha256':hash_file(Path(__file__).with_name('uv.lock'))},
        'parameters':PARAMETERS,'annotation_rules':RULES,'normalized_analysis_view':view,'numeric':facts,
        'task_status_counts':{kind:dict(Counter(r['status'] for r in results if r['role']==kind)) for kind in ['genetic_response','chemical_response']},
        'assignment_status_counts':combined.genetic_response_assignment_status.value_counts().to_dict(),'assignment_limitations':combined.assignment_limitation.value_counts().to_dict(),
        'pooled_types':combined.inferred_type.value_counts().to_dict(),'uncalibrated_records':len(combined),
        'methods':{'protocol':PROTOCOL,'genetic':'Same effector/library/drug/dose, matched hash culture wells; only supported single-genotype assignments, no join collisions',
            'chemical':'Within fixed effector/library/genotype, each nonzero drug dose versus DMSO; exact guide combination matched; all genes describe chemical response',
            'sampling':'20 target half-splits and 20 disjoint control splits; conditional cell sampling only; insufficient counts remain not estimable',
            'annotation':'Every original RNA cell; fixed conservative human markers; per-condition NTC expression-bin background or explicit observed-RNA fallback',
            'uncertain_records':'Preserved in RNA/QC/annotation denominators; source-inferred labels and assignment limitations exported independently',
            'source_hash_UMI':'CDS hash_umis_W is total hash UMI, not assigned-hash UMI; full raw evidence independently verified'},
        'limitations':['A172 是转化背景；神经/胶质参考匹配未校准，source cell line 不是每细胞细类型真值。',
            'CRISPRa 单列真实作用角色；不将激活当作 CRISPRi。药物和培养孔混杂，组合 barcode 不作独立培养重复。',
            'DMSO 源 dose=0 是条件代码；全部处理 vehicle 为 0.1% v/v。给药 96 小时来自论文方法，hash 末尾字段本身不单独证明时间。',
            '源 guide 调用可复现仍不是 ground truth；多靶标、文库/模式冲突、hash 弱及连接键碰撞保留不可估计原因。',
            '论文描述的 18,585 个 HPRT1 实验细胞与本地 CDS 同时含 HPRT1/MMR 标签的对应关系未确证；以全部本地对象为分母。',
            '未校准推断概率为空；无 NTC 的条件仅有明示回退背景，不能当作处理前状态。'],
        'exposure':'All eight source files, all observed RNA counts, source guide/hash inference, all applicable conditional endpoints examined',
        'references':['https://pmc.ncbi.nlm.nih.gov/articles/PMC10879025/','https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE225775','https://github.com/cole-trapnell-lab/sci-Plex-GxE/tree/63eed120697611d8baf4a207a797a147308a6674'],
        'tables':[{'title':'全部来源声明条件','rows':coverage},{'title':'全部遗传与药物响应','columns':sorted({k for row in compact for k in row}),'rows':compact},
            {'title':'全部记录组成与赋值资格','rows':assignment_composition},{'title':'合格遗传任务类型组成','rows':type_tasks},{'title':'全部状态覆盖与背景','rows':state_coverage}],
        'reproduce':sys.argv}
    report['artifacts']=[{'file':str(p.relative_to(output)),'sha256':hash_file(p)} for p in sorted(output.rglob('*')) if p.is_file()]
    write_json(output/'report.json',report);(output/'report.html').write_text(render(serial(report)))
    (output/'SHA256SUMS').write_text(''.join(f'{hash_file(p)}  {p.relative_to(output)}\n' for p in sorted(output.rglob('*')) if p.is_file() and p.name!='SHA256SUMS'))
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['cache','references','evidence','output']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    try:r=run(a.cache,a.references,a.evidence,a.output)
    except Exception as e:
        if a.output.exists():write_json(a.output/'failure.json',{'status':'failed','error':str(e),'type':type(e).__name__})
        raise
    print(json.dumps({'status':r['status'],'bundle_id':r['bundle_id']}))
