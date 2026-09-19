#!/usr/bin/env python
"""All-construct response and per-cell annotation for verified CRISPRi sources.

Consumes source structure bundles, preserving native feature and construct axes.
Each experiment runs independently with only its own matched non-targeting cells.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from rna import hash_file, value_hash, quantiles
from response import grouped_moments, control_half_means, correlation
from profile_responses import PARAMETERS, build_cache, task_result, serial, write_json
from annotation import RULES, marker_model, state_model, annotate, validate_records
from render import render

ROOT = Path(__file__).resolve().parents[2]
CODE = ['profile_crispri.py', 'profile_responses.py', 'response.py', 'rna.py', 'annotation.py', 'render.py']
WORK = {}


def source_context(structure, context):
    report = json.loads((structure / 'report.json').read_text())
    if report['status'] != 'completed':
        raise ValueError('completed_structure_required')
    for artifact in report['artifacts']:
        if artifact['file'].startswith(context + '/') and hash_file(structure / artifact['file']) != artifact['sha256']:
            raise ValueError('changed_structure_sidecar')
    directory = structure / context
    cells = pd.read_parquet(directory / 'cells.parquet')
    mapping = pd.read_parquet(directory / 'gene_mapping.parquet')
    genes = pd.read_parquet(directory / 'genes.parquet')
    tasks = pd.read_parquet(directory / 'tasks.parquet')
    digest = cells.input_sha256.iloc[0]
    paths = [ROOT / p for p, h in report['input_sha256'].items() if h == digest]
    if len(paths) != 1 or hash_file(paths[0]) != digest:
        raise ValueError('source_identity_mismatch')
    validate_records(cells, digest, len(cells))
    if cells.gene_axis_sha256.nunique() != 1 or cells.gene_axis_sha256.iloc[0] != value_hash(genes.source_gene.tolist()):
        raise ValueError('source_gene_axis_mismatch')
    if mapping.source_gene_id.tolist() != genes.source_gene.tolist():
        raise ValueError('source_mapping_order_mismatch')
    return report, cells, mapping, genes, tasks, paths[0]


def start_output(output, identity, resume):
    if output.resolve().is_relative_to((ROOT/'data/raw').resolve()):
        raise ValueError('raw_output_forbidden')
    if output.exists():
        if not resume or json.loads((output/'identity.json').read_text()) != identity:
            raise ValueError('resume_identity_mismatch')
        if (output/'report.json').exists():
            raise ValueError('completed_results_immutable')
    else:
        output.mkdir(parents=True)
        write_json(output/'identity.json', identity)


def finish(output, report, start):
    report.update(schema_version=2, completed_at=datetime.now(timezone.utc).isoformat(), duration_seconds=time.monotonic()-start,
        code_commit=report.get('code_commit') or subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        code={n:hash_file(Path(__file__).with_name(n)) for n in CODE},
        runtime={'python':sys.version,'packages':{p:importlib.metadata.version(p) for p in ['numpy','scipy','pandas','h5py','pyarrow']},
                 'uv_lock_sha256':hash_file(Path(__file__).with_name('uv.lock'))}, reproduce=sys.argv)
    report['artifacts']=[{'file':str(p.relative_to(output)),'sha256':hash_file(p)} for p in sorted(output.rglob('*'))
                         if p.is_file() and 'cache' not in p.relative_to(output).parts and p.name not in ['report.json','report.html','SHA256SUMS','failure.json']]
    write_json(output/'report.json',report)
    (output/'report.html').write_text(render(serial(report)))
    (output/'SHA256SUMS').write_text(''.join(f'{hash_file(p)}  {p.relative_to(output)}\n' for p in sorted(output.rglob('*'))
        if p.is_file() and 'cache' not in p.relative_to(output).parts and p.name!='SHA256SUMS'))
    return report


def run_task(item):
    w=WORK
    task, ids = item
    directory=w['output']/('task-'+value_hash(task)[:20])
    directory.mkdir(exist_ok=True)
    record=w['metadata'].loc[task]
    if (directory/'result.json').exists():
        saved=json.loads((directory/'result.json').read_text())
        if saved['identity']!=w['identity']:
            raise ValueError('task_resume_identity_mismatch')
        for name,h in saved['hashes'].items():
            if hash_file(directory/name)!=h: raise ValueError('task_result_changed')
        return saved['summary']
    target_name=str(record.source_target_ensembl)
    seed=PARAMETERS['seed'] ^ int(value_hash([w['context'],task])[:8],16)
    summary,genes,samples=task_result(target_name,w['log'][ids],w['thin'][ids],w['cells'].iloc[ids],
        w['control'],w['control_batches'],w['batches'],w['controls'],w['thin_controls'],w['halves'],
        w['genes'],w['conflicts'],seed,control_pools=w['control_pools'])
    summary.update(task=task,context=w['context'],target=str(record.source_target_gene),
                   target_ensembl=target_name,source_transcript=str(record.source_transcript),
                   source_guide_id=str(record.source_guide_id),n_cells=len(ids))
    if genes is None: genes=pd.DataFrame({'source_gene':w['genes'],'status':summary['status']})
    genes.to_parquet(directory/'genes.parquet',index=False,compression='zstd')
    summary['gene_results']={'file':str((directory/'genes.parquet').relative_to(w['output'])),
                             'sha256':hash_file(directory/'genes.parquet')}
    write_json(directory/'resampling.json',samples)
    write_json(directory/'result.json',{'identity':w['identity'],'summary':summary,
        'hashes':{name:hash_file(directory/name) for name in ['genes.parquet','resampling.json']}})
    return summary


def construct_agreement(summaries, output):
    by_gene=defaultdict(list)
    for s in summaries: by_gene[s['target']].append(s)
    rows=[]
    for gene,group in sorted(by_gene.items()):
        usable=[s for s in group if s['status']=='completed' and s['n_cells']>=PARAMETERS['consistency_min_cells']]
        correlations=[]
        for i,a in enumerate(usable):
            aa=pd.read_parquet(output/a['gene_results']['file'],columns=['source_gene','is_target','effect_all_matched_cells'])
            for b in usable[:i]:
                bb=pd.read_parquet(output/b['gene_results']['file'],columns=['source_gene','is_target','effect_all_matched_cells'])
                if aa.source_gene.tolist()!=bb.source_gene.tolist():raise ValueError('construct_axis_mismatch')
                keep=~(aa.is_target | bb.is_target)
                value=correlation(aa.effect_all_matched_cells[keep],bb.effect_all_matched_cells[keep])
                if value is not None:correlations.append(value)
        rows.append({'target':gene,'constructs':len(group),'usable_constructs':len(usable),
            'status':'completed' if correlations else 'not_estimable',
            'reason':None if correlations else 'fewer_than_two_nonconstant_construct_effects_with_10_cells',
            'pairwise_correlation':quantiles(correlations), 'tasks':[s['task'] for s in group],
            'interpretation':'Distinct promoters/guide pairs may be different interventions; no independent replicate claim'})
    return rows


def responses(structure,context,output,resume=False,workers=4):
    t0=time.monotonic()
    run_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    source,cells,mapping,genes,tasks,raw=source_context(structure,context)
    identity={'structure_sha256':hash_file(structure/'report.json'),'context':context,'parameters':PARAMETERS,
              'code':{n:hash_file(Path(__file__).with_name(n)) for n in CODE},
              'uv_lock_sha256':hash_file(Path(__file__).with_name('uv.lock'))}
    start_output(output,identity,resume)
    before=(raw.stat().st_size,raw.stat().st_mtime_ns,raw.stat().st_ctime_ns)
    log,thin=build_cache(raw,output/'cache',cells.input_sha256.iloc[0],PARAMETERS['seed'] ^ int(value_hash(context)[:8],16))
    ids=np.flatnonzero(cells.source_target_gene=='non-targeting')
    if not len(ids):raise ValueError('registered_source_lacks_non_targeting_control')
    batches=sorted(cells.source_batch.astype(str).unique())
    control,control_thin=log[ids],thin[ids]
    control_batches=cells.source_batch.iloc[ids].astype(str).to_numpy()
    controls=grouped_moments(control,control_batches,batches)
    thin_controls=grouped_moments(control_thin,control_batches,batches)
    halves=control_half_means(control,control_batches,batches,PARAMETERS['resampling_repetitions'],PARAMETERS['seed']+10)
    np.savez_compressed(output/'baseline.npz',gene_ids=genes.source_gene.to_numpy(dtype=str),
        mean_logCP10K=control.mean(axis=0,dtype=np.float64),var_logCP10K=control.var(axis=0,ddof=1,dtype=np.float64),
        detection=(control>0).mean(axis=0),n_NTC=len(control))
    overall_baseline=control.mean(axis=0,dtype=np.float64)
    np.savez_compressed(output/'baseline_by_GEM.npz',gene_ids=genes.source_gene.to_numpy(dtype=str),
        GEM_groups=np.asarray(batches),mean_logCP10K=np.stack([c['mean'] for c in controls]),
        n_NTC=np.asarray([c['n'] for c in controls]))
    baseline_rows=[{'source_batch':b,'n_NTC':c['n'],
        'RMS_from_pooled_NTC':float(np.sqrt(np.mean((c['mean']-overall_baseline)**2))) if c['n'] else None,
        'correlation_to_pooled_NTC':correlation(c['mean'],overall_baseline) if c['n'] else None,
        'interpretation':'Baseline variation across GEMs; biological and technical contributions are not identified'}
        for b,c in zip(batches,controls)]
    write_json(output/'baseline_by_GEM.json',baseline_rows)
    metadata=tasks.set_index('task')
    task_ids=[(name,group.index.to_numpy()) for name,group in cells.groupby('source_task',sort=True)
              if not metadata.loc[name,'is_control']]
    conflicts=set(mapping.loc[(mapping.symbol_vs_ensembl=='conflict') | mapping.many_to_one_mapping | mapping.mapped_symbol.isna(),'source_gene_id'])
    WORK.update(output=output,identity=identity,metadata=metadata,context=context,log=log,thin=thin,cells=cells,
                control=control,control_batches=control_batches,batches=batches,controls=controls,thin_controls=thin_controls,
                halves=halves,genes=genes.source_gene.tolist(),conflicts=conflicts,
                control_pools={b:np.flatnonzero(control_batches==b) for b in batches})
    summaries=[]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(run_task,item) for item in task_ids]):
            result=future.result()
            summaries.append(result)
            if len(summaries)%50==0:print(f'{context}: {len(summaries)}/{len(task_ids)} complete response tasks',flush=True)
    summaries.sort(key=lambda x:x['task'])
    write_json(output/'tasks.json',summaries)
    agreements=construct_agreement(summaries,output)
    write_json(output/'construct_agreement.json',agreements)
    resamples=[]
    for s in summaries:
        directory=output/Path(s['gene_results']['file']).parent
        resamples.extend({'task':s['task'],'target':s['target'],**r} for r in json.loads((directory/'resampling.json').read_text()))
    pd.DataFrame(resamples).to_parquet(output/'resampling.parquet',index=False,compression='zstd')
    changed=before!=(raw.stat().st_size,raw.stat().st_mtime_ns,raw.stat().st_ctime_ns)
    compact=[{'task':s['task'],'target':s['target'],'n_cells':s['n_cells'],'status':s['status'],
              'RMS':s.get('downstream_RMS'),'DE_status':s.get('DE',{}).get('status'),
              'DEG_BH':s.get('DEG_BH_excluding_target'),'target_RNA':s.get('target_RNA'),
              'stability':s.get('stability'),'depth_sensitivity':s.get('depth_sensitivity')} for s in summaries]
    report={'bundle_id':context+'-response-'+uuid.uuid4().hex,'title':context+' 全构件扰动响应',
        'status':'completed' if len(summaries)==len(task_ids) and not changed else 'failed',
        'identity':identity,'code_commit':run_commit,'input_sha256':{str(raw.relative_to(ROOT)):cells.input_sha256.iloc[0]},'inputs_unchanged':not changed,
        'context':context,'expected_tasks':len(task_ids),'actual_tasks':len(summaries),'n_cells':len(cells),'n_NTC_cells':len(ids),
        'task_status_counts':dict(Counter(s['status'] for s in summaries)),
        'DE_status_counts':dict(Counter(s.get('DE',{}).get('status','not_estimable') for s in summaries)),
        'methods':{'protocol':'https://github.com/yjcyxky/virtual-cell-challenge/issues/7#issuecomment-5744629187',
            'parameters':PARAMETERS,'worker_threads':workers,'control':'Same experiment and GEM; target-GEM-composition weighted NTC',
            'task':'Source gene_transcript; native Ensembl axis; guide pair is one construct',
            'DE':'Conditional cell-sampling Welch-Satterthwaite; task-wise BH and BY over detected native genes',
            'exposure':'All source endpoint counts and all construct responses explored; not untouched validation'},
        'limitations':['无已确认独立培养重复；细胞抽样区间与半分稳定性不能解释为跨实验复现。',
            'NTC 变异包括生理、培养、捕获和测量差异，不能整体称为纯技术噪声。',
            '启动子／转录本构件分别评估；同基因构件的相关性描述干预差异而非重复实验。',
            'target RNA 比率不是蛋白活性或每细胞干预剂量真值。未测量基因不是零值。'],
        'tables':[{'title':'全部任务','rows':compact}, {'title':'同基因构件间一致性','rows':agreements},
                  {'title':'NTC 基线 GEM 差异','rows':baseline_rows}]}
    WORK.clear()
    return finish(output,report,t0)


def annotate_context(structure,context,response,references,output,resume=False,chunk=1024):
    t0=time.monotonic()
    run_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    source,cells,mapping,genes,tasks,raw=source_context(structure,context)
    parent=json.loads((response/'report.json').read_text())
    if parent['status']!='completed' or parent['context']!=context:raise ValueError('completed_same_context_response_required')
    for line in (references/'SHA256SUMS').read_text().splitlines():
        h,name=line.split('  ',1)
        if hash_file(references/name)!=h:raise ValueError('reference_hash_mismatch')
    reference=json.loads((references/'gene_sets.json').read_text())
    identity={'structure_sha256':hash_file(structure/'report.json'),'response_sha256':hash_file(response/'report.json'),
        'reference_sha256':hash_file(references/'gene_sets.json'),'context':context,'rules':RULES,
        'code':{n:hash_file(Path(__file__).with_name(n)) for n in CODE}}
    start_output(output,identity,resume)
    if not (output/'references').exists():shutil.copytree(references,output/'references')
    cache_identity=json.loads((response/'cache/identity.json').read_text())
    if cache_identity['identity']['input_sha256']!=cells.input_sha256.iloc[0] or hash_file(response/'cache/logcp.npy')!=cache_identity['hashes']['logcp.npy']:
        raise ValueError('normalized_cache_identity_mismatch')
    log=np.load(response/'cache/logcp.npy',mmap_mode='r')
    if log.shape!=(len(cells),len(genes)):raise ValueError('normalized_cache_shape_mismatch')
    safe=mapping.mapped_symbol.where(~mapping.many_to_one_mapping & (mapping.symbol_vs_ensembl!='conflict'))
    resolved=[str(x) if pd.notna(x) else None for x in safe]
    targets=dict(zip(mapping.source_gene_id,resolved))
    model=marker_model(resolved,reference['profiles'])
    ids=np.flatnonzero(cells.source_target_gene=='non-targeting')
    baseline=np.zeros(len(genes))
    for a in range(0,len(ids),chunk):baseline+=log[ids[a:a+chunk]].sum(axis=0,dtype=np.float64)
    baseline/=len(ids)
    weights,coverage=state_model(resolved,reference['states'],baseline)
    write_json(output/'type_coverage.json',model['coverage']);write_json(output/'state_coverage.json',coverage)
    np.savez_compressed(output/'scoring_weights.npz',type_weights=model['weights'],state_weights=weights,baseline_mean=baseline,valid_gene_axis=model['valid_axis'])
    parts=[]
    for a in range(0,len(log),chunk):
        b=min(a+chunk,len(log));block=np.asarray(log[a:b])
        target_symbols=[targets.get(str(x)) for x in cells.source_gene_id.iloc[a:b]]
        labels,scores=annotate(block,target_symbols,model)
        for name,values in scores.items():labels[name]=values
        states=block@weights
        for j,(name,details) in enumerate(zip(reference['states'],coverage)):
            labels['state__'+name]=states[:,j] if details['status']=='completed' else np.nan
        parts.append(labels)
        if a%(chunk*100)==0:print(f'{context}: annotation {b}/{len(log)}',flush=True)
    combined=pd.concat([cells,pd.concat(parts,ignore_index=True)],axis=1)
    combined['inference_method']='broad-marker-rank-and-expression-v1'
    combined['inference_reference_match']='uncalibrated normal-human broad markers applied to transformed/immortalized cell line'
    expected=next(row.get('expected_lineages',[]) for row in source['tables'][0]['rows'] if row['context']==context)
    combined['source_context_mismatch']=~combined.inferred_lineage.isin(expected+['unknown'])
    combined['inference_exposure']=np.where(combined.source_target_gene=='non-targeting','NTC_baseline_RNA','perturbed_endpoint_RNA')
    combined['future_prediction_availability']=np.where(combined.source_target_gene=='non-targeting','only_if_matched_baseline_measured','unavailable_before_target_endpoint_measurement')
    combined.to_parquet(output/'cells.parquet',index=False,compression='zstd')
    composition,states,within,task_summaries=state_composition(combined,list(reference['states']))
    for name,frame in [('composition',composition),('states',states),('within_type_states',within)]:frame.to_parquet(output/(name+'.parquet'),index=False,compression='zstd')
    write_json(output/'tasks.json',task_summaries)
    # Reader opened every original file in read-only mode; rehash closes the identity chain.
    unchanged=hash_file(raw)==cells.input_sha256.iloc[0]
    compact_states=states[['task','target','state','difference','q10','median','q90','status']]
    report={'bundle_id':context+'-annotation-'+uuid.uuid4().hex,'title':context+' 逐细胞推断与组成状态分布',
        'status':'completed' if unchanged else 'failed','inputs_unchanged':unchanged,'identity':identity,'context':context,'code_commit':run_commit,
        'record_count':len(combined),'uncalibrated_records':int((combined.confidence_calibration=='uncalibrated').sum()),
        'pooled_types':combined.inferred_type.value_counts().to_dict(),'pooled_lineages':combined.inferred_lineage.value_counts().to_dict(),
        'method_conflicts':int(combined.method_conflict.sum()),'target_marker_label_changes':int(combined.target_marker_label_changed.sum()),
        'source_context_mismatches':int(combined.source_context_mismatch.sum()),
        'methods':{'protocol':'https://github.com/yjcyxky/virtual-cell-challenge/issues/7#issuecomment-5744629187','rules':RULES,
            'type':'Fixed human broad marker hypotheses, correlated rank/expression scores, conservative agreement and lineage fallback; no truth labels',
            'state':'Fixed within-context NTC mean bins; 12 expression modules; separate type and state',
            'within_type':'Endpoint inferred-type strata with >=10 target cells total across shared GEMs, each with >=2 same-type NTC; descriptive state differences only; conditioning on an outcome can bias causal interpretation',
            'exposure':'All measured endpoint transcriptomes; perturbed-state proxies unavailable before endpoint measurement'},
        'limitations':['所有类型及状态为未校准 RNA 推断，不能作 ground truth。宽 marker 参考不能充分表示肿瘤或永生化细胞；亚型 unknown。',
            '状态得分是表达代理，不是实际周期阶段、通路活性或分化轨迹；类型变化不能区分选择、死亡、增殖、捕获与转换。',
            '同推断类型内差异仍可能受扰动改变 marker、终点分层偏差、测量及未测因素影响。',
            '各实验独立固定背景基因；跨实验状态分数不能直接当作同一标尺的因果效应。'],
        'tables':[{'title':'全部构件的类型与不确定性','rows':task_summaries},
            {'title':'类型参考覆盖','rows':[{k:v for k,v in c.items() if k not in ['measured_symbols','unmeasured_or_ambiguous']} for c in model['coverage']]},
            {'title':'状态参考覆盖','rows':[{k:v for k,v in c.items() if k!='background_source_rows'} for c in coverage]},
            {'title':'全部任务状态分布','rows':json.loads(compact_states.to_json(orient='records'))}]}
    return finish(output,report,t0)


def state_composition(frame,state_names):
    columns=['state__'+s for s in state_names]
    control=frame[frame.source_target_gene=='non-targeting']
    composition_base=control.groupby(['source_batch','inferred_type'],observed=True).size().unstack(fill_value=0)
    composition_base=composition_base.div(composition_base.sum(axis=1),axis=0)
    state_base=control.groupby('source_batch',observed=True)[columns].mean()
    type_counts=control.groupby(['source_batch','inferred_type'],observed=True).size()
    type_states=control.groupby(['source_batch','inferred_type'],observed=True)[columns].mean()
    composition,states,within,tasks=[],[],[],[]
    for task,group in frame.groupby('source_task',observed=True,sort=True):
        target=str(group.source_target_gene.iloc[0]);batch_n=group.source_batch.value_counts()
        shared=batch_n.index.intersection(state_base.index)
        total=int(batch_n.loc[shared].sum());weights=batch_n.loc[shared]/total if total else batch_n.loc[shared].astype(float)
        expected_comp=composition_base.loc[shared].mul(weights,axis=0).sum() if total else pd.Series(dtype=float)
        proportions=group.inferred_type.value_counts(normalize=True)
        for label in sorted(set(proportions.index)|set(expected_comp.index)):
            composition.append({'task':task,'target':target,'inferred_type':label,'n_cells':len(group),
                'fraction':float(proportions.get(label,0)),'matched_NTC_fraction':float(expected_comp.get(label,0)) if total else None,
                'difference':float(proportions.get(label,0)-expected_comp.get(label,0)) if total else None,
                'status':'completed' if total else 'not_estimable','reason':None if total else 'no_same_GEM_NTC'})
        matched_group=group[group.source_batch.isin(shared)]
        expected=state_base.loc[shared].mul(weights,axis=0).sum(min_count=1) if total else pd.Series(np.nan,index=columns)
        quant=group[columns].quantile([.1,.5,.9])
        mean=matched_group[columns].mean()
        for name in columns:
            values=group[name].to_numpy(dtype=float);depth=group.computed_total_counts.to_numpy(dtype=float)
            rho=float(spearmanr(values,depth).statistic) if np.isfinite(values).all() and np.ptp(values)>0 and np.ptp(depth)>0 else None
            states.append({'task':task,'target':target,'state':name,'mean':float(mean[name]),'matched_NTC_mean':float(expected[name]),
                'difference':float(mean[name]-expected[name]),'q10':float(quant.loc[.1,name]),'median':float(quant.loc[.5,name]),'q90':float(quant.loc[.9,name]),
                'spearman_with_library_size':rho,'target_cells_matched':total,'target_cells_unmatched':len(group)-total,
                'status':'completed' if np.isfinite(mean[name]) and np.isfinite(expected[name]) else 'not_estimable',
                'reason':None if np.isfinite(mean[name]) and np.isfinite(expected[name]) else 'missing_state_reference_or_matched_control'})
        for label,sub in group.groupby('inferred_type',observed=True):
            counts=sub.source_batch.value_counts()
            eligible=[b for b in counts.index if type_counts.get((b,label),0)>=2]
            n=int(counts.loc[eligible].sum()) if eligible else 0
            if n>=10:
                w=counts.loc[eligible]/n
                baseline=type_states.loc[[(b,label) for b in eligible]].copy();baseline.index=eligible
                base=baseline.mul(w,axis=0).sum(min_count=1);observed=sub[sub.source_batch.isin(eligible)][columns].mean()
            for name in columns:
                complete=n>=10 and pd.notna(base[name]) and pd.notna(observed[name])
                within.append({'task':task,'target':target,'inferred_type':label,'state':name,'target_cells_used':n,
                    'shared_GEM_groups':len(eligible),'difference':float(observed[name]-base[name]) if complete else None,
                    'status':'completed' if complete else 'not_estimable',
                    'reason':None if complete else 'fewer_than_10_target_cells_in_type_GEM_strata_with_2_NTC_or_unavailable_state',
                    'causal_interpretation':False,'truth_label':False})
        tasks.append({'task':task,'target':target,'n_cells':len(group),'type_counts':group.inferred_type.value_counts().to_dict(),
            'lineage_counts':group.inferred_lineage.value_counts().to_dict(),'unknown_type_fraction':float((group.inferred_type=='unknown').mean()),
            'method_conflict_fraction':float(group.method_conflict.mean()),'mixed_marker_fraction':float(group.mixed_marker_signal.mean()),
            'target_marker_label_changed':int(group.target_marker_label_changed.sum()),'source_context_mismatch_count':int(group.source_context_mismatch.sum()),
            'confidence_calibration':'uncalibrated','probability_correct':None,'status':'completed'})
    return pd.DataFrame(composition),pd.DataFrame(states),pd.DataFrame(within),tasks


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('part',choices=['response','annotation'])
    for name in ['structure','output']:parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--context',required=True);parser.add_argument('--resume',action='store_true')
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--response',type=Path);parser.add_argument('--references',type=Path)
    args=parser.parse_args()
    try:
        if args.part=='response':result=responses(args.structure,args.context,args.output,args.resume,args.workers)
        else:
            if args.response is None or args.references is None:parser.error('annotation requires --response and --references')
            result=annotate_context(args.structure,args.context,args.response,args.references,args.output,args.resume)
    except Exception as exc:
        if args.output.exists() and not (args.output/'report.json').exists():
            write_json(args.output/'failure.json',{'status':'failed','error':f'{type(exc).__name__}: {exc}'})
        raise
    print(json.dumps({'status':result['status'],'bundle_id':result['bundle_id']}))
