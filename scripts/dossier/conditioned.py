"""Full source-condition assessment with bounded temporary normalized views.

The input is an explicit row selection plus identity sidecars. No row is dropped
for sample-size thresholds; applicability is reported per diagnostic.
"""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor,as_completed
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
from rna import RNAFile,hash_file,value_hash,quantiles
from profile_responses import PARAMETERS,write_json,serial,task_result
from response import grouped_moments,control_half_means,correlation,matched_effect
from annotation import RULES,marker_model,state_model,annotate
from profile_crispri import state_composition
from render import render

CODE=['conditioned.py','profile_responses.py','response.py','annotation.py','rna.py','profile_crispri.py','render.py']


def normalize_selection(path,rows,directory,seed,chunk=512):
    rows=np.asarray(rows,dtype=np.int64)
    if not len(rows) or (np.diff(rows)<=0).any():raise ValueError('row_selection_must_be_unique_sorted_nonempty')
    directory.mkdir()
    rng=np.random.default_rng(seed)
    with RNAFile(path) as source:
        if rows[0]<0 or rows[-1]>=source.shape[0]:raise ValueError('row_selection_out_of_bounds')
        selection=np.full(source.shape[0],-1,np.int64);selection[rows]=np.arange(len(rows))
        log=np.lib.format.open_memmap(directory/'log.npy',mode='w+',dtype='float32',shape=(len(rows),source.shape[1]))
        thin=np.lib.format.open_memmap(directory/'thin.npy',mode='w+',dtype='float32',shape=log.shape)
        for start,matrix in source.blocks(chunk):
            local=selection[start:start+matrix.shape[0]];keep=local>=0
            if not keep.any():continue
            matrix=matrix[keep];dest=local[keep]
            if not np.isfinite(matrix.data).all() or (matrix.data<0).any() or (matrix.data!=np.floor(matrix.data)).any():raise ValueError('count_view_requires_valid_counts')
            sums=np.asarray(matrix.sum(axis=1)).ravel();scale=np.divide(10000,sums,out=np.zeros_like(sums),where=sums>0)
            thinned=matrix.copy();thinned.data=rng.binomial(matrix.data.astype(np.int64),np.repeat(np.minimum(1,scale),np.diff(matrix.indptr))).astype(float)
            thinned.eliminate_zeros();total=np.asarray(thinned.sum(axis=1)).ravel();tscale=np.divide(10000,total,out=np.zeros_like(total),where=total>0)
            thinned.data=np.log1p(thinned.data*np.repeat(tscale,np.diff(thinned.indptr)))
            matrix.data=np.log1p(matrix.data*np.repeat(scale,np.diff(matrix.indptr)))
            log[dest]=matrix.toarray();thin[dest]=thinned.toarray()
        log.flush();thin.flush();del log,thin
    identity={'selected_rows_sha256':value_hash(rows.tolist()),'seed':seed,'log_sha256':hash_file(directory/'log.npy'),'thin_sha256':hash_file(directory/'thin.npy')}
    return np.load(directory/'log.npy',mmap_mode='r'),np.load(directory/'thin.npy',mmap_mode='r'),identity


def response_task(name,ids,log,thin,cells,control,cbatches,batches,controls,thin_controls,halves,genes,conflicts,pools,context,output,target_native):
    seed=PARAMETERS['seed']^int(value_hash([context,name])[:8],16)
    native=target_native.get(name,name)
    result,features,samples=task_result(native,log[ids],thin[ids],cells.iloc[ids],control,cbatches,batches,
        controls,thin_controls,halves,genes,conflicts,seed,control_pools=pools)
    result.update(task=name,target=name,source_target_feature=native,context=context,n_cells=len(ids))
    directory=output/('task-'+value_hash(name)[:20]);directory.mkdir()
    if features is None:features=pd.DataFrame({'source_gene':genes,'status':result['status']})
    features.to_parquet(directory/'genes.parquet',index=False,compression='zstd')
    result['gene_results']={'file':str((directory/'genes.parquet').relative_to(output)),'sha256':hash_file(directory/'genes.parquet')}
    write_json(directory/'result.json',result);write_json(directory/'resampling.json',samples)
    return result,[{'task':name,**x} for x in samples]


def assess_context(path,cells,genes,mapping,references,output,context,expected_lineages,protocol,source_identity,workers=2,retain_cache=False):
    if output.exists():raise ValueError('fresh_context_output_required')
    started=time.monotonic();commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip();output.mkdir(parents=True)
    for line in (references/'SHA256SUMS').read_text().splitlines():
        digest,name=line.split('  ',1)
        if hash_file(references/name)!=digest:raise ValueError('changed_annotation_reference')
    if cells.row_index.duplicated().any() or not cells.row_index.is_monotonic_increasing:raise ValueError('unordered_or_duplicate_source_row')
    if cells.source_context.nunique()!=1 or cells.source_context.iloc[0]!=context:raise ValueError('mixed_condition_context')
    if len(genes)!=len(mapping) or genes.source_gene.tolist()!=mapping.source_gene.tolist():raise ValueError('gene_mapping_axis_mismatch')
    rawstat=path.stat();cells=cells.copy().reset_index(drop=True)
    log,thin,cache_id=normalize_selection(path,cells.row_index.to_numpy(),output/'temporary-view',PARAMETERS['seed']^int(value_hash(context)[:8],16))
    if log.shape!=(len(cells),len(genes)):raise ValueError('context_view_shape_mismatch')
    ntc=np.flatnonzero(cells.source_target_gene=='non-targeting');batches=sorted(cells.source_batch.astype(str).unique());cbatches=cells.source_batch.iloc[ntc].astype(str).to_numpy()
    control,control_thin=log[ntc],thin[ntc];controls=grouped_moments(control,cbatches,batches);thin_controls=grouped_moments(control_thin,cbatches,batches)
    halves=control_half_means(control,cbatches,batches,PARAMETERS['resampling_repetitions'],PARAMETERS['seed']+10)
    pools={b:np.flatnonzero(cbatches==b) for b in batches};safe=mapping.mapped_symbol.where(~mapping.many_to_one_mapping&(mapping.symbol_vs_ensembl!='conflict'))
    resolved=[str(x) if pd.notna(x) else None for x in safe];conflicts=set(mapping.loc[safe.isna(),'source_gene'])
    target_native={symbol:native for symbol,native in zip(resolved,genes.source_gene) if symbol is not None}
    conflicts.update(name for name,group in mapping.dropna(subset=['mapped_symbol']).groupby('mapped_symbol') if len(group)>1)
    tasks=[(name,group.index.to_numpy()) for name,group in cells.groupby('source_task',observed=True,sort=True) if group.source_target_gene.iloc[0]!='non-targeting']
    summaries=[];samples=[]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(response_task,name,ids,log,thin,cells,control,cbatches,batches,controls,thin_controls,halves,
            genes.source_gene.tolist(),conflicts,pools,context,output,target_native) for name,ids in tasks]
        for future in as_completed(futures):
            result,draws=future.result();summaries.append(result);samples.extend(draws)
    summaries.sort(key=lambda x:x['task']);write_json(output/'tasks.json',summaries)
    pd.DataFrame(samples,columns=['task','replicate','half_1_cells','half_2_cells','half_effect_correlation','half_1_rms','half_2_rms','null_arm_cells','null_target_cells_not_matched','null_rms','null_reference_intersection','interpretation']).to_parquet(output/'resampling.parquet',index=False)
    baseline=control.mean(axis=0,dtype=np.float64) if len(ntc) else np.asarray(log).mean(axis=0,dtype=np.float64)
    np.savez_compressed(output/'baseline.npz',gene_ids=genes.source_gene.to_numpy(dtype=str),mean_logCP10K=baseline,
        n_NTC=len(ntc),baseline_type='matched_context_NTC' if len(ntc) else 'pooled_endpoint_fallback_no_controls')
    np.savez_compressed(output/'baseline_by_stratum.npz',gene_ids=genes.source_gene.to_numpy(dtype=str),strata=np.asarray(batches),mean_logCP10K=np.stack([c['mean'] for c in controls]),n_NTC=np.asarray([c['n'] for c in controls]))
    reference=json.loads((references/'gene_sets.json').read_text());model=marker_model(resolved,reference['profiles']);weights,state_coverage=state_model(resolved,reference['states'],baseline)
    parts=[]
    for a in range(0,len(log),512):
        b=min(a+512,len(log));block=np.asarray(log[a:b]);labels,scores=annotate(block,cells.source_target_gene.iloc[a:b].tolist(),model)
        for name,values in scores.items():labels[name]=values
        state=block@weights
        for i,(name,cov) in enumerate(zip(reference['states'],state_coverage)):labels['state__'+name]=state[:,i] if cov['status']=='completed' else np.nan
        labels['source_context_mismatch']=~labels.inferred_lineage.isin([*expected_lineages,'unknown'])
        parts.append(labels)
    combined=pd.concat([cells,pd.concat(parts,ignore_index=True)],axis=1)
    combined['state_inference_method']='fixed_NTC_expression_matched_background' if len(ntc) else 'pooled_endpoint_background_due_to_absent_controls'
    combined['available_before_endpoint']=False;combined['truth_label']=False
    combined.to_parquet(output/'cells.parquet',index=False,compression='zstd')
    composition,states,within,type_tasks=state_composition(combined,list(reference['states']))
    within=within.rename(columns={'shared_GEM_groups':'shared_technical_strata'})
    for frame in [composition,states,within]:
        if 'reason' in frame:frame['reason']=frame.reason.str.replace('GEM','technical_stratum',regex=False)
    composition.to_parquet(output/'composition.parquet',index=False);states.to_parquet(output/'states.parquet',index=False);within.to_parquet(output/'within-type-states.parquet',index=False)
    write_json(output/'type-tasks.json',type_tasks);write_json(output/'type-coverage.json',model['coverage']);write_json(output/'state-coverage.json',state_coverage)
    np.savez_compressed(output/'scoring-weights.npz',type_weights=model['weights'],state_weights=weights,baseline_mean=baseline,valid_gene_axis=model['valid_axis'])
    mapping.to_parquet(output/'gene-mapping.parquet',index=False)
    # Source label inconsistencies are retained; the diagnostic restricts only its own analysis arm.
    sensitivity=[]
    for task,group in combined.groupby('source_task',observed=True):
        if group.source_target_gene.iloc[0]=='non-targeting':continue
        allids=group.index.to_numpy();consistent=allids[~group.source_label_inconsistent.to_numpy()]
        affected_control=ntc[combined.source_label_inconsistent.iloc[ntc].to_numpy()]
        row={'task':task,'source_inconsistent_target_cells':int(group.source_label_inconsistent.sum()),'source_inconsistent_NTC_cells':len(affected_control)}
        if not row['source_inconsistent_target_cells'] and not len(affected_control):row.update(status='not_applicable',reason='no_inconsistent_labels_in_task_or_control')
        elif not len(consistent):row.update(status='not_estimable',reason='no_consistent_source_label_target_cells')
        else:
            valid_ntc=ntc[~combined.source_label_inconsistent.iloc[ntc].to_numpy()]
            c=grouped_moments(log[valid_ntc],combined.source_batch.iloc[valid_ntc].astype(str).to_numpy(),batches)
            effect,meta=matched_effect(grouped_moments(log[consistent],combined.source_batch.iloc[consistent].astype(str).to_numpy(),batches),c)
            original=next(s for s in summaries if s['task']==task)
            if effect is None or original['status']!='completed':row.update(status='not_estimable',reason='no_defensible_consistent_label_control')
            else:
                original_effect=pd.read_parquet(output/original['gene_results']['file']).effect_all_matched_cells.to_numpy();mask=genes.source_gene.to_numpy()!=target_native.get(task,task)
                row.update(status='completed',RMS=float(np.sqrt(np.mean(effect[mask]**2))),correlation_with_all_source_labels=correlation(effect[mask],original_effect[mask]),
                    interpretation='Source-label-consistent subset sensitivity only; original records retained and not asserted incorrect')
        sensitivity.append(row)
    write_json(output/'label-sensitivity.json',sensitivity)
    compact=[{'task':s['task'],'target':s['target'],'n_cells':s['n_cells'],'status':s['status'],'DE':s.get('DE'),
        'RMS':s.get('downstream_RMS'),'DEG_BH':s.get('DEG_BH_excluding_target'),'target_RNA':s.get('target_RNA'),'stability':s.get('stability'),'guide_and_stratum_consistency':s.get('consistency')} for s in summaries]
    report={'schema_version':2,'bundle_id':'condition-'+uuid.uuid4().hex,'title':context+' 全量条件评估','status':'completed',
        'context':context,'n_cells':len(cells),'n_NTC':len(ntc),'actual_tasks':len(tasks),'task_status_counts':dict(Counter(s['status'] for s in summaries)),
        'DE_status_counts':dict(Counter(s.get('DE',{}).get('status','not_estimable') for s in summaries)),
        'pooled_types':combined.inferred_type.value_counts().to_dict(),'uncalibrated_records':len(combined),
        'source_label_inconsistent_records':int(combined.source_label_inconsistent.sum()),'source_identity':source_identity,'analysis_view':cache_id,
        'inputs_unchanged':(rawstat.st_size,rawstat.st_mtime_ns,rawstat.st_ctime_ns)==(path.stat().st_size,path.stat().st_mtime_ns,path.stat().st_ctime_ns),
        'code_commit':commit,'code':{n:hash_file(Path(__file__).with_name(n)) for n in CODE},
        'runtime':{'python':sys.version,'numpy':np.__version__,'pandas':pd.__version__,'uv_lock_sha256':hash_file(Path(__file__).with_name('uv.lock'))},
        'reference_sha256':hash_file(references/'gene_sets.json'),'parameters':PARAMETERS,'annotation_rules':RULES,
        'methods':{'protocol':protocol,'control':'Same declared condition and exact registered technical stratum; target-stratum-composition weighted NTC',
            'response':'All source target genes; independent guide identities retained for guide consistency; no independent culture replication assumed',
            'annotation':'Frozen human marker agreement + conservative unknown; 12 RNA state proxies using context NTC background',
            'within_type':'Descriptive endpoint strata, >=10 targets across strata each with >=2 same inferred-type NTC; no causal mediation claim',
            'exposure':'All endpoint counts, responses and source-derived scores explored; endpoint inference unavailable as pre-intervention input',
            'cache':'Exact row selection recorded; normalized float32 and seeded expected-depth thinning temporary arrays hashed; derived view may be deleted after complete export'},
        'limitations':['未确证独立培养重复；DE 和重采样描述条件细胞抽样，不作生物重复显著性解释。',
            '源标签不一致保留并单列敏感性；按作者标签分层不表示恢复了物理来源真值。',
            'NTC 仍处在对应刺激条件；扰动响应不是刺激相对于未刺激的效应。',
            '类型和状态全部未校准，probability_correct 为空，不是 ground truth。',
            'NTC 大小短缺逐重采样记录，不能把不等臂零效应参照用作精确显著性检验。'],
        'tables':[{'title':'全部响应任务','rows':compact},{'title':'类型组成与不确定性','rows':type_tasks},{'title':'源标签一致性敏感性','rows':sensitivity}],
        'completed_at':datetime.now(timezone.utc).isoformat(),'duration_seconds':time.monotonic()-started}
    if not report['inputs_unchanged']:raise ValueError('count_cache_changed_during_assessment')
    del log,thin,control,control_thin
    if not retain_cache:shutil.rmtree(output/'temporary-view')
    report['artifacts']=[{'file':str(p.relative_to(output)),'sha256':hash_file(p)} for p in sorted(output.rglob('*')) if p.is_file() and 'temporary-view' not in p.parts]
    write_json(output/'report.json',report);(output/'report.html').write_text(render(serial(report)))
    (output/'SHA256SUMS').write_text(''.join(f'{hash_file(p)}  {p.relative_to(output)}\n' for p in sorted(output.rglob('*')) if p.is_file() and p.name!='SHA256SUMS' and 'temporary-view' not in p.parts))
    return report
