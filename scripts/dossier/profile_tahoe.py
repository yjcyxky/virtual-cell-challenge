#!/usr/bin/env python
"""Complete all local Tahoe chemical-condition diagnostics and evidence dossier."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,as_completed
from datetime import datetime,timezone
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
import h5py
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from chemical_diagnostics import PARAMETERS,distribution_contrast
from rna import hash_file,value_hash,quantiles
from profile_responses import write_json,serial
from render import render
ROOT=Path(__file__).resolve().parents[2]
CODE=['profile_tahoe.py','chemical_diagnostics.py','rna.py','response.py','profile_responses.py','render.py']
PROTOCOL='https://github.com/yjcyxky/virtual-cell-challenge/issues/17#issuecomment-5745359164'


def prepare_vectors(annotations,output):
    report=json.loads((annotations/'report.json').read_text());n=report['actual_cells']
    first=pq.ParquetFile(annotations/'shards/000/cells.parquet').schema_arrow.names
    names=[c for c in first if c.startswith('state__')]+['log1p_total_counts','log1p_detected_genes'];states=names[:-2]
    types=sorted(report['pooled_types']);lookup={t:i for i,t in enumerate(types)}
    directory=output/'temporary-vectors';directory.mkdir();vectors=np.lib.format.open_memmap(directory/'vectors.npy',mode='w+',dtype='float32',shape=(n,len(names)))
    labels=np.lib.format.open_memmap(directory/'types.npy',mode='w+',dtype='int16',shape=(n,));condition=np.empty(n,np.int32)
    hashes=[];barcodes=[];records=[];shard_ids=[];rows=[];position=0
    columns=['record_id','row_index','BARCODE_SUB_LIB_ID','computed_count_sha256','condition_index','inferred_type','computed_total_counts','computed_detected_genes','probability_correct','truth_label',*states]
    for folder in sorted((annotations/'shards').iterdir()):
        frame=pd.read_parquet(folder/'cells.parquet',columns=columns);stop=position+len(frame)
        if not np.array_equal(frame.row_index,np.arange(len(frame))) or frame.record_id.duplicated().any() or frame.probability_correct.notna().any() or frame.truth_label.any():raise ValueError('invalid_source_annotation_identity_or_truth_status')
        vectors[position:stop,:-2]=frame[states].to_numpy(dtype=np.float32);vectors[position:stop,-2]=np.log1p(frame.computed_total_counts);vectors[position:stop,-1]=np.log1p(frame.computed_detected_genes)
        labels[position:stop]=frame.inferred_type.map(lookup).to_numpy();condition[position:stop]=frame.condition_index
        hashes.extend(frame.computed_count_sha256);barcodes.extend(frame.BARCODE_SUB_LIB_ID);records.extend(frame.record_id);shard_ids.extend([int(folder.name)]*len(frame));rows.extend(frame.row_index);position=stop
    if position!=n:raise ValueError('incomplete_annotation_coverage')
    identity=pd.DataFrame({'record_id':records,'source_barcode':barcodes,'count_sha256':hashes,'shard_index':shard_ids,'source_row_index':rows})
    duplicate_id=identity.record_id.duplicated(keep=False);duplicate_barcode=identity.source_barcode.duplicated(keep=False);duplicate_counts=identity.count_sha256.duplicated(keep=False)
    identity.loc[duplicate_id|duplicate_barcode|duplicate_counts].to_parquet(output/'cross-shard-duplicate-candidates.parquet',index=False)
    audit={'records':n,'duplicate_record_id_rows':int(duplicate_id.sum()),'duplicate_barcode_rows':int(duplicate_barcode.sum()),
        'identical_count_candidate_rows':int(duplicate_counts.sum()),'identical_count_candidate_groups':identity.loc[duplicate_counts,'count_sha256'].nunique(),
        'interpretation':'All source identities checked across shards; count equality alone does not prove the same physical cell; no records removed'}
    if duplicate_id.any() or duplicate_barcode.any():raise ValueError('unexpected_cross_shard_source_identity_reuse')
    write_json(output/'duplicate-audit.json',audit)
    order=np.argsort(condition,kind='stable');sizes=np.bincount(condition,minlength=int(condition.max())+1);np.save(directory/'order.npy',order);np.save(directory/'pointers.npy',np.r_[0,np.cumsum(sizes)])
    vectors.flush();labels.flush();del vectors,labels
    write_json(directory/'metadata.json',{'names':names,'types':types,'n_cells':n,'order':'source shard index, original row index; condition order is separate index only'})
    write_json(output/'analysis-view.json',{'metadata':json.loads((directory/'metadata.json').read_text()),'files':{p.name:hash_file(p) for p in directory.iterdir() if p.is_file()}})
    return audit


def analyze_background(background,counts_string,annotations_string,output_string,identity):
    counts,annotations,output=Path(counts_string),Path(annotations_string),Path(output_string);key=str(background['background_index']).zfill(3);destination=output/'conditions'/key
    if (destination/'identity.json').exists():
        old=json.loads((destination/'identity.json').read_text())
        if old['run_identity']!=identity:raise ValueError('condition_checkpoint_identity_changed')
        for name,digest in old['artifacts'].items():
            if hash_file(destination/name)!=digest:raise ValueError('condition_checkpoint_artifact_changed')
        return old['summary']
    start=time.monotonic();directory=output/'temporary-vectors';meta=json.loads((directory/'metadata.json').read_text())
    vector=np.load(directory/'vectors.npy',mmap_mode='r');type_ids=np.load(directory/'types.npy',mmap_mode='r');type_names=np.asarray(meta['types'])
    order=np.load(directory/'order.npy',mmap_mode='r');ptr=np.load(directory/'pointers.npy',mmap_mode='r')
    conditions=pd.read_parquet(annotations/'conditions.parquet');selected=conditions.loc[conditions.background_index==background['background_index']]
    def ids(i):return order[ptr[i]:ptr[i+1]]
    vehicle_conditions=selected.loc[selected.source_drug=='DMSO_TF'];controls=np.concatenate([ids(i) for i in vehicle_conditions.condition_index]) if len(vehicle_conditions) else np.empty(0,int)
    if len(controls)!=background['n_DMSO']:raise ValueError('local_vehicle_size_mismatch')
    control_vectors=np.asarray(vector[controls]);control_types=type_names[type_ids[controls]]
    summaries=[];state=[];composition=[];within=[];draws=[]
    with tempfile.TemporaryDirectory(prefix='condition-'+key+'-',dir=output) as temporary:
        temporary=Path(temporary)
        with h5py.File(counts/'condition-profiles.h5','r') as h,h5py.File(temporary/'gene-effects.h5','w') as target:
            profiles=h['mean_logCP10K'];g=profiles.shape[1];mean=np.zeros(g)
            if len(controls):
                for row in vehicle_conditions.itertuples():mean+=profiles[row.condition_index].astype(float)*len(ids(row.condition_index))/len(controls)
            delta=target.create_dataset('effect_vs_plate_line_DMSO',shape=(len(selected),g),dtype='float32',chunks=(1,g),compression='gzip',compression_opts=1,shuffle=True,fillvalue=np.nan)
            target['condition_index']=selected.condition_index.to_numpy();target.attrs['interpretation']='All 62,710 source genes; difference of means of cell log1p(CP10K); not bulk RNA or culture-level inference'
            for j,row in enumerate(selected.to_dict('records')):
                i=row['condition_index'];chosen=ids(i)
                if len(chosen)!=row['observed_cells'] or len(chosen)!=row['eligible_profile_cells']:raise ValueError('condition_cell_count_mismatch')
                if row['source_drug']=='DMSO_TF':
                    other_conditions=vehicle_conditions.loc[vehicle_conditions.condition_index!=i]
                    control=np.concatenate([ids(c) for c in other_conditions.condition_index]) if len(other_conditions) else np.empty(0,int)
                    role='DMSO_sample_vs_other_same_plate_line_DMSO_samples';cv=np.asarray(vector[control]);ct=type_names[type_ids[control]]
                else:control=controls;role='chemical_vs_same_plate_line_DMSO';cv=control_vectors;ct=control_types
                result,st,co,wi,dr=distribution_contrast(vector[chosen],cv,type_names[type_ids[chosen]],ct,meta['names'],PARAMETERS['seed']^int(value_hash(['Tahoe',i])[:8],16))
                if len(chosen) and len(control):
                    if row['source_drug']=='DMSO_TF':
                        base=np.zeros(g)
                        for c in other_conditions.condition_index:base+=profiles[c].astype(float)*len(ids(c))/len(control)
                    else:base=mean
                    effect=profiles[i].astype(float)-base;delta[j]=effect
                    result['full_gene_RMS']=float(np.sqrt(np.mean(effect**2)))
                else:result['full_gene_RMS']=None
                summaries.append({**row,**result,'comparison_role':role,'gene_effect_row':j,'gene_effect_file':'conditions/'+key+'/gene-effects.h5'})
                for collection,values in [(state,st),(composition,co),(within,wi),(draws,dr)]:collection.extend([{'condition_index':i,**r} for r in values])
        for name,values in [('diagnostics',summaries),('state-distributions',state),('composition-contrasts',composition),('within-type-states',within),('resampling',draws)]:
            pd.DataFrame(values).to_parquet(temporary/(name+'.parquet'),index=False)
        summary={'background_index':background['background_index'],'status':'completed','conditions':len(selected),
            'diagnostic_status_counts':dict(Counter(r['status'] for r in summaries)),'stability_status_counts':dict(Counter(r['stability_status'] for r in summaries)),
            'resampling_rows':len(draws),'null_reference_intersections':sum(r['null_reference_intersection'] for r in draws),
            'null_size_shortfall_rows':sum(r['null_target_cells_not_matched']>0 for r in draws),'duration_seconds':time.monotonic()-start}
        write_json(temporary/'identity.json',{'run_identity':identity,'summary':summary,'artifacts':{p.name:hash_file(p) for p in temporary.iterdir() if p.is_file()}});os.rename(temporary,destination)
    return summary


def run(counts,annotations,cache,evidence,output,workers=4,resume=False):
    started=time.monotonic();commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    cr=json.loads((counts/'report.json').read_text());ar=json.loads((annotations/'report.json').read_text());prepared=json.loads((cache/'identity.json').read_text())
    if cr['status']!='completed' or ar['status']!='completed' or ar['source_count_report_sha256']!=hash_file(counts/'report.json'):raise ValueError('completed_consistent_phases_required')
    for directory,report in [(counts,cr),(annotations,ar)]:
        for name,digest in report['artifacts'].items():
            if hash_file(directory/name)!=digest:raise ValueError('upstream_artifact_changed')
    identity={'count_report_sha256':hash_file(counts/'report.json'),'annotation_report_sha256':hash_file(annotations/'report.json'),
        'metadata_identity_sha256':hash_file(cache/'identity.json'),'code':{n:hash_file(Path(__file__).with_name(n)) for n in CODE},
        'parameters':PARAMETERS,'uv_lock_sha256':hash_file(Path(__file__).with_name('uv.lock'))}
    if output.exists():
        if not resume or json.loads((output/'identity.json').read_text())!=identity or (output/'report.json').exists():raise ValueError('resume_identity_mismatch_or_completed_output')
        view=json.loads((output/'analysis-view.json').read_text())
        for name,digest in view['files'].items():
            if hash_file(output/'temporary-vectors'/name)!=digest:raise ValueError('temporary_view_changed')
    else:
        output.mkdir(parents=True);(output/'conditions').mkdir();write_json(output/'identity.json',identity)
        shutil.copytree(evidence,output/'evidence');shutil.copytree(annotations,output/'annotations')
        (output/'counts').mkdir();(output/'metadata').mkdir()
        for name in ['report.json','identity.json','shard-results.json','genes.parquet','gene-mapping.parquet','gene-coverage.parquet','conditions.parquet','condition-profiles.h5']:shutil.copy2(counts/name,output/'counts'/name)
        for p in cache.glob('*'):
            if p.is_file():shutil.copy2(p,output/'metadata'/p.name)
        for name in ['cell_line_metadata.parquet','sample_metadata.parquet','drug_metadata.parquet']:
            shutil.copy2(ROOT/'data/raw/tahoe100m/metadata'/name,output/'metadata'/name)
        prepare_vectors(annotations,output)
    backgrounds=json.loads((annotations/'backgrounds.json').read_text());results=[];failed=[]
    with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn')) as pool:
        futures={pool.submit(analyze_background,b,str(counts),str(annotations),str(output),identity):b for b in backgrounds}
        for future in as_completed(futures):
            b=futures[future]
            try:results.append(future.result())
            except Exception as e:failed.append({'background_index':b['background_index'],'error':str(e),'type':type(e).__name__})
            if (len(results)+len(failed))%20==0:print(f'Tahoe conditions {len(results)}/700 groups completed, {len(failed)} failed',flush=True)
    if failed:
        write_json(output/('failures-'+str(time.time_ns())+'.json'),failed);raise RuntimeError('incomplete_condition_analysis')
    results.sort(key=lambda r:r['background_index']);write_json(output/'condition-group-results.json',results)
    table=pd.concat([pd.read_parquet(output/'conditions'/str(r['background_index']).zfill(3)/'diagnostics.parquet') for r in results],ignore_index=True).sort_values('condition_index')
    if len(table)!=cr['local_conditions'] or table.condition_index.duplicated().any():raise ValueError('incomplete_local_condition_scope')
    table.to_parquet(output/'all-condition-diagnostics.parquet',index=False)
    samples=pd.read_parquet(output/'metadata/sample_metadata.parquet');missing_samples=samples.loc[~samples['sample'].isin(table['sample'])]
    missing_samples.to_parquet(output/'metadata-samples-without-local-expression.parquet',index=False)
    controls=table.loc[table.comparison_role=='DMSO_sample_vs_other_same_plate_line_DMSO_samples'];drugs=table.loc[table.comparison_role=='chemical_vs_same_plate_line_DMSO']
    numeric=Counter()
    for row in json.loads((counts/'shard-results.json').read_text()):numeric.update(row['numeric'])
    for file,digest in cr['input_sha256'].items():
        if hash_file(ROOT/file)!=digest:raise ValueError('raw_input_changed_during_condition_assessment')
    exposure={'source':'All downloaded counts and endpoint annotations observed','cell_line':'Author-assigned source labels, not per-cell verified genotype',
        'MOA_and_drug_targets':'Source metadata includes GPT-assisted annotations; not independent mechanistic ground truth',
        'replication':'Source sample/plate/sublibrary labels retained, independent culture replication unresolved',
        'time_and_media':'Exposure duration and detailed culture history not verified by the retrieved frozen HF card/tutorial; no time-effect inference',
        'licenses':'Registered SOURCE description CC BY 4.0 versus frozen current HF card CC0 retained as conflicting source statements'}
    write_json(output/'exposure-and-limitations.json',exposure)
    # Temporary row-indexed arrays are reproducible from exported annotations; hash remains.
    shutil.rmtree(output/'temporary-vectors')
    report={'schema_version':2,'bundle_id':'tahoe-dossier-'+uuid.uuid4().hex,'title':'Tahoe 全部本地分片条件响应与细胞注释评估','status':'completed',
        'actual_cells':cr['actual_cells'],'local_expression_shards':300,'global_obs_rows':100648790,'HF_card_expression_release_cells':95624334,
        'local_samples':table['sample'].nunique(),'source_sample_metadata_rows':1344,'local_cell_lines':table.cell_line_id.nunique(),'local_plates':table.plate.nunique(),
        'metadata_samples_without_local_expression':missing_samples['sample'].tolist(),
        'local_conditions':len(table),'numeric':dict(numeric),'count_phase':{k:v for k,v in cr.items() if k not in ['artifacts','input_sha256','identity']},
        'annotation_phase':{k:v for k,v in ar.items() if k not in ['artifacts','identity']},'duplicate_audit':json.loads((output/'duplicate-audit.json').read_text()),
        'chemical_status_counts':drugs.status.value_counts().to_dict(),'vehicle_status_counts':controls.status.value_counts().to_dict(),
        'stability_status_counts':table.stability_status.value_counts().to_dict(),'resampling_rows':sum(r['resampling_rows'] for r in results),
        'null_reference_intersections':sum(r['null_reference_intersections'] for r in results),'null_size_shortfall_rows':sum(r['null_size_shortfall_rows'] for r in results),
        'input_sha256':cr['input_sha256'],'inputs_unchanged':True,'code_commit':commit,'identity':identity,'parameters':PARAMETERS,'exposure':exposure,
        'methods':{'protocol':PROTOCOL,'scope':'All 300 local shards and 8,467,330 original rows; metadata universe and release total are separate denominators',
            'count_representation':'Each first CLS token (1,-2) verified separately; all biological counts scanned, source token/gene axis retained',
            'condition_comparison':'All sample×line conditions vs same plate/line local DMSO; vehicle samples compared to other distinct DMSO sample records only',
            'distribution':'All 12 fixed RNA proxies and log1p library size/detected genes: means/quantiles, type proportions and within-inferred-type descriptions',
            'stability':'20 target half splits and disjoint local vehicle arm draws where >=20 targets and >=2 controls; no culture inference or calibrated p-values',
            'gene_effects':'All 62,710 source features, difference of per-cell logCP10K means; chemical exposure, not CRISPRi target knockdown',
            'publication':'Includes every annotated original row and all diagnostic outputs; large intermediate sparse count-sum caches remain reproducible from fixed sources and registered count-phase hashes'},
        'limitations':['本地数据只包含部分释放的表达记录；不能把全局元数据行数或分片比例当作实际细胞覆盖率。',
            '基因 token 的首位 CLS=(1,-2) 是格式标记，其余所有生物计数均已检查；不得整体忽略负值。',
            '同 plate+cell line 没有本地 DMSO 的条件不可估响应；pooled endpoint 状态背景仅用于明确标注的推断回退。',
            '类型/状态全部未校准，probability_correct 为空；癌细胞、健康参考、药物诱导状态可能不匹配；unknown 不等于已知新类型。',
            '源 cell line、细胞周期、药物 MOA/target 与驱动变异注释有各自证据等级，不能当作逐细胞真值。',
            '真实培养重复、精确培养历史与本地处理时长尚未验证；plate/sample/sublibrary 不自动等于生物重复。',
            '同源基线差异与药物响应分开；细胞级条件抽样、组合效应、捕获选择不能识别因果机制。'],
        'tables':[{'title':'全部本地条件比较','columns':list(table.columns),'rows':table.to_dict('records')},
            {'title':'本地 DMSO 与状态背景覆盖','rows':backgrounds}],
        'completed_at':datetime.now(timezone.utc).isoformat(),'duration_seconds':time.monotonic()-started,'reproduce':sys.argv}
    report['artifacts']=[{'file':str(p.relative_to(output)),'sha256':hash_file(p)} for p in sorted(output.rglob('*')) if p.is_file()]
    write_json(output/'report.json',report);(output/'report.html').write_text(render(serial(report)))
    (output/'SHA256SUMS').write_text(''.join(f'{hash_file(p)}  {p.relative_to(output)}\n' for p in sorted(output.rglob('*')) if p.is_file() and p.name!='SHA256SUMS'))
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['counts','annotations','cache','evidence','output']:p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--workers',type=int,default=4);p.add_argument('--resume',action='store_true');a=p.parse_args()
    r=run(a.counts,a.annotations,a.cache,a.evidence,a.output,a.workers,a.resume);print(json.dumps({k:v for k,v in r.items() if k not in ['tables','artifacts','input_sha256','count_phase','annotation_phase','identity']}))
