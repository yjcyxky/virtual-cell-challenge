#!/usr/bin/env python
"""Complete read-only scBaseCount matrix and per-record inference assessment."""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import uuid
import h5py
import numpy as np
import pandas as pd
from scipy import sparse
from annotation import marker_model, state_model, annotate, RULES
from rna import RNAFile, hash_file, value_hash, mapping_audit, quantiles
from scperturb_audit import numeric_audit
from profile_responses import serial, write_json
from render import render

ROOT=Path(__file__).resolve().parents[2]
CODE=['profile_background.py','rna.py','scperturb_audit.py','annotation.py','profile_responses.py','render.py']


def inference_eligibility(record):
    tax=str(record.get('upstream_tax_id',''))
    if tax not in ['9606','','None','nan']:
        return {'status':'not_applicable','reason':'upstream_species_conflicts_with_human_reference_axis','evidence':tax}
    name=str(record.get('library_name',''))
    title=str(record.get('experiment_title',''))
    # Specific assay/library descriptions outrank generic archive RNA-Seq strategy.
    explicit=re.search(r'(?i)(?:^|[_\W])(ADT|HTO|TCR|gdTCR|BCR|VDJ|sgRNA)(?:[_\W]|$)',name)
    has_gex=bool(re.search(r'(?i)\bGEX\b|gene expression|mRNA',title+' '+name))
    title_specific=re.search(r'(?i)\b(?:TCR|BCR|VDJ|ADT|sgRNA)\b',title)
    if explicit or (title_specific and not has_gex):
        return {'status':'not_applicable','reason':'source_library_is_targeted_receptor_guide_or_antibody_assay',
                'evidence':name if explicit else title}
    if re.search(r'(?i)\bATAC|ChIP|bisulfite',title+' '+name):
        return {'status':'not_applicable','reason':'source_library_is_genomic_assay','evidence':title+' | '+name}
    uncertain=tax!='9606' or str(record.get('library_source','')) not in ['TRANSCRIPTOMIC','TRANSCRIPTOMIC SINGLE CELL'] or bool(title_specific)
    return {'status':'conditional','reason':'provenance_or_library_ambiguity' if uncertain else 'source_supports_human_RNA',
            'evidence':'Archive species, library source/strategy and exact library descriptions; counts alone do not prove modality'}


def matrix_in_memory(source):
    x=source.x
    if source.encoding=='dense':return sparse.csr_matrix(x[:])
    cls=sparse.csc_matrix if source.encoding=='csc_matrix' else sparse.csr_matrix
    matrix=cls((x['data'][:],x['indices'][:],x['indptr'][:]),shape=source.shape).tocsr()
    matrix.sum_duplicates();matrix.eliminate_zeros();matrix.sort_indices()
    return matrix


def unknown_labels(n,reason):
    return pd.DataFrame({'inferred_lineage':['unknown']*n,'inferred_type':['unknown']*n,
        'inferred_subtype':['unknown']*n,'inference_status':['not_applicable']*n,'inference_reason':[reason]*n,
        'confidence_calibration':['uncalibrated']*n,'probability_correct':[np.nan]*n,
        'method_conflict':[False]*n,'mixed_marker_signal':[False]*n,
        'target_marker_label_changed':[False]*n,'target_is_reference_marker':[False]*n})


def assess_file(record,output_string,references_string,identity):
    output,references=Path(output_string),Path(references_string)
    raw=ROOT/record['source_file'];directory=output/record['experiment_accession']
    directory.mkdir(exist_ok=True)
    expected=record['source_file_sha256_from_inventory']
    if hash_file(raw)!=expected:raise ValueError('source_inventory_hash_mismatch')
    if (directory/'report.json').exists():
        previous=json.loads((directory/'report.json').read_text())
        if previous['identity']!=identity:raise ValueError('resume_identity_mismatch')
        for a in previous['artifacts']:
            if hash_file(directory/a['file'])!=a['sha256']:raise ValueError('changed_completed_file_result')
        return previous['summary']
    started=time.monotonic();before=(raw.stat().st_size,raw.stat().st_mtime_ns,raw.stat().st_ctime_ns)
    reference=json.loads((references/'gene_sets.json').read_text())
    eligibility=inference_eligibility(record)
    with RNAFile(raw) as source:
        if 'SRX_accession' in source.obs and not (source.obs.SRX_accession == record['experiment_accession']).all():
            raise ValueError('source_experiment_label_mismatch')
        numeric={'X':numeric_audit(source.x)}
        for name,node in source.handle['layers'].items():numeric['layers/'+name]=numeric_audit(node)
        matrix=matrix_in_memory(source)
        obs=source.obs.copy().add_prefix('source_').reset_index(drop=True)
        obs.insert(0,'source_barcode',source.obs.index.astype(str))
        n,g=source.shape;native=source.var.index.astype(str).tolist();axis=value_hash(native)
        obs.insert(0,'row_index',np.arange(n));obs.insert(0,'input_sha256',expected)
        obs.insert(0,'record_id',[value_hash([expected,i]) for i in range(n)])
        obs['source_experiment']=record['experiment_accession']
        obs['gene_axis_sha256']=axis
        obs['source_sample_accessions']='|'.join(record.get('sample_accessions',[]))
        obs['source_study_accessions']='|'.join(record.get('study_accessions',[]))
        source_symbols=source.var.gene_symbols.astype(str).tolist()
        mapping=mapping_audit(source_symbols,pd.read_csv(ROOT/'data/raw/networks/hgnc_complete_set.txt',sep='\t',low_memory=False),
            pd.read_csv(ROOT/'data/raw/arc_vcc2026_controls/gene_names.csv').gene_name.tolist(),native)
        mapping.insert(0,'source_gene_id',native)
        mapping.to_parquet(directory/'gene_mapping.parquet',index=False,compression='zstd')
        source_obs_count=n
    total=np.asarray(matrix.sum(axis=1)).ravel().astype(float)
    detected=np.asarray((matrix>0).sum(axis=1)).ravel()
    counts_hashes=[];valid=np.ones(n,dtype=bool)
    for i in range(n):
        lo,hi=matrix.indptr[i:i+2];values=matrix.data[lo:hi]
        valid[i]=bool(np.isfinite(values).all() and (values>=0).all() and (values==np.floor(values)).all())
        digest=hashlib.sha256(axis.encode());digest.update(matrix.indices[lo:hi].astype('<i8').tobytes());digest.update(values.astype('<f8').tobytes())
        counts_hashes.append(digest.hexdigest())
    obs['computed_total_counts']=total;obs['computed_detected_genes']=detected
    obs['computed_numeric_valid']=valid;obs['computed_count_sha256']=counts_hashes
    comparable=valid & (total>0)
    gene_sum=np.asarray(matrix.sum(axis=0)).ravel()
    pd.DataFrame({'source_gene_id':native,'source_symbol':source_symbols,'computed_sum':gene_sum,
        'computed_detected_cells':np.asarray((matrix>0).sum(axis=0)).ravel()}).to_parquet(directory/'genes.parquet',index=False)
    safe=mapping.mapped_symbol.where(~mapping.many_to_one_mapping & (mapping.symbol_vs_ensembl!='conflict'))
    resolved=[str(x) if pd.notna(x) else None for x in safe]
    model=marker_model(resolved,reference['profiles'])
    baseline=np.zeros(g);chunk=256
    if eligibility['status']=='conditional' and comparable.any():
        for a in range(0,n,chunk):
            b=min(a+chunk,n);ids=np.flatnonzero(comparable[a:b])+a
            x=matrix[ids].astype(float)
            x.data=np.log1p(x.data*np.repeat(10000/total[ids],np.diff(x.indptr)))
            baseline+=np.asarray(x.sum(axis=0)).ravel()
        baseline/=comparable.sum()
        state_weights,state_coverage=state_model(resolved,reference['states'],baseline)
    else:
        state_weights=np.zeros((g,len(reference['states'])),dtype=np.float32)
        state_coverage=[{'state':name,'status':'not_applicable','reason':eligibility['reason'] if eligibility['status']!='conditional' else 'no_numeric_valid_nonzero_libraries'} for name in reference['states']]
    write_json(directory/'type_coverage.json',model['coverage']);write_json(directory/'state_coverage.json',state_coverage)
    np.savez_compressed(directory/'scoring_weights.npz',type_weights=model['weights'],state_weights=state_weights,
        baseline_mean=baseline,valid_gene_axis=model['valid_axis'])
    parts=[]
    for a in range(0,n,chunk):
        b=min(a+chunk,n)
        part=unknown_labels(b-a,eligibility['reason'])
        for name in reference['states']:part['state__'+name]=np.nan
        local=np.flatnonzero(comparable[a:b])
        if eligibility['status']=='conditional' and len(local):
            ids=local+a;x=matrix[ids].astype(float)
            x.data=np.log1p(x.data*np.repeat(10000/total[ids],np.diff(x.indptr)))
            log=x.toarray().astype(np.float32)
            labels,scores=annotate(log,[None]*len(ids),model)
            for name,values in scores.items():labels[name]=values
            state_values=log@state_weights
            for j,(name,details) in enumerate(zip(reference['states'],state_coverage)):
                labels['state__'+name]=state_values[:,j] if details['status']=='completed' else np.nan
            for name in labels:
                if name not in part:part[name]=np.nan if labels[name].dtype.kind in 'fiu' else None
                part.loc[local,name]=labels[name].to_numpy()
        bad=~valid[a:b];zero=valid[a:b] & (total[a:b]==0)
        part.loc[bad,'inference_reason']='invalid_numeric_counts'
        part.loc[zero,'inference_reason']='zero_library'
        parts.append(part)
    combined=pd.concat([obs,pd.concat(parts,ignore_index=True)],axis=1)
    combined['inference_method']='broad-marker-rank-and-expression-v1'
    combined['inference_reference_match']='human broad marker reference; tissue, protocol and disease shift uncalibrated'
    combined['inference_exposure']='observed_sample_RNA_unverified_baseline_or_endpoint'
    combined['future_prediction_availability']='not_certified_as_matched_NTC; only_if_corresponding_RNA_already_measured'
    combined['library_inference_eligibility']=eligibility['status']
    combined['library_inference_limitation']=eligibility['reason']
    combined.to_parquet(directory/'cells.parquet',index=False,compression='zstd')
    state_summary=[{'state':name,'distribution':quantiles(combined['state__'+name]),
        'status':'completed' if combined['state__'+name].notna().any() else 'not_estimable',
        'baseline':'pooled observed file, not certified NTC'} for name in reference['states']]
    differences={}
    for source_column,computed in [('source_umi_count_Unique',total),('source_gene_count_Unique',detected)]:
        differences[source_column]=int((~np.isclose(combined[source_column].to_numpy(),computed,rtol=1e-6,atol=1e-5)).sum()) if source_column in combined else None
    changed=before!=(raw.stat().st_size,raw.stat().st_mtime_ns,raw.stat().st_ctime_ns)
    if changed:raise ValueError('source_changed_during_read_only_assessment')
    summary={'experiment_accession':record['experiment_accession'],'status':'completed','n_cells':n,'n_genes':g,
        'input_sha256':expected,'gene_axis_sha256':axis,'numeric':numeric,'source_metadata_cells':record['source_cells_from_metadata'],
        'metadata_count_matches':int(record['source_cells_from_metadata'])==source_obs_count,'numeric_invalid_cells':int((~valid).sum()),
        'zero_libraries':int((total==0).sum()),'library_size':quantiles(total),'detected_genes':quantiles(detected),
        'duplicate_barcodes':int(combined.source_barcode.duplicated().sum()),'duplicate_count_profiles':int(combined.computed_count_sha256.duplicated().sum()),
        'source_QC_disagreements':differences,'inference_eligibility':eligibility,
        'inferred_types':combined.inferred_type.value_counts().to_dict(),'inferred_lineages':combined.inferred_lineage.value_counts().to_dict(),
        'inference_status_counts':combined.inference_status.value_counts().to_dict(),'unknown_fraction':float((combined.inferred_type=='unknown').mean()),
        'method_conflicts':int(combined.method_conflict.sum()),'source_labeled_cells':int((combined.source_cell_type.fillna('')!='').sum()),
        'confidence_calibration':'uncalibrated','probability_correct':None,'states':state_summary,
        'baseline_eligibility':'not_certified','independent_biological_repeats':None,
        'supervised_study_overlap':list(record.get('supervised_overlap',[])),
        'duration_seconds':time.monotonic()-started,'inputs_unchanged':True}
    write_json(directory/'provenance.json',record)
    report={'identity':identity,'summary':summary,'artifacts':[{'file':p.name,'sha256':hash_file(p)} for p in sorted(directory.iterdir()) if p.is_file() and p.name!='report.json']}
    write_json(directory/'report.json',report)
    return summary


def assess(provenance,references,output,resume=False,workers=4):
    t0=time.monotonic();run_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    prior=json.loads((provenance/'report.json').read_text())
    if prior['status']!='completed':raise ValueError('completed_provenance_required')
    for artifact in prior['artifacts']:
        if artifact['file']=='samples.parquet' and hash_file(provenance/artifact['file'])!=artifact['sha256']:raise ValueError('changed_provenance')
    records=pd.read_parquet(provenance/'samples.parquet')
    if len(records)!=1808 or records.experiment_accession.duplicated().any():raise ValueError('unexpected_scope')
    for line in (references/'SHA256SUMS').read_text().splitlines():
        digest,name=line.split('  ',1)
        if hash_file(references/name)!=digest:raise ValueError('changed_reference')
    identity={'provenance_sha256':hash_file(provenance/'report.json'),'samples_sha256':hash_file(provenance/'samples.parquet'),
        'reference_sha256':hash_file(references/'gene_sets.json'),'rules':RULES,'normalization_total':10000,'chunk':256,
        'code':{name:hash_file(Path(__file__).with_name(name)) for name in CODE},
        'uv_lock_sha256':hash_file(Path(__file__).with_name('uv.lock'))}
    if output.resolve().is_relative_to((ROOT/'data/raw').resolve()):raise ValueError('raw_output_forbidden')
    if output.exists():
        if not resume or json.loads((output/'identity.json').read_text())!=identity:raise ValueError('resume_identity_mismatch')
        if (output/'report.json').exists():raise ValueError('completed_results_immutable')
    else:
        output.mkdir(parents=True);write_json(output/'identity.json',identity);shutil.copytree(references,output/'references')
    results=[];failures=[]
    with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn')) as pool:
        futures={pool.submit(assess_file,row,str(output),str(references),identity):row['experiment_accession'] for row in records.to_dict('records')}
        for future in as_completed(futures):
            accession=futures[future]
            try:results.append(future.result())
            except Exception as exc:
                failure={'experiment_accession':accession,'status':'failed','error':f'{type(exc).__name__}: {exc}'}
                failures.append(failure);write_json(output/(accession+'-failure.json'),failure)
            if (len(results)+len(failures))%10==0:print(f'scBaseCount {len(results)}/1808 completed, {len(failures)} failed',flush=True)
    results.sort(key=lambda r:r['experiment_accession'])
    write_json(output/'samples.json',results);write_json(output/'failures.json',failures)
    compact=[{k:v for k,v in r.items() if k not in ['numeric','states']} for r in results]
    report={'schema_version':2,'bundle_id':'scbase-expression-'+uuid.uuid4().hex,'title':'scBaseCount 全表达输入及逐细胞推断',
        'status':'completed' if len(results)==1808 and not failures else 'failed','identity':identity,'code_commit':run_commit,
        'completed_at':datetime.now(timezone.utc).isoformat(),'duration_seconds':time.monotonic()-t0,
        'expected_files':1808,'completed_files':len(results),'failed_files':len(failures),'n_source_records':sum(r['n_cells'] for r in results),
        'methods':{'protocol':'https://github.com/yjcyxky/virtual-cell-challenge/issues/16#issuecomment-5744803720',
            'numeric':'Full X and all layers; CSC structural checks; Unique is the primary RNA count view; multimap fractions remain separate',
            'type':'17 fixed human broad profiles with conservative rank/expression gate, lineage fallback and unknown; uncalibrated',
            'state':'12 expression proxies with fixed whole-file pooled mean expression-bin controls; not NTC or universal cross-study calibration',
            'workers':workers,'cell_identity':'input SHA256 + row index; original study, sample, experiment and barcode retained',
            'duplicate_identity':'Count content hashes and source identities exported; barcode/shared sample alone never proves independent or duplicate cells'},
        'limitations':['本地矩阵可能来自 guide、抗体或受体靶向文库；RNA 推断资格与纯数值合格分开判断。',
            '源物种冲突与人类参考轴不能由表达推断修复；所有原始文件仍保留在工程核查分母中。',
            '所有类型与状态为未校准 RNA 推断；源 cell_type/ontology 独立保存，不是验证标签。',
            '样本整体不认证为 NTC 或 CRISPRi 监督；已知监督研究重叠不增加独立证据。',
            '状态分数用样本自身固定背景，只支持有方法限制的分布描述；不能恢复实际通路活性或真实周期阶段。'],
        'tables':[{'title':'全部样本','rows':compact},{'title':'失败输入','rows':failures}], 'reproduce':sys.argv}
    report['artifacts']=[{'file':str(p.relative_to(output)),'sha256':hash_file(p)} for p in sorted(output.rglob('*')) if p.is_file()]
    write_json(output/'report.json',report);(output/'report.html').write_text(render(serial(report)))
    (output/'SHA256SUMS').write_text(''.join(f'{hash_file(p)}  {p.relative_to(output)}\n' for p in sorted(output.rglob('*')) if p.is_file() and p.name!='SHA256SUMS'))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['provenance','references','output']:parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--resume',action='store_true');parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args();result=assess(args.provenance,args.references,args.output,args.resume,args.workers)
    print(json.dumps({'status':result['status'],'completed_files':result['completed_files'],'failed_files':result['failed_files']}))
    sys.exit(0 if result['status']=='completed' else 1)
