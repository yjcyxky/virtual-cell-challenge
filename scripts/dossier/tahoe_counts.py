#!/usr/bin/env python
"""Full Tahoe biological-count scan and all local sample/line expression profiles."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,as_completed
from datetime import datetime,timezone
import hashlib
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
from scipy import sparse
from rna import hash_file,value_hash,mapping_audit,quantiles
from profile_responses import write_json
from tahoe import decode,condition_eligibility
ROOT=Path(__file__).resolve().parents[2]


def scan_shard(record,cache_string,output_string,identity):
    cache,output=Path(cache_string),Path(output_string);key=str(record['shard_index']).zfill(3);directory=output/'shards'/key
    if (directory/'identity.json').exists():
        saved=json.loads((directory/'identity.json').read_text())
        if saved['run_identity']!=identity:raise ValueError('checkpoint_method_or_input_changed')
        for name,digest in saved['artifacts'].items():
            if hash_file(directory/name)!=digest:raise ValueError('checkpoint_artifact_changed')
        return saved['summary']
    started=time.monotonic();source=ROOT/record['file'];before=source.stat()
    if hash_file(source)!=record['sha256']:raise ValueError('source_shard_changed')
    frame=pd.read_parquet(cache/'shards'/(key+'.parquet'));n=len(frame)
    if n!=record['rows'] or not np.array_equal(frame.row_index,np.arange(n)):raise ValueError('metadata_row_identity_mismatch')
    genes=pd.read_parquet(output/'genes.parquet');tokens=genes.token_id.to_numpy();lookup=np.full(tokens.max()+1,-1,dtype=np.int32);lookup[tokens]=np.arange(len(genes));g=len(genes)
    axis=value_hash(genes.ensembl_id.tolist());condition_list=sorted(set(zip(frame['sample'],frame.cell_line_id)));lookup_condition={x:i for i,x in enumerate(condition_list)}
    groups=np.array([lookup_condition[x] for x in zip(frame['sample'],frame.cell_line_id)],dtype=np.int32);k=len(condition_list)
    observed_counts=np.bincount(groups,minlength=k);first_rows=frame.drop_duplicates(['sample','cell_line_id']).set_index(['sample','cell_line_id'])
    valid_condition,aliases=condition_eligibility(frame);frame['analysis_condition_eligible']=valid_condition;frame['analysis_condition_interpretation']=aliases
    total=np.zeros(n);detected=np.zeros(n,np.int32);valid_all=np.zeros(n,bool);hashes=[];numeric=Counter();summed=sparse.csr_matrix((k,g),dtype=np.float64);eligible_counts=np.zeros(k,np.int64);gene_total=np.zeros(g);gene_detected=np.zeros(g,np.int64)
    with tempfile.TemporaryDirectory(prefix='shard-'+key+'-',dir=output) as temporary:
        temporary=Path(temporary);position=0
        for batch in pq.ParquetFile(source).iter_batches(batch_size=512):
            count,valid,audit=decode(batch,lookup,g);numeric.update(audit);stop=position+len(batch)
            if batch.column(batch.schema.get_field_index('BARCODE_SUB_LIB_ID')).to_pylist()!=frame.BARCODE_SUB_LIB_ID.iloc[position:stop].tolist():raise ValueError('expression_to_metadata_order_mismatch')
            sizes=np.asarray(count.sum(axis=1)).ravel();total[position:stop]=sizes;detected[position:stop]=np.asarray((count>0).sum(axis=1)).ravel();valid_all[position:stop]=valid
            gene_total+=np.asarray(count.sum(axis=0)).ravel();gene_detected+=np.asarray((count>0).sum(axis=0)).ravel()
            for i in range(len(batch)):
                lo,hi=count.indptr[i:i+2];h=hashlib.sha256(axis.encode());h.update(count.indices[lo:hi].astype('<i8').tobytes());h.update(count.data[lo:hi].astype('<f8').tobytes());hashes.append(h.hexdigest())
            allowed=valid&valid_condition[position:stop]&(sizes>0);ids=np.flatnonzero(allowed);codes=groups[position:stop][ids]
            log=count[ids].copy();log.data=np.log1p(log.data*np.repeat(10000/sizes[ids],np.diff(log.indptr)))
            membership=sparse.csr_matrix((np.ones(len(ids)),(codes,np.arange(len(ids)))),shape=(k,len(ids)))
            summed=summed+membership@log;eligible_counts+=np.bincount(codes,minlength=k);position=stop
        if position!=n:raise ValueError('incomplete_source_scan')
        frame['input_sha256']=record['sha256'];frame['record_id']=[value_hash([record['sha256'],i]) for i in range(n)]
        frame['gene_axis_sha256']=axis;frame['computed_total_counts']=total;frame['computed_detected_genes']=detected;frame['computed_numeric_valid']=valid_all
        frame['computed_count_sha256']=hashes;frame['computed_eligible_expression_profile']=valid_all&valid_condition&(total>0)
        frame['source_UMI_QC_matches']=np.isclose(total,frame.source_obs_tscp_count.to_numpy(),rtol=0,atol=0)
        frame['source_gene_QC_matches']=detected==frame.source_obs_gene_count.to_numpy()
        frame.to_parquet(temporary/'cells.parquet',index=False,compression='zstd')
        condition_rows=[]
        for i,(sample,line) in enumerate(condition_list):
            first=first_rows.loc[(sample,line)]
            condition_rows.append({'local_condition_index':i,'sample':sample,'cell_line_id':line,'plate':first.plate,'source_drug':first.drug,
                'source_drugname_drugconc':first.source_sample_drugname_drugconc,'observed_cells':int(observed_counts[i]),'eligible_profile_cells':int(eligible_counts[i])})
        pd.DataFrame(condition_rows).to_parquet(temporary/'conditions.parquet',index=False)
        sparse.save_npz(temporary/'condition-log-sums.npz',summed,compressed=True)
        np.savez_compressed(temporary/'gene-totals.npz',raw_count_sum=gene_total,detected_cells=gene_detected)
        unchanged=(before.st_size,before.st_mtime_ns,before.st_ctime_ns)==(source.stat().st_size,source.stat().st_mtime_ns,source.stat().st_ctime_ns)
        if not unchanged:raise ValueError('raw_shard_changed_during_scan')
        summary={'shard_index':record['shard_index'],'file':record['file'],'input_sha256':record['sha256'],'status':'completed','n_cells':n,
            'n_conditions':k,'n_genes':g,'numeric':dict(numeric),'numeric_invalid_cells':int((~valid_all).sum()),'zero_libraries':int((total==0).sum()),
            'profile_eligible_cells':int(eligible_counts.sum()),'unresolved_condition_cells':int((~valid_condition).sum()),
            'explicit_whitespace_alias_cells':int((aliases=='explicit_trailing_whitespace_alias').sum()),
            'source_UMI_QC_disagreements':int((~frame.source_UMI_QC_matches).sum()),'source_gene_QC_disagreements':int((~frame.source_gene_QC_matches).sum()),
            'library_size':quantiles(total),'detected_genes':quantiles(detected),'inputs_unchanged':True,'duration_seconds':time.monotonic()-started}
        write_json(temporary/'identity.json',{'run_identity':identity,'summary':summary,'artifacts':{p.name:hash_file(p) for p in temporary.iterdir() if p.is_file()}})
        os.rename(temporary,directory)
    return summary


def aggregate(output,records,genes):
    """Reduce verified sparse shard profiles into one bounded, disk-backed matrix."""
    parts=[pd.read_parquet(output/'shards'/str(r['shard_index']).zfill(3)/'conditions.parquet') for r in records]
    design=pd.concat(parts,ignore_index=True);conditions=design.groupby(['sample','cell_line_id','plate','source_drug','source_drugname_drugconc'],dropna=False,observed=True).agg(
        observed_cells=('observed_cells','sum'),eligible_profile_cells=('eligible_profile_cells','sum')).reset_index().sort_values(['sample','cell_line_id']).reset_index(drop=True)
    if conditions.duplicated(['sample','cell_line_id']).any():raise ValueError('sample_line_maps_to_conflicting_conditions')
    conditions.insert(0,'condition_index',np.arange(len(conditions)));conditions.to_parquet(output/'conditions.parquet',index=False)
    lookup={(r['sample'],r['cell_line_id']):r['condition_index'] for r in conditions.to_dict('records')}
    path=output/'temporary-profile-sums.npy';matrix=np.lib.format.open_memmap(path,mode='w+',dtype='float64',shape=(len(conditions),len(genes)))
    gene_raw=np.zeros(len(genes));gene_detected=np.zeros(len(genes),np.int64)
    for r,part in zip(records,parts):
        directory=output/'shards'/str(r['shard_index']).zfill(3);local=sparse.load_npz(directory/'condition-log-sums.npz')
        for row in part.to_dict('records'):
            i=row['local_condition_index'];lo,hi=local.indptr[i:i+2];global_i=lookup[(row['sample'],row['cell_line_id'])]
            matrix[global_i,local.indices[lo:hi]]+=local.data[lo:hi]
        sums=np.load(directory/'gene-totals.npz');gene_raw+=sums['raw_count_sum'];gene_detected+=sums['detected_cells']
    matrix.flush();genes=genes.copy();genes['computed_count_sum']=gene_raw;genes['computed_detected_cells']=gene_detected;genes.to_parquet(output/'gene-coverage.parquet',index=False)
    with h5py.File(output/'condition-profiles.h5','w') as handle:
        means=handle.create_dataset('mean_logCP10K',shape=matrix.shape,dtype='float32',chunks=(1,len(genes)),compression='gzip',compression_opts=1,shuffle=True)
        handle.create_dataset('condition_index',data=conditions.condition_index.to_numpy());handle.create_dataset('n_cells',data=conditions.eligible_profile_cells.to_numpy())
        handle.attrs['representation']='Arithmetic mean of eligible cell log1p(CP10K), float32 storage; sum accumulated in float64. Not bulk RNA-seq or biological replicate.'
        for i,n in enumerate(conditions.eligible_profile_cells):means[i]=matrix[i]/n if n else np.nan
    del matrix;path.unlink();return conditions


def run(cache,output,workers=4,resume=False):
    started=time.monotonic();commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip();prepared=json.loads((cache/'identity.json').read_text())
    if prepared['status']!='completed':raise ValueError('complete_metadata_preparation_required')
    for path,digest in prepared['input_sha256'].items():
        if '/metadata/' in path and hash_file(ROOT/path)!=digest:raise ValueError('raw_metadata_changed')
    for name,digest in prepared['artifacts'].items():
        if hash_file(cache/name)!=digest:raise ValueError('metadata_cache_changed')
    identity={'metadata_identity_sha256':hash_file(cache/'identity.json'),'code':{n:hash_file(Path(__file__).with_name(n)) for n in ['tahoe_counts.py','tahoe.py','rna.py','profile_responses.py']},
        'uv_lock_sha256':hash_file(Path(__file__).with_name('uv.lock')),'normalization':10000,'chunk':512}
    if output.exists():
        if not resume or json.loads((output/'identity.json').read_text())!=identity:raise ValueError('resume_identity_mismatch')
        if (output/'report.json').exists():raise ValueError('completed_output_immutable')
    else:
        output.mkdir(parents=True);(output/'shards').mkdir();write_json(output/'identity.json',identity)
    raw=ROOT/'data/raw/tahoe100m/metadata';genes=pd.read_parquet(raw/'gene_metadata.parquet').sort_values('token_id').reset_index(drop=True);vocabulary=json.loads((raw/'gene_vocabulary.json').read_text())
    if genes.token_id.duplicated().any() or genes.ensembl_id.duplicated().any() or dict(zip(genes.ensembl_id,genes.token_id))!=vocabulary:raise ValueError('conflicting_gene_vocabulary')
    genes.to_parquet(output/'genes.parquet',index=False)
    records=json.loads((cache/'shards.json').read_text());summaries=[];failures=[]
    with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn')) as pool:
        futures={pool.submit(scan_shard,row,str(cache),str(output),identity):row for row in records}
        for future in as_completed(futures):
            row=futures[future]
            try:summaries.append(future.result())
            except Exception as e:failures.append({'shard_index':row['shard_index'],'status':'failed','error':str(e),'type':type(e).__name__})
            if (len(summaries)+len(failures))%5==0:print(f'Tahoe counts {len(summaries)}/300 completed, {len(failures)} failed',flush=True)
    if failures:
        write_json(output/('failures-'+str(time.time_ns())+'.json'),failures);raise RuntimeError('incomplete_Tahoe_count_scan')
    summaries.sort(key=lambda r:r['shard_index']);write_json(output/'shard-results.json',summaries)
    conditions=aggregate(output,records,genes)
    hgnc=ROOT/'data/raw/networks/hgnc_complete_set.txt';official=ROOT/'data/raw/arc_vcc2026_controls/gene_names.csv'
    mapping=mapping_audit(genes.gene_symbol.tolist(),pd.read_csv(hgnc,sep='\t',low_memory=False),pd.read_csv(official).gene_name.tolist(),genes.ensembl_id.tolist())
    mapping.insert(0,'token_id',genes.token_id);mapping.insert(0,'source_ensembl_id',genes.ensembl_id);mapping.to_parquet(output/'gene-mapping.parquet',index=False)
    report={'status':'completed','phase':'all_counts_and_conditional_profiles_only','bundle_id':'tahoe-counts-'+uuid.uuid4().hex,'identity':identity,
        'code_commit':commit,'completed_at':datetime.now(timezone.utc).isoformat(),'duration_seconds':time.monotonic()-started,
        'actual_cells':sum(r['n_cells'] for r in summaries),'completed_shards':len(summaries),'failed_shards':0,'local_conditions':len(conditions),
        'input_sha256':prepared['input_sha256'],'inputs_unchanged':all(r['inputs_unchanged'] for r in summaries),
        'gene_reference_sha256':{str(p.relative_to(ROOT)):hash_file(p) for p in [hgnc,official]},
        'representation':'CLS checked separately; all biological counts scanned; no source gene/cell deletion; eligibility affects derived profiles only',
        'reproduce':sys.argv}
    report['artifacts']={str(p.relative_to(output)):hash_file(p) for p in sorted(output.rglob('*')) if p.is_file()}
    write_json(output/'report.json',report);return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['cache','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--workers',type=int,default=4);p.add_argument('--resume',action='store_true');a=p.parse_args()
    r=run(a.cache,a.output,a.workers,a.resume);print(json.dumps({k:v for k,v in r.items() if k not in ['artifacts','input_sha256','identity','reproduce']}))
