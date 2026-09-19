#!/usr/bin/env python
"""Verified, read-only Seurat counts adapter; derived H5AD is an analysis cache."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import h5py
import numpy as np
import pandas as pd
from rna import RNAFile,hash_file
from profile_responses import write_json

ROOT=Path(__file__).resolve().parents[2]
RROOT=Path('/home/jy001/micromamba/envs/biominer/lib/R')


def frame(group,df):
    group.attrs['_index']='_index';group.attrs['encoding-type']='dataframe';group.attrs['encoding-version']='0.2.0'
    group.attrs['column-order']=np.asarray(df.columns,dtype=h5py.string_dtype())
    group.create_dataset('_index',data=np.asarray(df.index.astype(str),dtype=object),dtype=h5py.string_dtype())
    for name in df:
        values=df[name]
        if values.dtype.kind in 'biuf':group.create_dataset(name,data=values.to_numpy())
        else:
            node=group.create_group(name);node.attrs['encoding-type']='categorical';node.attrs['encoding-version']='0.2.0';node.attrs['ordered']=False
            cats=pd.Categorical(values);node.create_dataset('codes',data=cats.codes)
            node.create_dataset('categories',data=np.asarray(cats.categories.astype(str),dtype=object),dtype=h5py.string_dtype())


def make_h5ad(export,destination):
    if destination.exists():raise ValueError('fresh_H5AD_required')
    if (export/'COMPLETE').read_text().strip()!='completed':raise ValueError('incomplete_R_export')
    shape=pd.read_csv(export/'shape.csv').iloc[0];n,g,nnz=int(shape.cells),int(shape.genes),int(shape.nnz)
    types=pd.read_csv(export/'metadata_types.csv').set_index('field')['class']
    string_fields={name:'string' for name,kind in types.items() if kind in ('character','factor')}
    meta=pd.read_csv(export/'cells.csv',keep_default_na=False,na_values=['__VCC_SOURCE_NA__'],index_col='source_barcode',dtype={'source_barcode':str,**string_fields})
    for name,kind in types.items():
        if kind in ('character','factor'):meta[name]=meta[name].astype('string')
    genes=pd.read_csv(export/'genes.csv',keep_default_na=False).gene.astype(str).tolist()
    if len(meta)!=n or len(genes)!=g or meta.index.duplicated().any() or len(set(genes))!=g:raise ValueError('export_axis_mismatch')
    digests={}
    with h5py.File(destination,'w') as h:
        h.attrs['encoding-type']='anndata';h.attrs['encoding-version']='0.1.0'
        frame(h.create_group('obs'),meta);frame(h.create_group('var'),pd.DataFrame(index=genes))
        matrix=h.create_group('X');matrix.attrs['encoding-type']='csr_matrix';matrix.attrs['encoding-version']='0.1.0';matrix.attrs['shape']=(n,g)
        for slot,name,dtype,length in [('p','indptr','<i4',n+1),('i','indices','<i4',nnz),('x','data','<f8',nnz)]:
            path=export/f'counts.{slot}.bin'
            if path.stat().st_size!=length*np.dtype(dtype).itemsize:raise ValueError('binary_matrix_length_mismatch')
            array=np.memmap(path,mode='r',dtype=dtype,shape=(length,));dataset=matrix.create_dataset(name,shape=(length,),dtype=dtype,chunks=True)
            before=hash_file(path);digest=hashlib.sha256()
            for lo in range(0,length,1000000):
                dataset[lo:lo+1000000]=array[lo:lo+1000000]
                digest.update(dataset[lo:lo+1000000].astype(dtype).tobytes())
            if before!=digest.hexdigest():raise ValueError('binary_to_HDF5_value_mismatch')
            digests[name]=before
    with RNAFile(destination) as source:
        if source.shape!=(n,g) or source.obs.index.tolist()!=meta.index.tolist() or source.var.index.tolist()!=genes:raise ValueError('H5AD_axis_roundtrip_mismatch')
        for name in meta:
            a=meta[name].fillna('__VCC_SOURCE_NA__').astype(str).tolist()
            b=source.obs[name].fillna('__VCC_SOURCE_NA__').astype(str).tolist()
            if a!=b:raise ValueError('H5AD_metadata_roundtrip_mismatch: '+name)
    return {'shape':[n,g],'counts_stored_values':nnz,'binary_array_sha256':digests,'all_values_exactly_preserved':True,
        'layer_audit':pd.read_csv(export/'layers.csv').to_dict('records'),'source_metadata_types':types.to_dict()}


def convert(source,output,expected):
    started=time.monotonic();commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    before=hash_file(source)
    if before!=expected:raise ValueError('source_hash_mismatch_to_inventory')
    if output.exists():raise ValueError('fresh_analysis_cache_required')
    output.mkdir(parents=True)
    export=output/'R-export';env=os.environ.copy();env['R_HOME']=str(RROOT);env['LD_LIBRARY_PATH']=str(RROOT/'lib')+':'+str(RROOT.parent)
    cmd=['/lib/ld-linux-aarch64.so.1',str(RROOT/'bin/exec/R'),'--vanilla','--slave','-f',str(Path(__file__).with_name('export_seurat_readonly.R')),'--args',str(source.resolve()),str(export.resolve())]
    with (output/'R.log').open('w') as log:subprocess.run(cmd,env=env,check=True,stdout=log,stderr=subprocess.STDOUT)
    result=make_h5ad(export,output/'counts.h5ad')
    if hash_file(source)!=before:raise ValueError('source_changed_during_conversion')
    report=dict(result,status='completed',source_file=str(source),source_sha256=before,inputs_unchanged=True,
        output_sha256=hash_file(output/'counts.h5ad'),code_commit=commit,
        code={p.name:hash_file(p) for p in [Path(__file__),Path(__file__).with_name('export_seurat_readonly.R')]},
        semantics='Source RNA/counts only; exact transpose view, no filtering/normalization; RNA/data audited separately',
        completed_at=datetime.now(timezone.utc).isoformat(),duration_seconds=time.monotonic()-started,reproduce=sys.argv)
    write_json(output/'identity.json',report)
    print(json.dumps({'source':source.name,'status':'completed','shape':result['shape'],'seconds':report['duration_seconds']}),flush=True)
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--inventory',type=Path,required=True)
    a=p.parse_args();inv=json.loads((a.inventory/'report.json').read_text());rows=[r for r in inv['file_results'] if r['source_id']=='jiang2025' and r['file']==a.source.name]
    if len(rows)!=1:raise ValueError('one_inventory_record_required')
    convert(a.source,a.output,rows[0]['sha256'])
