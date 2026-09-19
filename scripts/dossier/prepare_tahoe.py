#!/usr/bin/env python
"""Build a verified local-scope Tahoe metadata cache; no expression inference here."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from rna import hash_file,value_hash
from profile_responses import write_json
from tahoe import metadata_join,parse_compounds
ROOT=Path(__file__).resolve().parents[2]


def prepare(inventory,output):
    start=time.monotonic();commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    inputs={r['file']:r['sha256'] for r in json.loads((inventory/'report.json').read_text())['file_results'] if r['source_id']=='tahoe100m'}
    shards=sorted(p for p in inputs if '/data/train-' in p)
    if len(inputs)!=306 or len(shards)!=300:raise ValueError('wrong_registered_Tahoe_scope')
    for path,digest in inputs.items():
        if hash_file(ROOT/path)!=digest:raise ValueError('source_hash_mismatch')
    output.mkdir(parents=True,exist_ok=False);local=[];headers=[]
    for shard,path in enumerate(shards):
        f=pq.ParquetFile(ROOT/path);cols=[c for c in f.schema_arrow.names if c not in ['genes','expressions']]
        frame=f.read(columns=cols).to_pandas();frame['shard_index']=shard;frame['row_index']=np.arange(len(frame));local.append(frame)
        headers.append({'shard_index':shard,'file':path,'sha256':inputs[path],'rows':len(frame),'row_groups':f.metadata.num_row_groups})
    local=pd.concat(local,ignore_index=True);write_json(output/'shards.json',headers)
    local.to_parquet(output/'local-expression-identities.parquet',index=False,compression='zstd')
    repeated=local[local.BARCODE_SUB_LIB_ID.duplicated(keep=False)].copy();repeated.to_parquet(output/'repeated-local-barcodes.parquet',index=False)
    keys=pd.Index(local.BARCODE_SUB_LIB_ID.unique());print(f'Local scope: {len(local)} records, {len(keys)} barcode keys',flush=True)
    meta=ROOT/'data/raw/tahoe100m/metadata';source=pq.ParquetFile(meta/'obs_metadata.parquet');writer=None;seen=0;selected=0
    for batch in source.iter_batches(batch_size=250000):
        values=batch.column(batch.schema.get_field_index('BARCODE_SUB_LIB_ID')).to_numpy(zero_copy_only=False)
        keep=keys.get_indexer(values)>=0;table=pa.Table.from_batches([batch]).filter(pa.array(keep));seen+=len(batch);selected+=len(table)
        if writer is None:writer=pq.ParquetWriter(output/'local-obs-metadata.parquet',table.schema,compression='zstd')
        writer.write_table(table)
        if seen%10000000==0:print(f'Global metadata {seen}/{source.metadata.num_rows}; matched {selected}',flush=True)
    writer.close()
    observations=pd.read_parquet(output/'local-obs-metadata.parquet');samples=pd.read_parquet(meta/'sample_metadata.parquet')
    doses=[{'sample':r['sample'],'source_drugname_drugconc':r['drugname_drugconc'],'components':parse_compounds(r['drugname_drugconc'])} for r in samples.to_dict('records')]
    write_json(output/'sample-dose-interpretation.json',doses)
    joined=metadata_join(local,observations,samples);directory=output/'shards';directory.mkdir()
    summaries=[]
    for i,group in joined.groupby('shard_index',sort=True):
        group=group.sort_values('row_index');path=directory/(str(i).zfill(3)+'.parquet');group.to_parquet(path,index=False,compression='zstd')
        summaries.append({'shard_index':int(i),'n_cells':len(group),'metadata_obs_missing':int((~group.metadata_obs_present).sum()),
            'metadata_sample_missing':int((~group.metadata_sample_present).sum()),'condition_mismatches':int((~group.metadata_condition_consistent).sum()),
            'artifact':str(path.relative_to(output)),'sha256':hash_file(path)})
    coverage=joined.groupby(['plate','sample','cell_line_id','drug','source_sample_drugname_drugconc'],dropna=False,observed=True).size().reset_index(name='local_records')
    coverage.to_parquet(output/'local-condition-coverage.parquet',index=False)
    for path,digest in inputs.items():
        if hash_file(ROOT/path)!=digest:raise ValueError('raw_source_changed')
    result={'status':'completed','phase':'local_metadata_preparation_only','code_commit':commit,'code':{n:hash_file(Path(__file__).with_name(n)) for n in ['prepare_tahoe.py','tahoe.py','rna.py']},
        'input_sha256':inputs,'inputs_unchanged':True,'actual_local_records':len(local),'unique_local_barcode_keys':len(keys),
        'repeated_local_barcode_records':len(repeated),'global_obs_records_read':seen,'local_obs_records_selected':selected,
        'shard_join_results':summaries,'local_conditions':len(coverage),'dose_semantics':'DMSO source concentration 0 uM is a vehicle code, not a measured vehicle concentration',
        'completed_at':datetime.now(timezone.utc).isoformat(),'duration_seconds':time.monotonic()-start,'reproduce':sys.argv}
    result['artifacts']={str(p.relative_to(output)):hash_file(p) for p in sorted(output.rglob('*')) if p.is_file()};write_json(output/'identity.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--inventory',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    try:r=prepare(a.inventory,a.output)
    except Exception as e:
        if a.output.exists():write_json(a.output/'failure.json',{'status':'failed','error':str(e),'type':type(e).__name__})
        raise
    print(json.dumps({k:v for k,v in r.items() if k not in ['input_sha256','artifacts','shard_join_results','code']}))
