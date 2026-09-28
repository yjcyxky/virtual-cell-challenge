"""Immutable full-population moments and a reproducible likelihood sampling pool."""
import gc
import json
import os
import numpy as np
import pandas as pd
from data import ROOT, write_json, ref, NTC
from challenge import ChallengeIdentity
from rna import RNAFile, hash_file
from vcc_mechanism.inputs import verified


def load_moments(config, contexts, statistics):
    manifest=json.loads(verified(config['moment_reuse']).read_text())
    source=json.loads(verified(manifest['source_metrics']).read_text())
    if source['status']!='completed' or not source['evaluation_completed']:
        raise ValueError('Moments require a completed provenance source')
    result=[]
    for c,s in zip(contexts,statistics,strict=True):
        path=verified(manifest['files'][c])
        m=dict(np.load(path))
        for k in ('labels','positions','counts'):
            np.testing.assert_array_equal(m[k],s[k])
        if c=='H1' and len(m['labels'])!=1: raise ValueError('H1 moments must be NTC only')
        result.append(m)
    return result


def choose_cells(frame, labels, seed, cap, ntc_cap):
    """Uniform within-condition reservoir; no quality or response-based selection."""
    allowed=(frame.eligible & frame.target.isin(labels[1:])) | frame.ntc_pool.eq('input')
    frame=frame.loc[allowed].copy()
    rng=np.random.default_rng(seed); chosen=[]
    for label in labels:
        candidates=frame.index[frame.target.eq(label)].to_numpy()
        if not len(candidates): raise ValueError(f'Empty likelihood condition {label}')
        n=ntc_cap if label==NTC else cap
        chosen.extend(rng.choice(candidates,min(n,len(candidates)),replace=False))
    result=frame.loc[chosen].sort_values('source_row')
    if (result.target.eq(NTC) & ~result.ntc_pool.eq('input')).any():
        raise ValueError('Scoring NTC entered likelihood pool')
    return result


def cell_pool(output, genes, contexts, statistics, config):
    spec=config['cvae']; directory=output/'cache/cells'; directory.mkdir(exist_ok=True)
    identity=ChallengeIdentity(genes,pd.read_csv(verified(config['identities']),sep='\t',low_memory=False))
    audit=json.loads(verified(config['benchmark']['data_audit']).read_text())
    source_refs={r['path']:r for r in audit['inputs']}
    pools=[]; records=[]
    for ci,(context,s) in enumerate(zip(contexts,statistics,strict=True)):
        path=directory/f'{context}.npy'; metadata=directory/f'{context}.json'; rowpath=directory/f'{context}-rows.parquet'
        if metadata.exists():
            record=json.loads(metadata.read_text()); verified(record['matrix_ref']); verified(record['rows_ref'])
            rows=pd.read_parquet(rowpath)
        else:
            frame=pd.read_parquet(output/f'cache/{context}-identities.parquet')
            rows=choose_cells(frame,s['labels'],spec['pool_seed']+ci,spec['cells_per_condition'],spec['ntc_pool_cells'])
            if rows.file.nunique()!=1: raise ValueError('Expected one training raw file')
            rawpath=ROOT/'data/raw'/rows.file.iloc[0]; before=rawpath.stat()
            source_ref=source_refs[str(rawpath.relative_to(ROOT))]
            print(f'Preparing likelihood pool {context}: {len(rows)} cells, raw source verification',flush=True)
            if hash_file(rawpath)!=source_ref['sha256']: raise ValueError('Raw likelihood source changed')
            temporary=path.with_suffix('.tmp')
            matrix=np.lib.format.open_memmap(temporary,mode='w+',dtype=np.uint16,shape=(len(rows),len(s['positions'])))
            wanted=rows.source_row.to_numpy(); offset=0
            with RNAFile(rawpath) as source:
                mapping=identity.features(source.var.gene_name.astype(str),source.var.index.astype(str))
                measured=mapping.loc[mapping.measured].sort_values('official_position')
                np.testing.assert_array_equal(measured.official_position,s['positions'])
                cols=measured.source_position.to_numpy()
                for start,block in source.blocks(config['data']['chunk_rows']):
                    end=np.searchsorted(wanted,start+block.shape[0])
                    if end>offset:
                        values=block[wanted[offset:end]-start][:,cols].toarray()
                        if not np.isfinite(values).all() or (values<0).any() or (values>65535).any() or (values!=np.floor(values)).any():
                            raise ValueError('Likelihood cache requires exact uint16 counts')
                        matrix[offset:end]=values.astype(np.uint16); offset=end
            if offset!=len(rows): raise ValueError('Incomplete likelihood pool')
            matrix.flush(); del matrix; temporary.replace(path)
            after=rawpath.stat()
            if (before.st_ino,before.st_size,before.st_mtime_ns)!=(after.st_ino,after.st_size,after.st_mtime_ns):
                raise ValueError('Raw input changed during extraction')
            rows.to_parquet(rowpath,index=False)
            record={'context':context,'matrix_ref':ref(path),'rows_ref':ref(rowpath),'source_ref':source_ref,
                    'cells':len(rows),'conditions':len(s['labels']),'pool_seed':spec['pool_seed']+ci,
                    'selection':'uniform without replacement within allowed condition; no optional QC',
                    'full_population_auxiliary_cells':int(s['counts'].sum())}
            write_json(metadata,record)
            with rawpath.open('rb') as handle: os.posix_fadvise(handle.fileno(),0,0,os.POSIX_FADV_DONTNEED)
            gc.collect()
        label_index={label:i for i,label in enumerate(s['labels'])}
        groups=rows.target.map(label_index).to_numpy()
        groups_list=[np.flatnonzero(groups==i) for i in range(len(s['labels']))]
        if any(not len(g) for g in groups_list): raise ValueError('Incomplete condition coverage')
        pools.append({'matrix':np.load(path,mmap_mode='r'),'groups':groups_list})
        records.append(record)
        print(f'Likelihood pool ready: {context}, {record["cells"]} cells',flush=True)
    write_json(output/'cache/cell-pools.json',{'records':records,'heldout_response_used':False})
    return pools,records
