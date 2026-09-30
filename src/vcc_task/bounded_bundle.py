"""Use official baseline/anchor functions with disk-backed CSR and bounded templates.

No scorer formula is copied. One PCG64(0) stream is passed through the public
baseline function's np.default_rng(seed) calls. The first panel is checked against
one canonical full-template call before any anchor is accepted.
"""
from contextlib import contextmanager
import gc
import json
from pathlib import Path
import shutil

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from scipy import sparse
from cell_eval2.baseline import generic_response_profile, build_baseline_prediction
from cell_eval2.real_bundle import build_real_bundle
from challenge import scorer_config

from .common import ROOT, NTC, ref, verified, write_json
from .count_store import CountWriter
from .capability import reference_unavailable


class OfficialSeedStream(np.random.Generator):
    """Generator accepted by np.default_rng, with the original seed for metadata."""
    def __init__(self):
        super().__init__(np.random.PCG64(0))

    def __int__(self):
        return 0


@contextmanager
def mapped_reference(path,directory):
    directory=Path(directory); directory.mkdir(parents=True,exist_ok=True)
    meta=ad.read_h5ad(path,backed='r')
    obs,var=meta.obs.copy(),meta.var.copy(); meta.file.close()
    arrays=[]
    with h5py.File(path,'r') as source:
        for key,dtype in [('data',source['X/data'].dtype),('indices',np.int64),('indptr',np.int64)]:
            dataset=source['X/'+key]
            target=directory/(key+'.npy')
            output=np.lib.format.open_memmap(target,mode='w+',dtype=dtype,shape=dataset.shape)
            for start in range(0,len(dataset),1_048_576):
                output[start:start+1_048_576]=dataset[start:start+1_048_576]
            output.flush(); del output
            arrays.append(np.load(target,mmap_mode='r'))
    matrix=sparse.csr_matrix((len(obs),len(var)),dtype=arrays[0].dtype)
    matrix.data,matrix.indices,matrix.indptr=arrays
    matrix.has_sorted_indices=True; matrix.has_canonical_format=True
    data=ad.AnnData(matrix,obs=obs,var=var)
    try:
        yield data
    finally:
        del data,matrix,arrays
        gc.collect()
        # These are decoded I/O caches. The immutable compressed real file and
        # its hash remain the source; no prediction or research artifact is removed.
        shutil.rmtree(directory)


def official_baseline(real,profile,path,chunk_targets,verify_full):
    labels=real.obs.target_gene.astype(str).to_numpy()
    ntc_rows=np.flatnonzero(labels==NTC)
    targets=sorted(set(labels)-{NTC})
    groups=real.obs.groupby('target_gene',observed=True).indices
    expected=np.concatenate([ntc_rows,*[np.asarray(groups[t]) for t in targets]])
    if not np.array_equal(expected,np.arange(real.n_obs)):
        raise ValueError('Full reference is not in the frozen canonical row order')
    control=real.X[ntc_rows]
    writer=CountWriter(path,real.var.copy(),allow_fractional=True)
    writer.append(control.astype(np.float32))
    stream=OfficialSeedStream()
    for start in range(0,len(targets),chunk_targets):
        current=targets[start:start+chunk_targets]
        indices=np.concatenate([groups[t] for t in current])
        template=ad.AnnData(sparse.vstack([control,sparse.csr_matrix((len(indices),real.n_vars),dtype=control.dtype)],format='csr'),
            obs=pd.concat([real.obs.iloc[ntc_rows],real.obs.iloc[indices]]),var=real.var.copy())
        generated=build_baseline_prediction(profile,template,pert_col='target_gene',control=NTC,
                                             emit='dispersed',seed=stream)
        writer.append(generated.X[len(ntc_rows):])
    writer.finish(real.obs.copy())
    check={'canonical_full_checked':False,'baseline_ref':ref(path),
           'seed':0,'rng':'one PCG64(0) state across sorted groups','chunk_targets':chunk_targets}
    if verify_full:
        canonical=build_baseline_prediction(profile,real,pert_col='target_gene',control=NTC,
                                             emit='dispersed',seed=0)
        stored=ad.read_h5ad(path,backed='r')
        if not stored.obs.equals(canonical.obs) or not stored.var.equals(canonical.var):
            raise ValueError('Chunked official baseline metadata mismatch')
        differences=0
        for start in range(0,real.n_obs,1024):
            differences+=(stored.X[start:start+1024]!=canonical.X[start:start+1024]).nnz
        stored.file.close()
        if differences:
            raise ValueError(f'Chunked official baseline differs in {differences} entries')
        check.update(canonical_full_checked=True,different_entries=0,checked_shape=list(real.shape))
        del canonical; gc.collect()
    return check


def bounded_bundle(real,directory,bundle_id,runtime,*,chunk_targets,verify_full):
    directory=Path(directory); marker=directory.parent/(directory.name+'-result.json')
    if marker.exists():
        record=json.loads(marker.read_text())
        for item in record['files']:
            verified(item)
        return record
    original=directory; attempt=0
    while directory.exists() and any(directory.iterdir()):
        attempt+=1; directory=original.with_name(original.name+f'-attempt-{attempt}')
    baseline_path=directory.parent/'official-baseline.h5ad'
    with mapped_reference(real,directory.parent/'decoded-reference-cache') as mapped:
        profile=generic_response_profile(mapped,pert_col='target_gene',control=NTC,exclude_target_gene=True)
        equivalence=official_baseline(mapped,profile,baseline_path,chunk_targets,verify_full)
        eq_path=directory.parent/'baseline-equivalence.json'; write_json(eq_path,equivalence)
        try:
            build_real_bundle(mapped,str(baseline_path),config=scorer_config(**runtime),
                outdir=str(directory),bundle_id=bundle_id,base_seed=0,n_splits=5)
        except ValueError as exc:
            if not reference_unavailable(exc):
                raise
            result={'available':False,'official_rejection':str(exc),'files':[]}
        else:
            result={'available':True,'files':[ref(p) for p in sorted(directory.rglob('*')) if p.is_file()]}
    result.update(bundle_directory=str(directory.relative_to(ROOT)),preserved_incomplete_attempts=attempt,
                  baseline_equivalence=equivalence)
    result['files'] += [ref(baseline_path),ref(eq_path)]
    write_json(marker,result)
    return result
