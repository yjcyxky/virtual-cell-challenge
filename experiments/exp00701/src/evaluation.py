"""All-target official six-metric evaluation with fixed complete NTC pools."""
from dataclasses import replace
import gc
import json
import os
import time
from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd
import polars as pl
from scipy import sparse
from cell_eval2 import EvalConfig,compute_metrics,aggregate_metrics_wide,score_metrics
from cell_eval2.baseline import build_run_meta,generic_response_profile,_emission_scale,_emit_scaled_resample
from cell_eval2.run import metric_output_names
from cell_eval2.real_bundle import build_real_bundle,read_real_bundle
from cell_eval2.competition import competition_members
from common import digest,event,write_json
from generation import Generator,target_composition
from training import predict,fit_id
from data import NTC

EVALUATION_PROTOCOL = 'cell-eval2-dispersed-exclude-target-v1'


def score_prediction(prediction,real,cfg,bundle,directory):
    """The same pinned metric and scaling implementation for both reference protocols."""
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    raw=compute_metrics(prediction,real,config=cfg)
    raw.write_parquet(directory/'raw-metrics.parquet')
    aggregate=aggregate_metrics_wide(raw,metrics=metric_output_names(cfg));aggregate.write_csv(directory/'aggregate.csv')
    meta=build_run_meta(cfg,real,prediction);write_json(directory/'run-meta.json',meta)
    scored=score_metrics(aggregate,real_bundle=str(bundle),user_meta=meta);scored.write_csv(directory/'scores.csv')
    members=list(competition_members());selected=scored.filter(pl.col('metric').is_in(members))
    components=dict(zip(selected['metric'].to_list(),selected['from_replicate'].to_list()))
    if set(components)!=set(members) or not all(v is not None and np.isfinite(v) for v in components.values()):
        raise ValueError('official_score_is_not_finite')
    score=float(scored.filter(pl.col('metric')=='avg_score')['from_replicate'].item())
    if not np.isclose(score,np.mean(list(components.values())),atol=1e-12):raise ValueError('official_average_mismatch')
    return {'score':score,'components':components}


def write_count_matrix(directory, blocks, shape, dtype):
    """Stream exact counts into a memory-mapped CSR, without a full dense/COO copy."""
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    dtype=np.dtype(dtype)
    indptr=np.lib.format.open_memmap(directory/'indptr.npy',mode='w+',dtype=np.int64,shape=(shape[0]+1,))
    indptr[0]=0;row=0;nnz=0
    with (directory/'data.bin').open('wb') as values, (directory/'indices.bin').open('wb') as indices:
        for block in blocks:
            block=block.tocsr() if sparse.issparse(block) else np.asarray(block)
            if block.ndim!=2 or block.shape[1]!=shape[1] or row+block.shape[0]>shape[0]:
                raise ValueError('count_matrix_block_shape_mismatch')
            if block.dtype!=dtype:
                raise ValueError('count_matrix_dtype_mismatch')
            csr=block if sparse.issparse(block) else sparse.csr_matrix(block)
            csr.data.tofile(values)
            csr.indices.astype(np.int64,copy=False).tofile(indices)
            indptr[row+1:row+block.shape[0]+1]=np.add(csr.indptr[1:],nnz,dtype=np.int64)
            row+=block.shape[0];nnz+=csr.nnz
    if row!=shape[0]:raise ValueError('incomplete_count_matrix')
    indptr.flush();del indptr
    metadata={'shape':list(shape),'dtype':dtype.str,'nnz':int(nnz),'index_dtype':'<i8',
              'files':{name:digest(directory/name) for name in ['data.bin','indices.bin','indptr.npy']}}
    write_json(directory/'matrix.json',metadata)
    return read_count_matrix(directory)


def read_count_matrix(directory):
    directory=Path(directory);meta=json.loads((directory/'matrix.json').read_text())
    nnz=meta['nnz'];dtype=np.dtype(meta['dtype'])
    matrix=sparse.csr_matrix(tuple(meta['shape']),dtype=dtype)
    # Assign validated streaming buffers directly: scipy's tuple constructor may
    # downcast small int64 indices and materialize otherwise file-backed arrays.
    matrix.data=np.memmap(directory/'data.bin',mode='r',dtype=dtype,shape=(nnz,)) if nnz else np.empty(0,dtype)
    matrix.indices=np.memmap(directory/'indices.bin',mode='r',dtype=np.int64,shape=(nnz,)) if nnz else np.empty(0,np.int64)
    matrix.indptr=np.load(directory/'indptr.npy',mmap_mode='r')
    if len(matrix.indptr)!=matrix.shape[0]+1 or matrix.indptr[0]!=0 or matrix.indptr[-1]!=nnz:
        raise ValueError('invalid_count_matrix_pointers')
    matrix.has_sorted_indices=True;matrix.has_canonical_format=True
    return matrix


def matrix_bytes(matrix):
    if sparse.issparse(matrix):
        return matrix.data.nbytes+matrix.indices.nbytes+matrix.indptr.nbytes
    return matrix.nbytes


def stream_baseline(profile,real,*,pert_col,control,seed,directory,block_rows=256):
    """File-backed execution of the pinned official dispersed emission kernel.

    layout() supplies controls first, then sorted contiguous target groups. Verify
    that contract rather than silently permuting rows or restarting the RNG per block.
    Parity tests compare values AND diagnostics with build_baseline_prediction.
    """
    genes=np.asarray(real.var.index).astype(str)
    if not np.array_equal(genes,profile.genes):raise ValueError('baseline_gene_axis_mismatch')
    labels=real.obs[pert_col].to_numpy().astype(str)
    control_rows=np.flatnonzero(labels==control)
    targets,sizes=np.unique(labels[labels!=control],return_counts=True)
    expected=np.concatenate([np.repeat(control,len(control_rows)),np.repeat(targets,sizes)])
    if not len(control_rows) or not np.array_equal(labels,expected):
        raise ValueError('baseline_requires_controls_then_sorted_target_groups')
    ctrl=real.X[control_rows].tocsr()
    ctrl_pb=np.asarray(ctrl.astype(np.float64).mean(axis=0)).ravel()
    scale,scale_diag=_emission_scale(profile.values,ctrl_pb,genes)
    rng=np.random.default_rng(seed)
    diag={'emit':'dispersed','seed':int(seed),**scale_diag,'n_explicit_zeros_removed':0,
          'n_rows':int(sizes.sum()),'max_row_total':0.,'max_scaled_noncontrol_row_total':0.,
          'max_row_total_full_prediction':0.}
    def blocks():
        for start in range(0,len(control_rows),block_rows):
            block=ctrl[start:start+block_rows].astype(np.float32)
            diag['max_row_total_full_prediction']=max(diag['max_row_total_full_prediction'],float(block.sum(axis=1).max()))
            yield block
        for i,size in enumerate(sizes):
            for start in range(0,int(size),block_rows):
                block,source,kernel=_emit_scaled_resample(ctrl,[min(block_rows,int(size)-start)],scale,rng)
                diag['n_explicit_zeros_removed']+=kernel['n_explicit_zeros_removed']
                diag['max_row_total']=max(diag['max_row_total'],kernel['max_row_total'])
                diag['max_scaled_noncontrol_row_total']=diag['max_row_total']
                diag['max_row_total_full_prediction']=max(diag['max_row_total_full_prediction'],kernel['max_row_total'])
                yield block
            if i%200==0:event('official_baseline_emission',target_index=i,targets=len(targets))
    matrix=write_count_matrix(directory,blocks(),real.shape,np.float32)
    return ad.AnnData(matrix,obs=real.obs.copy(),var=real.var.copy(),uns={'baseline_emission':diag})

def release(path):
    gc.collect()
    if hasattr(os,'posix_fadvise') and Path(path).exists():
        with Path(path).open('rb') as f: os.posix_fadvise(f.fileno(),0,0,os.POSIX_FADV_DONTNEED)
    try:
        import cupy
        cupy.get_default_memory_pool().free_all_blocks()
    except ImportError: pass
    import torch
    if torch.cuda.is_available(): torch.cuda.empty_cache()

def wait_for_memory(real,context,device,anchor=False):
    # Two disjoint half-panels total one full panel; allow another half for the
    # slice/copy transient. Full reference and baseline are read-only file mappings.
    # Unified CPU/GPU memory is shared with other active experiments on this host.
    # Admission delays execution; it never changes the panel or metric definition.
    groups=real.obs.target.nunique()
    required=(int(1.5*matrix_bytes(real.X)) if anchor else groups*real.n_vars*8*10)+(12<<30)
    gpu_required=groups*real.n_vars*8*12+(2<<30)
    while True:
        memory={line.split()[0].rstrip(':'):int(line.split()[1])*1024
                for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith(('MemAvailable:','MemTotal:'))}
        available=memory['MemAvailable']
        if required>memory.get('MemTotal',float('inf')):
            raise MemoryError(f'evaluation_requirement_exceeds_host_capacity:{required}')
        gpu_available=None;gpu_error=None
        if device=='cuda':
            import cupy
            try:gpu_available=cupy.cuda.runtime.memGetInfo()[0]
            except cupy.cuda.runtime.CUDARuntimeError as error:
                if error.status!=2:raise
                gpu_available=0;gpu_error=str(error)
        if available>=required and (gpu_available is None or gpu_available>=gpu_required):return
        event('waiting_for_evaluation_resources',context=context,anchor=anchor,
              available_gib=available/(1<<30),required_gib=required/(1<<30),
              gpu_available_gib=None if gpu_available is None else gpu_available/(1<<30),
              gpu_required_gib=gpu_required/(1<<30),gpu_error=gpu_error)
        time.sleep(30)

class Evaluation:
    def __init__(self,data,config,output,tracked):
        self.data,self.config,self.output,self.tracked=data,config,output,tracked
        if config.get('evaluation_protocol')!=EVALUATION_PROTOCOL:
            raise ValueError('evaluation_protocol_changed_new_run_required')

    def protocol(self):
        return {'id':EVALUATION_PROTOCOL,'baseline_emit':'dispersed','exclude_target_gene':True,
                'baseline_seed':self.config['baseline_seed']}

    def layout(self,context):
        stats=self.data.contexts[context]
        ids=np.load(stats['folder']/'evaluation-rows.npy')
        cells=self.data.cells(context)
        obs=pd.DataFrame({'target':cells.loc[ids,'target'].to_numpy()},index=[f'{context}-{i}' for i in ids])
        var=pd.DataFrame(index=np.asarray(self.data.genes)[stats['gene_indices']])
        return ids,obs,var

    def settings(self,context):
        directory=self.output/'cache/official'/context;directory.mkdir(parents=True,exist_ok=True)
        cfg=EvalConfig.from_preset('vcc2026')
        cfg=replace(cfg,device=self.config['official_device'],num_threads=self.config['threads'],
                    pert_chunk=64,cache_real=str(directory/'real-cache'),
                    de=replace(cfg.de,backend=self.config['official_de_backend']))
        return directory,cfg

    def prepare(self,context):
        directory,cfg=self.settings(context)
        ids,obs,var=self.layout(context)
        path=directory/'reference'
        if not (directory/'reference.json').exists():
            source=self.data.counts(context)
            values=write_count_matrix(path,(source[ids[start:start+256]] for start in range(0,len(ids),256)),
                                      (len(ids),len(var)),np.uint16)
            del values,source
            write_json(directory/'reference.json',{'shape':[len(ids),len(var)],'sha256':digest(path/'matrix.json'),
                       'evaluation_protocol':self.protocol(),'format':'csr',
                       'targets':'all usable targets, no overlap restriction',
                       'NTC':'entire context NTC pool; same source baseline is available to prediction',
                       'perturbation_cells':f'fixed stratified sample, at most {self.config["reference_cells_per_target"]} per target; no replacement',
                       'scoring':'pinned official vcc2026 method on context-native measured genes, not leaderboard-equivalent'})
        reference=json.loads((directory/'reference.json').read_text())
        if reference.get('evaluation_protocol')!=self.protocol():
            raise ValueError('reference_protocol_changed_new_run_required')
        real=ad.AnnData(read_count_matrix(path),obs=obs,var=var)
        bundle=directory/'bundle'
        if not (bundle/'manifest.json').exists():
            if bundle.exists() and any(bundle.iterdir()):
                # An interrupted atomic official build may leave an incomplete directory.
                raise ValueError(f'incomplete_official_bundle_requires_inspection:{bundle}')
            event('official_anchor_start',context=context,shape=real.shape)
            wait_for_memory(real,context,cfg.device,anchor=True)
            event('official_baseline_profile',context=context)
            profile=generic_response_profile(real,pert_col=cfg.pert_col,control=cfg.control,
                                             exclude_target_gene=True,target_gene_map=cfg.target_gene_map)
            event('official_baseline_prediction',context=context,emit='dispersed')
            arm=stream_baseline(profile,real,pert_col=cfg.pert_col,control=cfg.control,
                                seed=self.config['baseline_seed'],directory=directory/'baseline')
            event('official_bundle_build',context=context)
            write_json(directory/'baseline-protocol.json',self.protocol())
            build_real_bundle(real,arm,config=cfg,outdir=str(bundle),bundle_id=f'exp00701-{context}',
                              base_seed=self.config['replicate_seed'],n_splits=self.config['replicate_splits'])
            del arm;gc.collect()
            event('official_anchor_ready',context=context)
        if json.loads((directory/'baseline-protocol.json').read_text())!=self.protocol():
            raise ValueError('bundle_protocol_changed_new_run_required')
        manifest=read_real_bundle(bundle).manifest
        if manifest['rule_digest'] is None: raise ValueError('invalid_competition_bundle')
        return real,cfg,bundle

    def score(self,context,training_contexts,iteration,booster,features,kind='model',keep=False):
        name=fit_id(training_contexts) if training_contexts else 'baseline'
        directory=self.output/'predictions'/name/context/f'{kind}-{iteration:04d}'
        done=directory/'metrics.json'
        if done.exists():
            result=json.loads(done.read_text())
            if result.get('evaluation_protocol')!=self.protocol():
                raise ValueError('cached_score_protocol_changed_new_run_required')
            return result
        directory.mkdir(parents=True,exist_ok=True)
        real,cfg,bundle=self.prepare(context)
        wait_for_memory(real,context,cfg.device)
        stats=self.data.contexts[context];targets=stats['targets'];axis=stats['gene_indices']
        if kind=='zero': response=np.zeros_like(stats['response'])
        elif kind=='shared': response=self.shared(context,training_contexts)
        else: response=predict(booster,features,context,targets,axis)
        np.savez_compressed(directory/'predicted-response.npz',targets=targets,genes=np.array(self.data.genes)[axis],
                            response=response,prediction_target=self.config['prediction_target'])
        matrix_path=self.output/'cache'/f'prediction-{context}'
        cells_per=self.config['cells_per_prediction']
        perturbation_rows=len(targets)*cells_per
        control_ids=stats['control_rows']
        generator=Generator(self.data,context,self.config)
        # Official validate_pair requires equal label sets even with control_source=real.
        # Copy NTC unchanged for structural compatibility; the comparator still uses real NTC.
        source=self.data.counts(context)
        generated_means=np.empty(response.shape,np.float64)
        def blocks():
            for i,target in enumerate(targets):
                generated=generator.generate(str(target),response[i])
                generated_means[i]=(generated/generated.sum(1,dtype=np.float64)[:,None]*1e6).mean(0)
                yield generated
                if i%200==0:event('prediction',context=context,model=name,round=iteration,target_index=i,targets=len(targets))
            for start in range(0,len(control_ids),256):
                yield source[control_ids[start:start+256]].astype(np.uint32)
        counts=write_count_matrix(matrix_path,blocks(),(perturbation_rows+len(control_ids),len(axis)),np.uint32)
        del generator,source
        labels=np.concatenate([np.repeat(targets,cells_per),np.repeat(NTC,len(control_ids))])
        obs=pd.DataFrame({'target':labels},index=[f'prediction-{i}' for i in range(counts.shape[0])])
        prediction=ad.AnnData(counts,obs=obs,var=pd.DataFrame(index=np.array(self.data.genes)[axis]))
        event('official_score_start',context=context,training_contexts=training_contexts,round=iteration,kind=kind)
        scored_result=score_prediction(prediction,real,cfg,bundle,directory)
        score,components=scored_result['score'],scored_result['components']
        seen=set().union(*(set(self.data.contexts[c]['targets']) for c in training_contexts)) if training_contexts else set()
        auxiliary=[]
        epsilon=self.config['lfc_epsilon']
        generated_lfc=np.log2((generated_means+epsilon)/(stats['baseline_cpm']+epsilon))
        for i,target in enumerate(targets):
            own=axis!=self.data.lookup[str(target)]
            de_axis=own&(stats['baseline_cpm']>5)
            error=float(np.mean((response[i,own]-stats['response'][i,own])**2))
            projected_cpm=target_composition(stats['baseline_cpm'],response[i],epsilon)*1e6
            projected_lfc=np.log2((projected_cpm+epsilon)/(stats['baseline_cpm']+epsilon))
            auxiliary.append({'target':str(target),'seen_in_training':str(target) in seen,'log2fc_mse':error,
                              'log2fc_mae_cpm_gt5':float(np.mean(np.abs(response[i,de_axis]-stats['response'][i,de_axis]))),
                              'generated_log2fc_mae_cpm_gt5':float(np.mean(np.abs(generated_lfc[i,de_axis]-stats['response'][i,de_axis]))),
                              'projection_log2fc_mae_cpm_gt5':float(np.mean(np.abs(projected_lfc[de_axis]-response[i,de_axis]))),
                              'emission_log2fc_mae_cpm_gt5':float(np.mean(np.abs(generated_lfc[i,de_axis]-projected_lfc[de_axis])))})
        np.savez_compressed(directory/'generated-response.npz',targets=targets,genes=np.array(self.data.genes)[axis],
                            mean_cpm=generated_means,log2fc=generated_lfc)
        pd.DataFrame(auxiliary).to_parquet(directory/'target-support.parquet',index=False)
        result={'score':score,'components':components,'context':context,'training_contexts':sorted(training_contexts),
                'round':iteration,'kind':kind,'targets':len(targets),'seen_targets':sum(t in seen for t in targets),
                'prediction_seed':self.config['prediction_seed'],'reference':str(bundle),
                'prediction_target':self.config['prediction_target'],
                'evaluation_protocol':self.protocol(),
                'panel':'all usable context targets','readout_axis':'context-native measured genes',
                'scaled_score_scope':'local panel only; not leaderboard-equivalent'}
        self.tracked.log({f'eval/{name}/{context}/{kind}/round':iteration,f'eval/{name}/{context}/{kind}/overall':score,
                          **{f'eval/{name}/{context}/{kind}/{k}':v for k,v in components.items()}})
        event('official_score',**result)
        del prediction,counts,real;gc.collect()
        for name in ['data.bin','indices.bin','indptr.npy']:release(matrix_path/name)
        if keep:
            matrix_path.replace(directory/'predicted-counts')
            result['counts_sha256']=digest(directory/'predicted-counts/matrix.json')
        write_json(done,result)
        return result

    def shared(self,context,training_contexts):
        target_stats=self.data.contexts[context];axis=target_stats['gene_indices']
        result=np.zeros_like(target_stats['response']);counts=np.zeros_like(result)
        rows={str(t):i for i,t in enumerate(target_stats['targets'])}
        positions={int(g):i for i,g in enumerate(axis)}
        for source in training_contexts:
            stats=self.data.contexts[source]
            overlap=[(j,positions[int(g)]) for j,g in enumerate(stats['gene_indices']) if int(g) in positions]
            src,dst=np.asarray(overlap,dtype=int).T
            for j,target in enumerate(stats['targets']):
                if str(target) in rows:
                    result[rows[str(target)],dst]+=stats['response'][j,src];counts[rows[str(target)],dst]+=1
        return np.divide(result,counts,out=np.zeros_like(result),where=counts>0)
