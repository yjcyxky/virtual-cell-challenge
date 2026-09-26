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
from cell_eval2 import EvalConfig,compute_metrics,aggregate_metrics_wide,score_metrics
from cell_eval2.baseline import build_run_meta,generic_response_profile
from cell_eval2.run import metric_output_names
from cell_eval2.real_bundle import build_real_bundle,read_real_bundle
from cell_eval2.competition import competition_members
from common import digest,event,write_json
from generation import Generator
from training import predict,fit_id
from data import NTC

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
    # The official anchor makes two copied half-panels and temporary DE tables.
    # Unified CPU/GPU memory is shared with other active experiments on this host.
    # Admission delays execution; it never changes the panel or metric definition.
    groups=real.obs.target.nunique()
    required=(2*real.X.nbytes if anchor else groups*real.n_vars*8*10)+(12<<30)
    gpu_required=groups*real.n_vars*8*12+(2<<30)
    while True:
        available=int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines()
                           if line.startswith('MemAvailable:')))*1024
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
        path=directory/'reference.npy'
        if not (directory/'reference.json').exists():
            source=self.data.counts(context)
            values=np.lib.format.open_memmap(path,mode='w+',dtype=np.uint16,shape=(len(ids),len(var)))
            for start in range(0,len(ids),256): values[start:start+256]=source[ids[start:start+256]]
            values.flush();del values,source
            write_json(directory/'reference.json',{'shape':[len(ids),len(var)],'sha256':digest(path),
                       'targets':'all usable targets, no overlap restriction',
                       'NTC':'entire context NTC pool; same source baseline is available to prediction',
                       'perturbation_cells':f'fixed stratified sample, at most {self.config["reference_cells_per_target"]} per target; no replacement',
                       'scoring':'pinned official vcc2026 method on context-native measured genes, not leaderboard-equivalent'})
        real=ad.AnnData(np.load(path,mmap_mode='r'),obs=obs,var=var)
        bundle=directory/'bundle'
        if not (bundle/'manifest.json').exists():
            if bundle.exists() and any(bundle.iterdir()):
                # An interrupted atomic official build may leave an incomplete directory.
                raise ValueError(f'incomplete_official_bundle_requires_inspection:{bundle}')
            event('official_anchor_start',context=context,shape=real.shape)
            wait_for_memory(real,context,cfg.device,anchor=True)
            profile=generic_response_profile(real,pert_col=cfg.pert_col,control=cfg.control,exclude_target_gene=False)
            baseline_path=directory/'mean-response-baseline.npy'
            baseline=np.lib.format.open_memmap(baseline_path,mode='w+',dtype=np.float32,shape=real.shape)
            for start in range(0,len(real),256): baseline[start:start+256]=profile.values
            controls=np.flatnonzero(obs.target.eq(NTC))
            for start in range(0,len(controls),256): baseline[controls[start:start+256]]=real.X[controls[start:start+256]]
            baseline.flush()
            arm=ad.AnnData(baseline,obs=obs.copy(),var=var.copy())
            build_real_bundle(real,arm,config=cfg,outdir=str(bundle),bundle_id=f'exp006-{context}',
                              base_seed=self.config['replicate_seed'],n_splits=self.config['replicate_splits'])
            del arm,baseline;gc.collect();baseline_path.unlink()
            event('official_anchor_ready',context=context)
        manifest=read_real_bundle(bundle).manifest
        if manifest['rule_digest'] is None: raise ValueError('invalid_competition_bundle')
        return real,cfg,bundle

    def score(self,context,training_contexts,iteration,booster,features,kind='model',keep=False):
        name=fit_id(training_contexts) if training_contexts else 'baseline'
        directory=self.output/'predictions'/name/context/f'{kind}-{iteration:04d}'
        done=directory/'metrics.json'
        if done.exists(): return json.loads(done.read_text())
        directory.mkdir(parents=True,exist_ok=True)
        real,cfg,bundle=self.prepare(context)
        wait_for_memory(real,context,cfg.device)
        stats=self.data.contexts[context];targets=stats['targets'];axis=stats['gene_indices']
        if kind=='zero': response=np.zeros_like(stats['response'])
        elif kind=='shared': response=self.shared(context,training_contexts)
        else: response=predict(booster,features,context,targets,axis)
        np.savez_compressed(directory/'predicted-response.npz',targets=targets,genes=np.array(self.data.genes)[axis],response=response)
        matrix_path=self.output/'cache'/f'prediction-{context}.npy'
        cells_per=self.config['cells_per_prediction']
        perturbation_rows=len(targets)*cells_per
        control_ids=stats['control_rows']
        counts=np.lib.format.open_memmap(matrix_path,mode='w+',dtype=np.uint32,shape=(perturbation_rows+len(control_ids),len(axis)))
        generator=Generator(self.data,context,self.config)
        for i,target in enumerate(targets):
            counts[i*cells_per:(i+1)*cells_per]=generator.generate(str(target),response[i])
            if i%200==0:event('prediction',context=context,model=name,round=iteration,target_index=i,targets=len(targets))
        # Official validate_pair requires equal label sets even with control_source=real.
        # Copy NTC unchanged for structural compatibility; the comparator still uses real NTC.
        source=self.data.counts(context)
        for start in range(0,len(control_ids),256):
            counts[perturbation_rows+start:perturbation_rows+start+256]=source[control_ids[start:start+256]]
        counts.flush();del generator,source
        labels=np.concatenate([np.repeat(targets,cells_per),np.repeat(NTC,len(control_ids))])
        obs=pd.DataFrame({'target':labels},index=[f'prediction-{i}' for i in range(len(counts))])
        prediction=ad.AnnData(counts,obs=obs,var=pd.DataFrame(index=np.array(self.data.genes)[axis]))
        event('official_score_start',context=context,training_contexts=training_contexts,round=iteration,kind=kind)
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
        seen=set().union(*(set(self.data.contexts[c]['targets']) for c in training_contexts)) if training_contexts else set()
        auxiliary=[]
        for i,target in enumerate(targets):
            own=axis!=self.data.lookup[str(target)]
            error=float(np.mean((response[i,own]-stats['response'][i,own])**2))
            auxiliary.append({'target':str(target),'seen_in_training':str(target) in seen,'response_mse':error})
        pd.DataFrame(auxiliary).to_parquet(directory/'target-support.parquet',index=False)
        result={'score':score,'components':components,'context':context,'training_contexts':sorted(training_contexts),
                'round':iteration,'kind':kind,'targets':len(targets),'seen_targets':sum(t in seen for t in targets),
                'prediction_seed':self.config['prediction_seed'],'reference':str(bundle),
                'panel':'all usable context targets','readout_axis':'context-native measured genes',
                'scaled_score_scope':'local panel only; not leaderboard-equivalent'}
        self.tracked.log({f'eval/{name}/{context}/{kind}/round':iteration,f'eval/{name}/{context}/{kind}/overall':score,
                          **{f'eval/{name}/{context}/{kind}/{k}':v for k,v in components.items()}})
        event('official_score',**result)
        del prediction,counts,real,raw;gc.collect();release(matrix_path)
        if keep: matrix_path.replace(directory/'predicted-counts.npy')
        else: matrix_path.unlink()
        if keep: result['counts_sha256']=digest(directory/'predicted-counts.npy')
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
