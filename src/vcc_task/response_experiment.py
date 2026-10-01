"""One registered fit, full downstream prediction diagnostics, then official scores."""
import gc
import importlib.metadata
import json
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import polars as pl
from scipy import sparse
import torch
from analyze_submitted_counts import describe, clean, row_hashes
from research import execution_metadata

from .common import ROOT, NTC, ref, verified, write_json, stable_seed
from .counts import profile, validate_counts
from .count_store import CountWriter
from .response_data import source_tasks, functional_features, feature_rows, shared_responses, decode_composition, split_config, center_numpy
from .response_models import fit_ridge, fit_mlp, save_state, ResponseMLP
from .population_emission import fit_population, draw_population, state_record
from .prediction_diagnostics import diagnose_panel, read_group
from .frozen_scoring import score_frozen
from .score_support import metric_support
from .capability import pearson
from .official import annotated, de_table, de_summary


def fitted_predictor(job):
    cfg=job.config
    axis,tasks=source_tasks(cfg)
    labels,shared,mass,sources,template,observed=shared_responses(axis,tasks)
    if cfg['response_gauge']!='zero_mean_over_source_observed_readouts_excluding_own':
        raise ValueError('Unknown response gauge')
    template[observed]-=template[observed].mean()
    descriptor_path=job.output/'cache/functional-descriptors.npz'
    if descriptor_path.exists():
        with np.load(descriptor_path) as data:
            features={key:data[key] for key in data.files}
        lookup={str(t):i for i,t in enumerate(features['symbols'])}
    else:
        lookup,features=functional_features(cfg,tasks)
        temporary=descriptor_path.with_suffix('.partial.npz')
        np.savez_compressed(temporary,**features); temporary.replace(descriptor_path)
    state_path=job.output/'checkpoints/training-state.pt'
    model_path=job.output/'checkpoints/model.pt'
    mode=cfg['model']['kind']
    state=None
    fitted=None
    if model_path.exists():
        stored=torch.load(model_path,map_location='cpu',weights_only=False)
        if stored['research']!=job.research:
            raise ValueError('Fitted model has a different research identity')
        verified(stored['descriptor_ref'])
        if labels!=stored['shared_labels']:
            raise ValueError('Frozen source target identities changed')
        if stored['shared_values'] is not None:
            shared=stored['shared_values']
        mass=np.unpackbits(stored['shared_support_bits'],count=stored['shared_support_size']).reshape(len(labels),len(axis))
        sources,template,observed=stored['source_contexts'],stored['source_template'],stored['observed_readouts']
        state=stored['model_state']
    elif mode=='ridge':
        state=fit_ridge(tasks,lookup,features['scores'],len(axis),cfg['ridge'],state_path,job.run)
        state={k:v.cpu() if torch.is_tensor(v) else v for k,v in state.items()}
    elif mode=='mlp':
        fitted,state=fit_mlp(tasks,lookup,features['scores'],len(axis),cfg['mlp'],state_path,job.run,cfg['seed'])
        state={'model':{k:v.cpu() for k,v in fitted.state_dict().items()},'epoch':state['epoch'],
               'complete':state['complete'],'training_loss':state['training_loss']}
    else:
        state={'complete':True,'kind':mode,'independent_source_estimation':True}
    if mode=='mlp':
        fitted=ResponseMLP(features['scores'].shape[1],len(axis),cfg['mlp']['hidden'])
        fitted.load_state_dict(state['model']); fitted.eval()
    if not model_path.exists():
        save_state(model_path,{'research':job.research,'axis':axis,'model_state':state,
            'mode':mode,'shared_labels':labels,'shared_values':shared if mode in ('shared','knn') else None,
            'shared_support_bits':np.packbits(mass>0),'shared_support_size':mass.size,
            'source_contexts':sources,'source_template':template,'observed_readouts':observed,
            'source_gene_positions':{t['context']:t['positions'] for t in tasks},
            'inference_descriptors':{k:v for k,v in features.items() if k!='raw' or mode=='knn'},
            'descriptor_ref':ref(descriptor_path),'prior_ref':cfg['prior_ref'],
            'decoder':cfg['decoder'],'response_gauge':cfg['response_gauge'],
            'source_task_counts':{t['context']:len(t['labels']) for t in tasks}})
    mass=mass>0
    if mode not in ('shared','knn'):
        shared=None
    canonical_scores=np.empty_like(features['scores'])
    canonical_scores[features['assignment']]=features['scores']
    radius=np.sqrt(np.mean(canonical_scores**2/np.maximum(features['pca_variance'],1e-12),axis=1))
    source_indices=np.asarray([lookup[t] for t in features['source_covered_targets']])
    radius_p95=float(np.quantile(radius[source_indices],cfg['diagnostics']['prior_support_strata']['source_radius_quantile']))
    training_summary={'contexts':{t['context']:{'targets':len(t['labels']),
        'cells':int(t['cells'].sum()),'readouts':len(t['positions']),
        'estimable_dose_targets':len(t['dose']),'dose_used_in_loss':False} for t in tasks},
        'unique_targets':len(labels),'observed_readouts':int(observed.sum()),'official_readouts':len(axis),
        'prior_covered_training_targets':int(sum(t in lookup for t in labels)),
        'prior_missing_training_targets':int(sum(t not in lookup for t in labels)),
        'missing_training_descriptor_rule':'zero PCA coordinates, response retained in loss; inference uses source template',
        'pca_source_targets':len(features['source_covered_targets']),
        'scikit_learn_version':importlib.metadata.version('scikit-learn'),
        'pca_variance_explained':float(features['pca_variance_ratio'].sum()),
        'shuffled_prior':cfg['model']['shuffled_prior'],
        'descriptor_identity_preserved_fraction':float(np.mean(features['assignment']==np.arange(len(features['assignment'])))),
        'descriptor_shuffle_blocks':int(len(np.unique(features['shuffle_block']))),
        'prior_missingness_patterns':int(len(np.unique(features['missingness_group']))),
        'shuffle_preserves_each_target_finite_mask':bool(np.array_equal(features['missingness_group'],features['missingness_group'][features['assignment']])),
        'source_prior_whitened_pca_radius_quantiles':np.quantile(radius[source_indices],[0,.25,.5,.75,.95,1]).tolist(),
        'source_prior_radius_p95':radius_p95,
        'prior_radius_scope':'Canonical gene descriptors, regardless of shuffle; PCA axes/scales fitted only on source training targets; empirical support diagnostic, not a formal OOD test',
        'optimization':{k:v for k,v in state.items() if not torch.is_tensor(v) and k not in ('model',)}}
    for scope in ('source','evaluation'):
        selected=np.asarray([lookup[t] for t in features[scope+'_covered_targets']],dtype=np.int64)
        training_summary[scope+'_descriptor_identity_preserved_fraction']=float(
            np.mean(features['assignment'][selected]==selected)) if len(selected) else None
    write_json(job.output/'diagnostics/training-summary.json',training_summary)
    label_lookup={t:i for i,t in enumerate(labels)}
    if mode=='knn':
        valid_donors=[t for t in labels if t in lookup]
        donor_indices=np.asarray([label_lookup[t] for t in valid_donors])
        donor_features=features['raw'][[lookup[t] for t in valid_donors]]
    source_dose={target:{task['context']:task['dose'][target] for task in tasks if target in task['dose']}
                 for target in labels}
    gauge_lookup={str(axis[p]):i for i,p in enumerate(np.flatnonzero(observed))}

    def predict(targets,measured_positions):
        covered=np.asarray([t in lookup for t in targets])
        values=np.zeros((len(targets),len(axis)),dtype=np.float32)
        if mode=='shared':
            for i,t in enumerate(targets):
                if t in label_lookup:
                    values[i]=shared[label_lookup[t]]
        elif mode=='template':
            values[:]=template
        elif mode=='knn':
            for i,t in enumerate(targets):
                if t not in lookup:
                    values[i]=template; continue
                cosine=donor_features@features['raw'][lookup[t]]
                selected=np.argsort(-cosine,kind='stable')[:cfg['model']['neighbors']]
                rows=donor_indices[selected]
                observed_neighbor=mass[rows]>0
                values[i]=np.divide((shared[rows]*observed_neighbor).sum(0),observed_neighbor.sum(0),
                    out=template.copy(),where=observed_neighbor.sum(0)>0)
        elif mode=='ridge':
            x=feature_rows(targets,lookup,features['scores'])
            x=np.column_stack((x,np.ones(len(x),dtype=np.float32)))
            values=x@state['weights'].numpy()
            values[~covered]=template
        elif mode=='mlp':
            x=feature_rows(targets,lookup,features['scores'])
            with torch.no_grad():
                values=np.concatenate([fitted(torch.from_numpy(x[i:i+256])).numpy() for i in range(0,len(x),256)])
            values[~covered]=template
        else:
            raise ValueError('Unregistered response model')
        # The measured training loss cannot identify a common log-response shift.
        # Fix it before assigning neutral zero latent values to untrained genes.
        own=np.asarray([gauge_lookup.get(str(target),-1) for target in targets])
        if mode=='shared':
            for i,target in enumerate(targets):
                if target in label_lookup:
                    direct=mass[label_lookup[target]]
                    if direct.any():
                        values[i,direct]-=values[i,direct].mean()
                    values[i,~direct]=0
        else:
            values[:,observed]=center_numpy(values[:,observed],own)
        values[:,~observed]=0
        if not np.isfinite(values).all():
            raise ValueError('Nonfinite learned response')
        coverage={}
        for i,t in enumerate(targets):
            direct=mass[label_lookup[t]]>0 if t in label_lookup else np.zeros(len(axis),dtype=bool)
            coverage[t]={'seen_target':t in label_lookup,'prior_available':t in lookup,
                'descriptor_identity_preserved':bool(features['assignment'][lookup[t]]==lookup[t]) if t in lookup else None,
                'prior_observed_cell_line_fraction':float(features['observed_fraction'][lookup[t]]) if t in lookup else None,
                'prior_canonical_mean_dependency':float(features['canonical_dependency_mean'][lookup[t]]) if t in lookup else None,
                'prior_canonical_whitened_radius':float(radius[lookup[t]]) if t in lookup else None,
                'prior_radius_above_source_p95':bool(radius[lookup[t]]>radius_p95) if t in lookup else False,
                'direct_source_contexts':sources.get(t,[]),'direct_supervised_readouts':int(direct.sum()),
                'globally_supervised_readouts':int(observed.sum()),
                'direct_measured_supervised_readouts':int(direct[measured_positions].sum()),
                'measured_untrained_readouts':int((~observed[measured_positions]).sum()),
                'source_estimable_dose_json':json.dumps(source_dose.get(t,{}),sort_keys=True),
                'zero_learned_response':bool(np.count_nonzero(values[i])==0),
                'functional_extrapolation':bool(t not in label_lookup and covered[i] and mode in ('ridge','mlp','knn')),
                'missing_prior_template_fallback':bool(not covered[i] and mode in ('ridge','mlp','knn')),
                'fixed_on_target_rule':cfg['decoder']['own_residual']}
        direct_masks=np.asarray([mass[label_lookup[t]]>0 if t in label_lookup else np.zeros(len(axis),dtype=bool)
                                  for t in targets])
        return values,coverage,direct_masks,observed
    del tasks
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return axis,predict,template,training_summary,model_path


def emitter_null(job,context,ntc,real,covariance_positions):
    directory=job.output/'diagnostics'/context/'null'
    marker=directory/'result.json'
    if marker.exists():
        result=json.loads(marker.read_text())
        for item in result['files']:
            verified(item)
        return result
    directory.mkdir(parents=True,exist_ok=True)
    reference=ad.read_h5ad(real,backed='r')
    control=read_group(reference.X,np.flatnonzero(reference.obs.target_gene.eq(NTC))).tocsr()
    reference.file.close()
    half=control.shape[0]//2
    if control.shape[0]-half<job.config['prediction_cells']:
        raise ValueError('Insufficient disjoint NTC null support')
    blocks=[]; labels=[]; checks=[]; distribution_rows=[]
    ntc_x=sparse.csr_matrix(ntc.X)
    baseline=profile(ntc_x)
    desired,_=decode_composition(baseline,np.zeros(ntc.n_vars),own=-1,**job.config['decoder'])
    if not np.array_equal(desired,baseline):
        raise ValueError('Zero-response decoder identity failed')
    zero=fit_population(ntc_x,desired-baseline,**job.config['emitter'])
    null_reference=control[:half].astype(np.float64)
    _,null_moments=describe(null_reference)
    null_hashes=set(row_hashes(null_reference))
    z=null_reference[:,covariance_positions].toarray()
    z=np.log1p(z*(10_000/null_moments['depth'])[:,None])
    null_moments['cov']=np.cov(z,rowvar=False)
    for repeat in range(job.config['diagnostics']['null_repeats']):
        seed=stable_seed(job.config['seed'],context,'model-null',repeat)
        cells=job.config['prediction_cells']
        emitted=draw_population(ntc_x,zero,seed,cells)
        original=ntc_x[np.random.default_rng(seed).integers(0,ntc.n_obs,cells)]
        true_null=control[half:][np.random.default_rng(seed).choice(control.shape[0]-half,cells,replace=False)]
        checks.append((emitted!=original).nnz==0)
        for name,matrix in [('zero',emitted),('input',original),('real_null',true_null)]:
            validate_counts(matrix); blocks.append(matrix); labels.extend([f'{name}_{repeat}']*matrix.shape[0])
            summary,moments=describe(matrix.astype(np.float64),null_moments,covariance_positions,null_hashes)
            summary.update(kind=name,repeat=repeat,seed=seed,reference_cells=half,
                bulk_response_rms=float(np.sqrt(np.mean((profile(matrix)-profile(null_reference))**2))),
                mean_log_response_rms=float(np.sqrt(np.mean((moments['logmean']-null_moments['logmean'])**2))))
            distribution_rows.append(clean(summary))
    if not all(checks):
        raise ValueError('Zero-response emitter identity failed')
    matrix=sparse.vstack(blocks,format='csr')
    counts=directory/'counts.h5ad'; annotated(matrix,ntc.var_names,labels,context+'-null').write_h5ad(counts,compression='gzip')
    write_json(directory/'counts-checked.json',{'hard_constraints_passed':True,'zero_identity':True,'prediction_ref':ref(counts)})
    table,backend=de_table(matrix,control[:half],ntc.var_names,labels,job.config['runtime']['num_threads'])
    de_path=directory/'de.parquet'; table.write_parquet(de_path)
    summary=de_summary(table)
    distribution_path=directory/'distribution-statistics.parquet'
    pd.DataFrame(distribution_rows).to_parquet(distribution_path,index=False)
    result={'zero_identity':True,'zero_decoder_identity':True,'official_backend':backend,'de_summary':summary,
        'distribution_reference':'First half of disjoint scoring NTC; real_null drawn without replacement from second half; three repeats are sampling diagnostics, not biological replicates',
        'covariance_gene_positions':list(covariance_positions),'distribution_statistics_ref':ref(distribution_path),
        'added_zero_de':[summary[f'zero_{r}']['significant_genes']-summary[f'input_{r}']['significant_genes']
                         for r in range(job.config['diagnostics']['null_repeats'])],
        'files':[ref(counts),ref(de_path),ref(directory/'counts-checked.json'),ref(distribution_path)]}
    write_json(marker,result)
    return result


def diagnostic_identity(job,context,reference,generated,checkpoint):
    """Guard all diagnostic reuse by exact inputs, checkpoint, code and config."""
    directory=job.output/'diagnostics'/context
    directory.mkdir(parents=True,exist_ok=True)
    identity={'context':context,'panel_ref':reference,
        'prediction_ref':generated['prediction_ref'],'response_ref':generated['response_ref'],
        'checkpoint_ref':ref(checkpoint),'config_ref':ref(job.config_path),
        'code_refs':execution_metadata(ROOT,job.config['run_id'])['code_refs']}
    path=directory/'stage-identity.json'
    if path.exists():
        if json.loads(path.read_text())!=identity:
            raise ValueError('Diagnostic inputs/code/config changed; existing diagnostics cannot be reused')
    elif (directory/'diagnostics.json').exists():
        raise ValueError('Existing diagnostics have no complete provenance guard')
    else:
        write_json(path,identity)
    return path


def finish_diagnostics(job,context,generated,diagnostic,null,readout_ref,identity_path):
    """Record local matrix capacity separately from submission archive limits."""
    with h5py.File(verified(generated['prediction_ref'])) as handle:
        matrix=handle['X']; rows,columns=map(int,matrix.attrs['shape'])
        stored=len(matrix['data']); pointer=matrix['indptr']
        if (stored>rows*columns or len(matrix['indices'])!=stored or len(pointer)!=rows+1
                or int(pointer[-1])!=stored or pointer.dtype.itemsize<8):
            raise ValueError('Invalid sparse storage size/pointer capacity')
        detected=np.diff(pointer[:])
    if not (diagnostic['hard_constraints_passed'] and null['zero_identity'] and null['zero_decoder_identity']):
        raise ValueError('Incomplete count/null acceptance before official scoring')
    directory=job.output/'diagnostics'/context
    target_rows=pd.read_parquet(directory/'per-target.parquet')
    # CountWriter emits input NTC first, then sorted, contiguous target groups.
    offset=rows-int(target_rows.n.sum()); detected_rows=[]
    for row in target_rows.itertuples():
        selected=detected[offset:offset+row.n]
        if len(selected)!=row.n:
            raise ValueError('Diagnostic group offsets differ from generated counts')
        detected_rows.append({'target':row.target,'detected_mean':float(selected.mean()),
            'detected_cv':float(selected.std(ddof=1)/selected.mean()),'cells':row.n})
        offset+=row.n
    if offset!=rows:
        raise ValueError('Diagnostic cell coverage incomplete')
    detected_path=directory/'detected-cv.parquet'
    pd.DataFrame(detected_rows).to_parquet(detected_path,index=False)
    derived_path=directory/'derived-gene-moments.h5'
    with h5py.File(directory/'gene-moments.h5','r') as source, h5py.File(derived_path,'w') as output:
        for name in ('genes','targets'):
            source.copy(name,output)
        shape=source['mean'].shape
        datasets={name:output.create_dataset(name,shape,dtype='f4',chunks=(1,columns),compression='gzip',compression_opts=1)
            for name in ('raw_fano','mean_log1p_CP10k_response','mean_CP10k_log2FC_pc0p01')}
        ntc_cp=source['ntc_cpmean'][:]; ntc_log=source['ntc_logmean'][:]
        for i in range(shape[0]):
            mean=source['mean'][i]; var=source['var'][i]
            datasets['raw_fano'][i]=np.divide(var,mean,out=np.full_like(mean,np.nan),where=mean>0)
            datasets['mean_log1p_CP10k_response'][i]=source['logmean'][i]-ntc_log
            datasets['mean_CP10k_log2FC_pc0p01'][i]=np.log2((source['cpmean'][i]+.01)/(ntc_cp+.01))
        output.attrs['raw_fano_undefined']='NaN where raw mean is zero; no variance-to-mean interpretation there'
        output.attrs['normalization']='CP10k uses each cell library; bulk log1p CP50K remains separately stored in gene-moments.h5'
    result={'status':'completed','hard_constraints_passed':True,'identity_ref':ref(identity_path),
        'rows':rows,'measured_columns':columns,'stored_elements':stored,
        'storage_limit':rows*columns,'storage_limit_basis':'Canonical local CSR logical capacity with int64 pointers; variable-size local panel is not a competition submission archive',
        'official_submission_nnz_limit_applicable':False,
        'stage_order':'prediction frozen; counts, null distribution/DE, residual/pathway and readout diagnostics complete; official scoring next',
        'files':[ref(directory/'diagnostics.json'),ref(directory/'null/result.json'),readout_ref,
            ref(detected_path),ref(derived_path)]}
    path=directory/'pre-score-completed.json'; write_json(path,result)
    job.run.log({f'diagnostics/{context}/completed':1,
        f'diagnostics/{context}/stored_elements':stored,
        f'diagnostics/{context}/added_zero_de_max':max(null['added_zero_de'])})
    return path


def count_predictions(job,context,axis,targets,ntc,latent,coverage,direct_masks,observed,template,checkpoint):
    directory=job.output/'predictions'/context
    directory.mkdir(parents=True,exist_ok=True)
    marker=directory/'generated.json'
    if marker.exists():
        result=json.loads(marker.read_text())
        for item in result['files']:
            verified(item)
        return result
    cfg=job.config
    lookup={g:i for i,g in enumerate(axis)}
    positions=np.asarray([lookup[g] for g in ntc.var_names])
    ntc_x=sparse.csr_matrix(ntc.X,dtype=np.float64)
    baseline=profile(ntc_x)
    latent_path=directory/'full-axis-response.npz'
    np.savez_compressed(latent_path,genes=axis.astype(str),targets=np.asarray(targets),response=latent,
                        measured_positions=positions,source_template=template,
                        direct_supervision=direct_masks,global_supervision=observed)
    writer=CountWriter(directory/'counts.h5ad',ntc.var.copy())
    writer.append(ntc_x)
    observations=[pd.DataFrame({'target_gene':NTC},index=ntc.obs_names)]
    emission=[]
    repeat_blocks=[]; repeat_labels=[]; repeat_diagnostics=[]
    repeat_targets=sorted(targets,key=lambda t:stable_seed(cfg['seed'],context,'repeat-target',t))[:cfg['diagnostics']['repeat_targets']]
    projected_template,_=decode_composition(baseline,template[positions],own=-1,**cfg['decoder'])
    for i,target in enumerate(targets):
        own=np.flatnonzero(ntc.var_names==target)
        desired,detail=decode_composition(baseline,latent[i,positions],own=int(own[0]) if own.size else -1,**cfg['decoder'])
        fitted=fit_population(ntc_x,desired-baseline,**cfg['emitter'])
        seed=stable_seed(cfg['seed'],cfg['fit_scope']['outer_split'],context,target,'prediction')
        counts=draw_population(ntc_x,fitted,seed,cfg['prediction_cells'])
        writer.append(counts)
        observations.append(pd.DataFrame({'target_gene':target},index=[f'{context}-{target}-{r}' for r in range(cfg['prediction_cells'])]))
        detail.update(target=target,seed=seed,**state_record(fitted))
        detail['sample_bulk_rmse']=float(np.sqrt(np.mean((profile(counts)-desired)**2)))
        emission.append(detail)
        if target in repeat_targets:
            means=[]; variances=[]
            for repeat in range(cfg['diagnostics']['generation_repeats']):
                repeated=draw_population(ntc_x,fitted,stable_seed(seed,'repeat',repeat),cfg['prediction_cells'])
                x=repeated.astype(np.float64)
                x=sparse.diags(10_000/np.asarray(x.sum(1)).ravel())@x
                mean=np.asarray(x.mean(0)).ravel()
                variance=np.maximum(np.asarray(x.power(2).sum(0)).ravel()-repeated.shape[0]*mean**2,0)/(cfg['prediction_cells']-1)
                means.append(mean); variances.append(variance)
                repeat_blocks.append(repeated); repeat_labels.extend([f'{target}__repeat_{repeat}']*cfg['prediction_cells'])
            expected=np.mean(variances,axis=0)/cfg['prediction_cells']; observed_variance=np.var(means,axis=0,ddof=1)
            allowed=(expected>1e-10)&~np.isin(ntc.var_names,targets)
            repeat_diagnostics.append({'target':target,'repeats':cfg['diagnostics']['generation_repeats'],
                'variance_ratio':float(observed_variance[allowed].sum()/expected[allowed].sum()),'valid_genes':int(allowed.sum())})
        if (i+1)%100==0:
            print(f'{context} generated {i+1}/{len(targets)} targets',flush=True)
    obs=pd.concat(observations); obs['target_gene']=obs.target_gene.astype('category')
    prediction=writer.finish(obs)
    repeat_path=directory/'independent-repeats.h5ad'
    annotated(sparse.vstack(repeat_blocks,format='csr'),ntc.var_names,repeat_labels,context+'-repeat').write_h5ad(repeat_path,compression='gzip')
    write_json(directory/'emission.json',{'records':emission,'independent_repeats':repeat_diagnostics,
        'source_coverage':coverage,'decoder':cfg['decoder'],'template_bulk':(projected_template-baseline).tolist()})
    result={'prediction_ref':ref(prediction),'response_ref':ref(latent_path),'checkpoint_ref':ref(checkpoint),
        'source_coverage':coverage,'template_bulk':(projected_template-baseline).tolist(),
        'files':[ref(prediction),ref(latent_path),ref(repeat_path),ref(directory/'emission.json')]}
    write_json(marker,result)
    return result


def de_diagnostics(score_directory):
    """Linear-time joins/grouping over official DE tables, not a substitute scorer."""
    r=pl.scan_parquet(score_directory/'de_real.parquet').select('target','feature',
        pl.col('p_adj').alias('p_real'),pl.col('log2_fold_change').alias('lfc_real'))
    p=pl.scan_parquet(score_directory/'de_pred.parquet').select('target','feature',
        pl.col('p_adj').alias('p_pred'),pl.col('log2_fold_change').alias('lfc_pred'))
    both=r.join(p,on=['target','feature'],how='full',coalesce=True).filter(pl.col('target')!=pl.col('feature'))
    rs=(pl.col('p_real')<.05).fill_null(False); ps=(pl.col('p_pred')<.05).fill_null(False)
    valid=pl.col('lfc_real').is_finite().fill_null(False)&(pl.col('lfc_real')!=0)
    same=pl.col('lfc_real').sign()==pl.col('lfc_pred').sign()
    summary=both.group_by('target').agg(rs.sum().alias('real_significant'),ps.sum().alias('pred_significant'),
        (rs&ps).sum().alias('overlap'),(ps&~rs).sum().alias('false_positive_calls'),
        (rs&~ps).sum().alias('missed_calls'),(ps&valid).sum().alias('pred_adjudicable'),
        (rs&pl.col('p_pred').is_null()).sum().alias('reference_significant_absent_from_pred_table'),
        (ps&valid&same).sum().alias('correct_direction')).collect(engine='streaming')
    path=score_directory/'de-call-diagnostics.parquet'; summary.write_parquet(path)
    return ref(path)


def readout_diagnostics(directory,generated,ntc,real,targets):
    with np.load(verified(generated['response_ref'])) as latent:
        positions=latent['measured_positions']
        direct=latent['direct_supervision'][:,positions]
        globally=latent['global_supervision'][positions]
    with np.load(directory/'responses.npz') as data:
        prediction=data['prediction'].astype(float)
        truth=data['real'].astype(float)
        genes=data['genes']
        if list(data['targets'])!=list(targets):
            raise ValueError('Readout diagnostic target order mismatch')
    reference=ad.read_h5ad(real,backed='r')
    score_origin=profile(read_group(reference.X,np.flatnonzero(reference.obs.target_gene.eq(NTC))))
    reference.file.close()
    prediction+=profile(ntc.X)-score_origin
    allowed=~np.isin(genes,targets)
    pred_mean=prediction[:,allowed].mean(0)
    true_mean=truth[:,allowed].mean(0)
    # These label-derived quantities are EVALUATION ONLY. Predictions/checkpoint
    # are already frozen, and none of these values reaches the predictor.
    norm=np.linalg.norm(true_mean)
    direction=true_mean/norm if norm>1e-12 else np.zeros_like(true_mean)
    rows=[]
    for i,target in enumerate(targets):
        entry={'target':target,'real_response_rms':float(np.sqrt(np.mean(truth[i,allowed]**2)))}
        pc=prediction[i,allowed]-pred_mean
        rc=truth[i,allowed]-true_mean
        pr=pc-(pc@direction)*direction
        rr=rc-(rc@direction)*direction
        entry.update(centroid_removed_pearson=pearson(pc,rc),
            centroid_removed_rmse=float(np.sqrt(np.mean((pc-rc)**2))),
            evaluator_only_target_template_residual_pearson=pearson(pr,rr),
            evaluator_only_target_template_residual_rmse=float(np.sqrt(np.mean((pr-rr)**2))))
        total_energy=float(np.square(truth[i,allowed]).sum())
        for name,mask in [('direct',direct[i]&allowed),('indirect',(~direct[i])&globally&allowed),
                          ('untrained',(~globally)&allowed)]:
            entry[name+'_genes']=int(mask.sum())
            entry[name+'_response_rmse']=float(np.sqrt(np.mean((prediction[i,mask]-truth[i,mask])**2))) if mask.any() else None
            entry[name+'_observed_real_response_rms']=float(np.sqrt(np.mean(truth[i,mask]**2))) if mask.any() else None
            entry[name+'_observed_real_energy_fraction']=float(np.square(truth[i,mask]).sum()/total_energy) if total_energy else None
            entry[name+'_response_pearson']=pearson(prediction[i,mask],truth[i,mask]) if mask.any() else None
        rows.append(entry)
    result=pd.DataFrame(rows)
    path=directory/'readout-coverage.parquet'; result.to_parquet(path,index=False)
    q25,q75=np.quantile(result.real_response_rms,[.25,.75])
    centering={'scope':'Evaluator only: independently remove prediction/real across-target centroids, then project both off the REAL mean response direction; label-derived direction never used in fitting/generation',
        'real_mean_direction_defined':bool(norm>1e-12),
        'mean_centroid_removed_pearson':float(result.centroid_removed_pearson.mean()),
        'mean_target_template_residual_pearson':float(result.evaluator_only_target_template_residual_pearson.mean()),
        'mean_target_template_residual_rmse':float(result.evaluator_only_target_template_residual_rmse.mean())}
    for key,value in list(centering.items()):
        if isinstance(value,float) and not np.isfinite(value): centering[key]=None
    return ref(path),{'weak_observed_response_quartile':result.loc[result.real_response_rms<=q25,'target'].tolist(),
                     'strong_observed_response_quartile':result.loc[result.real_response_rms>=q75,'target'].tolist()},centering


def run_response(job):
    cfg=job.config
    split=split_config(cfg)
    torch.set_num_threads(cfg['runtime']['num_threads'])
    torch.set_float32_matmul_precision('highest')
    axis,predict,template,training,checkpoint=fitted_predictor(job)
    panels={}; diagnostics=[]
    for context,reference in cfg['evaluation_refs'].items():
        panel_record=json.loads(verified(reference).read_text())
        real=verified(panel_record['real_ref'])
        ntc=ad.read_h5ad(verified(panel_record['input_ntc_ref']))
        targets=sorted(split['evaluation_targets'][context])
        axis_index={g:i for i,g in enumerate(axis)}
        measured_positions=np.asarray([axis_index[g] for g in ntc.var_names])
        latent,coverage,direct_masks,observed=predict(targets,measured_positions)
        generated=count_predictions(job,context,axis,targets,ntc,latent,coverage,direct_masks,observed,template,checkpoint)
        prediction=verified(generated['prediction_ref'])
        directory=job.output/'diagnostics'/context
        identity_path=diagnostic_identity(job,context,reference,generated,checkpoint)
        diagnostic=diagnose_panel(prediction,ntc,directory,seed=cfg['seed'],checkpoint_ref=ref(checkpoint),
            expected_targets=targets,expected_cells=cfg['prediction_cells'],source_coverage=coverage,
            template=np.asarray(generated['template_bulk']),real=real,pathway_ref=cfg['pathway_ref'],
            covariance_genes=cfg['diagnostics']['covariance_genes'])
        null=emitter_null(job,context,ntc,real,diagnostic['covariance_gene_positions'])
        readout_ref,response_strata,centering=readout_diagnostics(directory,generated,ntc,real,targets)
        completed_diagnostics=finish_diagnostics(job,context,generated,diagnostic,null,readout_ref,identity_path)
        metadata=pd.read_parquet(directory/'per-target.parquet')
        strata={'seen':[t for t in targets if coverage[t]['seen_target']],
            'unseen':[t for t in targets if not coverage[t]['seen_target']],
            'prior_missing':[t for t in targets if not coverage[t]['prior_available']],
            'prior_radius_above_source_p95':[t for t in targets if coverage[t]['prior_radius_above_source_p95']],
            'prior_measured_fraction_below_0p9':[t for t in targets if coverage[t]['prior_available'] and coverage[t]['prior_observed_cell_line_fraction']<cfg['diagnostics']['prior_support_strata']['minimum_observed_fraction']],
            'official':split['official_overlap_targets'][context],
            'n_lt_100':metadata.loc[metadata.real_n<100,'target'].tolist(),
            'n_100_399':metadata.loc[(metadata.real_n>=100)&(metadata.real_n<400),'target'].tolist(),
            'n_ge_400':metadata.loc[metadata.real_n>=400,'target'].tolist()}
        strata.update(response_strata)
        scores=score_frozen(prediction,real,panel_record['bundle'],job.output/'predictions'/context/'score',
                            cfg['runtime'],diagnostic,strata)
        raw_table=pl.read_parquet(job.output/'predictions'/context/'score/raw.parquet')
        support=metric_support(raw_table,scores['raw'])
        valid_counts={name:details['finite_raw_targets'] for name,details in support.items()}
        de_ref=de_diagnostics(job.output/'predictions'/context/'score')
        panels[context]={'scores':scores,'diagnostic_ref':ref(directory/'diagnostics.json'),
            'pre_score_completed_ref':ref(completed_diagnostics),
            'null_ref':ref(directory/'null/result.json'),'generation_ref':ref(job.output/'predictions'/context/'generated.json'),
            'de_diagnostics_ref':de_ref,'measured_genes':ntc.n_vars,'targets':len(targets),
            'valid_targets_per_metric':valid_counts,
            'metric_support':support,
            'readout_diagnostics_ref':readout_ref,
            'evaluator_only_centering':centering,
            'residual_mean_pearson':diagnostic['decomposition']['mean_residual_pearson'],
            'residual_mean_rmse':diagnostic['decomposition']['mean_residual_rmse']}
        diagnostics.extend([directory/'diagnostics.json',directory/'null/result.json',identity_path,
            completed_diagnostics,directory/'null/distribution-statistics.parquet'])
        logged={f'evaluation/{context}/raw/{k}':v for k,v in scores['raw'].items() if v is not None}
        logged.update({f'evaluation/{context}/normalized/{k}':v for k,v in (scores['normalized'] or {}).items() if v is not None})
        if scores['Overall'] is not None:
            logged[f'evaluation/{context}/Overall']=scores['Overall']
        job.run.log(logged)
        del ntc,latent; gc.collect()
    manifest=job.output/'predictions/manifest.json'; write_json(manifest,panels)
    raw_pds=[p['scores']['raw']['pds_cosine'] for p in panels.values()]
    defined=[p['scores']['Overall'] for p in panels.values() if p['scores']['Overall'] is not None]
    job.complete({'summary':{'raw_pds':float(np.mean(raw_pds)),
        'mean_Overall':float(np.mean(defined)) if defined else None,'defined_Overall_panels':len(defined)},
        'training':training,'panels':panels,'diagnostics_completed':True,
        'submission_decision':'defer_until_matched_batch_comparison_complete'},checkpoint,manifest,diagnostics)
