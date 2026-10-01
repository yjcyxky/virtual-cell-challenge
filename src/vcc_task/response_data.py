"""Source-only functional descriptors and masked compositional response labels."""
import json
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from .common import NTC, verified


def split_config(config):
    splits=json.loads(verified(config['benchmark']['split_manifest']).read_text())
    matches=[s for s in splits if s['id']==config['fit_scope']['outer_split']]
    if len(matches)!=1 or matches[0]['target_partition']!=config['fit_scope']['target_partition']:
        raise ValueError('Outer fit scope differs from the frozen split manifest')
    return matches[0]


def center_numpy(values, own):
    values=np.asarray(values,dtype=np.float64).copy()
    present=own>=0
    rows=np.flatnonzero(present)
    total=values.sum(1)
    total[rows]-=values[rows,own[rows]]
    values-= (total/(values.shape[1]-present))[:,None]
    values[rows,own[rows]]=0
    return values.astype(np.float32)


def source_tasks(config):
    """Construct labels only from allowed source contexts and training targets.

    A compressed source statistics archive contains other rows; row selection
    precedes any response computation, weighting or representation fitting.
    """
    split=split_config(config)
    if set(config['source_observations'])!=set(split['training_contexts']):
        raise ValueError('Source artifact set must exactly match this fit scope')
    axis=pd.read_csv(verified(config['benchmark']['gene_axis'])).gene_name.to_numpy()
    lookup={str(g):i for i,g in enumerate(axis)}
    tasks=[]
    for context in split['training_contexts']:
        item=config['source_observations'][context]
        with np.load(verified(item['statistics_ref'])) as statistics:
            labels=statistics['labels']
            keep=np.isin(labels,split['training_targets'])
            labels=labels[keep]
            positions=statistics['positions']
            baseline=statistics['bulk'][np.flatnonzero(statistics['labels']==NTC)[0]]
            response=statistics['bulk'][keep]-baseline
            index={int(p):i for i,p in enumerate(positions)}
            own=np.asarray([index.get(lookup.get(str(t),-1),-1) for t in labels])
            values=center_numpy(response,own)
            counts=statistics['counts'][keep]
        dose=pd.read_parquet(verified(item['dose_ref']))
        dose=dose.loc[dose.target.isin(labels)]
        reliable=dose.loc[dose.reliability.eq('estimable')]
        dosage={str(t):float(np.average(g.knockdown_fraction_unclipped,weights=g.cells))
                for t,g in reliable.groupby('target',sort=False)}
        tasks.append({'context':context,'labels':labels.astype(str),'positions':positions,
            'own':own,'values':values,'cells':counts,'dose':dosage})
    return axis,tasks


def functional_features(config, tasks):
    """PCA/imputation see only unique source training target descriptors.

    Shuffling AFTER this fit preserves the exact PCA and feature distribution of
    every source context and the evaluation set, and each target's original
    finite-entry mask. These groups use no held-out response measurements.
    """
    frame=pd.read_csv(verified(config['prior_ref']),index_col=0)
    symbols=np.asarray([str(c).rsplit(' (',1)[0] for c in frame.columns])
    unique,count=np.unique(symbols,return_counts=True)
    duplicate=set(unique[count>1])
    keep=np.asarray([s not in duplicate for s in symbols])
    symbols=symbols[keep]
    features=frame.to_numpy(dtype=np.float32)[:,keep].T.copy()
    finite=np.isfinite(features)
    observed_fraction=finite.mean(1)
    if np.any(observed_fraction==0):
        raise ValueError('Entirely unobserved DepMap descriptor cannot count as covered')
    dependency_mean=np.where(finite,features,0).sum(1,dtype=np.float64)/finite.sum(1)
    _,missingness_group=np.unique(np.packbits(finite,axis=1),axis=0,return_inverse=True)
    lookup={str(t):i for i,t in enumerate(symbols)}
    training=sorted(set().union(*(set(t['labels']) for t in tasks)))
    indices=np.asarray([lookup[t] for t in training if t in lookup],dtype=np.int64)
    if len(indices)<2:
        raise ValueError('Insufficient source targets with external functional descriptors')
    medians=np.nanmedian(features[indices],axis=0)
    medians=np.nan_to_num(medians,nan=0,posinf=0,neginf=0)
    features=np.where(np.isfinite(features),features,medians[None,:])
    pca=PCA(n_components=min(config['model']['pca_components'],len(indices)-1,features.shape[1]),
            svd_solver='randomized',random_state=config['seed'])
    pca.fit(features[indices])
    scores=pca.transform(features).astype(np.float32)
    # Raw cosine kNN uses the original imputed descriptor, without response fitting.
    norms=np.linalg.norm(features,axis=1,keepdims=True)
    raw=features/np.maximum(norms,1e-12)
    assignment=np.arange(len(symbols))
    memberships=np.zeros(len(symbols),dtype=np.int64)
    for bit,task in enumerate(tasks):
        for target in task['labels']:
            if target in lookup:
                memberships[lookup[target]] |= 1 << bit
    evaluation=set().union(*map(set,split_config(config)['evaluation_targets'].values()))
    for target in evaluation:
        if target in lookup:
            memberships[lookup[target]] |= 1 << len(tasks)
    _,shuffle_block=np.unique(np.column_stack((memberships,missingness_group)),axis=0,return_inverse=True)
    if config['model']['shuffled_prior']:
        rng=np.random.default_rng(config['seed'])
        for block in np.unique(shuffle_block):
            group=np.flatnonzero(shuffle_block==block)
            assignment[group]=rng.permutation(group)
    if (not np.array_equal(missingness_group,missingness_group[assignment])
            or not np.array_equal(memberships,memberships[assignment])):
        raise ValueError('Descriptor shuffle changed source/evaluation membership or a finite-entry mask')
    record={'symbols':symbols,'scores':scores[assignment], 'raw':raw[assignment],
        'cell_line_ids':frame.index.to_numpy(dtype=str),
        'source_covered_targets':np.asarray([t for t in training if t in lookup]),
        'evaluation_covered_targets':np.asarray(sorted(evaluation.intersection(lookup))),
        'pca_components':pca.components_.astype(np.float32),'pca_mean':pca.mean_.astype(np.float32),
        'pca_variance_ratio':pca.explained_variance_ratio_.astype(np.float32),
        'pca_variance':pca.explained_variance_.astype(np.float32),
        'imputation_median':medians.astype(np.float32),'assignment':assignment,
        'shuffle_membership':memberships,'shuffle_block':shuffle_block,
        'missingness_group':missingness_group,'observed_fraction':observed_fraction,
        'canonical_dependency_mean':dependency_mean,
        'excluded_ambiguous_symbols':np.asarray(sorted(duplicate),dtype=str)}
    return lookup,record


def feature_rows(labels,lookup,features):
    result=np.zeros((len(labels),features.shape[1]),dtype=np.float32)
    for i,target in enumerate(labels):
        if target in lookup:
            result[i]=features[lookup[target]]
    return result


def shared_responses(axis,tasks):
    labels=sorted(set().union(*(set(t['labels']) for t in tasks)))
    lookup={t:i for i,t in enumerate(labels)}
    sums=np.zeros((len(labels),len(axis)),dtype=np.float32)
    mass=np.zeros_like(sums)
    sources={t:[] for t in labels}
    for task in tasks:
        rows=np.asarray([lookup[t] for t in task['labels']])
        positions=task['positions']
        for j,row in enumerate(rows):
            valid=np.ones(len(positions),dtype=bool)
            if task['own'][j]>=0:
                valid[task['own'][j]]=False
            ix=positions[valid]
            sums[row,ix]+=task['values'][j,valid]
            mass[row,ix]+=1
            sources[task['labels'][j]].append(task['context'])
    values=np.divide(sums,mass,out=np.zeros_like(sums),where=mass>0)
    # Context-balanced no-target-information response, with observed-gene masks.
    template=np.zeros(len(axis),dtype=np.float64); support=np.zeros(len(axis))
    for task in tasks:
        positions=task['positions']; valid=np.ones_like(task['values'],dtype=bool)
        rows=np.flatnonzero(task['own']>=0); valid[rows,task['own'][rows]]=False
        mean=(task['values']*valid).sum(0)/np.maximum(valid.sum(0),1)
        observed=valid.any(0)
        template[positions[observed]]+=mean[observed]; support[positions[observed]]+=1
    template=np.divide(template,support,out=np.zeros_like(template),where=support>0)
    return labels,values,mass,sources,template.astype(np.float32),support>0


def decode_composition(ntc_bulk, latent, *, own=-1, own_residual=.2, total=50_000):
    """Invert a smoothed compositional log response on the ACTUAL measured axis.

    Restrict the full-axis latent vector before this call. q+ is renormalized on
    this measured set; pseudocount removal and positivity projection are recorded.
    Own-target residual is a fixed rule, explicitly outside learned downstream loss.
    """
    baseline=np.asarray(ntc_bulk,dtype=np.float64)
    latent=np.asarray(latent,dtype=np.float64)
    if own<0 and np.count_nonzero(latent)==0:
        # The exact identity must not become a tiny nonzero effect after an
        # exp/log round trip, which would unnecessarily activate smoothing/IPF.
        return baseline.copy(),{'pseudocount_projection_fraction':0.,
            'fixed_own_residual':None,'decoded_bulk_rms':0.}
    active=np.ones(len(baseline),dtype=bool)
    fixed=own_residual*np.expm1(baseline[own])/total if own>=0 else 0.
    if own>=0:
        active[own]=False
    logq=(baseline+latent)[active]
    q=np.exp(np.clip(logq-logq.max(),-80,0))
    q*= (total*(1-fixed)+len(q))/q.sum()
    negative=q<1
    downstream=np.maximum(q-1,0)
    if downstream.sum()<=0:
        raise ValueError('Nonpositive decoded population')
    abundance=np.zeros(len(baseline))
    abundance[active]=downstream*(1-fixed)/downstream.sum()
    if own>=0:
        abundance[own]=fixed
    bulk=np.log1p(total*abundance)
    return bulk,{'pseudocount_projection_fraction':float(negative.mean()),
                 'fixed_own_residual':own_residual if own>=0 else None,
                 'decoded_bulk_rms':float(np.sqrt(np.mean((bulk-baseline)**2)))}
