"""Descriptive response comparison and endpoint mixture identities, not causal models."""
from collections import Counter
import numpy as np
import pandas as pd


def safe_symbols(mapping):
    symbols=mapping.mapped_symbol
    conflict=mapping.get('symbol_vs_ensembl',pd.Series('',index=mapping.index)).eq('conflict')
    counts=Counter(symbols.dropna())
    good=symbols.notna() & symbols.map(counts).eq(1) & ~conflict
    uncertain=set(symbols[~good].dropna())
    for field in ['candidates','ensembl_candidates']:
        if field in mapping:
            for value in mapping.loc[~good,field].dropna():uncertain.update(str(value).split('|'))
    return symbols.where(good & ~symbols.isin(uncertain)).tolist()


def compare_vectors(a,b,minimum=100):
    a=np.asarray(a,dtype=float);b=np.asarray(b,dtype=float)
    if a.shape!=b.shape:raise ValueError('unaligned_comparison_vectors')
    valid=np.isfinite(a)&np.isfinite(b);n=int(valid.sum())
    if n<minimum:return {'status':'not_estimable','reason':'fewer_than_'+str(minimum)+'_common_finite_genes','genes':n,'correlation':None,'difference_RMS':None}
    a=a[valid];b=b[valid];ac=a-a.mean();bc=b-b.mean()
    denominator=float(np.linalg.norm(ac)*np.linalg.norm(bc))
    constant=np.ptp(a)==0 or np.ptp(b)==0 or denominator==0
    return {'status':'completed','reason':None,'genes':n,
            'correlation':None if constant else float(np.clip(np.dot(ac,bc)/denominator,-1,1)),
            'correlation_status':'not_estimable_zero_variance' if constant else 'completed',
            'difference_RMS':float(np.sqrt(np.mean((a-b)**2))),
            'RMS_A':float(np.sqrt(np.mean(a*a))),'RMS_B':float(np.sqrt(np.mean(b*b)))}


def endpoint_cycle_partition(s,g2m,control):
    """NTC-derived boundaries; both proxies must be observed, no invented state."""
    scores=np.asarray(s,dtype=float)+np.asarray(g2m,dtype=float)
    valid=np.isfinite(scores);control=np.asarray(control,dtype=bool)
    reference=scores[valid&control]
    labels=np.full(len(scores),'unknown_cycle_proxy',dtype=object)
    if len(reference)<3:return labels,{'status':'not_estimable','reason':'fewer_than_3_finite_NTC_scores','boundaries':None,'finite_NTC':len(reference)}
    q=np.quantile(reference,[1/3,2/3])
    if not q[0]<q[1]:return labels,{'status':'not_estimable','reason':'coincident_NTC_tercile_boundaries','boundaries':q.tolist(),'finite_NTC':len(reference)}
    bins=np.searchsorted(q,scores[valid],side='right')
    labels[valid]=np.array(['low_cycle_RNA_proxy','middle_cycle_RNA_proxy','high_cycle_RNA_proxy'])[bins]
    return labels,{'status':'completed','reason':None,'boundaries':q.tolist(),'finite_NTC':len(reference),
                   'available_before_endpoint':False,'calibrated_cell_cycle_stage':False}


def mixture_identity(target_mean,control_mean,target_counts,control_counts,batches,minimum_targets=10):
    """Symmetric mean decomposition on observed common state × batch support.

    One row is a target-present state/technical stratum. The caller passes all
    such rows, including unsupported controls as NaNs with zero control count.
    Nothing is imputed for an absent state. Batch weights are retained target
    cell proportions; control-state weights renormalize on common support.
    """
    tm=np.asarray(target_mean,dtype=float);cm=np.asarray(control_mean,dtype=float)
    nt=np.asarray(target_counts,dtype=float);nc=np.asarray(control_counts,dtype=float)
    batches=np.asarray(batches)
    if tm.shape!=cm.shape or tm.ndim!=2 or len(tm)!=len(nt) or len(nt)!=len(nc) or len(nt)!=len(batches):raise ValueError('mixture_group_dimensions')
    if (nt<0).any() or (nc<0).any():raise ValueError('negative_group_counts')
    keep=(nt>0)&(nc>=2)&np.isfinite(tm).all(axis=1)&np.isfinite(cm).all(axis=1)
    retained=int(nt[keep].sum());observed=int(nt.sum())
    meta={'observed_target_cells':observed,'supported_target_cells':retained,'supported_target_fraction':retained/observed if observed else None,
          'observed_target_state_strata':len(nt),'supported_target_state_strata':int(keep.sum()),'minimum_NTC_per_state_stratum':2,
          'causal_interpretation':False,'independent_biological_replicates':None}
    if retained<minimum_targets:return None,{**meta,'status':'not_estimable','reason':'fewer_than_'+str(minimum_targets)+'_supported_target_cells'}
    tm=tm[keep];cm=cm[keep];nt=nt[keep];nc=nc[keep];batches=batches[keep]
    ntb={b:nt[batches==b].sum() for b in np.unique(batches)}
    ncb={b:nc[batches==b].sum() for b in np.unique(batches)}
    pt=nt/np.array([ntb[b] for b in batches]);pc=nc/np.array([ncb[b] for b in batches])
    weights=np.array([ntb[b]/retained for b in batches])
    target=np.sum((weights*pt)[:,None]*tm,axis=0)
    control=np.sum((weights*pc)[:,None]*cm,axis=0)
    total=target-control
    composition=np.sum((weights*(pt-pc)/2)[:,None]*(tm+cm),axis=0)
    within=np.sum((weights*(pt+pc)/2)[:,None]*(tm-cm),axis=0)
    error=float(np.max(np.abs(total-composition-within)))
    if error>1e-9:raise ValueError('mixture_identity_numerical_failure')
    return {'total':total,'composition':composition,'within':within,'target_mean':target,'control_mean':control},\
        {**meta,'status':'completed','reason':None,'supported_technical_strata':len(ntb),'maximum_identity_residual':error}
