"""Descriptive chemical endpoint contrasts; cells are not biological replicates."""
import numpy as np
from response import correlation

PARAMETERS={'seed':20260919,'repetitions':20,'minimum_stability_target_cells':20,
            'within_type_minimum_target_cells':10,'within_type_minimum_control_cells':2}


def distribution_contrast(target,control,target_types,control_types,names,seed):
    """Compare explicitly matched arms in fixed RNA proxy/log-QC coordinates.

    Each null draw permutes one control pool and takes disjoint equal-sized arms.
    Draw indices refer only to that control pool; overlap is checked explicitly.
    """
    target=np.asarray(target,dtype=float);control=np.asarray(control,dtype=float)
    n,m=len(target),len(control);summary={'n_target':n,'n_control':m,'independent_biological_replicates':None}
    columns=[];composition=[];within=[];draws=[]
    if not n or not m:
        summary.update(status='not_estimable',reason='missing_target_or_same_stratum_vehicle',stability_status='not_estimable')
        return summary,columns,composition,within,draws
    available=np.isfinite(target).all(axis=0)&np.isfinite(control).all(axis=0)
    effect=np.full(len(names),np.nan);effect[available]=target[:,available].mean(axis=0)-control[:,available].mean(axis=0)
    for j,name in enumerate(names):
        a=target[:,j];a=a[np.isfinite(a)];b=control[:,j];b=b[np.isfinite(b)]
        r={'variable':name,'n_target_finite':len(a),'n_control_finite':len(b),'status':'completed' if len(a) and len(b) else 'not_estimable'}
        if len(a) and len(b):
            qa=np.quantile(a,[.1,.25,.5,.75,.9]);qb=np.quantile(b,[.1,.25,.5,.75,.9])
            r.update(target_mean=float(a.mean()),control_mean=float(b.mean()),mean_difference=float(a.mean()-b.mean()),
                target_q10=float(qa[0]),target_median=float(qa[2]),target_q90=float(qa[4]),control_q10=float(qb[0]),control_median=float(qb[2]),control_q90=float(qb[4]),
                median_difference=float(qa[2]-qb[2]),target_IQR=float(qa[3]-qa[1]),control_IQR=float(qb[3]-qb[1]))
        columns.append(r)
    a_types=np.asarray(target_types);b_types=np.asarray(control_types)
    for label in sorted(set(a_types)|set(b_types)):
        a=a_types==label;b=b_types==label;nt,nc=int(a.sum()),int(b.sum())
        composition.append({'inferred_type':label,'target_cells':nt,'control_cells':nc,'target_fraction':nt/n,'control_fraction':nc/m,'fraction_difference':nt/n-nc/m,'truth_label':False})
        for j,name in enumerate(names):
            aa=target[a,j];aa=aa[np.isfinite(aa)];bb=control[b,j];bb=bb[np.isfinite(bb)]
            ok=len(aa)>=PARAMETERS['within_type_minimum_target_cells'] and len(bb)>=PARAMETERS['within_type_minimum_control_cells']
            within.append({'inferred_type':label,'variable':name,'n_target':len(aa),'n_control':len(bb),
                'status':'completed' if ok else 'not_estimable','reason':None if ok else 'insufficient_same_inferred_type_cells',
                'mean_difference':float(aa.mean()-bb.mean()) if ok else None,'interpretation':'endpoint descriptive stratum, not causal mediation'})
    summary.update(status='completed',complete_dimensions=int(available.sum()),proxy_and_logQC_RMS=float(np.sqrt(np.mean(effect[available]**2))) if available.any() else None)
    if n<PARAMETERS['minimum_stability_target_cells'] or m<2 or not available.any():
        summary.update(stability_status='not_estimable',stability_reason='requires_20_targets_2_controls_and_finite_dimensions')
        return summary,columns,composition,within,draws
    rng=np.random.default_rng(seed);t=target[:,available];c=control[:,available];control_mean=c.mean(axis=0)
    for repetition in range(PARAMETERS['repetitions']):
        order=rng.permutation(n);left,right=np.array_split(order,2);ea=t[left].mean(axis=0)-control_mean;eb=t[right].mean(axis=0)-control_mean
        cp=rng.permutation(m);size=min(n,m//2);ca,cb=cp[:size],cp[size:2*size]
        intersection=np.intersect1d(ca,cb).size
        if intersection:raise ValueError('null_control_arms_overlap')
        null=c[ca].mean(axis=0)-c[cb].mean(axis=0)
        draws.append({'replicate':repetition,'half_1_cells':len(left),'half_2_cells':len(right),'half_reference_cells':m,
            'half_effect_correlation':correlation(ea,eb),'half_1_RMS':float(np.sqrt(np.mean(ea**2))),'half_2_RMS':float(np.sqrt(np.mean(eb**2))),
            'null_arm_cells':size,'null_target_cells_not_matched':n-size,'null_reference_intersection':int(intersection),
            'null_RMS':float(np.sqrt(np.mean(null**2))),'interpretation':'conditional cell sampling in RNA proxy/log-QC space; not independent culture replication'})
    summary.update(stability_status='completed',resamples=len(draws),null_target_size_shortfall=n-min(n,m//2))
    return summary,columns,composition,within,draws
