"""Generate counts from predicted log2FC of arithmetic per-cell CPM means."""
import numpy as np
from common import seed

def target_composition(baseline_cpm,response,epsilon):
    """Independent LFC predictions must be projected onto a legal CPM composition."""
    desired=np.maximum(0,(np.asarray(baseline_cpm,dtype=np.float64)+epsilon)*np.exp2(np.asarray(response,dtype=np.float64))-epsilon)
    if not np.isfinite(desired).all() or desired.sum()<=0: raise ValueError('invalid_predicted_profile')
    desired/=desired.sum()
    return desired

def generate_counts(templates,baseline_cpm,response,rng,epsilon=1e-9):
    """Calibrate unweighted per-cell proportions, then restore template depths.

    The source templates carry heterogeneity. Column constraints concern mean CPM,
    not depth-weighted count sums; rounding can introduce small residual errors.
    """
    desired=target_composition(baseline_cpm,response,epsilon)
    counts=np.asarray(templates,dtype=np.float64)
    depths=counts.sum(1)
    if np.any(depths<=0):raise ValueError('empty_template_cell')
    values=counts/depths[:,None]
    observed=values.mean(0)
    factors=np.divide(desired,observed,out=np.zeros_like(desired),where=observed>0)
    values*=factors
    # A gene missing from this finite bag can still have a nonzero full-pool response.
    missing=(observed==0)&(desired>0)
    values[:,missing]=desired[missing]
    # IPF in probability space preserves each cell's depth and the unweighted mean.
    for _ in range(128):
        row_sums=values.sum(1)
        if np.any(row_sums<=0):raise ValueError('unreachable_cell_composition')
        values/=row_sums[:,None]
        values*=np.divide(desired*len(values),values.sum(0),out=np.zeros_like(desired),where=values.sum(0)>0)
        if np.max(np.abs(values.sum(1)-1))<1e-6:break
    else:raise ValueError('lfc_composition_calibration_did_not_converge')
    values*=depths[:,None]
    floor=np.floor(values)
    if values.max(initial=0)>1000000: raise ValueError('prediction_exceeds_official_cell_count_limit')
    result=(floor+(rng.random(values.shape)<(values-floor))).astype(np.uint32)
    if np.any(result.sum(1,dtype=np.uint64)==0): raise ValueError('empty_predicted_cell')
    if result.sum(1,dtype=np.uint64).max(initial=0)>1000000: raise ValueError('prediction_exceeds_official_cell_count_limit')
    return result

class Generator:
    def __init__(self,data,context,config):
        self.context,self.config=context,config
        stats=data.contexts[context];frame=data.cells(context);controls=stats['control_rows']
        strata=(frame.loc[controls,'batch']+'|'+frame.loc[controls,'guide']).to_numpy()
        _,inverse=np.unique(strata,return_inverse=True)
        order=np.argsort(inverse,kind='stable')
        splits=np.flatnonzero(np.diff(inverse[order]))+1
        self.pools=[controls[rows] for rows in np.split(order,splits)]
        sizes=np.array([len(pool) for pool in self.pools])
        self.expected=config['cells_per_prediction']*sizes/sizes.sum()
        self.counts=data.counts(context)
        self.baseline=stats['baseline_cpm']

    def generate(self,target,response):
        rng=np.random.default_rng(seed(self.config['prediction_seed'],self.context,target,'templates'))
        allocation=np.floor(self.expected).astype(int)
        extra=self.config['cells_per_prediction']-allocation.sum()
        fraction=self.expected-allocation
        if extra: allocation[rng.choice(len(allocation),extra,replace=False,p=fraction/fraction.sum())]+=1
        indices=np.concatenate([rng.choice(self.pools[i],n,replace=n>len(self.pools[i])) for i,n in enumerate(allocation) if n])
        templates=np.asarray(self.counts[indices])
        rounding=np.random.default_rng(seed(self.config['prediction_seed'],self.context,target,'rounding'))
        return generate_counts(templates,self.baseline,response,rounding,self.config['lfc_epsilon'])
