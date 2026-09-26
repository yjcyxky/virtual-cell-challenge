"""NTC-template count generation anchored to the full-pool predicted profile."""
import numpy as np
from common import seed

def generate_counts(templates,baseline,response,rng):
    """Retain template heterogeneity/depth; calibrate aggregate profile before rounding.

    Additive log-CP50K predictions are first projected onto nonnegative CP50K
    profiles. Resampled-template noise is not added as a second baseline shift.
    """
    desired=np.expm1(np.maximum(0,np.asarray(baseline,dtype=np.float64)+response))
    if not np.isfinite(desired).all() or desired.sum()<=0: raise ValueError('invalid_predicted_profile')
    desired/=desired.sum()
    counts=np.asarray(templates,dtype=np.float64)
    depths=counts.sum(1)
    total=depths.sum()
    observed=counts.sum(0)
    factors=np.divide(desired*total,observed,out=np.zeros_like(desired),where=observed>0)
    values=counts*factors
    # A gene missing from this finite bag can still have a nonzero full-pool response.
    missing=(observed==0)&(desired>0)
    values[:,missing]=depths[:,None]*desired[missing]
    # Iterative proportional fitting preserves per-cell depths and the desired group profile.
    for _ in range(12):
        values*=np.divide(depths,values.sum(1),out=np.ones_like(depths),where=values.sum(1)>0)[:,None]
        values*=np.divide(desired*total,values.sum(0),out=np.zeros_like(desired),where=values.sum(0)>0)
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
        self.baseline=stats['baseline']

    def generate(self,target,response):
        rng=np.random.default_rng(seed(self.config['prediction_seed'],self.context,target,'templates'))
        allocation=np.floor(self.expected).astype(int)
        extra=self.config['cells_per_prediction']-allocation.sum()
        fraction=self.expected-allocation
        if extra: allocation[rng.choice(len(allocation),extra,replace=False,p=fraction/fraction.sum())]+=1
        indices=np.concatenate([rng.choice(self.pools[i],n,replace=n>len(self.pools[i])) for i,n in enumerate(allocation) if n])
        templates=np.asarray(self.counts[indices])
        rounding=np.random.default_rng(seed(self.config['prediction_seed'],self.context,target,'rounding'))
        return generate_counts(templates,self.baseline,response,rounding)
