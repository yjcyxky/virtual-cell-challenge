"""Sparse evaluation of the existing nonnegative-RNA marker scoring equations.

Implicit zeros receive their average tied rank. This avoids a cells-by-all-genes
dense allocation without restricting the measured gene universe.
"""
import numpy as np
from scipy import sparse
from scipy.stats import rankdata
from annotation import decide


def annotate_sparse(log, targets, model):
    x=sparse.csr_matrix(log,dtype=np.float32,copy=True)
    x.sum_duplicates();x.eliminate_zeros();x.sort_indices()
    if not np.isfinite(x.data).all() or (x.data<0).any():
        raise ValueError('sparse_marker_scoring_requires_finite_nonnegative_expression')
    valid=np.flatnonzero(model['valid_axis']);n,k=len(targets),len(model['names'])
    if x.shape[0]!=n:raise ValueError('target_row_mismatch')
    if not len(valid):return decide(np.zeros((n,k)),np.zeros((n,k)),np.zeros((n,k)),model),{}
    y=x[:,valid].tocsr();g=len(valid);weights=model['weights'][valid];membership=model['membership'][valid]
    positive=np.diff(y.indptr);zero_rank=(g-positive+1)/(2*g)-.5
    delta=y.copy()
    for row in range(n):
        lo,hi=y.indptr[row:row+2]
        delta.data[lo:hi]=(rankdata(y.data[lo:hi],method='average')+g-positive[row])/g-.5-zero_rank[row]
    rs=np.asarray(delta@weights)+zero_rank[:,None]*weights.sum(axis=0)
    mean=np.asarray(y.sum(axis=1,dtype=np.float64)).ravel()/g
    second=np.asarray(y.astype(np.float64).power(2).sum(axis=1)).ravel()/g
    std=np.sqrt(np.maximum(0,second-mean**2))
    es=np.divide(np.asarray(y@weights)-mean[:,None]*weights.sum(axis=0),std[:,None],out=np.zeros((n,k)),where=std[:,None]>0)
    present=y.copy();present.data=np.ones_like(present.data);detected=np.asarray(present@membership)
    result=decide(rs,es,detected,model);rs2,es2,d2=rs.copy(),es.copy(),detected.copy()
    local_lookup={int(source):i for i,source in enumerate(valid)};targets=np.asarray(targets)
    for target in set(targets):
        source=model['lookup'].get(target)
        if source is None:continue
        j=local_lookup[source];rows=np.flatnonzero(targets==target);w=weights[j]
        values=y[rows,j].toarray().ravel();r=zero_rank[rows]+delta[rows,j].toarray().ravel()
        z=np.divide(values-mean[rows],std[rows],out=np.zeros(len(rows)),where=std[rows]>0)
        rs2[rows]=(rs[rows]-r[:,None]*w)/np.maximum(1-w,1e-8)
        es2[rows]=(es[rows]-z[:,None]*w)/np.maximum(1-w,1e-8)
        d2[rows]-=(values[:,None]>0)*membership[j]
    sensitivity=decide(rs2,es2,d2,model)
    result['target_excluded_type']=sensitivity.inferred_type
    result['target_marker_label_changed']=result.inferred_type!=sensitivity.inferred_type
    result['target_is_reference_marker']=[t in model['lookup'] and bool(model['membership'][model['lookup'][t]].any()) for t in targets]
    scores={'rank__'+name:rs[:,j] for j,name in enumerate(model['names'])}
    scores.update({'expression__'+name:es[:,j] for j,name in enumerate(model['names'])})
    return result,scores
