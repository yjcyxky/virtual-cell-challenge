"""Measured-axis, per-target diagnostics saved before official scoring."""
import json
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from scipy import sparse
from analyze_submitted_counts import describe, clean, row_hashes

from .common import NTC, ref, verified, write_json, stable_seed
from .counts import profile, validate_counts
from .capability import pearson, pathway_vectors


def read_group(matrix, rows):
    """Use one CSR span for a contiguous group; preserve arbitrary index order."""
    rows=np.asarray(rows,dtype=np.int64)
    if len(rows) and np.all(np.diff(rows)==1):
        return matrix[int(rows[0]):int(rows[-1])+1]
    return matrix[rows]


def response_geometry(values, seed):
    values = np.asarray(values, dtype=np.float64)
    if values.shape[1] == 0:
        return {'defined':False, 'reason':'no non-target measured readouts'}
    centroid = values.mean(0)
    energy = float(np.square(values).sum())
    residual = values-centroid
    total = float(np.square(residual).sum())
    # Exact participation rank from trace(K) and ||K||_F, without eigendecomposing
    # a target-by-target matrix. Rows are blocked to bound peak memory.
    small = residual if len(residual) <= residual.shape[1] else residual.T
    trace_square = 0.
    for start in range(0,len(small),256):
        gram = small[start:start+256] @ small.T
        trace_square += float(np.square(gram).sum())
    rng = np.random.default_rng(seed)
    pairs = rng.integers(0,len(values),size=(min(20_000,len(values)**2),2))
    pairs = pairs[pairs[:,0] != pairs[:,1]]
    norm = np.linalg.norm(values,axis=1)
    denominator = norm[pairs[:,0]]*norm[pairs[:,1]]
    good = denominator>1e-12
    pairs = pairs[good]
    cosine = np.concatenate([np.einsum('ij,ij->i',values[p[:,0]],values[p[:,1]])
                             for p in np.array_split(pairs,max(1,(len(pairs)+255)//256))])/denominator[good]
    return {'defined':True, 'targets':len(values), 'genes':values.shape[1],
        'shared_centroid_energy_fraction':len(values)*float(centroid@centroid)/energy if energy else None,
        'centered_participation_rank':total**2/trace_square if trace_square else None,
        'response_rms_quantiles':np.quantile(np.sqrt(np.mean(values**2,axis=1)),[0,.25,.5,.75,1]).tolist(),
        'pair_cosine_quantiles':np.quantile(cosine,[0,.25,.5,.75,1]).tolist() if len(cosine) else None,
        'valid_sampled_pairs':len(cosine), 'pair_sampling_seed':seed}


def diagnose_panel(prediction, ntc, directory, *, seed, checkpoint_ref, expected_targets,
                   expected_cells=None, source_coverage=None, template=None, real=None,
                   pathway_ref=None, covariance_genes=128):
    """Labels and true responses are diagnostic-only; no return value trains a model."""
    directory = Path(directory); directory.mkdir(parents=True,exist_ok=True)
    marker = directory/'diagnostics.json'
    prediction_ref = ref(prediction)
    if marker.exists():
        result = json.loads(marker.read_text())
        if result['prediction_ref'] != prediction_ref:
            raise ValueError('Frozen diagnostic prediction changed')
        for item in result['files']:
            verified(item)
        return result
    pred = ad.read_h5ad(prediction, backed='r')
    # AnnData returns a new sparse accessor on every .X access. Keep one so its
    # cached CSR row pointer is loaded once, including for historical stores.
    pred_x = pred.X
    if not pred.var_names.equals(ntc.var_names):
        raise ValueError('Diagnostic measured gene axis/order mismatch')
    ntc_x = sparse.csr_matrix(ntc.X,dtype=np.float64)
    validate_counts(ntc_x)
    _, origin = describe(ntc_x)
    hashes = set(row_hashes(ntc_x))
    targets = sorted(expected_targets)
    groups = pred.obs.groupby('target_gene',observed=True).indices
    if set(groups)-{NTC} != set(targets):
        raise ValueError('Prediction target panel mismatch')
    genes = pred.var_names.to_numpy()
    allowed = ~np.isin(genes,targets)
    positions = np.flatnonzero(allowed)
    positions = positions[np.argsort(origin['logvar'][positions])[-covariance_genes:]]
    z = ntc_x[:,positions].toarray()
    z = np.log1p(z*(10_000/origin['depth'])[:,None])
    origin['cov'] = np.cov(z,rowvar=False)
    moments_path = directory/'gene-moments.h5'
    moment_file = h5py.File(moments_path,'w')
    moment_file.create_dataset('genes',data=genes.astype('S'))
    moment_file.create_dataset('targets',data=np.asarray(targets,dtype='S'))
    names = ['mean','var','cpmean','cpvar','logmean','logvar','zero_fraction','bulk']
    moment_data = {key:moment_file.create_dataset(key,(len(targets),len(genes)),dtype='f4',
                      chunks=(1,len(genes)),compression='gzip',compression_opts=1) for key in names}
    for key in ['mean','var','cpmean','cpvar','logmean','logvar']:
        moment_file.create_dataset('ntc_'+key,data=origin[key])
    bulk_origin = profile(ntc_x)
    rows, bulk_values = [], []
    real_ad = ad.read_h5ad(real,backed='r') if real is not None else None
    real_x = real_ad.X if real_ad is not None else None
    real_groups = real_ad.obs.groupby('target_gene',observed=True).indices if real_ad is not None else None
    real_origin = profile(read_group(real_x,real_groups[NTC])) if real_ad is not None else None
    real_values = []
    for i,target in enumerate(targets):
        matrix = sparse.csr_matrix(read_group(pred_x,groups[target]),dtype=np.float64)
        depth = validate_counts(matrix)
        if expected_cells is not None and len(depth) != expected_cells:
            raise ValueError('Generated cell count differs from registered count')
        summary, moments = describe(matrix,origin,positions,hashes)
        moments['zero_fraction'] = 1-np.bincount(matrix.indices,minlength=len(genes))/matrix.shape[0]
        moments['bulk'] = profile(matrix)
        for key in names:
            moment_data[key][i] = moments[key]
        delta = moments['bulk']-bulk_origin
        bulk_values.append(delta)
        own = np.flatnonzero(genes==target)
        valid = own.size and origin['cpmean'][own[0]]>.05
        summary.update(target=target, measured_genes=len(genes),
            on_target_residual_cpm_ratio=float(moments['cpmean'][own[0]]/origin['cpmean'][own[0]]) if valid else None,
            own_target_measured=bool(own.size),
            positive_downstream_bulk_fraction=float(np.mean(delta[allowed]>0)),
            negative_downstream_bulk_fraction=float(np.mean(delta[allowed]<0)),
            zero_delta=bool(np.count_nonzero(delta)==0),
            bulk_response_rms=float(np.sqrt(np.mean(delta[allowed]**2))),
            low_ntc_expression_detected=int(((origin['cpmean']<.05)&(moments['cpmean']>0)&allowed).sum()))
        if source_coverage is not None:
            summary.update(source_coverage[target])
        if real_ad is not None:
            truth = read_group(real_x,real_groups[target])
            response = profile(truth)-real_origin
            real_values.append(response)
            summary.update(real_n=truth.shape[0], real_n_ge_400=truth.shape[0]>=400,
                response_rmse=float(np.sqrt(np.mean((moments['bulk'][allowed]-real_origin[allowed]-response[allowed])**2))),
                response_pearson=pearson(moments['bulk'][allowed]-real_origin[allowed],response[allowed]))
        rows.append(clean(summary))
        if (i+1)%250 == 0:
            print(f'counts diagnostics {i+1}/{len(targets)}',flush=True)
    if NTC in groups:
        validate_counts(read_group(pred_x,groups[NTC]))
    moment_file.close(); pred.file.close()
    if real_ad is not None:
        real_ad.file.close()
    table_path = directory/'per-target.parquet'
    pd.DataFrame(rows).to_parquet(table_path,index=False)
    values = np.asarray(bulk_values)
    arrays_path = directory/'responses.npz'
    np.savez_compressed(arrays_path,targets=np.asarray(targets),genes=genes.astype(str),
                        prediction=values.astype(np.float32),real=np.asarray(real_values,dtype=np.float32))
    files = [ref(moments_path),ref(table_path),ref(arrays_path)]
    decomposition = None
    if template is not None and real_values:
        direction = np.asarray(template)[allowed].astype(float)
        direction /= max(np.linalg.norm(direction),1e-12)
        a,b = np.asarray(real_values)[:,allowed],(values+bulk_origin-real_origin)[:,allowed]
        ar,br = a-np.outer(a@direction,direction),b-np.outer(b@direction,direction)
        residuals = pd.DataFrame({'target':targets,
            'real_template_coefficient':a@direction,'pred_template_coefficient':b@direction,
            'residual_rms_error':np.sqrt(np.mean((ar-br)**2,axis=1)),
            'residual_pearson':[pearson(x,y) for x,y in zip(ar,br)],
            'real_residual_rms':np.sqrt(np.mean(ar**2,axis=1)),
            'pred_residual_rms':np.sqrt(np.mean(br**2,axis=1))})
        residual_path=directory/'residuals.parquet'; residuals.to_parquet(residual_path,index=False)
        files.append(ref(residual_path))
        decomposition={'template_scope':'source-only supplied direction; orthogonal projection',
            'mean_residual_pearson':clean(residuals.residual_pearson.mean()),
            'mean_residual_rmse':float(residuals.residual_rms_error.mean())}
        if pathway_ref is not None and allowed.sum()>=5:
            vectors, records = pathway_vectors(pathway_ref,genes[allowed])
            vectors = sparse.csr_matrix(vectors)
            rr,pr = (vectors@ar.T).T,(vectors@br.T).T
            for i,record in enumerate(records):
                record.update(residual_pearson=pearson(rr[:,i],pr[:,i]),
                    residual_rmse=float(np.sqrt(np.mean((rr[:,i]-pr[:,i])**2))))
            pathway_path=directory/'pathways.parquet'; pd.DataFrame(records).to_parquet(pathway_path,index=False)
            files.append(ref(pathway_path))
            decomposition['pathways']=len(records)
    result={'status':'completed','hard_constraints_passed':True,'prediction_ref':prediction_ref,
        'checkpoint_ref':checkpoint_ref,'ntc_reference_scope':'same measured axis; permitted input NTC',
        'targets':len(targets),'covariance_gene_positions':positions.tolist(),
        'geometry':response_geometry(values[:,allowed],stable_seed(seed,'geometry')),
        'decomposition':decomposition,'files':files,
        'stage_order':'prediction frozen -> full count diagnostics -> official six metrics'}
    write_json(marker,result)
    return result
