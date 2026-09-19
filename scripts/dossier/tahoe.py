"""Exact Tahoe token decoding and source metadata joins, with explicit CLS handling."""
import ast
import numpy as np
import pandas as pd
from scipy import sparse


def decode(batch,token_columns,n_genes):
    genes=batch.column(batch.schema.get_field_index('genes'));expressions=batch.column(batch.schema.get_field_index('expressions'))
    gp=genes.offsets.to_numpy();ep=expressions.offsets.to_numpy()
    if not np.array_equal(np.diff(gp),np.diff(ep)) or (np.diff(gp)<1).any():raise ValueError('misaligned_or_empty_Tahoe_sequences')
    ids=genes.values.to_numpy()[gp[0]:gp[-1]];values=expressions.values.to_numpy()[ep[0]:ep[-1]];starts=gp[:-1]-gp[0]
    if not (ids[starts]==1).all() or not (values[starts]==-2).all():raise ValueError('unexpected_CLS_encoding')
    keep=np.ones(len(ids),bool);keep[starts]=False;ids=ids[keep];values=values[keep]
    if (ids<0).any() or (ids>=len(token_columns)).any():raise ValueError('gene_token_out_of_bounds')
    indices=token_columns[ids]
    if (indices<0).any():raise ValueError('unknown_or_special_token_in_gene_counts')
    ptr=np.r_[0,np.cumsum(np.diff(gp)-1)];rows=np.repeat(np.arange(len(batch)),np.diff(ptr))
    bad=~np.isfinite(values)|(values<0)|(values!=np.floor(values))
    valid=np.bincount(rows,weights=bad.astype(int),minlength=len(batch))==0
    # Repeated biological tokens are a structural error, not a reason to silently sum counts.
    matrix=sparse.csr_matrix((values.astype(np.float64),indices.astype(np.int32),ptr),shape=(len(batch),n_genes))
    before=matrix.nnz;matrix.sum_duplicates()
    if matrix.nnz!=before:raise ValueError('duplicate_gene_tokens_in_cell')
    matrix.sort_indices();matrix.eliminate_zeros()
    return matrix,valid,{'CLS_markers':len(batch),'biological_values':len(values),'nonfinite':int((~np.isfinite(values)).sum()),
        'negative':int((values<0).sum()),'noninteger':int((np.isfinite(values)&(values!=np.floor(values))).sum())}


def parse_compounds(value):
    result=ast.literal_eval(value)
    if not isinstance(result,list) or not result:raise ValueError('invalid_compound_dose_list')
    rows=[]
    for item in result:
        if not isinstance(item,tuple) or len(item)!=3:raise ValueError('invalid_compound_dose_tuple')
        name,dose,unit=item
        if not isinstance(name,str) or not isinstance(unit,str) or not isinstance(dose,(int,float)) or not np.isfinite(dose) or dose<0:raise ValueError('invalid_compound_dose_value')
        rows.append({'source_compound':name,'source_dose_value':float(dose),'source_dose_unit':unit,'vehicle_code_not_molar_dose':name=='DMSO_TF'})
    return rows


def metadata_join(local,observations,samples):
    if observations.BARCODE_SUB_LIB_ID.duplicated().any() or samples['sample'].duplicated().any():raise ValueError('nonunique_metadata_join_key')
    obs=observations.add_prefix('source_obs_').rename(columns={'source_obs_BARCODE_SUB_LIB_ID':'BARCODE_SUB_LIB_ID'})
    result=local.merge(obs,on='BARCODE_SUB_LIB_ID',how='left',validate='many_to_one',sort=False)
    sm=samples.add_prefix('source_sample_').rename(columns={'source_sample_sample':'sample'})
    result=result.merge(sm,on='sample',how='left',validate='many_to_one',sort=False)
    result['metadata_obs_present']=result.source_obs_sample.notna();result['metadata_sample_present']=result.source_sample_plate.notna()
    for source,target in [('plate','plate'),('sample','sample'),('drug','drug'),('cell_line_id','cell_line')]:
        result['metadata_match_'+source]=result[source]==result['source_obs_'+target]
    result['sample_condition_match']=(result.plate==result.source_sample_plate)&(result.drug==result.source_sample_drug)
    result['metadata_condition_consistent']=result[[c for c in result if c.startswith('metadata_match_')]+['sample_condition_match']].all(axis=1)
    return result


def condition_eligibility(frame):
    """A documented whitespace alias is separate from unchanged source labels."""
    identity=frame[['metadata_match_plate','metadata_match_sample','metadata_match_cell_line_id']].all(axis=1)
    sample_plate=frame.plate==frame.source_sample_plate
    names=(frame.drug.str.strip()==frame.source_obs_drug.str.strip())&(frame.drug.str.strip()==frame.source_sample_drug.str.strip())
    valid=identity&sample_plate&names&frame.metadata_obs_present&frame.metadata_sample_present
    reason=np.where(frame.metadata_condition_consistent,'exact_source_labels',np.where(valid,'explicit_trailing_whitespace_alias','unresolved_metadata_condition_conflict'))
    return valid.to_numpy(dtype=bool),reason
