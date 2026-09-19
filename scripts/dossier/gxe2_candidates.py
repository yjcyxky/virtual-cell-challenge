#!/usr/bin/env python
"""Annotate every source-threshold GxE2 candidate lacking a final author CDS label."""
import argparse
from collections import Counter
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import time
import numpy as np
import pandas as pd
from annotation import marker_model,state_model,RULES
from sparse_annotation import annotate_sparse
from rna import RNAFile,hash_file,mapping_audit,quantiles
from profile_responses import write_json
ROOT=Path(__file__).resolve().parents[2]


def run(records,references,output):
    start=time.monotonic();report=json.loads((records/'report.json').read_text());path=records/'non-CDS-candidates.h5ad'
    if report['status']!='completed' or hash_file(path)!=report['artifacts'][path.name]:raise ValueError('verified_candidate_view_required')
    for line in (references/'SHA256SUMS').read_text().splitlines():
        digest,name=line.split('  ',1)
        if hash_file(references/name)!=digest:raise ValueError('reference_changed')
    output.mkdir(parents=True,exist_ok=False);reference=json.loads((references/'gene_sets.json').read_text())
    with RNAFile(path) as source:
        n,g=source.shape
        if n!=report['coverage']['non_CDS_source_threshold_candidates']:raise ValueError('candidate_scope_changed')
        cells=source.obs.reset_index(names='source_barcode')
        if cells.source_CDS_member.any() or not cells.source_500_UMI_threshold_met.all() or cells.record_id.duplicated().any():raise ValueError('candidate_applicability_or_identity_mismatch')
        mapping=mapping_audit(source.var.gene_name.tolist(),pd.read_csv(ROOT/'data/raw/networks/hgnc_complete_set.txt',sep='\t',low_memory=False),pd.read_csv(ROOT/'data/raw/arc_vcc2026_controls/gene_names.csv').gene_name.tolist(),source.var.index.astype(str).tolist())
        mapping.insert(0,'source_gene_id',source.var.index.astype(str));mapping.to_parquet(output/'gene-mapping.parquet',index=False)
        safe=mapping.mapped_symbol.where(~mapping.many_to_one_mapping&(mapping.symbol_vs_ensembl!='conflict'));resolved=[str(x) if pd.notna(x) else None for x in safe]
        mean=np.zeros(g)
        for start,matrix in source.blocks(1024):
            if not np.isfinite(matrix.data).all() or (matrix.data<0).any() or (matrix.data!=np.floor(matrix.data)).any():raise ValueError('invalid_candidate_count')
            total=np.asarray(matrix.sum(axis=1)).ravel()
            if not np.array_equal(total,cells.computed_total_counts.iloc[start:start+len(total)].to_numpy()):raise ValueError('candidate_raw_record_QC_mismatch')
            matrix.data=np.log1p(matrix.data*np.repeat(10000/total,np.diff(matrix.indptr)));mean+=np.asarray(matrix.sum(axis=0)).ravel()
        mean/=n;weights,coverage=state_model(resolved,reference['states'],mean);model=marker_model(resolved,reference['profiles']);parts=[]
        for start,matrix in source.blocks(1024):
            total=np.asarray(matrix.sum(axis=1)).ravel();matrix.data=np.log1p(matrix.data*np.repeat(10000/total,np.diff(matrix.indptr)))
            labels,scores=annotate_sparse(matrix,[None]*matrix.shape[0],model)
            for name,values in scores.items():labels[name]=values
            values=np.asarray(matrix@weights)
            for j,(name,cov) in enumerate(zip(reference['states'],coverage)):labels['state__'+name]=values[:,j] if cov['status']=='completed' else np.nan
            parts.append(labels)
    combined=pd.concat([cells,pd.concat(parts,ignore_index=True)],axis=1)
    combined['state_background']='pooled_endpoint_candidates_without_CDS_not_NTC';combined['source_condition_status']='not_estimable_no_author_CDS_condition'
    combined['source_cell_line']=None;combined['source_genetic_assignment']=None;combined['available_before_endpoint']=False;combined['truth_label']=False
    if combined.probability_correct.notna().any() or combined.record_id.duplicated().any() or len(combined)!=n:raise ValueError('candidate_inference_identity_or_truth_violation')
    combined.to_parquet(output/'cells.parquet',index=False,compression='zstd');np.savez_compressed(output/'scoring-weights.npz',pooled_mean_logCP10K=mean,state_weights=weights,type_weights=model['weights'],valid_gene_axis=model['valid_axis'])
    write_json(output/'type-coverage.json',model['coverage']);write_json(output/'state-coverage.json',coverage)
    if hash_file(path)!=report['artifacts'][path.name]:raise ValueError('candidate_input_changed_during_inference')
    result={'status':'completed','phase':'all_source_threshold_candidates_without_CDS_inference','n_candidates':n,'raw_source_record_report_sha256':hash_file(records/'report.json'),
        'candidate_view_sha256':hash_file(path),'reference_sha256':hash_file(references/'gene_sets.json'),'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'code':{name:hash_file(Path(__file__).with_name(name)) for name in ['gxe2_candidates.py','rna.py','annotation.py','sparse_annotation.py']},'annotation_rules':RULES,
        'pooled_types':combined.inferred_type.value_counts().to_dict(),'uncalibrated_records':n,'state_distributions':{name:quantiles(combined['state__'+name]) for name in reference['states']},
        'inputs_unchanged':True,'limitations':['Source 500 UMI rule only establishes candidate applicability, not a validated cell identity.','No final source hash/guide/cell-line assignment is imputed.','Healthy human marker references and endpoint-based states are uncalibrated; probabilities are null.'],
        'completed_at':datetime.now(timezone.utc).isoformat()}
    result['artifacts']={str(p.relative_to(output)):hash_file(p) for p in output.rglob('*') if p.is_file()};write_json(output/'report.json',result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['records','references','output']:p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();r=run(a.records,a.references,a.output);print(json.dumps({k:v for k,v in r.items() if k not in ['artifacts','state_distributions','code']}))
