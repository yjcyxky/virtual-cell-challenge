#!/usr/bin/env python
"""Verify original Adamson counts/row order and expose lost GEM identities."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import numpy as np
import pandas as pd
from scipy.io import mmread
from profile_background import matrix_in_memory
from profile_responses import write_json
from rna import RNAFile,hash_file

ROOT=Path(__file__).resolve().parents[2]


def identity_rows(barcodes,metadata,obs):
    barcodes=pd.Series(barcodes).astype(str).reset_index(drop=True)
    source=list(obs.index.astype(str));obs=obs.reset_index(drop=True)
    if len(source)!=len(barcodes) or any(a.split('-')[0]!=b.split('-')[0] for a,b in zip(barcodes,source)):
        raise ValueError('original_barcode_row_scope_differs')
    if barcodes.duplicated().any() or metadata.index.duplicated().any():raise ValueError('original_full_identity_not_unique')
    recovered=metadata.reindex(barcodes).reset_index(drop=True);label=recovered['guide identity'];original=obs.perturbation
    equal=label.notna()&original.notna()&(label.astype(str)==original.astype(str))
    return pd.DataFrame({'row_index':np.arange(len(obs)),'collection_source_barcode':source,'original_full_barcode':barcodes,
        'original_GEM_group':barcodes.str.rsplit('-',n=1).str[-1].astype(int),'collection_source_guide_label':original,
        'original_GEO_guide_label':label,'source_guide_label_concordant':equal,
        'original_GEO_record_present':barcodes.isin(metadata.index),'original_GEO_good_coverage':recovered['good coverage'],
        'original_GEO_guide_UMI':recovered['UMI count'],'original_GEO_guide_reads':recovered['read count'],
        'identity_interpretation':'Original GEO identity after complete matrix and ordered gene-axis equality; collection label retained'})


def run(evidence,output):
    output.mkdir(parents=True,exist_ok=False);summaries=[]
    inputs=json.loads((evidence/'Adamson-original-count-retrieval.json').read_text())+json.loads((evidence/'Adamson-identities-retrieval.json').read_text())
    for record in inputs:
        if hash_file(evidence/record['file'])!=record['sha256']:raise ValueError('original_GEO_evidence_changed')
    for accession in ['GSM2406675_10X001','GSM2406677_10X005','GSM2406681_10X010']:
        file='AdamsonWeissman2016_'+accession+'.h5ad';path=ROOT/'data/raw/scperturb'/file;before=hash_file(path)
        matrix=mmread(evidence/(accession+'_matrix.mtx.txt.gz')).T.tocsr();matrix.sum_duplicates();matrix.eliminate_zeros();matrix.sort_indices()
        axis=pd.read_csv(evidence/(accession+'_genes.tsv.gz'),header=None,sep='\t');barcodes=pd.read_csv(evidence/(accession+'_barcodes.tsv.gz'),header=None)[0]
        metadata=pd.read_csv(evidence/(accession+'_cell_identities.csv.gz'),index_col=0)
        with RNAFile(path) as source:
            if source.shape!=matrix.shape or source.var.ensembl_id.astype(str).tolist()!=axis[0].astype(str).tolist():raise ValueError('Adamson_original_gene_axis_or_shape_changed')
            current=matrix_in_memory(source);difference=(current-matrix).tocsr();difference.eliminate_zeros()
            if difference.nnz:raise ValueError('Adamson_original_RNA_values_differ')
            rows=identity_rows(barcodes,metadata,source.obs)
            del current,difference,matrix
        if hash_file(path)!=before:raise ValueError('collection_input_changed')
        rows['original_chemical_background']=rows.original_GEM_group.map({1:'tunicamycin',2:'thapsigargin',3:'DMSO'}) if '10X005' in accession else 'source_single_experiment'
        rows.to_parquet(output/(accession+'.parquet'),index=False)
        summary={'file':file,'input_sha256':before,'cells':len(rows),'all_source_counts_exactly_equal':True,
            'original_full_barcodes_unique':True,'original_GEM_group_counts':rows.original_GEM_group.value_counts().to_dict(),
            'collection_vs_original_guide_concordant':int(rows.source_guide_label_concordant.sum()),
            'original_GEO_records_present':int(rows.original_GEO_record_present.sum()),
            'source_base_barcode_duplicates':int(barcodes.str.split('-').str[0].duplicated().sum()),
            'nonconcordant_collection_labels':rows.loc[~rows.source_guide_label_concordant,'collection_source_guide_label'].fillna('<missing>').value_counts().to_dict()}
        summaries.append(summary);print(json.dumps(summary),flush=True)
    report={'status':'completed','inputs':inputs,'files':summaries,'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'code_sha256':hash_file(Path(__file__)),'completed_at':datetime.now(timezone.utc).isoformat(),
        'chemical_mapping_evidence':{'repository':'https://github.com/thomasmaxwellnorman/perturbseq_demo','commit':(evidence/'perturbseq_demo/commit.txt').read_text().strip(),
            'file':'perturbseq_demo.ipynb','mapping':{'1':'Tm','2':'Thaps','3':'DMSO'}},
        'scope':'No collection annotation replacement; original identity and label concordance are separate analysis evidence.'}
    report['artifacts']={p.name:hash_file(p) for p in output.iterdir() if p.is_file()};write_json(output/'report.json',report)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--evidence',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.evidence,a.output)
