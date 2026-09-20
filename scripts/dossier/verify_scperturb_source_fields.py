#!/usr/bin/env python
"""Verify all original RNA obs/var values against the published scPerturb sidecars."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import h5py
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from rna import read_frame,hash_file
from profile_responses import write_json


def same_values(left,right):
    left=left.reset_index(drop=True).astype(object);right=right.reset_index(drop=True).astype(object)
    pd.testing.assert_series_equal(left.where(left.notna(),None),right.where(right.notna(),None),check_names=False,check_dtype=False,check_exact=True)


def run(cells,output):
    report=json.loads((cells/'report.json').read_text());rows=[];root=Path(__file__).resolve().parents[2]
    if report['status']!='completed':raise ValueError('completed_cell_phase_required')
    for entry in report['files']:
        file=entry['file'];folder=cells/file.removesuffix('.h5ad');component=json.loads((folder/'report.json').read_text())
        for name in ['cells.parquet','source-var.parquet']:
            if hash_file(folder/name)!=component['artifact_hashes'][name]:raise ValueError('frozen_source_sidecar_changed')
        with h5py.File(root/'data/raw/scperturb'/file,'r') as source:
            obs=read_frame(source['obs']);var=read_frame(source['var']).reset_index(names='source_var_index')
        expected=['source_obs__'+name for name in obs];actual=[n for n in pq.read_schema(folder/'cells.parquet').names if n.startswith('source_obs__')]
        if set(actual)!=set(expected):raise ValueError('source_obs_column_scope_changed')
        derived=pd.read_parquet(folder/'cells.parquet',columns=['source_barcode','row_index']+expected)
        if not np.array_equal(derived.row_index,np.arange(len(obs))):raise ValueError('source_row_order_changed')
        same_values(pd.Series(obs.index.astype(str)),derived.source_barcode)
        for name in obs:same_values(obs[name],derived['source_obs__'+name])
        derived_var=pd.read_parquet(folder/'source-var.parquet')
        if list(derived_var)!=list(var):raise ValueError('source_var_column_scope_changed')
        for name in var:same_values(var[name],derived_var[name])
        rows.append({'file':file,'records':len(obs),'original_obs_fields':list(obs),'original_var_fields':list(var),
            'obs_values_checked':int(obs.size),'var_values_checked':int(var.size),'all_values_and_missing_masks_equal':True,
            'source_barcode_matches_original_index':True,'cells_sidecar_sha256':component['artifact_hashes']['cells.parquet'],
            'source_var_sidecar_sha256':component['artifact_hashes']['source-var.parquet'],'registered_raw_input_sha256':entry['input_sha256']})
        print('source field verification '+str(len(rows))+'/51 '+file,flush=True)
    result={'status':'completed','files':rows,'RNA_files':len(rows),'RNA_records':sum(r['records'] for r in rows),
        'obs_values_checked':sum(r['obs_values_checked'] for r in rows),'var_values_checked':sum(r['var_values_checked'] for r in rows),
        'cell_phase_report_sha256':hash_file(cells/'report.json'),'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'code_sha256':hash_file(Path(__file__)),'completed_at':datetime.now(timezone.utc).isoformat(),
        'scope':'Read-only exact comparison of every decoded source obs/var value and missing mask. X count integrity is established separately by the frozen all-value cell and response phases.'}
    write_json(output,result);print(json.dumps({k:v for k,v in result.items() if k!='files'}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cells',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():p.error('fresh verification output required')
    run(a.cells,a.output)
