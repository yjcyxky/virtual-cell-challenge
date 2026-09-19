#!/usr/bin/env python
"""Stream all raw GxE2 barcode records; distinguish records from source cells."""
import argparse
from collections import Counter
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time
import h5py
import numpy as np
import pandas as pd
from scipy import sparse
from convert_seurat_cache import frame
from rna import read_frame,read_array,decode_categories,hash_file,value_hash,quantiles
from profile_responses import write_json


def annotation_slice(node,start,stop):
    if isinstance(node,h5py.Dataset):
        return node.asstr()[start:stop] if node.dtype.kind in 'OSU' else node[start:stop]
    if {'categories','codes'}<=set(node):
        return decode_categories(read_array(node['categories']),node['codes'][start:stop],node.name)
    raise ValueError('unsupported_bounded_annotation_encoding')


def run(cache,output,chunk=100000):
    started=time.monotonic();output.mkdir(parents=True,exist_ok=False);(output/'records').mkdir()
    identity=json.loads((cache/'identity.json').read_text());input_sha=identity['coordinate']['coordinate_sha256']
    source=cache/'coordinate-counts.h5ad'
    if hash_file(source)!=identity['coordinate']['cache_sha256']:raise ValueError('raw_coordinate_view_changed')
    n,g=identity['coordinate']['shape'];membership=np.full(n,-1,np.int32)
    metadata=pd.read_parquet(cache/'all-CDS-source-metadata.parquet');lookup=pd.Index(metadata.source_barcode)
    for folder in cache.glob('CDS-*'):
        comparison=pd.read_parquet(folder/'raw-count-comparison.parquet')
        ids=lookup.get_indexer(comparison.source_barcode)
        if (ids<0).any() or (membership[comparison.source_row]>=0).any() or comparison.differing_values.any():raise ValueError('ambiguous_or_different_CDS_count_membership')
        membership[comparison.source_row]=ids
    if (membership>=0).sum()!=len(metadata):raise ValueError('incomplete_source_CDS_membership')
    counts=Counter(); numeric=Counter(); umi_hist=Counter(); detected_hist=Counter(); totals_sum=0; all_features=np.zeros(g); all_detected=np.zeros(g,np.int64)
    selected_parts=[];candidate_parts=[];candidate_ptr=[0];destination=output/'non-CDS-candidates.h5ad'
    with h5py.File(source,'r') as h,h5py.File(destination,'w') as out:
        ptr=h['X/indptr'][:];var=read_frame(h['var']);axis_hash=value_hash(var.index.tolist());genes=var.index.to_numpy(dtype=str)
        out.attrs['encoding-type']='anndata';out.attrs['encoding-version']='0.1.0';frame(out.create_group('var'),var)
        x=out.create_group('X');x.attrs['encoding-type']='csr_matrix';x.attrs['encoding-version']='0.1.0'
        values=x.create_dataset('data',shape=(0,),maxshape=(None,),dtype='<f8',chunks=(1000000,));indices=x.create_dataset('indices',shape=(0,),maxshape=(None,),dtype='<i4',chunks=(1000000,))
        for start in range(0,n,chunk):
            stop=min(start+chunk,n);lo,hi=ptr[start],ptr[stop];d=h['X/data'][lo:hi];gi=h['X/indices'][lo:hi]
            bad=~np.isfinite(d)|(d<0)|(d!=np.floor(d))
            numeric.update(stored_values=len(d),nonfinite=int((~np.isfinite(d)).sum()),negative=int((d<0).sum()),noninteger=int((d!=np.floor(d)).sum()))
            if (gi<0).any() or (gi>=g).any():raise ValueError('gene_coordinate_out_of_bounds')
            invalid=np.bincount(np.repeat(np.arange(stop-start),np.diff(ptr[start:stop+1])),weights=bad.astype(int),minlength=stop-start)>0
            matrix=sparse.csr_matrix((d,gi,ptr[start:stop+1]-lo),shape=(stop-start,g));matrix.sum_duplicates();matrix.eliminate_zeros();matrix.sort_indices()
            sums=np.asarray(matrix.sum(axis=1)).ravel();detected=np.diff(matrix.indptr);source_member=membership[start:stop]>=0;candidate=(sums>=500)&~source_member
            all_features+=np.asarray(matrix.sum(axis=0)).ravel();all_detected+=np.bincount(matrix.indices,minlength=g)
            umi_hist.update(Counter(sums.astype(np.int64).tolist()));detected_hist.update(Counter(detected.tolist()));totals_sum+=int(sums.sum())
            barcodes=annotation_slice(h['obs/_index'],start,stop);sample=annotation_slice(h['obs/sample'],start,stop);hashes=[]
            for i in range(stop-start):
                a,b=matrix.indptr[i:i+2];digest=hashlib.sha256();digest.update(axis_hash.encode());digest.update(matrix.indices[a:b].astype('<i8').tobytes());digest.update(matrix.data[a:b].astype('<f8').tobytes());hashes.append(digest.hexdigest())
            qc=pd.DataFrame({'record_id':[value_hash([input_sha,i]) for i in range(start,stop)],'input_sha256':input_sha,'study_id':'McFaline2024_GxE2',
                'row_index':np.arange(start,stop),'source_barcode':barcodes,'source_sample':sample,'gene_axis_sha256':axis_hash,
                'computed_total_counts':sums,'computed_detected_genes':detected,'computed_numeric_valid':~invalid,'computed_count_sha256':hashes,
                'source_CDS_member':source_member,'source_500_UMI_threshold_met':sums>=500,
                'RNA_annotation_applicability':np.where(source_member,'source_CDS_cell',np.where(candidate,'source_threshold_candidate_without_CDS','not_established_source_cell'))})
            qc.to_parquet(output/'records'/f'{start//chunk:04d}.parquet',index=False,compression='zstd')
            if source_member.any():
                part=qc.loc[source_member].copy();part['CDS_metadata_row']=membership[start:stop][source_member];selected_parts.append(part)
            if candidate.any():
                part=qc.loc[candidate].copy();part['candidate_view_row']=np.arange(len(candidate_ptr)-1,len(candidate_ptr)-1+int(candidate.sum()));candidate_parts.append(part)
                selected=matrix[candidate];a=len(values);b=a+selected.nnz;values.resize((b,));indices.resize((b,));values[a:b]=selected.data;indices[a:b]=selected.indices
                candidate_ptr.extend((selected.indptr[1:]+candidate_ptr[-1]).tolist())
            counts.update(records=stop-start,source_CDS=int(source_member.sum()),non_CDS_source_threshold_candidates=int(candidate.sum()),not_established_source_cell=int((~source_member&~candidate).sum()),source_CDS_below_500=int((source_member&(sums<500)).sum()),zero_libraries=int((sums==0).sum()),invalid_records=int(invalid.sum()))
            if start//chunk%10==0:print('GxE2 RNA records '+str(stop)+'/'+str(n),flush=True)
        candidates=pd.concat(candidate_parts,ignore_index=True) if candidate_parts else pd.DataFrame()
        x.create_dataset('indptr',data=np.asarray(candidate_ptr,np.int64));x.attrs['shape']=(len(candidates),g)
        frame(out.create_group('obs'),candidates.set_index('source_barcode'))
    cells=pd.concat(selected_parts,ignore_index=True).sort_values('CDS_metadata_row').reset_index(drop=True)
    if not np.array_equal(cells.CDS_metadata_row,np.arange(len(metadata))) or not cells.source_barcode.equals(metadata.source_barcode):raise ValueError('source_metadata_order_mismatch')
    cells['source_reported_UMI']=metadata['n.umi'];cells['source_UMI_matches']=cells.computed_total_counts==metadata['n.umi']
    cells.to_parquet(output/'CDS-records.parquet',index=False);candidates.to_parquet(output/'non-CDS-candidate-records.parquet',index=False)
    genes_table=var.reset_index(names='source_gene');genes_table['computed_sum']=all_features;genes_table['computed_detected_records']=all_detected;genes_table.to_parquet(output/'genes.parquet',index=False)
    pd.DataFrame(sorted(umi_hist.items()),columns=['UMI','raw_barcode_records']).to_parquet(output/'all-record-UMI-histogram.parquet',index=False)
    pd.DataFrame(sorted(detected_hist.items()),columns=['detected_genes','raw_barcode_records']).to_parquet(output/'all-record-detected-histogram.parquet',index=False)
    if sum(umi_hist.values())!=n or sum(umi*number for umi,number in umi_hist.items())!=totals_sum or totals_sum!=identity['coordinate']['coordinate_count_sum']:raise ValueError('raw_count_scope_or_sum_mismatch')
    if hash_file(source)!=identity['coordinate']['cache_sha256']:raise ValueError('source_view_changed_during_scan')
    report={'status':'completed','phase':'all_raw_barcode_record_identity_and_QC','code_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'code_sha256':hash_file(Path(__file__)),'source_identity_sha256':hash_file(cache/'identity.json'),'coordinate_input_sha256':input_sha,'coordinate_cache_sha256':identity['coordinate']['cache_sha256'],
        'numeric':dict(numeric),'coverage':dict(counts),'source_CDS_UMI_disagreements':int((~cells.source_UMI_matches).sum()),'all_record_UMI_sum':totals_sum,
        'source_reported_500_UMI_starting_cells':1052205,'actual_500_UMI_records':int(counts['source_CDS']-counts['source_CDS_below_500']+counts['non_CDS_source_threshold_candidates']),
        'gene_axis_sha256':axis_hash,'genes':g,'inputs_unchanged':True,'interpretation':'43 million barcode count records are not 43 million established cells; source 500 UMI threshold is an applicability tag, not a new raw-data filter',
        'duration_seconds':time.monotonic()-started,'completed_at':datetime.now(timezone.utc).isoformat()}
    report['artifacts']={str(p.relative_to(output)):hash_file(p) for p in output.rglob('*') if p.is_file()};write_json(output/'report.json',report);return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cache',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    print(json.dumps(run(a.cache,a.output)),flush=True)
