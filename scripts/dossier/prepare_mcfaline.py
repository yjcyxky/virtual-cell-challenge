#!/usr/bin/env python
"""Prepare verified independent GxE1 count/metadata caches, never modifying sources."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import pandas as pd
from convert_seurat_cache import RROOT,make_h5ad
from mcfaline import coordinate_cache,compare_CDS,assigned_design
from profile_responses import write_json
from rna import RNAFile,hash_file
ROOT=Path(__file__).resolve().parents[2]


def prepare(inventory,output):
    start=time.monotonic();commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    if output.exists():raise ValueError('fresh_McFaline_cache_required')
    inv=json.loads((inventory/'report.json').read_text());files={r['file']:r['sha256'] for r in inv['file_results'] if r['source_id']=='mcfaline_figueroa2024' and 'GSM7056148' in r['file']}
    if len(files)!=8:raise ValueError('expected_all_eight_GxE1_inputs')
    for file,digest in files.items():
        if hash_file(ROOT/file)!=digest:raise ValueError('source_hash_mismatch')
    output.mkdir(parents=True);prefix=ROOT/'data/raw/mcfaline_figueroa2024/GSM7056148_sciPlexGxE_1_'
    path=lambda name:Path(str(prefix)+name)
    env=os.environ.copy();env['R_HOME']=str(RROOT);env['LD_LIBRARY_PATH']=str(RROOT/'lib')+':'+str(RROOT.parent)
    for name,directory in [('preprocessed_cds.rds.gz','R-export'),('sgRNATable_out.rds.gz','guide-export')]:
        with (output/(directory+'.log')).open('w') as log:
            subprocess.run(['/lib/ld-linux-aarch64.so.1',str(RROOT/'bin/exec/R'),'--vanilla','--slave','-f',str(Path(__file__).with_name('export_monocle_readonly.R')),'--args',str(path(name)),str((output/directory).resolve())],env=env,check=True,stdout=log,stderr=subprocess.STDOUT)
    raw=coordinate_cache(path('UMI.count.matrix.gz'),path('cell.annotations.txt.gz'),path('gene.annotations.txt.gz'),output/'coordinate-counts.h5ad')
    cds=make_h5ad(output/'R-export/CDS',output/'CDS-counts.h5ad')
    compare=compare_CDS(output/'coordinate-counts.h5ad',output/'CDS-counts.h5ad');compare.to_parquet(output/'CDS-count-comparison.parquet',index=False)
    with RNAFile(output/'CDS-counts.h5ad') as source:
        design=assigned_design(source.obs);design.to_parquet(output/'source-design.parquet',index=False)
        source.obs.reset_index(names='source_barcode').to_parquet(output/'CDS-source-metadata.parquet',index=False)
        metadata=source.obs.copy()
    guides=pd.read_csv(output/'guide-export/table.csv',keep_default_na=False)
    if (guides[['read_count','umi_count']]<0).any().any() or (guides.read_count<guides.umi_count).any():raise ValueError('guide_read_UMI_count_invalid')
    guides.to_parquet(output/'guide-capture.parquet',index=False)
    whitelist=pd.read_csv(path('sgRNA_sequences.txt.gz'),sep='\t');whitelist['captured_protospacer']=whitelist.gRNA_sequence.str.slice(1,20)
    if whitelist.captured_protospacer.duplicated().any():raise ValueError('ambiguous_source_guide_whitelist')
    whitelist.to_parquet(output/'guide-whitelist.parquet',index=False)
    by_sequence=whitelist.set_index('captured_protospacer');guidechecks=[]
    for barcode,r in metadata.iterrows():
        seq=[] if pd.isna(r.protospacer_sequence) else str(r.protospacer_sequence).split(',')
        resolved=[];modes=[];missing=[]
        for s in seq:
            if s not in by_sequence.index:missing.append(s)
            else:
                record=by_sequence.loc[s];resolved.append('NTC' if record.gene=='negative_control' else record.gene);modes.append(record.CRISPR)
        source_genes=[] if pd.isna(r.gene_id) else str(r.gene_id).split(',');source_modes=[] if pd.isna(r.CRISPR_gRNA) else str(r.CRISPR_gRNA).split(',')
        guidechecks.append({'source_barcode':barcode,'assigned_guide_components':len(seq),'unknown_whitelist_sequences':missing,
            'gene_labels_match_whitelist':sorted(resolved)==sorted(source_genes),'effector_labels_match_whitelist':sorted(modes)==sorted(source_modes),
            'assignment_is_source_inference':True})
    pd.DataFrame(guidechecks).to_parquet(output/'guide-label-checks.parquet',index=False)
    capture=guides.loc[guides.gRNA.isin(whitelist.captured_protospacer)].groupby(['new_cell','gRNA'],as_index=False).read_count.sum()
    capture['total_whitelisted_reads']=capture.groupby('new_cell').read_count.transform('sum')
    capture['read_fraction']=capture.read_count/capture.total_whitelisted_reads
    capture['source_rule_assigned']=(capture.read_count>=10)&(capture.read_fraction>=.3)
    capture.to_parquet(output/'guide-assignment-evidence.parquet',index=False)
    called=capture.loc[capture.source_rule_assigned].groupby('new_cell').gRNA.apply(set).to_dict()
    assignment_checks=[]
    for barcode,r in metadata.iterrows():
        source_call=set() if pd.isna(r.protospacer_sequence) else set(str(r.protospacer_sequence).split(','))
        assignment_checks.append({'source_barcode':barcode,'source_join_key':r.new_cell,
            'source_rule_call_matches':source_call==called.get(r.new_cell,set()),
            'source_assigned_sequences':sorted(source_call),'recomputed_sequences':sorted(called.get(r.new_cell,set()))})
    pd.DataFrame(assignment_checks).to_parquet(output/'guide-assignment-reproduction.parquet',index=False)
    hashes=pd.read_csv(path('hashTable.out.txt.gz'),sep='\t',header=None,names=['sample','barcode','hash','axis','umi'])
    sheet=pd.read_csv(path('hash_sample_sheet.txt.gz'),sep='\t',header=None,names=['hash','sequence','axis'])
    if sheet.hash.duplicated().any() or sheet.sequence.duplicated().any() or not set(hashes['hash'])<=set(sheet.hash) or (hashes.umi<0).any():raise ValueError('invalid_hash_mapping_or_counts')
    sheet.to_parquet(output/'hash-design.parquet',index=False)
    hash_counts=hashes.groupby(['barcode','hash']).umi.sum();hash_totals=hashes.groupby('barcode').umi.sum()
    hash_max=hash_counts.groupby(level=0).max()
    hash_checks=[]
    for barcode,r in metadata.iterrows():
        top=hash_counts.get((barcode,r.top_oligo_W));total=hash_totals.get(barcode)
        hash_checks.append({'source_barcode':barcode,'assigned_hash':r.top_oligo_W,'hash_in_whitelist':r.top_oligo_W in set(sheet.hash),
            'assigned_hash_raw_UMI':top,'all_hash_raw_UMI':total,'source_hash_umis_W':r.hash_umis_W,
            'assigned_hash_has_maximum_raw_UMI':top==hash_max.get(barcode),
            'source_UMI_field_matches_total':total==r.hash_umis_W,
            'source_hash_enrichment_ratio':r.top_to_second_best_ratio_W,
            'assignment_is_source_inference':True})
    pd.DataFrame(hash_checks).to_parquet(output/'hash-assignment-evidence.parquet',index=False)
    for file,digest in files.items():
        if hash_file(ROOT/file)!=digest:raise ValueError('original_source_changed')
    result={'status':'completed','completed_at':datetime.now(timezone.utc).isoformat(),'duration_seconds':time.monotonic()-start,
        'input_sha256':files,'inputs_unchanged':True,'code_commit':commit,'code':{n:hash_file(Path(__file__).with_name(n)) for n in ['prepare_mcfaline.py','mcfaline.py','export_monocle_readonly.R','convert_seurat_cache.py']},
        'coordinate':raw,'CDS':cds,'CDS_cells_compared':len(compare),'CDS_differing_values':int(compare.differing_values.sum()),
        'source_cells_without_CDS_metadata':raw['shape'][0]-len(compare),'genetic_assignment_status':design.genetic_response_assignment_status.value_counts().to_dict(),
        'assignment_limitations':design.assignment_limitation.value_counts().to_dict(),'guide_capture_rows':len(guides),
        'guide_whitelist_unknown_capture_rows':int((~guides.gRNA.isin(whitelist.captured_protospacer)).sum()),
        'guide_label_check_failures':int(sum(bool(x['unknown_whitelist_sequences']) or not x['gene_labels_match_whitelist'] or not x['effector_labels_match_whitelist'] for x in guidechecks)),
        'guide_source_rule_disagreements':sum(not x['source_rule_call_matches'] for x in assignment_checks),
        'guide_join_collision_cells':int(design.guide_join_key_collision.sum()),
        'hash_capture_rows':len(hashes),'hash_total_UMI_disagreements':sum(not x['source_UMI_field_matches_total'] for x in hash_checks),
        'source_enrichment_call_not_raw_UMI_maximum':sum(not x['assigned_hash_has_maximum_raw_UMI'] for x in hash_checks),
        'source_metadata_not_groundtruth':True,'reproduce':sys.argv}
    result['artifacts']={str(p.relative_to(output)):hash_file(p) for p in sorted(output.rglob('*')) if p.is_file()}
    write_json(output/'identity.json',result);return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--inventory',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    try:r=prepare(a.inventory,a.output)
    except Exception as e:
        if a.output.exists():write_json(a.output/'failure.json',{'status':'failed','error':str(e),'type':type(e).__name__})
        raise
    print(json.dumps({k:v for k,v in r.items() if k not in ['artifacts','CDS','input_sha256','code']}))
