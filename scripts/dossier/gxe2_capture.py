#!/usr/bin/env python
"""Audit every GxE2 capture record against source labels without replacing them."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import io
import json
from pathlib import Path
import subprocess
import time
import numpy as np
import pandas as pd
from scipy import sparse
from rna import hash_file
from profile_responses import write_json

ROOT = Path(__file__).resolve().parents[2]
PREFIX = 'GSM7056149_sciPlexGxE_2_'
CROSSWALK = {'01E': 'A', '02E': 'B', '03E': 'C', '04E': 'D'}


class CaptureText(io.TextIOBase):
    """Count literal NULs before the CSV C parser interprets them as empty."""
    def __init__(self, path):
        self.stream = gzip.open(path, 'rt'); self.nuls = 0; self.lines = 0

    def read(self, size=-1):
        text = self.stream.read(size)
        self.nuls += text.count('\x00'); self.lines += text.count('\n')
        return text

    def readable(self):
        return True

    def close(self):
        self.stream.close(); super().close()


def guide_calls(indices, values, names):
    """Reproduce the published top-five rule; ties are flagged, not resolved as truth."""
    if not len(values) or values.sum() == 0:
        return [], None, None, False
    order = np.lexsort((names[indices], -values))[:5]
    ordered = values[order]; ratios = ordered / values.sum()
    r12 = ordered[0] / ordered[1] if len(ordered) > 1 and ordered[1] else np.inf
    r23 = ordered[1] / ordered[2] if len(ordered) > 2 and ordered[2] else np.inf if len(ordered)>1 else np.nan
    keep = ratios > .3
    keep[0] |= r12 >= 2 or r23 >= 2
    called = sorted(names[indices[order[keep]]].tolist())
    tie = bool(len(ordered) > 1 and (np.diff(ordered) == 0).any())
    return called, float(ordered[0]) if called else None, float(ratios[0]) if called else None, tie


def run(cache, output):
    started = time.monotonic(); output.mkdir(parents=True, exist_ok=False)
    identity = json.loads((cache/'identity.json').read_text())
    source = ROOT/'data/raw/mcfaline_figueroa2024'
    paths = {suffix: source/(PREFIX+suffix) for suffix in ['gRNATable_reads.out.txt.gz', 'gRNASampleSheet.txt.gz', 'hashTable.out.txt.gz', 'hash_sample_sheet.txt.gz', 'hash_whitelist_rep1.txt.gz', 'hash_whitelist_rep2.txt.gz']}
    for path in paths.values():
        if hash_file(path) != identity['input_sha256'][str(path.relative_to(ROOT))]:
            raise ValueError('capture_input_changed')
    metadata = pd.read_parquet(cache/'all-CDS-source-metadata.parquet')
    if metadata.source_barcode.duplicated().any() or metadata.new_cell.duplicated().any():
        raise ValueError('ambiguous_source_CDS_identity')
    metadata[['source_barcode', 'new_cell']].to_parquet(output/'source-join-keys.parquet', index=False)
    index = pd.Index(metadata.new_cell); n = len(metadata)
    whitelist = pd.read_csv(paths['gRNASampleSheet.txt.gz'], sep='\t', header=None, names=['source_guide', 'source_sequence'])
    whitelist['analysis_guide_alias'] = whitelist.source_guide.replace({'originalIDTorder_1': 'random_region_35'})
    whitelist['gene'] = whitelist.analysis_guide_alias.str.split('_').str[0]
    whitelist['source_processing_sequence'] = [s[:19] if g == 'random' else s for s,g in zip(whitelist.source_sequence, whitelist.gene)]
    ambiguous_names=set(whitelist.loc[whitelist.analysis_guide_alias.duplicated(keep=False),'analysis_guide_alias'])
    if whitelist.groupby('analysis_guide_alias').gene.nunique().max()>1:raise ValueError('same_guide_name_multiple_genes')
    whitelist['guide_name_identifies_one_sequence']=~whitelist.analysis_guide_alias.isin(ambiguous_names)
    whitelist.to_parquet(output/'guide-whitelist.parquet', index=False)
    guide_names = whitelist.analysis_guide_alias.drop_duplicates().to_numpy(); guide_index = pd.Index(guide_names)
    matrix = sparse.csr_matrix((n, len(guide_names)), dtype=np.int64)
    counts = Counter(); guide_rows = Counter(); prefixes = Counter()
    # Only source-CDS matched capture counts are materialized as an analysis view.
    # Every source capture row still contributes to the scope and numeric audit.
    with CaptureText(paths['gRNATable_reads.out.txt.gz']) as stream:
        for batch in pd.read_csv(stream, sep='\t', header=None, names=['sample','barcode','guide','source_UMI_placeholder','read_count'], keep_default_na=False, dtype={'read_count':np.int64}, chunksize=2000000):
            prefix = batch.barcode.str.slice(0,3); mapped = prefix.map(CROSSWALK)
            if mapped.isna().any() or (batch.read_count<0).any() or batch.source_UMI_placeholder.ne('').any(): raise ValueError('unexpected_guide_capture_format')
            names = batch.guide.replace({'originalIDTorder_1':'random_region_35'})
            rows = index.get_indexer(mapped + '_' + batch.barcode.str.slice(4)); cols = guide_index.get_indexer(names)
            keep = (rows>=0)&(cols>=0)
            matrix += sparse.csr_matrix((batch.read_count.to_numpy()[keep], (rows[keep], cols[keep])), shape=matrix.shape)
            counts.update(raw_records=len(batch), raw_reads=int(batch.read_count.sum()), matched_CDS_records=int(keep.sum()), matched_CDS_reads=int(batch.read_count.to_numpy()[keep].sum()), unknown_guide_records=int((cols<0).sum()))
            prefixes.update(prefix.value_counts().to_dict()); guide_rows.update(names.value_counts().to_dict())
            # Sorted source barcodes are summarized without retaining tens of millions of strings.
            if counts['raw_records'] % 10000000 == 0: print('GxE2 guide capture '+str(counts['raw_records']), flush=True)
        counts['literal_NULs'] = stream.nuls; counts['physical_lines'] = stream.lines
    if counts['literal_NULs'] != counts['raw_records'] or counts['physical_lines'] != counts['raw_records']: raise ValueError('guide_placeholder_or_line_scope_mismatch')
    sparse.save_npz(output/'CDS-guide-read-counts.npz', matrix, compressed=True)
    pd.DataFrame([{'guide':g,'capture_records':v} for g,v in sorted(guide_rows.items())]).to_parquet(output/'all-capture-guide-coverage.parquet',index=False)
    comparisons=[]; gene_lookup=whitelist.set_index('analysis_guide_alias').gene.to_dict()
    for i, row in enumerate(metadata.itertuples()):
        lo,hi = matrix.indptr[i:i+2]
        calls,maximum,ratio,tie = guide_calls(matrix.indices[lo:hi], matrix.data[lo:hi], guide_names)
        original = sorted(str(row.gRNA_id).split(',')) if pd.notna(row.gRNA_id) else []
        source_genes = sorted(str(row.gene_id).split(',')) if pd.notna(row.gene_id) else []
        genes = sorted(gene_lookup.get(g,'__unknown__') for g in original)
        comparisons.append({'source_barcode':row.source_barcode,'source_join_key':row.new_cell,'recomputed_guides':','.join(calls), 'source_guides':','.join(original),
            'source_guide_call_matches':calls==original, 'source_gene_labels_match_whitelist':genes==source_genes,
            'recomputed_maxCount':maximum,'source_maxCount':row.gRNA_maxCount,
            'source_maxCount_matches': (maximum is None and pd.isna(row.gRNA_maxCount)) or (maximum is not None and maximum==row.gRNA_maxCount),
            'recomputed_topRatio':ratio,'source_topRatio':row.gRNA_topRatio,
            'source_topRatio_matches': (ratio is None and pd.isna(row.gRNA_topRatio)) or (ratio is not None and np.isclose(ratio,row.gRNA_topRatio,rtol=1e-9,atol=1e-12)),
            'top_five_read_count_tie':tie, 'total_whitelisted_reads':int(matrix.data[lo:hi].sum()),
            'source_call_has_ambiguous_sequence_name':bool(set(original)&ambiguous_names),
            'cross_library_key_evidence':'candidate_prefix_crosswalk_checked_against_source_labels_not_physical_truth'})
    checked=pd.DataFrame(comparisons);checked.to_parquet(output/'guide-source-reproduction.parquet',index=False)
    del matrix,comparisons
    # Hash oligos must be interpreted jointly with the two source replicate whitelists.
    sheets=pd.concat([pd.read_csv(paths['hash_whitelist_rep1.txt.gz'],sep='\t'),pd.read_csv(paths['hash_whitelist_rep2.txt.gz'],sep='\t')],ignore_index=True)
    if sheets.duplicated(['hash_oligo','replicate']).any():raise ValueError('ambiguous_hash_replicate_design')
    sheets.to_parquet(output/'hash-whitelist.parquet',index=False)
    hash_sheet=pd.read_csv(paths['hash_sample_sheet.txt.gz'],sep='\t',header=None,names=['hash','sequence','axis'])
    if hash_sheet.hash.duplicated().any() or hash_sheet.sequence.duplicated().any():raise ValueError('ambiguous_hash_capture_whitelist')
    hash_sheet.to_parquet(output/'hash-capture-whitelist.parquet',index=False)
    barcodes=pd.Index(metadata.source_barcode); hash_index=pd.Index(hash_sheet.hash)
    hashes=sparse.csr_matrix((n,len(hash_index)),dtype=np.int64); hash_counts=Counter()
    with CaptureText(paths['hashTable.out.txt.gz']) as stream:
        for batch in pd.read_csv(stream,sep='\t',header=None,names=['sample','barcode','hash','axis','UMI'],keep_default_na=False,dtype={'axis':np.int64,'UMI':np.int64},chunksize=2000000):
            rows=barcodes.get_indexer(batch.barcode);cols=hash_index.get_indexer(batch['hash']);keep=rows>=0
            if (cols<0).any() or (batch.UMI<0).any() or batch.axis.ne(1).any():raise ValueError('invalid_hash_capture')
            hashes+=sparse.csr_matrix((batch.UMI.to_numpy()[keep],(rows[keep],cols[keep])),shape=hashes.shape)
            hash_counts.update(raw_records=len(batch),raw_UMI=int(batch.UMI.sum()),matched_CDS_records=int(keep.sum()),matched_CDS_UMI=int(batch.UMI.to_numpy()[keep].sum()))
            if hash_counts['raw_records']%10000000==0:print('GxE2 hash capture '+str(hash_counts['raw_records']),flush=True)
        if stream.nuls or stream.lines!=hash_counts['raw_records']:raise ValueError('unexpected_hash_lines')
    sparse.save_npz(output/'CDS-hash-UMI-counts.npz',hashes,compressed=True)
    totals=np.asarray(hashes.sum(axis=1)).ravel();maxima=hashes.max(axis=1).toarray().ravel();assigned=hash_index.get_indexer(metadata.top_oligo)
    if (assigned<0).any():raise ValueError('source_hash_not_in_capture_whitelist')
    raw_assigned=np.asarray(hashes[np.arange(n),assigned]).ravel()
    hash_checks=metadata[['source_barcode','top_oligo','replicate','RT_well','cell_line','treatment','dose','hash_plate','hash_umis','top_to_second_best_ratio']].copy()
    hash_checks['total_raw_hash_UMI']=totals;hash_checks['assigned_raw_hash_UMI']=raw_assigned
    hash_checks['source_UMI_matches_total']=totals==metadata.hash_umis;hash_checks['assigned_hash_is_raw_UMI_maximum']=raw_assigned==maxima
    hash_checks['RT_replicate_matches']=np.where(metadata.RT_well<=192,'replicate_1','replicate_2')==metadata.replicate
    joined=metadata.merge(sheets,left_on=['top_oligo','replicate'],right_on=['hash_oligo','replicate'],how='left',validate='m:1',suffixes=('','_whitelist'))
    hash_checks['source_condition_matches_whitelist']=np.logical_and.reduce([joined[c].eq(joined[c+'_whitelist']).to_numpy() for c in ['cell_line','treatment','dose','hash_plate']])
    hash_checks.to_parquet(output/'hash-source-reproduction.parquet',index=False)
    for path in paths.values():
        if hash_file(path)!=identity['input_sha256'][str(path.relative_to(ROOT))]:raise ValueError('capture_changed_during_audit')
    report={'status':'completed','phase':'capture_identity_audit','code_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'code_sha256':hash_file(Path(__file__)),'adapter_identity_sha256':hash_file(cache/'identity.json'),
        'input_sha256':{str(p.relative_to(ROOT)):hash_file(p) for p in paths.values()},'inputs_unchanged':True,'CDS_cells':n,
        'guide_capture':dict(counts),'guide_prefix_records':dict(prefixes),'candidate_crosswalk':CROSSWALK,'crosswalk_is_inferred':True,
        'guide_check_disagreements':{c:int((~checked[c]).sum()) for c in ['source_guide_call_matches','source_gene_labels_match_whitelist','source_maxCount_matches','source_topRatio_matches']},
        'guide_top_five_tied_cells':int(checked.top_five_read_count_tie.sum()),'hash_capture':dict(hash_counts),
        'guide_names_with_multiple_source_sequences':sorted(ambiguous_names),'source_calls_with_ambiguous_sequence_name':int(checked.source_call_has_ambiguous_sequence_name.sum()),
        'hash_check_disagreements':{c:int((~hash_checks[c]).sum()) for c in ['source_UMI_matches_total','assigned_hash_is_raw_UMI_maximum','RT_replicate_matches','source_condition_matches_whitelist']},
        'duration_seconds':time.monotonic()-started,'completed_at':datetime.now(timezone.utc).isoformat()}
    report['artifacts']={str(p.relative_to(output)):hash_file(p) for p in output.rglob('*') if p.is_file()};write_json(output/'report.json',report);return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cache',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    print(json.dumps(run(a.cache,a.output)),flush=True)
