"""Conservative source-library and cell-record relations; never deduplicate inputs."""
import hashlib
from pathlib import Path
import re
import numpy as np
import pandas as pd
from rna import value_hash


def author_library_lookup(manifests,geo_tables):
    fastq={};accessions={}
    for context,frame in manifests.items():
        for row in frame.to_dict('records'):
            for value in row.values():
                if not isinstance(value,str):continue
                match=re.fullmatch(r'(.+)_R[12]_001.fastq.gz',Path(value).name)
                if not match:continue
                item=('replogle',context,int(row['gemgroup']),str(row['library']),'guide' if 'sgRNA' in value else 'RNA')
                fastq.setdefault(match[1],set()).add(item)
    for context,frame in geo_tables.items():
        for row in frame.to_dict('records'):
            mode='guide' if row['source_library'].endswith('_sgRNA') else 'RNA'
            for accession in re.findall(r'SRX\d+',row['source_relations']):
                accessions.setdefault(accession,set()).add(('nadig',context,int(row['gem_group']),row['source_library'],mode,row['source_GSM']))
    return fastq,accessions


def resolve_library(record,lookup):
    fastq,accessions=lookup;families=list(record['supervised_overlap'])
    if not families:return None
    candidates=fastq.get(str(record['library_name']).strip(),set()) if families==['replogle2022'] else accessions.get(record['experiment_accession'],set())
    if len(candidates)!=1:return {'status':'indeterminate','reason':'author_library_relation_missing_or_ambiguous','candidate_count':len(candidates)}
    item=next(iter(candidates));family,context,gem,library,modality=item[:5]
    return {'status':'completed','family':family,'context':context,'GEM':gem,'source_library':library,'modality':modality,
        'source_GSM':item[5] if len(item)>5 else None,'capture_key':value_hash([family,context,gem]),
        'evidence_kind':'exact_author_FASTQ_basename_to_GEM' if family=='replogle' else 'exact_GEO_SRX_relation_to_GEM'}


def barcode_core(values):
    """Only documented 10x bare sequence or sequence-GEM identifiers are eligible."""
    series=pd.Series(values).astype('string');valid=series.str.fullmatch(r'[ACGT]{16}(?:-[0-9]+)?',na=False)
    return series.str.replace(r'-[0-9]+$','',regex=True).where(valid)


def projected_fingerprints(matrix,source_axis,target_axis):
    if len(set(source_axis))!=len(source_axis) or len(set(target_axis))!=len(target_axis):raise ValueError('ambiguous_projection_gene_axis')
    lookup={gene:i for i,gene in enumerate(source_axis)}
    if not set(target_axis)<=set(source_axis):raise ValueError('unmeasured_genes_cannot_be_zero_filled')
    x=matrix[:,[lookup[gene] for gene in target_axis]].tocsr();x.sum_duplicates();x.eliminate_zeros();x.sort_indices()
    axis=value_hash(target_axis);fingerprints=[]
    for i in range(x.shape[0]):
        lo,hi=x.indptr[i:i+2];digest=hashlib.sha256(axis.encode());digest.update(x.indices[lo:hi].astype('<i8').tobytes());digest.update(x.data[lo:hi].astype('<f8').tobytes());fingerprints.append(digest.hexdigest())
    return fingerprints,np.asarray(x.sum(axis=1)).ravel(),np.diff(x.indptr),axis


def compare_pair(left,right,relation):
    """Exact barcodes alone never establish identity across independent libraries."""
    if left.source_barcode.duplicated().any() or right.source_barcode.duplicated().any():raise ValueError('ambiguous_within_file_barcode')
    matches=left.merge(right,on='source_barcode',suffixes=('_A','_B'),validate='one_to_one')
    equal=(matches.gene_axis_sha256_A==matches.gene_axis_sha256_B)&(matches.computed_count_sha256_A==matches.computed_count_sha256_B)
    same_capture=bool(relation.get('capture_A')) and relation.get('capture_A')==relation.get('capture_B')
    both_RNA=relation.get('RNA_A',False) and relation.get('RNA_B',False)
    same_sample=bool(relation.get('sample_A')) and relation.get('sample_A')==relation.get('sample_B')
    same_study=bool(relation.get('study_A')) and relation.get('study_A')==relation.get('study_B')
    matches['full_native_axis_and_counts_equal']=equal
    matches['confirmed_capture_barcode_link']=same_capture
    matches['confirmed_exact_RNA_expression_copy']=same_capture&both_RNA&equal
    matches['relation_status']='confirmed_capture_barcode_link' if same_capture else 'candidate_shared_sample_barcode' if same_sample and same_study else 'barcode_or_count_coincidence_not_identity_proof'
    if relation.get('capture_A') and relation.get('capture_B') and not same_capture:matches['relation_status']='distinct_documented_captures_barcode_not_identity'
    matches['physical_singlet_cell_truth']=False;matches['independent_biological_replicates']=None
    return matches
