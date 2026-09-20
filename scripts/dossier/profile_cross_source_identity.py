#!/usr/bin/env python
"""Audit all scBase source relations and all supervised-source library overlaps."""
import argparse
from collections import Counter
from datetime import datetime,timezone
from itertools import combinations
import json
from pathlib import Path
import shutil
import subprocess
import time
import numpy as np
import pandas as pd
from cross_source_identity import author_library_lookup,resolve_library,barcode_core,projected_fingerprints,compare_pair
from profile_background import matrix_in_memory
from profile_responses import write_json
from rna import RNAFile,hash_file,quantiles

ROOT=Path(__file__).resolve().parents[2]
BASE_COLUMNS=['record_id','row_index','source_barcode','gene_axis_sha256','computed_count_sha256','computed_total_counts','computed_detected_genes']


def checksum_index(folder):
    return {name:digest for digest,name in [line.split('  ',1) for line in (folder/'SHA256SUMS').read_text().splitlines()]}


def checked(folder,hashes,name):
    if name not in hashes or hash_file(folder/name)!=hashes[name]:raise ValueError('frozen_source_component_changed:'+str(folder/name))
    return folder/name


def run(base,provenance,replogle,nadig,output):
    started=time.monotonic();br=json.loads((base/'report.json').read_text());pr=json.loads((provenance/'report.json').read_text())
    if br['status']!='completed' or br['completed_files']!=1808 or br['failed_files'] or pr['status']!='completed':raise ValueError('complete_source_assessments_required')
    if br['identity']['provenance_sha256']!=hash_file(provenance/'report.json'):raise ValueError('provenance_not_shared')
    output.mkdir(parents=True,exist_ok=False);bh=checksum_index(base);ph=checksum_index(provenance)
    source_rows=pd.read_parquet(checked(provenance,ph,'samples.parquet')).to_dict('records')
    structure={family:folder for family,folder in [('replogle',replogle),('nadig',nadig)]};sh={key:checksum_index(folder) for key,folder in structure.items()}
    manifests={context:pd.read_csv(checked(replogle,sh['replogle'],'evidence/'+prefix+'_raw_files.csv')) for context,prefix in [('K562_essential','KD6'),('K562_gwps','KD8'),('rpe1','RD7')]}
    geo={context:pd.read_parquet(checked(nadig,sh['nadig'],context+'/GEO_library_associations.parquet')) for context in ['hepg2','jurkat']}
    lookup=author_library_lookup(manifests,geo);catalog=[];frames=[];by_accession={};known={}
    for index,record in enumerate(source_rows):
        accession=record['experiment_accession'];report=json.loads(checked(base,bh,accession+'/report.json').read_text());summary=report['summary']
        cells=pd.read_parquet(checked(base,bh,accession+'/cells.parquet'),columns=BASE_COLUMNS)
        if len(cells)!=summary['n_cells'] or cells.record_id.duplicated().any():raise ValueError('source_record_scope_changed')
        library=resolve_library(record,lookup)
        if library and library['status']!='completed':raise ValueError('supervised_library_relation_requires_explicit_evidence')
        if library:known[accession]=library
        eligibility=summary['inference_eligibility'];sample=record.get('sample_ref');study=record.get('study_ref')
        row={'file_index':index,'experiment_accession':accession,'stored_records':len(cells),'source_input_sha256':record['source_file_sha256_from_inventory'],
            'source_sample':sample if isinstance(sample,str) and sample else 'unresolved:'+accession,
            'source_study':study if isinstance(study,str) and study else 'unresolved:'+accession,
            'source_sample_aliases':record['sample_accessions'],'source_study_aliases':record['study_accessions'],
            'source_library_name':record.get('library_name'),'source_title':record.get('experiment_title'),'source_cell_line':record.get('source_cell_line'),
            'source_species':record.get('upstream_species'),'inference_eligibility':eligibility['status'],'inference_reason':eligibility['reason'],
            'RNA_eligible':eligibility['status']!='not_applicable','author_library_relation':library,'capture_key':library['capture_key'] if library else None,
            'within_file_barcode_duplicates':int(cells.source_barcode.duplicated().sum()),'source_cells_sidecar_sha256':bh[accession+'/cells.parquet'],
            'source_gene_axis_sha256':summary['gene_axis_sha256'],'independent_biological_replicates':None}
        cells['file_index']=np.int16(index);cells['study_index']=row['source_study'];frames.append(cells);catalog.append(row);by_accession[accession]=index
        if (index+1)%100==0:print('all scBase record identities '+str(index+1)+'/1808',flush=True)
    all_cells=pd.concat(frames,ignore_index=True);del frames
    if len(all_cells)!=13255146:raise ValueError('incomplete_scBase_record_scope')
    if len(known)!=48:raise ValueError('incomplete_registered_supervised_overlap_scope')
    all_cells.to_parquet(output/'all-scbase-record-identities.parquet',index=False,compression='zstd')
    pd.DataFrame(catalog).to_parquet(output/'scbase-source-catalog.parquet',index=False);write_json(output/'scbase-source-catalog.json',catalog)
    # Bare identifiers are deliberately not physical identities. Preserve the
    # complete distribution of collisions across source studies and files.
    collision=all_cells.groupby('source_barcode',sort=True).agg(stored_records=('record_id','size'),source_files=('file_index','nunique'),source_studies=('study_index','nunique')).reset_index()
    collision.to_parquet(output/'all-barcode-collision-denominators.parquet',index=False)
    exact_keys=['source_barcode','gene_axis_sha256','computed_count_sha256'];mask=all_cells.duplicated(exact_keys,keep=False)
    candidates=all_cells.loc[mask].copy();candidates['candidate_group']=candidates.groupby(exact_keys,sort=True).ngroup()
    candidates['interpretation']='Barcode and complete count equality candidate only; source library and modality relations required for confirmation.'
    candidates.to_parquet(output/'barcode-and-count-candidate-membership.parquet',index=False)
    # Evaluate every shared sample pair, not just the visibly similar libraries.
    pair_rows=[];pair_root=output/'shared-sample-pairs';pair_root.mkdir();groups={}
    for row in catalog:groups.setdefault(row['source_sample'],[]).append(row)
    for sample,group in sorted(groups.items()):
        if len(group)<2 or sample.startswith('unresolved:'):continue
        local={r['file_index']:all_cells.loc[all_cells.file_index==r['file_index'],BASE_COLUMNS] for r in group}
        for a,b in combinations(group,2):
            relation={'capture_A':a['capture_key'],'capture_B':b['capture_key'],'RNA_A':a['RNA_eligible'],'RNA_B':b['RNA_eligible'],
                'sample_A':a['source_sample'],'sample_B':b['source_sample'],'study_A':a['source_study'],'study_B':b['source_study']}
            name=a['experiment_accession']+'__'+b['experiment_accession'];left=local[a['file_index']];right=local[b['file_index']]
            row={'source_sample':sample,'experiment_A':a['experiment_accession'],'experiment_B':b['experiment_accession'],
                'records_A':len(left),'records_B':len(right),'capture_A':a['capture_key'],'capture_B':b['capture_key'],
                'RNA_A':a['RNA_eligible'],'RNA_B':b['RNA_eligible'],'source_study_A':a['source_study'],'source_study_B':b['source_study'],
                'library_A':a['source_library_name'],'library_B':b['source_library_name'],'independent_biological_replicates':None}
            if a['within_file_barcode_duplicates'] or b['within_file_barcode_duplicates']:
                row.update(status='indeterminate',reason='within_file_barcode_identity_ambiguous',shared_barcodes=None,confirmed_exact_RNA_expression_copies=None)
            else:
                pairs=compare_pair(left,right,relation);pairs.to_parquet(pair_root/(name+'.parquet'),index=False)
                row.update(status='completed',shared_barcodes=len(pairs),equal_native_count_vectors=int(pairs.full_native_axis_and_counts_equal.sum()),
                    confirmed_capture_barcode_links=int(pairs.confirmed_capture_barcode_link.sum()),confirmed_exact_RNA_expression_copies=int(pairs.confirmed_exact_RNA_expression_copy.sum()),
                    relation_status_counts=pairs.relation_status.value_counts().to_dict(),cell_evidence='shared-sample-pairs/'+name+'.parquet')
            pair_rows.append(row)
    pd.DataFrame(pair_rows).to_parquet(output/'all-shared-sample-library-pairs.parquet',index=False);write_json(output/'all-shared-sample-library-pairs.json',pair_rows)
    print('all shared sample library pairs '+str(len(pair_rows))+' completed',flush=True)
    # Original source native axes are wholly present in scBase. No unmeasured
    # features are filled with zero and no counts are normalized for identity.
    original={};gene_axes={}
    for family,context in [('replogle','K562_essential'),('replogle','K562_gwps'),('replogle','rpe1'),('nadig','hepg2'),('nadig','jurkat')]:
        folder=structure[family];hashes=sh[family]
        original[(family,context)]=pd.read_parquet(checked(folder,hashes,context+'/cells.parquet'),columns=BASE_COLUMNS+['source_batch','source_task','source_guide_id','source_target_gene'])
        genes=pd.read_parquet(checked(folder,hashes,context+'/genes.parquet'));gene_axes[(family,context)]=genes.source_gene.tolist()
    links_root=output/'supervised-record-links';links_root.mkdir();overlaps=[]
    source_by_accession={r['experiment_accession']:r for r in source_rows}
    for accession,library in known.items():
        record=source_by_accession[accession];family,context=library['family'],library['context'];source=original[(family,context)]
        source=source.loc[source.source_batch.astype(str)==str(library['GEM'])].copy()
        if len(source) and not source.source_barcode.str.endswith('-'+str(library['GEM'])).all():raise ValueError('original_barcode_GEM_suffix_contradiction')
        source['barcode_core']=barcode_core(source.source_barcode.to_numpy()).to_numpy()
        if source.barcode_core.isna().any() or source.barcode_core.duplicated().any():raise ValueError('original_capture_barcode_not_unique')
        current=all_cells.loc[all_cells.file_index==by_accession[accession],BASE_COLUMNS].copy();current['barcode_core']=barcode_core(current.source_barcode.to_numpy()).to_numpy()
        if current.barcode_core.isna().any() or current.barcode_core.duplicated().any():raise ValueError('scBase_supervised_capture_barcode_not_unique')
        target_axis=gene_axes[(family,context)];current['projected_source_axis_sha256']=None;current['projected_count_sha256']=None
        current['projected_counts']=np.nan;current['projected_detected_genes']=np.nan
        if library['modality']=='RNA':
            path=ROOT/record['source_file'];before=hash_file(path)
            if before!=record['source_file_sha256_from_inventory']:raise ValueError('raw_scBase_RNA_input_changed')
            with RNAFile(path) as raw:
                matrix=matrix_in_memory(raw);fingerprints,totals,detected,axis=projected_fingerprints(matrix,raw.var.index.astype(str).tolist(),target_axis)
                if raw.obs.index.astype(str).tolist()!=current.source_barcode.tolist():raise ValueError('source_row_order_not_shared')
            current['projected_source_axis_sha256']=axis;current['projected_count_sha256']=fingerprints;current['projected_counts']=totals;current['projected_detected_genes']=detected
            del matrix
            if hash_file(path)!=before:raise ValueError('raw_RNA_changed_during_comparison')
        joined=current.merge(source,on='barcode_core',suffixes=('_scbase','_original'),how='left',validate='one_to_one',indicator=True)
        joined['source_capture_key']=library['capture_key'];joined['source_modality']=library['modality'];matched=joined['_merge']=='both';joined.drop(columns='_merge',inplace=True)
        joined['confirmed_capture_barcode_link']=matched;joined['source_native_panel_counts_identical']=pd.Series([None]*len(joined),dtype='boolean')
        if library['modality']=='RNA':
            joined.loc[matched,'source_native_panel_counts_identical']=(joined.loc[matched,'projected_count_sha256']==joined.loc[matched,'computed_count_sha256_original']).to_numpy()
        joined['identity_status']=np.where(matched,'confirmed_same_capture_barcode','not_in_filtered_original_source_cell_universe')
        joined['count_interpretation']='Original RNA panel projection only; scBase full gene universe and read/quantification pipelines differ' if library['modality']=='RNA' else 'guide_assay_not_comparable_RNA_expression'
        joined['physical_singlet_cell_truth']=False;joined['independent_biological_replicates']=None
        joined.to_parquet(links_root/(accession+'.parquet'),index=False)
        ratio=(joined.loc[matched,'projected_counts']/joined.loc[matched,'computed_total_counts_original']) if library['modality']=='RNA' else np.array([])
        overlaps.append({'experiment_accession':accession,**library,'scBase_records':len(current),'original_capture_records':len(source),
            'matched_capture_barcode_records':int(matched.sum()),'unmatched_scBase_records':int((~matched).sum()),
            'original_measured_genes':len(target_axis),'unmeasured_original_genes_in_scBase':0 if library['modality']=='RNA' else None,
            'exact_original_panel_count_records':int(joined.source_native_panel_counts_identical.fillna(False).sum()) if library['modality']=='RNA' else None,
            'projected_count_ratio_to_original':quantiles(ratio),'full_native_axis_same':False,
            'source_input_sha256':record['source_file_sha256_from_inventory'],'cell_evidence':'supervised-record-links/'+accession+'.parquet',
            'interpretation':'Same source capture; these sequencing/reprocessing or cross-modal records are not independent experiments. Unmatched barcodes are not automatically additional established cells.'})
        print('supervised scBase library comparisons '+str(len(overlaps))+'/48 '+accession,flush=True)
    write_json(output/'supervised-library-overlaps.json',overlaps);pd.DataFrame(overlaps).to_parquet(output/'supervised-library-overlaps.parquet',index=False)
    # Canonical source-capture/barcode units quantify known re-observation only;
    # no claim is made that all unknown libraries or doublets are distinct cells.
    relation_rows=[]
    for row in catalog:
        if row['capture_key']:
            part=all_cells.loc[all_cells.file_index==row['file_index'],['record_id','source_barcode']].copy();part['capture_key']=row['capture_key'];part['RNA_eligible']=row['RNA_eligible'];part['experiment_accession']=row['experiment_accession'];relation_rows.append(part)
    capture_members=pd.concat(relation_rows,ignore_index=True);capture_members.to_parquet(output/'known-capture-record-membership.parquet',index=False)
    summaries={'scbase_files':len(catalog),'scbase_records':len(all_cells),'source_studies':len({r['source_study'] for r in catalog}),
        'source_samples':len(groups),'shared_sample_groups':sum(len(g)>1 and not k.startswith('unresolved:') for k,g in groups.items()),'shared_sample_file_pairs':len(pair_rows),
        'barcode_strings_spanning_multiple_studies':int((collision.source_studies>1).sum()),'records_with_cross_study_barcode_collisions':int(collision.loc[collision.source_studies>1,'stored_records'].sum()),
        'barcode_and_count_candidate_records':len(candidates),'barcode_and_count_candidate_groups':int(candidates.candidate_group.nunique()),
        'supervised_related_files':len(overlaps),'supervised_modality_files':dict(Counter(o['modality'] for o in overlaps)),
        'supervised_matched_RNA_capture_records':sum(o['matched_capture_barcode_records'] for o in overlaps if o['modality']=='RNA'),
        'supervised_matched_guide_capture_records':sum(o['matched_capture_barcode_records'] for o in overlaps if o['modality']=='guide'),
        'supervised_exact_original_RNA_panel_records':sum(o['exact_original_panel_count_records'] or 0 for o in overlaps),
        'known_capture_record_memberships':len(capture_members),'known_capture_barcode_units':len(capture_members.drop_duplicates(['capture_key','source_barcode'])),
        'independent_biological_replicates':None,'raw_inputs_unchanged':True}
    write_json(output/'summary.json',summaries)
    report={'status':'completed','phase':'all_scBase_source_relations_and_all_48_supervised_overlap_libraries','summary':summaries,
        'input_reports':{k:hash_file(v/'report.json') for k,v in [('scbase_expression',base),('scbase_provenance',provenance),('replogle_structure',replogle),('nadig_structure',nadig)]},
        'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'code':{n:hash_file(Path(__file__).with_name(n)) for n in ['profile_cross_source_identity.py','cross_source_identity.py','rna.py','profile_background.py']},
        'duration_seconds':time.monotonic()-started,'completed_at':datetime.now(timezone.utc).isoformat(),
        'limitations':['Same barcode or identical sparse counts alone do not establish identity across studies.','Known shared capture/barcode is not proof of a physical singlet or an independent culture.','scBase related files may be sequencing subsets or different assays; projection equality is not complete full-axis equality.','Unresolved study/sample/library provenance remains unknown; no source records are removed or relabeled.']}
    report['artifacts']={str(p.relative_to(output)):hash_file(p) for p in sorted(output.rglob('*')) if p.is_file()};write_json(output/'report.json',report);print(json.dumps(summaries))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['base','provenance','replogle','nadig','output']:p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();run(a.base,a.provenance,a.replogle,a.nadig,a.output)
