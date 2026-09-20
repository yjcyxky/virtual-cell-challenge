#!/usr/bin/env python
"""Freeze complete overlap evidence, coverage denominators and offline viewers."""
import argparse
from collections import Counter
from datetime import datetime,timezone
from itertools import combinations
import json
from pathlib import Path
import shutil
import subprocess
import sys
import uuid
import numpy as np
import pandas as pd
from profile_cross_source_coverage import Frozen, ROOT, BASE
from profile_responses import write_json,serial
from render import render
from render_cross_source import coverage_page,relationship_page
from rna import hash_file


def copy_component(folder,destination):
    report=json.loads((folder/'report.json').read_text())
    if report['status']!='completed':raise ValueError('unfinished_component')
    artifacts=report['artifacts']
    if isinstance(artifacts,list):artifacts={r['file']:r['sha256'] for r in artifacts}
    destination.mkdir(parents=True)
    for name,digest in artifacts.items():
        if Path(name).is_absolute() or '..' in Path(name).parts or (folder/name).is_symlink():raise ValueError('unsafe_path')
        if hash_file(folder/name)!=digest:raise ValueError('component_changed:'+name)
        target=destination/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(folder/name,target)
        if hash_file(target)!=digest:raise ValueError('copy_changed')
    shutil.copy2(folder/'report.json',destination/'report.json')
    return report


def compose(identities,coverage,output):
    identities=identities.resolve();coverage=coverage.resolve();output=output.resolve()
    output.mkdir(parents=True,exist_ok=False);frozen=Frozen()
    ir=copy_component(identities,output/'identities');cr=copy_component(coverage,output/'coverage')
    panels=json.loads((coverage/'panels.json').read_text());by_panel={p['panel_id']:p for p in panels}
    used=json.loads((coverage/'used-inputs.json').read_text())
    identity_catalog=str((identities/'scbase-source-catalog.json').relative_to(ROOT))
    if used['frozen_artifacts'][identity_catalog]!=hash_file(identities/'scbase-source-catalog.json'):raise ValueError('identity_coverage_not_shared')
    for path,digest in {**used['frozen_artifacts'],**used['raw_reference_hashes']}.items():
        if hash_file(ROOT/path)!=digest:raise ValueError('consumed_source_changed_after_coverage:'+path)
    nodes={};edges=[]
    def node(id,kind,**facts):
        if id not in nodes:nodes[id]={'id':id,'label':id,'kind':kind,**facts}
    def edge(a,b,relation,details=None,evidence=None):
        if a not in nodes or b not in nodes:raise ValueError('undefined_relation_node')
        edges.append({'a':a,'b':b,'relation':relation,'details':details or {},'evidence':evidence or []})
    for p in panels:
        node(p['panel_id'],'source_measurement_panel',family=p['family'],source_context=p['source_context'],
             source_records=p['source_records'],native_features=p['native_features'],metadata=p['source_metadata'],
             independent_biological_replicates=None,coverage='coverage/'+p['gene_coverage_file'])
    scbase=json.loads((identities/'scbase-source-catalog.json').read_text())
    for row in scbase:
        pid='scBase:'+row['experiment_accession'];study='study:'+row['source_study'];sample='sample:'+row['source_sample']
        node(study,'source_study_accession',aliases=row['source_study_aliases']);node(sample,'source_sample_accession',aliases=row['source_sample_aliases'])
        evidence=['identities/scbase-source-catalog.parquet']
        edge(pid,sample,'source_accession_association_not_replication',evidence=evidence)
        edge(sample,study,'source_accession_association_not_replication',evidence=evidence)
        relation=row['author_library_relation']
        if relation:
            capture='capture:'+relation['capture_key'];node(capture,'author_documented_capture',**relation)
            edge(pid,capture,'confirmed_source_capture_association',relation,['identities/supervised-record-links/'+row['experiment_accession']+'.parquet'])
    # Deduplicate repeated provenance edges, retaining all record-level evidence.
    unique={json.dumps(e,sort_keys=True):e for e in edges};edges=list(unique.values())
    overlaps=json.loads((identities/'supervised-library-overlaps.json').read_text())
    capture_original=set()
    for row in overlaps:
        original=row['family']+':'+row['context'];capture='capture:'+row['capture_key'];pid='scBase:'+row['experiment_accession']
        if (capture,original) not in capture_original:
            edge(capture,original,'author_GEM_within_original_context',{'GEM':row['GEM'],'source_library':row['source_library']},['identities/supervised-library-overlaps.parquet']);capture_original.add((capture,original))
        edge(pid,original,'confirmed_shared_capture_reprocessing_or_cross_modality',
             {k:row[k] for k in ['modality','scBase_records','original_capture_records','matched_capture_barcode_records','exact_original_panel_count_records','interpretation']},['identities/'+row['cell_evidence']])
    pairs=json.loads((identities/'all-shared-sample-library-pairs.json').read_text())
    for row in pairs:
        a='scBase:'+row['experiment_A'];b='scBase:'+row['experiment_B']
        relation='confirmed_capture_barcode_links' if row.get('confirmed_capture_barcode_links',0) else 'candidate_shared_sample_barcode_overlap' if row.get('shared_barcodes',0) else 'shared_sample_without_confirmed_record_identity'
        if row.get('confirmed_exact_RNA_expression_copies',0):relation='confirmed_exact_RNA_expression_copies'
        edge(a,b,relation,{k:row.get(k) for k in ['shared_barcodes','equal_native_count_vectors','confirmed_capture_barcode_links','confirmed_exact_RNA_expression_copies','relation_status_counts','status','reason']},['identities/'+row['cell_evidence']] if row.get('cell_evidence') else ['identities/all-shared-sample-library-pairs.parquet'])
    proof_dir=output/'exact-copy-proofs';proof_dir.mkdir()
    h1=BASE/'h1-structure-20260919-v3';dups=frozen.parquet(h1,'duplicate_groups.parquet')
    if len(dups)!=38176 or not dups.classification.eq('content_and_label_identical').all() or not dups.records.eq(3).all():raise ValueError('H1_duplicate_scope_changed')
    dups.to_parquet(proof_dir/'H1-NTC-copy-groups.parquet',index=False,compression='zstd')
    for a,b in combinations(['Training','Validation','Test'],2):
        edge('H1:'+a,'H1:'+b,'confirmed_exact_count_and_intervention_copy',{'shared_NTC_groups':38176,'global_H1_excess_records':76352,'physical_cell_truth':False},['exact-copy-proofs/H1-NTC-copy-groups.parquet'])
    sc=BASE/'scperturb-dossier-20260919';copy_rows=[]
    source_metadata=[]
    for p in panels:
        if p['family']!='scPerturb':continue
        meta=p['source_metadata'];study='study:'+meta['study_id'];node(study,'source_publication_identity')
        edge(p['panel_id'],study,'source_publication_association_not_replication',evidence=['coverage/panels.parquet'])
        for acc in meta['provenance']['accession_candidates_from_processing']:
            aid='study:'+acc;node(aid,'processing_code_accession_candidate',truth_status='candidate_provenance_not_proof_of_same_cells')
            edge(study,aid,'candidate_processing_code_study_accession',evidence=['coverage/panels.parquet'])
        if not p['confirmed_collection_copy']:continue
        name=p['panel_id'].removeprefix('scPerturb:');prefix='response/'+name+'/deep-response/'
        proof=frozen.json(sc,prefix+'copy-proof.json');links=frozen.parquet(sc,prefix+'exact-source-record-links.parquet')
        if proof['status']!='completed' or any(proof['mismatches'].values()) or len(links)!=proof['record_count'] or links.collection_record_id.duplicated().any() or links.original_record_id.duplicated().any():raise ValueError('incomplete_exact_copy_evidence')
        family='replogle' if name.startswith('Replogle') else 'nadig'
        context=name.removeprefix('ReplogleWeissman2022_' if family=='replogle' else 'NadigOConner2024_')
        destination=proof_dir/name;destination.mkdir();links.to_parquet(destination/'record-links.parquet',index=False,compression='zstd');write_json(destination/'copy-proof.json',proof)
        row={'collection':p['panel_id'],'original':family+':'+context,'exact_copied_records':len(links),'duplicated_construct_tasks':proof['actual_tasks'],'independent_additional_experiments':0,'source_response_release':proof['release']};copy_rows.append(row)
        edge(p['panel_id'],family+':'+context,'confirmed_exact_count_and_intervention_copy',row,['exact-copy-proofs/'+name+'/record-links.parquet','exact-copy-proofs/'+name+'/copy-proof.json'])
    if len(copy_rows)!=5 or sum(r['exact_copied_records'] for r in copy_rows)!=2956306:raise ValueError('incomplete_Replogle_Nadig_copy_scope')
    # Retain fixed source protocol summaries, including context/subline/culture
    # observability and explicit limitations, without silently pooling names.
    sources=[
        ('h1-structure-20260919-v3',3,'assessment-h1-structure-20260919'),
        ('official-controls-20260919',6,'assessment-official-controls-20260919'),
        ('replogle-dossier-20260919',7,'assessment-replogle-20260919'),
        ('nadig-dossier-20260919',8,'assessment-nadig-20260919'),
        ('jiang-dossier-20260919-v2',9,'assessment-jiang-20260919'),
        ('mcfaline-gxe1-dossier-20260919-v2',10,'assessment-mcfaline-gxe1-20260919'),
        ('mcfaline-gxe2-dossier-20260919',11,'assessment-mcfaline-gxe2-20260919'),
        ('mcfaline-chemical3-dossier-20260919-v2',12,'assessment-mcfaline-chemical3-20260919'),
        ('mcfaline-chemical4-dossier-20260919-v2',12,'assessment-mcfaline-chemical4-20260919'),
        ('scperturb-dossier-20260919',14,'assessment-scperturb-rna-20260919'),
        ('scbase-dossier-20260919-v2',16,'assessment-scbase-expression-20260919'),
        ('tahoe-dossier-20260919-v2',17,'assessment-tahoe-20260919'),
        ('priors-dossier-20260919-v4',18,'assessment-priors-20260919-v2')]
    (output/'source-protocols').mkdir()
    for name,issue,tag in sources:
        report=frozen.json(BASE/name,'report.json')
        selected={k:report[k] for k in ['bundle_id','status','completed_at','input_sha256','methods','references','limitations','exposure','identity','runtime','code_commit'] if k in report}
        selected['small_source_design_tables']=[t for t in report.get('tables',[]) if len(t['rows'])<=100]
        selected['source_report_sha256']=hash_file(BASE/name/'report.json')
        write_json(output/'source-protocols'/(name+'.json'),selected)
        source_metadata.append({'source_dossier':name,'issue':issue,'bundle_id':report['bundle_id'],'source_report_sha256':selected['source_report_sha256'],
                                'release':'https://github.com/yjcyxky/virtual-cell-challenge/releases/tag/'+tag,'protocol_file':'source-protocols/'+name+'.json'})
    # These three protein matrices remain explicitly outside RNA coverage; the
    # paired modalities must not increase the RNA-cell denominator.
    index=frozen.json(sc,'file-index.json');protein=[r for r in index if r['status']=='not_applicable']
    for r in protein:
        pid='scPerturb:'+r['file'].removesuffix('.h5ad')
        node(pid,'protein_matrix_not_RNA',source_facts=r)
        audit=frozen.json(sc,'source-audit/'+r['file']+'.json')
        proof='source-protocols/'+r['file']+'-paired-modality.json'
        write_json(output/proof,{'study_id':audit['study_id'],'relationships':audit['relationships'],'input_sha256':audit['input_sha256'],'interpretation':'Protein measurements are not extra RNA observations; barcode pairing alone does not prove physical singlets.'})
        sid='study:'+audit['study_id'];node(sid,'source_publication_identity')
        edge(pid,sid,'source_publication_association_not_replication',evidence=[proof])
        for relation in audit['relationships']:
            partner='scPerturb:'+relation['file'].removesuffix('.h5ad')
            if partner in nodes:edge(pid,partner,'candidate_paired_RNA_protein_modalities',relation,[proof])
    graph={'nodes':list(nodes.values()),'edges':edges,'interpretation':'Source association is distinct from record identity and independently randomized/cultured replication.'}
    # All graph drill-down evidence is included, and all edge endpoints exist.
    for e in edges:
        for path in e['evidence']:
            if not (output/path).is_file():raise ValueError('missing_graph_evidence:'+path)
    write_json(output/'relationships.json',graph);(output/'relationships.html').write_text(relationship_page(serial(graph)))
    matrix=np.load(coverage/'official-coverage-codes.npz',allow_pickle=False)
    target_frame=pd.read_parquet(coverage/'all-panel-target-coverage.parquet')
    target_names=pd.read_parquet(coverage/'target-identifiers.parquet').source_gene.tolist()
    full_index=pd.MultiIndex.from_product([[p['panel_id'] for p in panels],target_names],names=['panel_id','official_target'])
    indexed=target_frame.set_index(['panel_id','official_target']).reindex(full_index)
    if indexed.status.isna().any() or len(indexed)!=len(panels)*300:raise ValueError('incomplete_target_matrix')
    cols=['completed_CRISPRi_construct_tasks','completed_CRISPRi_tasks_after_confirmed_copy_exclusion','completed_DE_tasks']
    counts=indexed[cols].to_numpy(np.uint32).reshape(len(panels),300,3)
    (output/'coverage.html').write_text(coverage_page(serial(panels),matrix['official_genes'].tolist(),matrix['statuses'].tolist(),matrix['codes'],target_names,counts))
    denominators=[
        {'universe':'H1 source split records','stored_records':491046,'confirmed_excess_exact_copies':76352,'records_after_confirmed_copy_exclusion':414694,'independent_biological_replicates':None,'meaning':'38,176 shared NTC groups occur in all three stored splits; original source IDs retained.'},
        {'universe':'Replogle + Nadig originals','stored_records':2956306,'confirmed_excess_exact_copies':0,'records_after_confirmed_copy_exclusion':2956306,'independent_biological_replicates':None,'meaning':'Five original expression contexts; construct, guide and GEM are not automatic biological repeats.'},
        {'universe':'scPerturb RNA collection','stored_records':8802191,'confirmed_excess_exact_copies':2956306,'records_after_confirmed_copy_exclusion':5845885,'independent_biological_replicates':None,'meaning':'Count excludes only five proven original-source copies, not every possible relationship among other studies; remaining records not claimed mutually independent.'},
        {'universe':'scBase all files','stored_records':13255146,'confirmed_excess_exact_copies':0,'records_after_confirmed_copy_exclusion':None,'independent_biological_replicates':None,'meaning':'No confirmed full-native-axis RNA copies in audited shared-sample pairs; this is not proof that all records are independent. 227 files are outside eligible human RNA inference.'},
        {'universe':'Documented Replogle/Nadig capture memberships in scBase comparisons','stored_records':ir['summary']['known_capture_record_memberships'],'confirmed_excess_exact_copies':None,'records_after_confirmed_copy_exclusion':ir['summary']['known_capture_barcode_units'],'independent_biological_replicates':None,'meaning':'Unique capture × barcode keys within registered comparison universe only. Cross-assay/processing membership and low-complexity profiles are not verified physical singlet identities.'},
        {'universe':'Tahoe plates 6 and 14','stored_records':None,'confirmed_excess_exact_copies':None,'records_after_confirmed_copy_exclusion':None,'independent_biological_replicates':2,'meaning':'Two author-designated biological repeat plates; 4,738 paired chemical/vehicle × cell-line panels were assessed. Not a universal replicate count for Tahoe or other sources.'}]
    decisions=[
        {'use':'Independent supervision count','recommendation':'Exclude known H1 NTC cross-split copies and five scPerturb original-source copies from additional-evidence counts; preserve every stored record and record-link proof.'},
        {'use':'scBase related libraries','recommendation':'All 48 libraries are linked to author captures; 39 RNA and 9 guide. Keep reprocessing and cross-modal roles distinct; 49,476 matched RNA capture/barcode records have zero exact projected original-panel count matches.'},
        {'use':'Gene panel comparison','recommendation':'Use native mapping and per-source denominator; compare unique conflict-free canonical features only. 91 official labels remain unresolved; missing features are not zero and many-to-one features are not silently summed.'},
        {'use':'Target and control coverage','recommendation':'Use all construct tasks, source conditions and matching-control status; count coverage separately from completed response and DE. Source copy exclusion is separate from response eligibility.'},
        {'use':'Cell background names','recommendation':'Retain source line, effector, time, environment, guide library, sample/capture, panel and protocol. Shared K562/A172 labels do not establish the same subline, passage or independent culture.'},
        {'use':'Evidence still unknown','recommendation':'Per-cell scBase perturbations, ambiguous guides, source identity and missing time/dose remain unknown; follow #22–#27. Auxiliary protein/fitness/signature resources are separate evidence modalities.'}]
    report={'schema_version':2,'bundle_id':'cross-source-dossier-'+uuid.uuid4().hex,'title':'跨来源身份、独立证据边界与全量覆盖','status':'completed','completed_at':datetime.now(timezone.utc).isoformat(),
            'identity_summary':ir['summary'],'coverage_summary':{k:v for k,v in cr.items() if k not in ['artifacts','reproduce']},
            'relationship_nodes':len(nodes),'relationship_edges':len(edges),'verified_original_copy_records':2956306,'H1_excess_copy_records':76352,
            'independent_biological_replicates_global':None,'inputs_unchanged':True,
            'consumed_coverage_inputs_reverified':len(used['frozen_artifacts'])+len(used['raw_reference_hashes']),
            'component_report_sha256':{'identities':hash_file(identities/'report.json'),'coverage':hash_file(coverage/'report.json')},
            'methods':{'identity':'All 13,255,146 scBase records; all 641 shared-sample file pairs; all 48 explicitly author-mapped Replogle/Nadig libraries; complete source barcodes/native gene axes/count fingerprints; 5 prior exact-copy proofs and all H1 NTC copy groups retained.',
                       'coverage':'All source panels × 18,533 official labels and all source panels × 300 targets; literal coverage and unique conflict-free canonical mapping separated. Gene zero status has explicit file-wide QC scope, not arbitrary task absence.',
                       'target_identity':'Frozen HGNC unique symbol/alias resolution, with explicit source INTERVENTION Ensembl ID corroboration when supplied. Conflicts and unresolved names remain unknown. An expression-feature Ensembl ID is not substituted for missing intervention identity. Every task retains source name, intervention ID and resolution status.',
                       'conditions':'Every frozen observed/declared condition and every deep construct task; chemical, regulatory-locus, CRISPRa, mouse and auxiliary evidence roles remain separate.',
                       'view':'Offline gzip byte matrix contains every gene coverage entry. Source and gene filters plus 100-row pagination do not subsample. Graph contains every registered source/study/sample/capture edge; 30-neighbor pagination only controls drawing.',
                       'uncertainty':'Source association ≠ same record ≠ physical singlet ≠ independent biological replicate. No invented probability, per-cell genotype, culture identity or correction.'},
            'limitations':['技术副本的排除计数不是全局独立细胞数；未证实重叠不能自动视为独立。',
                          '计数完全一致也需要来源身份支持；不同研究可复用barcode。文库重处理、guide/RNA模态与捕获关联分别保留。',
                          '原生面板、归一化分母和实验条件仍不同；canonical gene交集只解决身份对齐，不消除测量差异。',
                          '原始来源标签、count矩阵与推断字段保持分离。类型／状态未校准，不作为ground truth。',
                          '全量来源结果已经被分析接触；不存在未接触的验证集，本票不产生训练语料或预测模型。'],
            'source_releases':source_metadata,'protein_not_applicable':protein,'evidence_issues':[22,23,24,25,26,27],
            'tables':[{'title':'记录分母与独立性边界','rows':denominators},{'title':'已证实的五个合集副本','rows':copy_rows},
                      {'title':'全部48个关联文库','rows':overlaps},{'title':'全部641个共享样本文件对','rows':pairs},
                      {'title':'全量来源基因与目标覆盖','columns':['panel_id','family','species','modality','source_records','native_features','official_literal_present','official_safely_measured','official_measured_all_zero','official_missing','official_unresolved','official_ambiguous_mapping','source_construct_tasks','completed_CRISPRi_construct_tasks','official_targets_with_CRISPRi_response','official_targets_after_copy_exclusion','confirmed_collection_copy','zero_QC_scope'],'rows':panels},
                      {'title':'全部条件文件与分母','rows':json.loads((coverage/'condition-index.json').read_text())},
                      {'title':'辅助资源的独立证据角色','rows':json.loads((coverage/'auxiliary-resources.json').read_text())},
                      {'title':'数据使用建议','rows':decisions}],
            'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            'code_sha256':{name:hash_file(Path(__file__).with_name(name)) for name in ['compose_cross_source_dossier.py','render_cross_source.py','cross_source_coverage.py','profile_cross_source_coverage.py']},
            'reproduce':sys.argv,'reproduction':'Run profile_cross_source_identity.py and profile_cross_source_coverage.py against the frozen input folders specified in their reports, then this composer with --identities --coverage --output FRESH. Locked scripts/dossier/uv.lock; raw counts and source labels remain unchanged.'}
    write_json(output/'source-protocol-index.json',source_metadata);write_json(output/'record-denominators.json',denominators)
    write_json(output/'composition-input-hashes.json',frozen.used)
    report['artifacts']=[{'file':str(p.relative_to(output)),'sha256':hash_file(p)} for p in sorted(output.rglob('*')) if p.is_file()]
    report['artifacts'].sort(key=lambda x:(x['file'] not in ['coverage.html','relationships.html','relationships.json','record-denominators.json','coverage/all-source-tasks.parquet','coverage/all-panel-target-coverage.parquet'],x['file']))
    write_json(output/'report.json',report);(output/'report.html').write_text(render(serial(report)))
    (output/'SHA256SUMS').write_text(''.join(hash_file(p)+'  '+str(p.relative_to(output))+'\n' for p in sorted(output.rglob('*')) if p.is_file() and p.name!='SHA256SUMS'))
    print(json.dumps({k:report[k] for k in ['bundle_id','status','relationship_nodes','relationship_edges','verified_original_copy_records','H1_excess_copy_records']}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['identities','coverage','output']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    try:compose(a.identities,a.coverage,a.output)
    except Exception as error:
        if a.output.exists():write_json(a.output/'failure.json',{'status':'failed','error':repr(error),'code_sha256':hash_file(Path(__file__))})
        raise
