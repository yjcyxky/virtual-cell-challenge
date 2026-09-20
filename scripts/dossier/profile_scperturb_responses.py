#!/usr/bin/env python
"""Execute every frozen scPerturb RNA design and verify reusable source responses."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import subprocess
import time
import pandas as pd
from profile_responses import write_json
from rna import hash_file
from scperturb_response_core import describe_conditions,deep_responses

ROOT=Path(__file__).resolve().parents[2]
CODE=['profile_scperturb_responses.py','scperturb_response_core.py','chemical_diagnostics.py','profile_background.py','profile_responses.py','response.py','rna.py']


def verify_copied_rows(original,collection):
    fields={'source_barcode':'source_barcode','gene_axis_sha256':'source_identity_axis_sha256',
        'computed_count_sha256':'computed_on_source_identity_axis_sha256','source_task':'source_obs__gene_transcript',
        'source_guide_id':'source_obs__guide_id','source_batch':'source_obs__batch','source_target_gene':'source_obs__gene'}
    if len(original)!=len(collection):raise ValueError('copy_record_scope_differs')
    mismatches={a:int((original[a].astype(str).to_numpy()!=collection[b].astype(str).to_numpy()).sum()) for a,b in fields.items()}
    if any(mismatches.values()):raise ValueError('source_copy_identity_or_intervention_differs:'+json.dumps(mismatches))
    if original.source_barcode.duplicated().any() or collection.source_barcode.duplicated().any():raise ValueError('copy_source_barcode_ambiguity')
    return mismatches


def reuse_source_responses(file,cells,directory):
    prefix='replogle' if file.startswith('Replogle') else 'nadig';study='ReplogleWeissman2022_' if prefix=='replogle' else 'NadigOConner2024_';context=file.removeprefix(study).removesuffix('.h5ad')
    base=ROOT/'data/assessments';annotation=base/(prefix+'-annotation-'+context+'-20260919');response=base/(prefix+'-response-'+context+'-20260919')
    ar=json.loads((annotation/'report.json').read_text());rr=json.loads((response/'report.json').read_text())
    for folder,report in [(annotation,ar),(response,rr)]:
        if report['status']!='completed':raise ValueError('original_completed_response_required')
        for item in report['artifacts']:
            if hash_file(folder/item['file'])!=item['sha256']:raise ValueError('published_source_artifact_changed')
    original=pd.read_parquet(annotation/'cells.parquet');mismatches=verify_copied_rows(original,cells);directory.mkdir()
    links=pd.DataFrame({'collection_record_id':cells.record_id,'original_record_id':original.record_id,
        'original_input_sha256':original.input_sha256,'collection_input_sha256':cells.input_sha256,'source_barcode':original.source_barcode,
        'full_count_identity_sha256':original.computed_count_sha256,'source_construct_task':original.source_task,'source_guide':original.source_guide_id,'source_batch':original.source_batch})
    links.to_parquet(directory/'exact-source-record-links.parquet',index=False)
    tasks=json.loads((response/'tasks.json').read_text());release='https://github.com/yjcyxky/virtual-cell-challenge/releases/tag/assessment-'+prefix+'-20260919'
    for task in tasks:
        task['external_response_evidence']={'release':release,'archive_gene_result_path':'response/'+context+'/'+task['gene_results']['file'],
            'sha256':task['gene_results']['sha256'],'original_response_report_sha256':hash_file(response/'report.json')}
    write_json(directory/'tasks.json',tasks);shutil.copy2(response/'report.json',directory/'original-response-report.json')
    shutil.copy2(annotation/'report.json',directory/'original-identity-report.json')
    result={'status':'completed','execution_mode':'reused_after_every_record_count_axis_barcode_construct_guide_target_and_batch_verified',
        'file':file,'record_count':len(cells),'mismatches':mismatches,'original_input_sha256':original.input_sha256.iloc[0],
        'collection_input_sha256':cells.input_sha256.iloc[0],'original_annotation_report_sha256':hash_file(annotation/'report.json'),
        'original_response_report_sha256':hash_file(response/'report.json'),'original_response_bundle_id':rr['bundle_id'],'release':release,
        'actual_tasks':len(tasks),'task_status_counts':rr['task_status_counts'],'DE_status_counts':rr['DE_status_counts'],
        'interpretation':'Same observations and intervention assignments, not independent replication; all source promoter/transcript constructs retained, including unmapped targets.'}
    write_json(directory/'copy-proof.json',result);return result


def verified_completed_files(source,identity):
    """Reuse immutable completed components after an equivalent driver optimization."""
    previous=json.loads((source/'identity.json').read_text())
    for key in ['design_report_sha256','cell_phase_report_sha256','uv_lock_sha256']:
        if previous[key]!=identity[key]:raise ValueError('reuse_input_identity_changed:'+key)
    allowed={'profile_scperturb_responses.py','scperturb_response_core.py'}
    if set(previous['code'])!=set(identity['code']):raise ValueError('reuse_code_scope_changed')
    for name,digest in previous['code'].items():
        if name not in allowed and digest!=identity['code'][name]:raise ValueError('reuse_analysis_method_changed:'+name)
    files={}
    for path in sorted(source.glob('*/report.json')):
        report=json.loads(path.read_text())
        if report.get('status')!='completed':continue
        if report['identity']!=previous:raise ValueError('reuse_component_identity_changed')
        for name,digest in report['artifacts'].items():
            if Path(name).is_absolute() or '..' in Path(name).parts:raise ValueError('unsafe_reuse_component_path')
            if hash_file(path.parent/name)!=digest:raise ValueError('reuse_component_artifact_changed:'+name)
        files[path.parent.name]=hash_file(path)
    return {'source_identity_sha256':hash_file(source/'identity.json'),'source_identity':previous,'completed_file_reports':files,
        'reason':'Verified completed files retained from interrupted run; equivalent batched eligibility gate validated against prior real-data outputs. Component code identities are preserved.'}


def run(design_root,cells_root,output,workers=2,resume=False,reuse_from=None):
    started=time.monotonic();manifest=json.loads((design_root/'report.json').read_text());cell_phase=json.loads((cells_root/'report.json').read_text())
    if manifest['status']!='completed' or cell_phase['status']!='completed':raise ValueError('all_design_and_cell_phases_required')
    identity={'design_report_sha256':hash_file(design_root/'report.json'),'cell_phase_report_sha256':hash_file(cells_root/'report.json'),
        'code':{n:hash_file(Path(__file__).with_name(n)) for n in CODE},'uv_lock_sha256':hash_file(Path(__file__).with_name('uv.lock'))}
    reuse=verified_completed_files(reuse_from,identity) if reuse_from else None
    if reuse:identity['reused_completed_components']=reuse
    if output.exists():
        if not resume or json.loads((output/'identity.json').read_text())!=identity or (output/'report.json').exists():raise ValueError('response_resume_identity_changed_or_completed')
    else:output.mkdir(parents=True);write_json(output/'identity.json',identity)
    results=[]
    for record in manifest['files']:
        file=record['file']
        if record['status']=='not_applicable':results.append(record);continue
        directory=output/file.removesuffix('.h5ad')
        copied=reuse is not None and directory.name in reuse['completed_file_reports']
        if copied and not directory.exists():shutil.copytree(reuse_from/directory.name,directory)
        if (directory/'report.json').exists():
            old=json.loads((directory/'report.json').read_text())
            expected=reuse['source_identity'] if copied else identity
            if old['identity']!=expected:raise ValueError('completed_file_response_identity_changed')
            if copied and hash_file(directory/'report.json')!=reuse['completed_file_reports'][directory.name]:raise ValueError('reused_component_report_changed')
            for name,digest in old['artifacts'].items():
                if hash_file(directory/name)!=digest:raise ValueError('completed_file_response_artifact_changed')
            results.append(old);continue
        folder=design_root/record['directory'];cell_folder=cells_root/file.removesuffix('.h5ad');cr=json.loads((cell_folder/'report.json').read_text())
        for name in ['cells.parquet','gene-mapping.parquet']:
            if hash_file(cell_folder/name)!=cr['artifact_hashes'][name]:raise ValueError('completed_cell_phase_changed')
        if hash_file(folder/'row-applicability.parquet')!=record['artifacts']['row-applicability.parquet']:raise ValueError('frozen_design_changed')
        cells=pd.read_parquet(cell_folder/'cells.parquet');design=pd.read_parquet(folder/'row-applicability.parquet');mapping=pd.read_parquet(cell_folder/'gene-mapping.parquet')
        if len(cells)!=record['cells'] or not cells.row_index.equals(design.row_index):raise ValueError('cell_design_record_scope_differs')
        directory.mkdir();description=describe_conditions(cells,design,directory/'descriptions')
        if file.startswith(('Replogle','Nadig')):deep=reuse_source_responses(file,cells,directory/'deep-response')
        elif design.deep_response_candidate.any():
            path=ROOT/'data/raw/scperturb'/file
            if hash_file(path)!=record['input_sha256']:raise ValueError('raw_source_changed')
            deep=deep_responses(file,path,cells,design,mapping,directory/'deep-response',workers)
            if hash_file(path)!=record['input_sha256']:raise ValueError('raw_source_changed_during_response')
        else:deep={'status':'not_applicable','reason':'not_a_qualified_single_target_CRISPRi_source','actual_tasks':0}
        report={'status':'completed','file':file,'cells':len(cells),'identity':identity,'source_design_report_sha256':hash_file(folder/'report.json'),
            'source_cell_report_sha256':hash_file(cell_folder/'report.json'),'description':description,'deep_response':deep,
            'artifacts':{str(p.relative_to(directory)):hash_file(p) for p in directory.rglob('*') if p.is_file()},'completed_at':datetime.now(timezone.utc).isoformat()}
        write_json(directory/'report.json',report);results.append(report);print('scPerturb response files '+str(sum(r['status']=='completed' for r in results))+'/51 '+file,flush=True)
    report={'status':'completed','phase':'all_51_RNA_conditions_and_all_eligible_CRISPRi_tasks','identity':identity,'files':results,
        'actual_RNA_files':sum(r['status']=='completed' for r in results),'actual_RNA_records':sum(r.get('cells',0) for r in results),
        'actual_deep_response_tasks':sum(r.get('deep_response',{}).get('actual_tasks',0) for r in results),
        'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'duration_seconds':time.monotonic()-started,'completed_at':datetime.now(timezone.utc).isoformat()}
    write_json(output/'report.json',report)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['design','cells','output']:p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--workers',type=int,default=2);p.add_argument('--resume',action='store_true');p.add_argument('--reuse-from',type=Path)
    a=p.parse_args();run(a.design,a.cells,a.output,a.workers,a.resume,a.reuse_from)
