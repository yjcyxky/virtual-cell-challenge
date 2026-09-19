#!/usr/bin/env python
"""Verify full scBase expression coverage and expose study/sample composition evidence."""
import argparse
from collections import Counter,defaultdict
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
import uuid
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from publish_assessment import verified_files
from profile_responses import write_json,serial
from rna import hash_file,quantiles
from render import render


def compose(source,provenance,output):
    original=json.loads((source/'report.json').read_text())
    if original['status']!='completed' or original['completed_files']!=1808 or original['failed_files']!=0:raise ValueError('full_successful_expression_scope_required')
    if hash_file(provenance/'report.json')!=original['identity']['provenance_sha256']:raise ValueError('provenance_identity_mismatch')
    files=verified_files(source);output.mkdir(parents=True,exist_ok=False)
    for name in files:
        target=output/'expression'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source/name,target)
    samples=json.loads((source/'samples.json').read_text());by_study=defaultdict(list);by_sample=defaultdict(list)
    records=[];states=[];identity_rows=[];numeric=defaultdict(Counter);types=Counter();lineages=Counter();eligibility=Counter()
    state_names=list(json.loads((source/'references/gene_sets.json').read_text())['states'])
    for item in samples:
        accession=item['experiment_accession'];directory=source/accession
        if not item['inputs_unchanged']:raise ValueError('source_changed')
        origin=json.loads((directory/'provenance.json').read_text());schema=pq.read_schema(directory/'cells.parquet')
        needed=['row_index','record_id','input_sha256','source_experiment','source_barcode','source_cell_type','inferred_type','inferred_lineage','inferred_subtype','confidence_calibration','probability_correct']
        if not set(needed+['state__'+name for name in state_names])<=set(schema.names):raise ValueError('incomplete_cell_annotation_contract')
        cells=pd.read_parquet(directory/'cells.parquet',columns=needed)
        if len(cells)!=item['n_cells'] or not np.array_equal(cells.row_index,np.arange(len(cells))) or cells.record_id.duplicated().any():raise ValueError('cell_identity_coverage_mismatch')
        if not (cells.input_sha256==item['input_sha256']).all() or not (cells.source_experiment==accession).all():raise ValueError('cell_source_identity_mismatch')
        if cells.probability_correct.notna().any() or not (cells.confidence_calibration=='uncalibrated').all():raise ValueError('false_calibrated_confidence')
        if cells.inferred_type.value_counts().to_dict()!=item['inferred_types']:raise ValueError('type_summary_mismatch')
        types.update(item['inferred_types']);lineages.update(item['inferred_lineages'])
        eligibility[item['inference_eligibility']['reason']]+=1
        row={**{k:v for k,v in item.items() if k not in ['numeric','states']},
            'source_studies':origin.get('study_accessions',[]),'source_samples':origin.get('sample_accessions',[]),
            'source_title':origin.get('experiment_title'),'source_library':origin.get('library_name'),
            'report':f'expression/{accession}/report.json'}
        records.append(row)
        identity_rows.append({'experiment':accession,'records':len(cells),'row_sequence_verified':True,'unique_record_ids':True,
            'source_identity_verified':True,'source_labels_separate':True,'uncalibrated_probability_null':True,'all_state_columns_present':True})
        for study in origin.get('study_accessions',[]) or ['unresolved']:
            by_study[study].append(row)
        for sample in origin.get('sample_accessions',[]):by_sample[sample].append(row)
        for state in item['states']:
            states.append({'experiment':accession,'studies':row['source_studies'],'state':state['state'],'status':state['status'],
                'distribution':state['distribution'],'baseline':state['baseline'],'cross_sample_calibration':'unavailable; sample-specific background'})
        for layer,values in item['numeric'].items():
            numeric[layer].update({key:values[key] for key in ['stored_values_checked','nonfinite','negative','noninteger']})
    studies=[]
    for name,group in sorted(by_study.items()):
        counts=Counter();coarse=Counter();source_samples=set()
        for row in group:counts.update(row['inferred_types']);coarse.update(row['inferred_lineages']);source_samples.update(row['source_samples'])
        total=sum(r['n_cells'] for r in group)
        studies.append({'source_study':name,'files':len(group),'source_samples':len(source_samples),'stored_records':total,
            'inferred_types':dict(counts),'inferred_lineages':dict(coarse),'unknown_fraction':counts.get('unknown',0)/total,
            'sample_median_library_size_distribution':quantiles([r['library_size']['median'] for r in group]),
            'sample_unknown_fraction_distribution':quantiles([r['unknown_fraction'] for r in group]),
            'non_applicable_files':sum(r['inference_eligibility']['status']=='not_applicable' for r in group),
            'interpretation':'Stored record composition, no cross-library deduplication or independent biological replication claim'})
    candidates=[]
    for name,group in sorted(by_sample.items()):
        if len(group)<2:continue
        candidates.append({'source_sample':name,'experiments':[r['experiment_accession'] for r in group],
            'studies':sorted(set(s for r in group for s in r['source_studies'])),'library_names':[r['source_library'] for r in group],
            'stored_records':sum(r['n_cells'] for r in group),'gene_axes':sorted(set(r['gene_axis_sha256'] for r in group)),
            'candidate_kind':'Shared biological sample: possible sequencing/library repeat or distinct assay',
            'same_cells_proven':False,'independent_biological_replicates':False,'next_check':'Exact study/library/cell/count reconciliation in #19'})
    for name,rows in [('studies',studies),('samples',records),('sample-states',states),('shared-sample-candidates',candidates),('annotation-contract',identity_rows)]:
        write_json(output/(name+'.json'),rows)
    overview={'files':len(samples),'source_records':sum(r['n_cells'] for r in samples),'distinct_study_labels':len(by_study),
        'all_metadata_cell_counts_match':all(r['metadata_count_matches'] for r in samples),
        'all_source_inputs_unchanged':all(r['inputs_unchanged'] for r in samples),'numeric_invalid_cells':sum(r['numeric_invalid_cells'] for r in samples),
        'zero_libraries':sum(r['zero_libraries'] for r in samples),'inferred_types':dict(types),'inferred_lineages':dict(lineages),
        'method_conflicts':sum(r['method_conflicts'] for r in samples),'eligibility_file_counts':dict(eligibility),
        'source_QC_disagreements':dict(sum((Counter({k:v for k,v in r['source_QC_disagreements'].items() if v is not None}) for r in samples),Counter())),
        'shared_sample_groups':len(candidates),'all_probabilities_null':True,'not_deduplicated':True}
    write_json(output/'overview.json',overview)
    report={'schema_version':2,'bundle_id':'scbase-dossier-'+uuid.uuid4().hex,'title':'scBaseCount 全表达、样本与研究组成评估',
        'status':'completed','completed_at':datetime.now(timezone.utc).isoformat(),'inputs_unchanged':True,
        'expression_bundle':original['bundle_id'],'expression_report_sha256':hash_file(source/'report.json'),
        'provenance_report_sha256':hash_file(provenance/'report.json'),'identity':original['identity'],
        'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'composer_sha256':hash_file(Path(__file__)),
        'methods':{**original['methods'],'aggregation':'Types/counts pooled descriptively by source study; sample states retain each file-specific background. Shared BioSample groups are candidates, not proven duplicates.',
            'contract_verification':'Every source row, sequential source index, within-file unique identity, input/experiment labels, separate source/inferred fields, all state columns and null uncalibrated probability checked'},
        'limitations':original['limitations']+['研究汇总按来源关系计数，未去重；共享样本不证明相同细胞或技术重复，更不增加独立生物重复。',
            '每个样本的状态背景不同；状态分位数不能直接作为跨研究的统一活性尺度。研究深度分布是样本中位数的分布，不是合并细胞中位数。'],
        'exposure':'All 1808 local count matrices and all applicable per-cell RNA endpoints explored; source labels not independent ground truth',
        'tables':[{'title':'全量覆盖与核验','rows':[overview]},
            {'title':'全部数值层','rows':[{'layer':k,**dict(v)} for k,v in numeric.items()]},
            {'title':'全部研究组成','rows':studies},{'title':'全部样本组成与资格','rows':records},
            {'title':'全部样本状态代理分布','rows':states},{'title':'共享源样本候选','rows':candidates}],
        'reproduce':sys.argv}
    report['artifacts']=[{'file':str(p.relative_to(output)),'sha256':hash_file(p)} for p in sorted(output.rglob('*')) if p.is_file()]
    write_json(output/'report.json',report);(output/'report.html').write_text(render(serial(report)))
    (output/'SHA256SUMS').write_text(''.join(f'{hash_file(p)}  {p.relative_to(output)}\n' for p in sorted(output.rglob('*')) if p.is_file() and p.name!='SHA256SUMS'))
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['source','provenance','output']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();r=compose(a.source,a.provenance,a.output);print(json.dumps({'status':r['status'],'bundle_id':r['bundle_id']}))
