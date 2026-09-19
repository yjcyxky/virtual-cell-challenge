#!/usr/bin/env python
"""Use explicit McFarland hash treatment/time labels without editing source obs."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import shutil
import subprocess
import pandas as pd
from rna import hash_file,value_hash,RNAFile
from profile_responses import write_json

ROOT=Path(__file__).resolve().parents[2]


def refine_rows(obs,design):
    result=design.copy();obs=obs.reset_index(drop=True);counts=Counter()
    for i,row in obs.iterrows():
        original_time=str(row['time'])
        if original_time=='72, 96':
            result.loc[i,['condition_identity_supported','control_eligible']]=False
            result.loc[i,'response_scope_reason']='pooled_source_time_label_does_not_identify_matched_72_or_96_hour_experiment'
            counts['unresolved_genetic_time_records']+=1
        if original_time!='3, 6, 12, 24, 48':continue
        tag=str(row.hash_tag);assignment=str(row.hash_assignment);match=re.fullmatch(r'(DMSO|Tram|Untreated)_(3|6|12|24|48)hr',tag)
        supported=match is not None and assignment==tag
        treatment=match[1] if supported else None;hours=int(match[2]) if supported else None
        bg=json.loads(result.response_background.iloc[i]);bg.append(['explicit_source_hash_hours',hours]);bg.append(['source_channel',str(row.channel)])
        result.loc[i,'response_background']=json.dumps(bg,separators=(',',':'))
        condition=[['original_collection_perturbation',str(row.perturbation)],['original_collection_time',original_time],
            ['original_collection_dose_value',str(row.dose_value)],['source_hash_tag',tag],['source_hash_assignment',assignment],
            ['explicit_hash_treatment',treatment],['explicit_hash_hours',hours]]
        result.loc[i,'analysis_condition']=json.dumps(condition,separators=(',',':'))
        result.loc[i,'control_eligible']=supported and treatment=='DMSO';result.loc[i,'condition_identity_supported']=supported
        result.loc[i,'control_rule']='explicit_source_DMSO_hash_same_hours_cell_line_and_channel'
        result.loc[i,'response_scope_reason']='collection_perturbation_time_dose_are_pooled_experiment_labels_use_explicit_hash_identity' if supported else 'source_hash_missing_multiplet_or_assignment_disagreement'
        counts['timecourse_records']+=1;counts['supported_timecourse_records' if supported else 'unsupported_timecourse_records']+=1
        if supported and treatment=='DMSO':counts['explicit_DMSO_hash_reference_records']+=1
    return result,dict(counts)


def run(source,output):
    report=json.loads((source/'report.json').read_text())
    if report['status']!='completed':raise ValueError('completed_design_required')
    for record in report['files']:
        if record['status']!='completed':continue
        for name,digest in record['artifacts'].items():
            if hash_file(source/record['directory']/name)!=digest:raise ValueError('original_design_artifact_changed')
    shutil.copytree(source,output);file='McFarlandTsherniak2020.h5ad';folder=output/file.removesuffix('.h5ad');record=next(r for r in report['files'] if r['file']==file)
    path=ROOT/'data/raw/scperturb'/file
    if hash_file(path)!=record['input_sha256']:raise ValueError('source_RNA_changed')
    with RNAFile(path) as r:design,summary=refine_rows(r.obs,pd.read_parquet(folder/'row-applicability.parquet'))
    design.to_parquet(folder/'row-applicability.parquet',index=False)
    conditions=design.groupby(['response_background','analysis_condition'],sort=True,dropna=False).agg(n_cells=('row_index','size'),control_cells=('control_eligible','sum'),
        supported_condition_cells=('condition_identity_supported','sum'),deep_candidate_cells=('deep_response_candidate','sum')).reset_index()
    conditions['condition_id']=[value_hash([file,a,b]) for a,b in zip(conditions.response_background,conditions.analysis_condition)]
    conditions.to_parquet(folder/'all-source-conditions.parquet',index=False)
    methods=json.loads((folder/'methods.json').read_text());methods['explicit_source_hash_refinement']=summary
    methods['limitations']+=['The original per-cell perturbation, time and dose fields in the time-course block describe the pooled experiment; source hash labels supply observed treatment/time identities without RNA-based inference.',
        'The 72,96 hour genetic block lacks a resolved per-record timepoint; descriptive distributions retained, matched-time contrasts not estimable.']
    write_json(folder/'methods.json',methods)
    record.update(all_source_conditions=len(conditions),eligible_controls=int(design.control_eligible.sum()),unsupported_condition_cells=int((~design.condition_identity_supported).sum()),
        response_backgrounds=design.response_background.nunique(),source_condition_limitation_counts=design.response_scope_reason.fillna('no_additional_identity_limitation').value_counts().to_dict(),
        source_hash_refinement=summary)
    record['artifacts']={p.name:hash_file(p) for p in folder.iterdir() if p.is_file() and p.name!='report.json'};write_json(folder/'report.json',record)
    report['identity']={'base_design_report_sha256':hash_file(source/'report.json'),'base_identity':report['identity'],'refinement_code_sha256':hash_file(Path(__file__))}
    report['phase']='frozen_54_file_response_design_with_explicit_McFarland_hash_timecourse';report['code_commit']=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    write_json(output/'identity.json',report['identity']);write_json(output/'report.json',report);print(json.dumps(summary),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.source,a.output)
