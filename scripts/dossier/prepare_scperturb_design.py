#!/usr/bin/env python
"""Freeze all reviewed scPerturb row and task applicability before responses."""
import argparse
from collections import defaultdict
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import shutil
import pandas as pd
from rna import RNAFile,hash_file,value_hash
from profile_responses import write_json
from scperturb_design import build_design

ROOT=Path(__file__).resolve().parents[2]


def gene_names(path):
    h=pd.read_csv(path,sep='\t',dtype=str).fillna('');approved=set(h.symbol);aliases=defaultdict(set)
    for row in h.itertuples():
        for field in ['alias_symbol','prev_symbol']:
            for name in getattr(row,field,'').split('|'):
                if name:aliases[name].add(row.symbol)
    result={n:n for n in approved};result.update({n:next(iter(v)) for n,v in aliases.items() if len(v)==1 and n not in approved});return result


def run(manifest,adamson,output,refine_from=None,refine_files=None):
    output.mkdir(parents=True,exist_ok=False);names=gene_names(ROOT/'data/raw/networks/hgnc_complete_set.txt');results=[]
    identity={'manifest_sha256':hash_file(manifest),'Adamson_identity_report_sha256':hash_file(adamson/'report.json'),
        'HGNC_sha256':hash_file(ROOT/'data/raw/networks/hgnc_complete_set.txt'),
        'code':{n:hash_file(Path(__file__).with_name(n)) for n in ['prepare_scperturb_design.py','scperturb_design.py','scperturb_cells.py']}}
    base={}
    if refine_from is not None:
        if not refine_files:raise ValueError('explicit_refined_file_scope_required')
        previous=json.loads((refine_from/'report.json').read_text())
        if previous['status']!='completed':raise ValueError('completed_base_design_required')
        base={r['file']:r for r in previous['files']};identity['base_design_report_sha256']=hash_file(refine_from/'report.json');identity['refined_files']=refine_files
    write_json(output/'identity.json',identity)
    for entry in json.loads(manifest.read_text()):
        file=entry['file']
        if entry['analysis']['status']=='not_applicable':results.append({'file':file,'status':'not_applicable','reason':entry['analysis']['reason']});continue
        if base and file not in refine_files:
            record=base[file]
            if record['status']!='completed' or record['input_sha256']!=entry['input_sha256']:raise ValueError('base_file_scope_changed')
            for name,digest in record['artifacts'].items():
                if hash_file(refine_from/record['directory']/name)!=digest:raise ValueError('base_design_artifact_changed')
            shutil.copytree(refine_from/record['directory'],output/record['directory']);results.append(record);continue
        path=ROOT/'data/raw/scperturb'/file
        if hash_file(path)!=entry['input_sha256']:raise ValueError('collection_input_changed')
        recover=pd.read_parquet(adamson/(file.replace('AdamsonWeissman2016_','').replace('.h5ad','.parquet'))) if file.startswith('Adamson') else None
        with RNAFile(path) as source:design,methods=build_design(file,source.obs,names,recover)
        directory=output/file.removesuffix('.h5ad');directory.mkdir();design.to_parquet(directory/'row-applicability.parquet',index=False)
        # Every source condition is represented, including unknown assignment and
        # multi-target rows that cannot identify an isolated gene intervention.
        conditions=design.groupby(['response_background','analysis_condition'],sort=True,dropna=False).agg(
            n_cells=('row_index','size'),control_cells=('control_eligible','sum'),supported_condition_cells=('condition_identity_supported','sum'),
            deep_candidate_cells=('deep_response_candidate','sum')).reset_index()
        conditions['condition_id']=[value_hash([file,a,b]) for a,b in zip(conditions.response_background,conditions.analysis_condition)]
        conditions.to_parquet(directory/'all-source-conditions.parquet',index=False)
        candidates=design.loc[design.deep_response_candidate].groupby(['biological_background_index','single_target','target_kind'],sort=True).agg(n_candidate_cells=('row_index','size')).reset_index()
        candidates.to_parquet(directory/'deep-candidate-tasks.parquet',index=False);write_json(directory/'methods.json',methods)
        summary={'file':file,'status':'completed','cells':len(design),'all_source_conditions':len(conditions),'deep_candidate_tasks':len(candidates),
            'deep_candidate_cells':int(design.deep_response_candidate.sum()),'eligible_controls':int(design.control_eligible.sum()),
            'unsupported_condition_cells':int((~design.condition_identity_supported).sum()),'response_backgrounds':design.response_background.nunique(),
            'source_condition_limitation_counts':design.response_scope_reason.fillna('no_additional_identity_limitation').value_counts().to_dict(),
            'target_kind_counts':design.target_kind.value_counts().to_dict(),'input_sha256':entry['input_sha256'],'directory':directory.name,
            'artifacts':{p.name:hash_file(p) for p in directory.iterdir() if p.is_file()}}
        write_json(directory/'report.json',summary);results.append(summary);print(file+' controls='+str(summary['eligible_controls'])+' deep='+str(len(candidates)),flush=True)
    report={'status':'completed','phase':'frozen_54_file_response_design','identity':identity,'files':results,
        'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'completed_at':datetime.now(timezone.utc).isoformat()}
    write_json(output/'report.json',report)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['manifest','adamson','output']:p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--refine-from',type=Path);p.add_argument('--refine-files',nargs='+')
    a=p.parse_args();run(a.manifest,a.adamson,a.output,a.refine_from,a.refine_files)
