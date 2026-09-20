#!/usr/bin/env python
"""Additional human RNA negative-guide baselines, distinct from vehicle/cutting controls."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pandas as pd
from scipy import sparse
from profile_cross_source_coverage import Frozen,ROOT,BASE
from profile_response_heterogeneity import task_eligibility
from profile_background import matrix_in_memory
from profile_responses import write_json
from rna import RNAFile,hash_file,value_hash

NEGATIVE_GUIDE_RULES={
 'author_demo_explicit_triple_negative_construct','source_CTRL_guide_mapping','explicit_source_scrambled_guides_only','explicit_negative_guide_pairs_only',
 'explicit_NTg_guide','explicit_non_targeting_guide_not_zero_guide_control','source_NonTarget_guide_mapping',
 'non_targeting_source_good_coverage_exact_or_likely_match','explicit_source_control_guide_combinations','explicit_non_targeting_guides',
 'source_Cas13_non_targeting_guide_class','explicit_NO_TARGET_guides','source_non_targeting_construct_mapping'}


def run(source,output):
    source=source.resolve();output=output.resolve();output.mkdir(parents=True,exist_ok=False);f=Frozen();raw={};catalog=[];scope=[]
    parent=f.json(source,'report.json');panels={p['panel_id']:p for p in f.json(source,'coverage/panels.json')};tasks=f.parquet(source,'coverage/all-source-tasks.parquet')
    primary=tasks.loc[[task_eligibility(r,panels[r['panel_id']])[0] for r in tasks.to_dict('records')]]
    def profile(pid,path,digest,cells,groups,rules):
        if hash_file(path)!=digest:raise ValueError('additional_baseline_count_input_changed')
        raw[str(path.relative_to(ROOT))]=digest;before=path.stat();mapping=f.parquet(source,'coverage/'+panels[pid]['native_mapping_file'])
        with RNAFile(path) as source_rna:
            genes=source_rna.var.index.astype(str).tolist()
            if genes!=mapping.source_gene.astype(str).tolist():raise ValueError('additional_baseline_native_gene_axis_changed')
            if not np.array_equal(cells.row_index,np.arange(len(cells))) or not np.array_equal(cells.source_barcode.astype(str),source_rna.obs.index.astype(str)):raise ValueError('additional_baseline_cell_axis_changed')
            if len(cells)!=source_rna.shape[0]:raise ValueError('additional_baseline_cell_count_changed')
            codes=np.full(len(cells),-1,dtype=int);keys=list(groups);counts=np.zeros(len(keys),dtype=int);sums=np.zeros((len(keys),len(genes)))
            for i,k in enumerate(keys):codes[np.asarray(groups[k],dtype=int)]=i
            memory=matrix_in_memory(source_rna) if source_rna.encoding=='csc_matrix' else None
            def blocks():
                if memory is None:yield from source_rna.blocks(1024)
                else:
                    for start in range(0,source_rna.shape[0],1024):yield start,memory[start:start+1024].astype(float)
            for start,x in blocks():
                local=codes[start:start+x.shape[0]];keep=local>=0
                if not keep.any():continue
                x=x[keep].astype(float);x.sum_duplicates();totals=np.asarray(x.sum(axis=1)).ravel();ids=np.flatnonzero(keep)+start
                expected=cells['computed_total_counts' if 'computed_total_counts' in cells else 'computed_total_expression'].iloc[ids].to_numpy()
                if not np.array_equal(totals,expected) or (totals<=0).any() or not np.isfinite(x.data).all() or (x.data<0).any() or (x.data!=np.floor(x.data)).any():raise ValueError('additional_baseline_native_count_contract_failed')
                x.data=np.log1p(x.data*np.repeat(10000/totals,np.diff(x.indptr))).astype(np.float32)
                local=local[keep];assignment=sparse.csr_matrix((np.ones(len(local)),(local,np.arange(len(local)))),shape=(len(keys),len(local)))
                sums+=(assignment@x).toarray();counts+=np.bincount(local,minlength=len(keys))
            del memory
        if (path.stat().st_size,path.stat().st_mtime_ns,path.stat().st_ctime_ns)!=(before.st_size,before.st_mtime_ns,before.st_ctime_ns):raise ValueError('additional_baseline_counts_mutated')
        for i,context in enumerate(keys):
            id=value_hash([pid,context]);folder=output/id;folder.mkdir();means=sums[i]/counts[i] if counts[i] else np.full(len(genes),np.nan)
            np.savez_compressed(folder/'baseline.npz',source_gene=np.asarray(genes,dtype=str),mean_logCP10K=means,n_NTC=int(counts[i]),baseline_id=id)
            mapping.to_parquet(folder/'native-gene-mapping.parquet',index=False)
            side=cells.iloc[np.asarray(groups[context],dtype=int)][['record_id','input_sha256','row_index','source_barcode']].copy();side['source_control_rule']=rules[context]
            side.to_parquet(folder/'source-negative-guide-records.parquet',index=False,compression='zstd')
            row={'context_id':id,'baseline_id':id,'panel_id':pid,'source_context':context,'NTC_cells':int(counts[i]),'source_control_rule':rules[context],
                 'status':'completed' if counts[i] else 'not_estimable','reason':None if counts[i] else 'no_verified_NTC_cells',
                 'reference_role':'source_negative_guide_reference; not proof of zero effective perturbation','source_metadata':panels[pid]['source_metadata']}
            catalog.append(row);write_json(folder/'report.json',row)
        print('additional negative-guide baselines '+pid+' '+str(len(keys)),flush=True)
    sc=BASE/'scperturb-dossier-20260919'
    for row in f.json(sc,'file-index.json'):
        name=row['file'].removesuffix('.h5ad');pid='scPerturb:'+name;p=panels.get(pid);reason=None
        if not p or not p['human_RNA_applicable']:reason='not_verified_human_RNA'
        elif p['confirmed_collection_copy']:reason='confirmed_collection_copy_no_new_baseline'
        elif row.get('expression_scale')!='log1p_CP10K_analysis_view':reason='source_scale_not_native_count_compatible'
        if reason:scope.append({'panel_id':pid,'status':'not_applicable','reason':reason});continue
        design=f.parquet(sc,'design/'+name+'/row-applicability.parquet');all_ctrl=design.loc[design.control_eligible]
        negative=design.control_eligible&design.control_rule.isin(NEGATIVE_GUIDE_RULES)
        covered=set(primary.loc[primary.panel_id.eq(pid),'source_background_index'].dropna().astype(int))
        cells=f.parquet(sc,'cells/'+name+'/cells.parquet')[['record_id','input_sha256','row_index','source_barcode','computed_total_expression','computed_numeric_valid']]
        if not np.array_equal(cells.row_index,design.row_index):raise ValueError('additional_baseline_source_design_axis_mismatch')
        groups={};rules={}
        for bg in sorted(design.loc[negative,'biological_background_index'].unique()):
            if int(bg) in covered:continue
            context='source_biological_background_'+str(int(bg));keep=negative&design.biological_background_index.eq(bg)&cells.computed_numeric_valid&cells.computed_total_expression.gt(0)
            groups[context]=np.flatnonzero(keep);rules[context]='|'.join(sorted(design.loc[keep,'control_rule'].unique()))
        scope.append({'panel_id':pid,'status':'completed' if groups else 'not_applicable','reason':None if groups else 'no_additional_verified_negative_guide_background_beyond_primary',
                      'source_reference_rules':all_ctrl.control_rule.value_counts().to_dict(),'negative_guide_control_records':int(negative.sum()),'additional_backgrounds':len(groups),
                      'intergenic_cutting_vehicle_untreated_are_NTC':False})
        if groups:
            audit=f.json(sc,'source-audit/'+row['file']+'.json');profile(pid,ROOT/audit['path'],audit['input_sha256'],cells,groups,rules)
    # GxE1 CRISPRa negative-guide backgrounds are valid RNA reference panels;
    # their perturbation responses remain outside the primary CRISPRi pairs.
    gxe1=BASE/'mcfaline-gxe1-dossier-20260919-v2';cells=f.parquet(gxe1,'cells.parquet');covered=set(primary.loc[primary.panel_id.eq('GxE1'),'source_condition'])
    contexts=sorted(set(tasks.loc[tasks.panel_id.eq('GxE1')&tasks.modality.eq('CRISPRa'),'source_condition'].dropna())-covered);groups={};rules={}
    for context in contexts:
        keep=cells.source_context.eq(context)&cells.genetic_response_eligible&cells.analysis_target.eq('NTC')
        groups[context]=np.flatnonzero(keep);rules[context]='source_NTC_same_effector_library_drug_dose'
    if groups:
        report=f.json(gxe1,'report.json');args=report['reproduce'];cache=ROOT/args[args.index('--cache')+1];identity=f.json(gxe1,'adapter/identity.json')
        profile('GxE1',cache/'coordinate-counts.h5ad',identity['artifacts']['coordinate-counts.h5ad'],cells,groups,rules)
    write_json(output/'baseline-contexts.json',catalog);write_json(output/'all-source-negative-guide-scope.json',scope)
    write_json(output/'consumed-inputs.json',{'frozen_artifacts':f.used,'raw_count_files':raw})
    report={'status':'completed','phase':'additional_verified_negative_guide_RNA_baselines','source_bundle_id':parent['bundle_id'],'contexts':catalog,
        'additional_contexts':len(catalog),'actual_NTC_contexts':sum(r['NTC_cells']>0 for r in catalog),'NTC_records':sum(r['NTC_cells'] for r in catalog),
        'negative_guide_rule_whitelist':sorted(NEGATIVE_GUIDE_RULES),'intergenic_cutting_vehicle_untreated_are_NTC':False,'input_mutations':0,
        'completed_at':datetime.now(timezone.utc).isoformat(),'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'code_sha256':hash_file(Path(__file__)),'reproduce':sys.argv}
    report['artifacts']={str(p.relative_to(output)):hash_file(p) for p in sorted(output.rglob('*')) if p.is_file()};write_json(output/'report.json',report)
    (output/'SHA256SUMS').write_text(''.join(hash_file(p)+'  '+str(p.relative_to(output))+'\n' for p in sorted(output.rglob('*')) if p.is_file() and p.name!='SHA256SUMS'))
    print(json.dumps({k:v for k,v in report.items() if k not in ['artifacts','contexts']}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    try:run(a.source,a.output)
    except Exception as error:
        if a.output.exists():write_json(a.output/'failure.json',{'status':'failed','error':repr(error),'code_sha256':hash_file(Path(__file__))})
        raise
