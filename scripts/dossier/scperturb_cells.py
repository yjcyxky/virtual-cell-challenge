#!/usr/bin/env python
"""All-file RNA identity and conservative per-cell inference, separate from controls."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,as_completed
from datetime import datetime,timezone
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import shutil
import subprocess
import sys
import time
import h5py
import numpy as np
import pandas as pd
from scipy import sparse
from annotation import marker_model,state_model,RULES
from sparse_annotation import annotate_sparse
from profile_background import matrix_in_memory
from prepare_mouse_references import mouse_symbol_index
from rna import RNAFile,hash_file,value_hash,mapping_audit,quantiles
from profile_responses import write_json,serial
from render import render

ROOT=Path(__file__).resolve().parents[2]
CODE=['scperturb_cells.py','rna.py','annotation.py','sparse_annotation.py','profile_background.py','prepare_mouse_references.py','profile_responses.py','render.py']


def biological_background(file,obs):
    fields=[c for c in ['cell_line','DepMap_ID','time','medium','age','patient'] if c in obs]
    if file.startswith('Zhao'):fields += [c for c in ['sample','tissue'] if c in obs]
    if file.startswith(('Frangieh','Datlinger','Shifrut')) and 'perturbation_2' in obs:fields.append('perturbation_2')
    if file.startswith('Papalexi') and 'arrayed' not in file and 'hto' in obs:fields.append('hto')
    frame=obs[fields].copy() if fields else pd.DataFrame({'source_file_scope':['whole_file']*len(obs)})
    frame=frame.astype(object).where(frame.notna(),None)
    keys=[json.dumps(list(r),ensure_ascii=False,separators=(',',':')) for r in frame.itertuples(index=False,name=None)]
    codes,unique=pd.factorize(np.asarray(keys,dtype=object),sort=True)
    return codes,[{'background_index':i,'source_fields':fields,'source_values':json.loads(key),'background_key':key} for i,key in enumerate(unique)]


def mapped_axis(file,var,species,references):
    native=var.index.astype(str).tolist()
    symbol_field=next((c for c in ['gene_name','gene_symbols','gene_symbol','symbol'] if c in var),None)
    symbols=var[symbol_field].astype(str).tolist() if symbol_field else native
    id_field=next((c for c in ['ensembl_id','ensembl_gene_id','gene_ids','gene_id'] if c in var),None)
    identifiers=var[id_field].astype(str).tolist() if id_field else None
    if species=='human':
        table=mapping_audit(symbols,pd.read_csv(ROOT/'data/raw/networks/hgnc_complete_set.txt',sep='\t',low_memory=False),pd.read_csv(ROOT/'data/raw/arc_vcc2026_controls/gene_names.csv').gene_name.tolist(),identifiers)
        table=table.rename(columns={'source_gene':'source_symbol'});table.insert(0,'source_gene',native)
        safe=table.mapped_symbol.where(~table.many_to_one_mapping&(table.symbol_vs_ensembl!='conflict'))
    else:
        mgi=pd.read_csv(references/'MGI/HOM_MouseHumanSequence.rpt',sep='\t',dtype=str);official,case=mouse_symbol_index(mgi)
        # Source RNA mouse symbols use their native case. Case alignment is explicit and unique.
        resolved=[s if s in official else case.get(s.upper()) for s in symbols];multiplicity=Counter(s for s in resolved if s)
        table=pd.DataFrame({'source_gene':native,'source_symbol':symbols,'mapped_symbol':resolved,
            'mapping_status':['exact_mouse_symbol' if s in official else 'unique_mouse_symbol_case_alignment' if r else 'unresolved_in_MGI_orthology_report' for s,r in zip(symbols,resolved)],
            'many_to_one_mapping':[multiplicity[r]>1 if r else False for r in resolved],'symbol_vs_ensembl':'not_verified_by_MGI_orthology_report'})
        safe=table.mapped_symbol.where(~table.many_to_one_mapping)
    identity_axis=identifiers if identifiers is not None and len(set(identifiers))==len(identifiers) and all(s.startswith(('ENSG','ENSMUSG')) for s in identifiers) else native
    return native,identity_axis,table,[str(s) if pd.notna(s) else None for s in safe]


def expression_view(matrix,count_compatible):
    x=matrix.astype(np.float64,copy=True)
    if count_compatible:
        totals=np.asarray(x.sum(axis=1)).ravel();scale=np.divide(10000.,totals,out=np.zeros_like(totals),where=totals>0)
        x.data=np.log1p(x.data*np.repeat(scale,np.diff(x.indptr)))
    return x.astype(np.float32)


def scan_file(entry,output_string,human_string,mouse_string,identity):
    started=time.monotonic();file=entry['file'];path=ROOT/'data/raw/scperturb'/file;output=Path(output_string)/file.removesuffix('.h5ad')
    if (output/'report.json').exists():
        old=json.loads((output/'report.json').read_text())
        if old['identity']!=identity:raise ValueError('scPerturb_cell_resume_identity_changed')
        for name,digest in old['artifact_hashes'].items():
            if hash_file(output/name)!=digest:raise ValueError('scPerturb_completed_cell_artifact_changed')
        return old['summary']
    if hash_file(path)!=entry['input_sha256']:raise ValueError('scPerturb_source_hash_changed')
    species=entry['species'][0]
    if len(entry['species'])!=1 or species not in ['human','mouse']:raise ValueError('unregistered_mixed_or_unknown_species')
    reference_path=Path(human_string if species=='human' else mouse_string);reference=json.loads((reference_path/'gene_sets.json').read_text())
    count_compatible=entry['matrices']['X']['nonnegative_integer_compatible'];transform_verified=count_compatible or file.startswith('Gehring')
    scale='log1p_CP10K_analysis_view' if count_compatible else 'source_verified_log1p_normalized_10000' if file.startswith('Gehring') else 'source_continuous_X_transform_unverified'
    output.mkdir(parents=True,exist_ok=False)
    with RNAFile(path) as source:
        n,g=source.shape;obs=source.obs.reset_index(drop=True);source.var.reset_index(names='source_var_index').to_parquet(output/'source-var.parquet',index=False)
        native,identity_axis,mapping,resolved=mapped_axis(file,source.var,species,reference_path);mapping.to_parquet(output/'gene-mapping.parquet',index=False)
        axis_hash=value_hash(native);identity_axis_hash=value_hash(identity_axis);codes,backgrounds=biological_background(file,obs);k=len(backgrounds)
        if k>1000:raise ValueError('unreviewed_high_cardinality_biological_background')
        qc=obs.add_prefix('source_');qc.insert(0,'source_barcode',source.obs.index.astype(str));qc.insert(0,'row_index',np.arange(n))
        qc.insert(0,'input_sha256',entry['input_sha256']);qc.insert(0,'record_id',[value_hash([entry['input_sha256'],i]) for i in range(n)])
        qc['study_id']=file.removesuffix('.h5ad');qc['background_index']=codes;qc['gene_axis_sha256']=axis_hash;qc['source_identity_axis_sha256']=identity_axis_hash
        means=np.zeros((k,g),float);sizes=np.bincount(codes,minlength=k);totals=np.empty(n);detected=np.empty(n,np.int32);valid=np.empty(n,bool)
        hashes=[];identity_hashes=[];numeric=Counter();gene_sums=np.zeros(g);gene_detected=np.zeros(g,np.int64)
        # CSC is converted once in an independent in-memory analysis view; dense arrays are streamed.
        memory=matrix_in_memory(source) if source.encoding=='csc_matrix' else None
        chunk=source.x.chunks[0] if source.encoding=='dense' and source.x.chunks else 1024
        chunk=max(256,min(chunk,8192))
        def blocks():
            if memory is None:yield from source.blocks(chunk)
            else:
                for a in range(0,n,1024):yield a,memory[a:a+1024].astype(float)
        for start,matrix in blocks():
            stop=start+matrix.shape[0];d=matrix.data;bad=~np.isfinite(d)|(d<0)
            numeric.update(stored_X_values=len(d),nonfinite=int((~np.isfinite(d)).sum()),negative=int((d<0).sum()),noninteger=int((np.isfinite(d)&(d!=np.floor(d))).sum()))
            valid[start:stop]=np.bincount(np.repeat(np.arange(stop-start),np.diff(matrix.indptr)),weights=bad.astype(int),minlength=stop-start)==0
            matrix.sum_duplicates();matrix.eliminate_zeros();matrix.sort_indices();totals[start:stop]=np.asarray(matrix.sum(axis=1)).ravel();detected[start:stop]=np.diff(matrix.indptr)
            gene_sums+=np.asarray(matrix.sum(axis=0)).ravel();gene_detected+=np.bincount(matrix.indices,minlength=g)
            for i in range(matrix.shape[0]):
                lo,hi=matrix.indptr[i:i+2];indices=matrix.indices[lo:hi].astype('<i8').tobytes();values=matrix.data[lo:hi].astype('<f8').tobytes()
                digest=hashlib.sha256(axis_hash.encode());digest.update(indices);digest.update(values);hashes.append(digest.hexdigest())
                if identity_axis_hash==axis_hash:identity_hashes.append(hashes[-1])
                else:
                    digest=hashlib.sha256(identity_axis_hash.encode());digest.update(indices);digest.update(values);identity_hashes.append(digest.hexdigest())
            if bad.any():raise ValueError('nonfinite_or_negative_RNA_expression')
            x=expression_view(matrix,count_compatible)
            for code in np.unique(codes[start:stop]):means[code]+=np.asarray(x[codes[start:stop]==code].sum(axis=0,dtype=float)).ravel()
        means/=sizes[:,None];qc['computed_total_expression']=totals;qc['computed_detected_genes']=detected;qc['computed_numeric_valid']=valid
        qc['computed_count_sha256']=hashes;qc['computed_on_source_identity_axis_sha256']=identity_hashes;qc['computed_total_is_UMI']=count_compatible
        pd.DataFrame({'source_gene':native,'source_identity_gene':identity_axis,'computed_sum':gene_sums,'computed_detected_cells':gene_detected}).to_parquet(output/'genes.parquet',index=False)
        weights=[];coverage=[]
        for code in range(k):
            w,c=state_model(resolved,reference['states'],means[code]);weights.append(w);coverage.extend([{'background_index':code,**item} for item in c])
            backgrounds[code].update(n_cells=int(sizes[code]),background_role='pooled_observed_endpoint_RNA_not_NTC',source_expression_scale=scale)
        np.savez_compressed(output/'state-backgrounds.npz',mean_expression=means,state_weights=np.stack(weights),background_cells=sizes)
        write_json(output/'backgrounds.json',backgrounds);write_json(output/'state-coverage.json',coverage)
        model=marker_model(resolved,reference['profiles']);write_json(output/'type-coverage.json',model['coverage']);parts=[]
        lookup=set(s for s in resolved if s);target=[str(t) if str(t) in lookup else None for t in obs.get('perturbation',pd.Series([None]*n))]
        for start,matrix in blocks():
            stop=start+matrix.shape[0];x=expression_view(matrix,count_compatible);labels,scores=annotate_sparse(x,target[start:stop],model)
            for name,values in scores.items():labels[name]=values
            state=np.empty((stop-start,len(reference['states'])))
            for code in np.unique(codes[start:stop]):
                local=codes[start:stop]==code;values=np.asarray(x[local]@weights[code]);cov=coverage[code*len(reference['states']):(code+1)*len(reference['states'])]
                for j,item in enumerate(cov):
                    if item['status']!='completed':values[:,j]=np.nan
                state[local]=values
            for j,name in enumerate(reference['states']):labels['state__'+name]=state[:,j]
            if not transform_verified:
                labels['uncalibrated_candidate_type_before_scale_downgrade']=labels.inferred_type
                labels['inferred_type']='unknown';labels['inferred_lineage']='unknown';labels['inferred_subtype']='unknown';labels['target_excluded_type']='unknown'
                labels['inference_status']='not_estimable';labels['inference_reason']='source_expression_transform_not_verified_for_reference_scoring'
            zero=totals[start:stop]<=0
            if zero.any():
                for name in ['inferred_type','inferred_lineage','inferred_subtype','target_excluded_type']:labels.loc[zero,name]='unknown'
                labels.loc[zero,'inference_status']='not_estimable';labels.loc[zero,'inference_reason']='zero_expression_record'
                for name in reference['states']:labels.loc[zero,'state__'+name]=np.nan
            parts.append(labels)
            if start and start%(chunk*100)==0:print(file+' annotated '+str(stop)+'/'+str(n),flush=True)
    combined=pd.concat([qc,pd.concat(parts,ignore_index=True)],axis=1);combined['inference_expression_scale']=scale
    combined['state_background']='pooled_same_source_biological_context_endpoints_not_control';combined['state_reference_derivation']='mouse_markers_and_explicit_one_to_one_orthology' if species=='mouse' else 'human_frozen_reference'
    combined['truth_label']=False;combined['available_before_endpoint']=False
    if combined.record_id.duplicated().any() or combined.probability_correct.notna().any() or len(combined)!=n:raise ValueError('annotation_identity_or_truth_violation')
    combined.to_parquet(output/'cells.parquet',index=False,compression='zstd')
    duplicate=combined.source_barcode.duplicated(keep=False);count_dups=combined.computed_count_sha256.duplicated(keep=False)
    combined.loc[duplicate|count_dups,['record_id','row_index','source_barcode','computed_count_sha256','source_identity_axis_sha256','computed_on_source_identity_axis_sha256']].to_parquet(output/'within-file-duplicate-candidates.parquet',index=False)
    state_summary=[]
    for code,group in combined.groupby('background_index',sort=True):
        for name in reference['states']:state_summary.append({'background_index':code,'state':name,'n_cells':len(group),'finite_scores':int(group['state__'+name].notna().sum()),'distribution':quantiles(group['state__'+name]),'expression_scale':scale})
    write_json(output/'state-distributions.json',state_summary)
    if hash_file(path)!=entry['input_sha256']:raise ValueError('scPerturb_source_changed_during_cell_analysis')
    summary={'file':file,'status':'completed','cells':n,'genes':g,'species':species,'count_compatible':count_compatible,'expression_scale':scale,'source_transform_verified':transform_verified,
        'backgrounds':len(backgrounds),'numeric':dict(numeric),'zero_expression_records':int((totals<=0).sum()),'pooled_types':combined.inferred_type.value_counts().to_dict(),
        'inference_status_counts':combined.inference_status.value_counts().to_dict(),'uncalibrated_records':len(combined),'method_conflicts':int(combined.method_conflict.sum()),
        'duplicate_source_barcode_records':int(duplicate.sum()),'identical_count_candidate_records':int(count_dups.sum()),'input_sha256':entry['input_sha256'],'inputs_unchanged':True,
        'reference_sha256':hash_file(reference_path/'gene_sets.json'),'duration_seconds':time.monotonic()-started}
    report={'schema_version':2,'bundle_id':'scPerturb-cells-'+value_hash([identity,file])[:24],'title':file+' 全记录身份与RNA推断','status':'completed','summary':summary,'identity':identity,
        'methods':{'type':'Conservative rank/expression marker agreement with unknown; species-specific frozen references; target-excluded marker sensitivity',
            'states':'12 fixed RNA proxies; expression-bin backgrounds from all observed endpoints within registered source biological contexts; not NTC or pre-intervention covariates',
            'continuous':'Provided nonnegative X retained without a second normalization/log; unknown final types when transform/reference scale is unverified',
            'identity':'Every source obs/var retained; input+row record IDs; complete ordered-native-axis and, when unique, source-Ensembl-axis count fingerprints; no copies removed'},
        'limitations':['类型/状态均未校准，概率为空，不是ground truth。','原始细胞系、细胞类型、Mixscape、周期等作者标签单独保留，不作为校准真值。',
            '状态背景由全部观察终点构造；不同背景/物种/变换的绝对状态值不在同一校准刻度。','此阶段只完成逐细胞及数值身份；对照资格与全任务响应在独立阶段执行。'],
        'tables':[{'title':'逐来源背景','rows':backgrounds},{'title':'类型参考覆盖','rows':model['coverage']},{'title':'各背景状态分布','rows':state_summary}],
        'artifact_hashes':{str(p.relative_to(output)):hash_file(p) for p in output.rglob('*') if p.is_file()},'completed_at':datetime.now(timezone.utc).isoformat()}
    report['artifacts']=[{'file':name,'sha256':digest} for name,digest in report['artifact_hashes'].items()]
    write_json(output/'report.json',report);(output/'report.html').write_text(render(serial(report)));return summary


def run(manifest,human,mouse,output,workers=2,resume=False):
    started=time.monotonic();entries=json.loads(manifest.read_text());identity={'manifest_sha256':hash_file(manifest),'human_reference_sha256':hash_file(human/'gene_sets.json'),'mouse_reference_sha256':hash_file(mouse/'gene_sets.json'),
        'code':{n:hash_file(Path(__file__).with_name(n)) for n in CODE},'annotation_rules':RULES,'uv_lock_sha256':hash_file(Path(__file__).with_name('uv.lock'))}
    if output.exists():
        if not resume or json.loads((output/'identity.json').read_text())!=identity or (output/'report.json').exists():raise ValueError('scPerturb_run_identity_changed_or_completed')
    else:output.mkdir(parents=True);write_json(output/'identity.json',identity)
    for reference in [human,mouse]:
        for line in (reference/'SHA256SUMS').read_text().splitlines():
            digest,name=line.split('  ',1)
            if hash_file(reference/name)!=digest:raise ValueError('frozen_reference_changed')
    applicable=[r for r in entries if r['analysis']['status']!='not_applicable'];excluded=[r for r in entries if r['analysis']['status']=='not_applicable']
    if len(applicable)!=51 or len(excluded)!=3:raise ValueError('frozen_51_RNA_3_protein_scope_changed')
    results=[];failures=[]
    with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn')) as pool:
        futures={pool.submit(scan_file,r,str(output),str(human),str(mouse),identity):r['file'] for r in applicable}
        for future in as_completed(futures):
            file=futures[future]
            try:results.append(future.result());print('scPerturb cell files '+str(len(results))+'/51 completed '+file,flush=True)
            except Exception as e:failures.append({'file':file,'status':'failed','error':str(e),'type':type(e).__name__});print('scPerturb FAILED '+file+' '+str(e),flush=True)
    if failures:write_json(output/('failures-'+str(time.time_ns())+'.json'),failures);raise RuntimeError('incomplete_scPerturb_RNA_cell_analysis')
    results.sort(key=lambda r:r['file']);report={'status':'completed','phase':'all_51_RNA_record_QC_and_conservative_cell_inference','identity':identity,'files':results,'excluded_protein_files':excluded,
        'actual_RNA_records':sum(r['cells'] for r in results),'count_compatible_RNA_files':sum(r['count_compatible'] for r in results),
        'pooled_types':dict(sum((Counter(r['pooled_types']) for r in results),Counter())),'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'duration_seconds':time.monotonic()-started,'completed_at':datetime.now(timezone.utc).isoformat(),'reproduce':sys.argv}
    write_json(output/'report.json',report);return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['manifest','human','mouse','output']:p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--workers',type=int,default=2);p.add_argument('--resume',action='store_true');a=p.parse_args()
    r=run(a.manifest,a.human,a.mouse,a.output,a.workers,a.resume);print(json.dumps({k:v for k,v in r.items() if k not in ['files','identity','excluded_protein_files']}))
