#!/usr/bin/env python
"""Add verified source protocol and plate6/14 repeat evidence to a frozen dossier."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
import uuid
import h5py
import numpy as np
import pandas as pd
from compose_crispri_dossier import verified_copy
from profile_responses import write_json,serial
from rna import hash_file,quantiles
from response import correlation
from tahoe import parse_compounds
from render import render


def canonical_condition(value):
    components=parse_compounds(value)
    return json.dumps(sorted((r['source_compound'].strip(),r['source_dose_value'],r['source_dose_unit']) for r in components),separators=(',',':'))


def repeat_evidence(source,output):
    conditions=pd.read_parquet(source/'counts/conditions.parquet');conditions=conditions.loc[conditions.plate.isin(['plate6','plate14'])].copy()
    conditions['comparison_condition']=conditions.source_drugname_drugconc.map(canonical_condition)
    keys=['cell_line_id','comparison_condition'];groups={(plate,line,condition):frame for (plate,line,condition),frame in conditions.groupby(['plate',*keys],observed=True)}
    union=sorted(set(zip(conditions.cell_line_id,conditions.comparison_condition)))
    rows=[];baseline_rows=[]
    with h5py.File(source/'counts/condition-profiles.h5','r') as h:
        profile=h['mean_logCP10K'];g=profile.shape[1]
        def mean(frame):
            n=int(frame.eligible_profile_cells.sum()) if frame is not None else 0
            if not n:return None,0
            total=np.zeros(g)
            for row in frame.itertuples():total+=profile[row.condition_index].astype(float)*row.eligible_profile_cells/n
            return total,n
        controls={}
        for (plate,line),frame in conditions.loc[conditions.source_drug=='DMSO_TF'].groupby(['plate','cell_line_id'],observed=True):controls[(plate,line)]=mean(frame)
        for line in sorted(conditions.cell_line_id.unique()):
            a,na=controls.get(('plate6',line),(None,0));b,nb=controls.get(('plate14',line),(None,0));ok=na>=20 and nb>=20
            baseline_rows.append({'cell_line_id':line,'plate6_DMSO_cells':na,'plate14_DMSO_cells':nb,'status':'completed' if ok else 'not_estimable',
                'baseline_gene_correlation':correlation(a,b) if ok else None,'baseline_difference_RMS':float(np.sqrt(np.mean((a-b)**2))) if ok else None})
        for line,key in union:
            f6=groups.get(('plate6',line,key));f14=groups.get(('plate14',line,key));a,na=mean(f6);b,nb=mean(f14)
            ca,nca=controls.get(('plate6',line),(None,0));cb,ncb=controls.get(('plate14',line),(None,0));ok=min(na,nb,nca,ncb)>=20
            vehicle=json.loads(key)[0][0]=='DMSO_TF'
            row={'cell_line_id':line,'complete_drug_dose_unit_key':key,'is_vehicle':vehicle,'plate6_cells':na,'plate14_cells':nb,'plate6_DMSO_cells':nca,'plate14_DMSO_cells':ncb,
                'plate6_sample_ids':f6['sample'].tolist() if f6 is not None else [],'plate14_sample_ids':f14['sample'].tolist() if f14 is not None else [],
                'status':'completed' if ok else 'not_estimable','reason':None if ok else 'requires_20_local_cells_in_each_condition_and_plate_line_vehicle',
                'declared_biological_repeats':2,'source_author_claim_not_independent_verification':True}
            if ok:
                ea=a-ca;eb=b-cb;nonzero=(ea!=0)|(eb!=0)
                row.update(mean_logRNA_correlation=correlation(a,b),mean_logRNA_difference_RMS=float(np.sqrt(np.mean((a-b)**2))),
                    plate6_response_RMS=float(np.sqrt(np.mean(ea**2))) if not vehicle else None,plate14_response_RMS=float(np.sqrt(np.mean(eb**2))) if not vehicle else None,
                    response_correlation=correlation(ea,eb) if not vehicle else None,response_difference_RMS=float(np.sqrt(np.mean((ea-eb)**2))) if not vehicle else None,
                    response_sign_agreement_among_either_nonzero=float(np.mean(np.sign(ea[nonzero])==np.sign(eb[nonzero]))) if nonzero.any() and not vehicle else None,
                    response_role='not_applicable_vehicle_baseline' if vehicle else 'same_drug_dose_line_different_author_declared_culture_replicate')
            rows.append(row)
    frame=pd.DataFrame(rows);frame.to_parquet(output/'plate6-plate14-repeat-comparisons.parquet',index=False);pd.DataFrame(baseline_rows).to_parquet(output/'plate6-plate14-baseline-comparisons.parquet',index=False)
    return rows,baseline_rows


def refine(source,evidence,output):
    if output.exists():raise ValueError('fresh_supplemented_dossier_required')
    protocol=json.loads((evidence/'protocol-evidence.json').read_text())
    if protocol['facts']['reported_biological_replicate_plates']!=['plate6','plate14'] or protocol['facts']['exposure_hours']!=24:raise ValueError('unregistered_source_protocol')
    report=verified_copy(source,output);shutil.copy2(source/'report.json',output/'calculation-report.json');shutil.copytree(evidence,output/'primary-protocol-evidence')
    rows,baselines=repeat_evidence(source,output)
    report['calculation_bundle_id']=report['bundle_id'];report['bundle_id']='tahoe-dossier-'+uuid.uuid4().hex
    report['source_protocol']=protocol;report['exposure']['time_and_media']='Author v3 reports 24h drug exposure of mixed suspension spheroids; precise medium not verified in retrieved sections; no time-series effect'
    report['exposure']['replication']='Author v3 explicitly identifies plate14 as biological replicate of plate6; other plate/sample/sublibrary labels are not biological repeats'
    report['exposure']['cell_line']='Source SNP-based deconvolution of scRNA libraries; retained source labels are not per-cell groundtruth'
    report['exposure']['source_vs_local_holdout']='Source scVI excludes plate14 from training; this dossier explores all local plate14 records and cannot claim it is untouched validation'
    report['limitations']=[x for x in report['limitations'] if not x.startswith('真实培养重复、精确培养历史')]
    report['limitations'].extend(['论文 v3 的索引正文说明 24h 暴露及 plate6/14 生物重复；直接全文请求仍受 403/429 限制，未声称已取得完整 Methods。',
        'plate6/14 配对仅有作者报告的两个培养重复；全基因响应比较区分条件均值与各自 DMSO 差分，不能从相似基线推出相同药物效应。',
        '不同 plate 的状态背景基因可能不同，新增重复比较使用共同原始基因轴上的 logRNA；不直接把跨板状态绝对分数视为同一刻度。'])
    frame=pd.DataFrame(rows);drug=frame.loc[(~frame.is_vehicle)&(frame.status=='completed')]
    report['repeat_summary']={'matched_condition_union':len(rows),'status_counts':frame.status.value_counts().to_dict(),
        'mean_logRNA_correlation':quantiles(drug.mean_logRNA_correlation),'response_correlation':quantiles(drug.response_correlation),
        'response_difference_RMS':quantiles(drug.response_difference_RMS),'interpretation':'All genes on identical source axis; float32 stored condition means weighted by local cells; no biological-replicate significance test'}
    report['tables'].extend([{'title':'作者报告的 plate6/14 全部条件重现性','columns':list(frame.columns),'rows':rows},{'title':'plate6/14 同系 DMSO 基线比较','rows':baselines}])
    report['supplement']={'created_at':datetime.now(timezone.utc).isoformat(),'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'code_sha256':hash_file(Path(__file__)),'calculation_report_sha256':hash_file(source/'report.json'),'protocol_evidence_sha256':hash_file(evidence/'protocol-evidence.json'),
        'reproduce':sys.argv,'change':'Original all-record and all-condition calculations preserved; new source protocol and all matched plate6/14 replicate comparisons appended'}
    report['artifacts']=[{'file':str(p.relative_to(output)),'sha256':hash_file(p)} for p in sorted(output.rglob('*')) if p.is_file() and p.name not in ['SHA256SUMS'] and p not in [output/'report.json',output/'report.html']]
    write_json(output/'report.json',report);(output/'report.html').write_text(render(serial(report)))
    (output/'SHA256SUMS').write_text(''.join(f'{hash_file(p)}  {p.relative_to(output)}\n' for p in sorted(output.rglob('*')) if p.is_file() and p.name!='SHA256SUMS'))
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['source','evidence','output']:p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();r=refine(a.source,a.evidence,a.output);print(json.dumps({'bundle_id':r['bundle_id'],'repeat_summary':r['repeat_summary']}))
