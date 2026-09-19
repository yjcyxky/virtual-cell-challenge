#!/usr/bin/env python
"""Expose every heterogeneous resource's diagnostics without changing calculations."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import subprocess
import uuid
from compose_crispri_dossier import verified_copy
from rna import hash_file
from profile_responses import write_json
from render import render


def refine(source,output):
    if output.exists():raise ValueError('fresh_presentation_bundle_required')
    report=verified_copy(source,output)
    shutil.copy2(source/'report.json',output/'calculation-report.json')
    rows=report['tables'][0]['rows'];decisions=report['tables'][1:]
    for row in rows:row.setdefault('coverage_status','resolved_identifier_coverage; unresolved_official_identifiers_separate')
    report['tables']=[{'title':'全部资源的用途与覆盖','rows':rows,'columns':['resource','status','semantics','official_axis_covered','official_targets_covered','coverage_status']},
        *[{'title':row['resource']+'：全部指标','rows':[row]} for row in rows],*decisions]
    report['calculation_bundle_id']=report['bundle_id'];report['bundle_id']='priors-dossier-'+uuid.uuid4().hex
    report['presentation']={'created_at':datetime.now(timezone.utc).isoformat(),'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'code_sha256':hash_file(Path(__file__)),'calculation_report_sha256':hash_file(source/'report.json'),
        'reproduce':['scripts/dossier/refine_prior_dossier.py','--source',str(source),'--output',str(output)],
        'change':'Resource-specific columns displayed in separate tables; all calculated values and sidecars unchanged'}
    report['artifacts'].append({'file':'calculation-report.json','sha256':hash_file(output/'calculation-report.json')})
    write_json(output/'report.json',report);(output/'report.html').write_text(render(report))
    (output/'SHA256SUMS').write_text(''.join(f'{hash_file(p)}  {p.relative_to(output)}\n' for p in sorted(output.rglob('*')) if p.is_file() and p.name!='SHA256SUMS'))
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=refine(a.source,a.output);print(r['bundle_id'])
