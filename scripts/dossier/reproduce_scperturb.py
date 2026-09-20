#!/usr/bin/env python
"""Rebuild all scPerturb RNA results from frozen references and downloaded inputs.

Run with uv run --project scripts/dossier --locked python. Original scPerturb
files stay in data/raw/scperturb. Verified Replogle/Nadig results from #7/#8 are
required at their original assessment paths for exact-source response reuse.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime,timezone


def commands(a):
    scripts=Path(__file__).resolve().parent;out=a.output
    def command(name,*args):return [sys.executable,str(scripts/name),*[str(x) for x in args]]
    manifest=a.audit/'analysis_manifest.json';adamson=out/'Adamson-original-identities';cells=out/'cells';base=out/'design-base';refined=out/'design-timecourse';design=out/'design';response=out/'response'
    return [
        command('scperturb_adamson.py','--evidence',a.adamson_evidence,'--output',adamson),
        command('scperturb_cells.py','--manifest',manifest,'--human',a.human_reference,'--mouse',a.mouse_reference,'--output',cells,'--workers',a.workers),
        command('prepare_scperturb_design.py','--manifest',manifest,'--adamson',adamson,'--output',base),
        command('refine_scperturb_timecourse.py','--source',base,'--output',refined),
        command('prepare_scperturb_design.py','--manifest',manifest,'--adamson',adamson,'--refine-from',refined,'--refine-files',
            'AdamsonWeissman2016_GSM2406675_10X001.h5ad','AdamsonWeissman2016_GSM2406677_10X005.h5ad','AdamsonWeissman2016_GSM2406681_10X010.h5ad','--output',design),
        command('profile_scperturb_responses.py','--design',design,'--cells',cells,'--output',response,'--workers',a.workers),
        command('compose_scperturb_dossier.py','--cells',cells,'--design',design,'--response',response,'--audit',a.audit,'--adamson',adamson,
            '--human-reference',a.human_reference,'--mouse-reference',a.mouse_reference,'--evidence',a.adamson_evidence,*a.additional_evidence,'--output',out/'dossier')]


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['audit','human-reference','mouse-reference','adamson-evidence','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--additional-evidence',type=Path,nargs='*',default=[]);p.add_argument('--workers',type=int,default=2);p.add_argument('--print-commands',action='store_true');a=p.parse_args()
    jobs=commands(a)
    if a.print_commands:print(json.dumps(jobs,indent=2));sys.exit(0)
    a.output.mkdir(parents=True,exist_ok=False);environment={**os.environ,'OPENBLAS_NUM_THREADS':'1'};records=[]
    for index,command in enumerate(jobs):
        item={'phase':index+1,'command':command,'started_at':datetime.now(timezone.utc).isoformat(),'status':'running'};records.append(item)
        (a.output/'execution.json').write_text(json.dumps(records,indent=2))
        result=subprocess.run(command,env=environment);item.update(exit_code=result.returncode,status='completed' if result.returncode==0 else 'failed',finished_at=datetime.now(timezone.utc).isoformat())
        (a.output/'execution.json').write_text(json.dumps(records,indent=2))
        if result.returncode:sys.exit(result.returncode)
