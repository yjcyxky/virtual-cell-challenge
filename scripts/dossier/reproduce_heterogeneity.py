#!/usr/bin/env python
"""Reproduce the registered read-only assessment in fresh analysis directories."""
import argparse
import os
from pathlib import Path
import subprocess
import sys


def run(source,work,output,workers):
    source=source.resolve();work=work.resolve();output=output.resolve()
    if not (source/'SHA256SUMS').is_file():raise ValueError('missing_frozen_source_bundle')
    if work.exists() or output.exists():raise ValueError('reproduction_requires_fresh_work_and_output_directories')
    work.mkdir(parents=True);code=Path(__file__).resolve().parent;env=os.environ.copy();env['OPENBLAS_NUM_THREADS']='1'
    def phase(name,*args):
        subprocess.run([sys.executable,str(code/name),*map(str,args)],check=True,env=env)
    comparisons=work/'comparisons';endpoint=work/'endpoint';supplements=work/'supplements';baselines=work/'baselines'
    phase('profile_response_heterogeneity.py','--source',source,'--output',comparisons,'--workers',workers)
    phase('profile_endpoint_decomposition.py','--source',source,'--output',endpoint)
    phase('profile_heterogeneity_supplements.py','--source',source,'--comparisons',comparisons,'--output',supplements)
    phase('profile_heterogeneity_baselines.py','--source',source,'--endpoint',endpoint,'--comparisons',comparisons,'--output',baselines)
    phase('compose_heterogeneity_dossier.py','--source',source,'--comparisons',comparisons,'--endpoint',endpoint,'--supplements',supplements,'--baselines',baselines,'--output',output)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['source','work','output']:p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--workers',type=int,default=4);a=p.parse_args();run(a.source,a.work,a.output,a.workers)
