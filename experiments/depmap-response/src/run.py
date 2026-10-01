"""Train one independent registered DepMap response fit and complete its evaluation."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT/'scripts/dossier')]
from research import bind
from vcc_task.run_context import RegisteredRun
from vcc_task.response_experiment import run_response


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--config',type=Path,required=True)
    cfg=json.loads(parser.parse_args().config.read_text())
    for key,value in cfg['environment'].items():
        if os.environ.get(key)!=value:
            raise ValueError(f'Execution environment differs from registration: {key}')
    research=bind(ROOT,cfg['run_id'],cfg,resume=os.environ.get('VCC_RESEARCH_RESUME')=='1')
    job=RegisteredRun(cfg,research)
    try:
        run_response(job)
    except BaseException:
        job.run.finish(exit_code=1); raise


if __name__=='__main__':
    main()
