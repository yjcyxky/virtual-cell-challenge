"""One run: validation, preprocessing, nested fits, official evaluation and export."""
import argparse
from datetime import datetime,timezone
import importlib.metadata
import itertools
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import threading
import time
import traceback
import numpy as np
import wandb
import yaml
from common import EXPERIMENT,ROOT,digest,write_json
from data import prepare
from features import Features,prepare_priors
from training import fit,fit_id,select_rounds
from evaluation import Evaluation
from submission import export

def git(*arguments):
    return subprocess.check_output(['git',*arguments],cwd=ROOT,text=True).strip()

def pipeline_identity():
    paths=['experiments/exp005/src','experiments/exp005/configs','experiments/exp005/pyproject.toml',
           'experiments/exp005/uv.lock','experiments/exp005/reproduce.sh','experiments/exp005/PLAN.md',
           'experiments/exp001-context-pair-xgb/src','experiments/exp002-response-transfer-validation/src','scripts/dossier']
    if git('diff','HEAD','--',*paths) or git('ls-files','--others','--exclude-standard','--',*paths):
        raise ValueError('formal_training_requires_committed_code_config_and_lock')
    return {'git_commit':git('rev-parse','HEAD'),'uv_lock_sha256':digest(EXPERIMENT/'uv.lock'),
            'code_tree':{path:git('rev-parse',f'HEAD:{path}') for path in paths},
            'environment':json.loads((EXPERIMENT/'.venv/experiment-environment.json').read_text()),
            'runtime':{p:importlib.metadata.version(p) for p in ['xgboost','cell-eval2','gpudge','numpy','pandas','anndata','wandb']}}

def report(output,result):
    path=EXPERIMENT/'REPORT.md'
    header=f'# exp005 结果\n\nRun `{output.name}`，状态：{result["status"]}。\n\nW&B：{result.get("wandb_url","offline")}\n\n'
    if 'outer' in result:
        header+='| 留出 context | 训练轮数 | 官方方法 Overall | 零响应 | 同靶点转移 | 靶点数 | 训练中已见 |\n|---|---:|---:|---:|---:|---:|---:|\n'
        for context,metrics in result['outer'].items():
            baselines=result.get('baselines',{}).get(context,{})
            zero=baselines.get('zero_response',{}).get('score',float('nan'))
            shared=baselines.get('shared_response',{}).get('score',float('nan'))
            header+=f'| {context} | {metrics["round"]} | {metrics["score"]:.6f} | {zero:.6f} | {shared:.6f} | {metrics["targets"]} | {metrics["seen_targets"]} |\n'
        header+='\n本地 context-native 面板分数，不等同 A/B/C 榜单。全部外层分数只报告，不用于轮数选择。\n'
    if 'error' in result:header+='\n阻塞/失败：'+result['error']+'\n'
    header+='\n限制：独立细胞背景仅五个且曾用于项目开发；NTC bag 不构成新背景。均值 residual 加 NTC 模板无法完整恢复扰动导致的新状态和分布。原始覆盖不等于合格监督覆盖；未测基因不作零标签。\n'
    path.write_text(header)

def run(args):
    if args.resume:
        output=EXPERIMENT/'outputs'/args.resume
        frozen=yaml.safe_load((output/'config.yaml').read_text());config=frozen['configuration']
        identity=pipeline_identity()
        if {k:v for k,v in frozen['identity'].items() if k!='git_commit'}!={k:v for k,v in identity.items() if k!='git_commit'}:
            raise ValueError('resume_code_or_environment_changed')
        identity=frozen['identity']
    else:
        config=yaml.safe_load(Path(args.config).read_text())
        run_id=args.run_id or datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')+'-residual-s'+str(config['seed'])
        output=EXPERIMENT/'outputs'/run_id
        if output.exists():raise ValueError('existing_run_requires_resume')
        identity=pipeline_identity();output.mkdir(parents=True)
        frozen={'run_id':run_id,'experiment_id':'exp005','configuration':config,'identity':identity,
                'source_run':None,'created_at':datetime.now(timezone.utc).isoformat()}
        (output/'config.yaml').write_text(yaml.safe_dump(frozen,allow_unicode=True,sort_keys=True))
    # One append-only log follows this run, including retries and resumed stages.
    log=(output/'train.log').open('a',buffering=1)
    class Tee:
        def __init__(self,stream):self.stream=stream
        def write(self,text):self.stream.write(text);log.write(text)
        def flush(self):self.stream.flush();log.flush()
    sys.stdout=Tee(sys.stdout);sys.stderr=Tee(sys.stderr)
    os.environ['WANDB_DIR']=str(output)
    options=dict(entity='yjcyxky',project='virtual-cell-challenge',group='exp005',id=output.name,
                 name=f'exp005-{output.name}',dir=str(output),config=frozen,resume='allow')
    try:tracked=wandb.init(**options)
    except wandb.errors.Error:
        tracked=wandb.init(**options,mode='offline')
    result={'status':'running','run_id':output.name,'wandb_url':tracked.url,'identity':identity,
            'wandb_sync':'offline' if tracked.offline else 'online'}
    write_json(output/'metrics.json',result);report(output,result)
    stop=threading.Event();start=time.monotonic()
    def monitor():
        while not stop.wait(30):
            usage=resource.getrusage(resource.RUSAGE_SELF)
            tracked.log({'resources/elapsed_seconds':time.monotonic()-start,'resources/max_rss_gib':usage.ru_maxrss/(1024**2),
                         'resources/cpu_seconds':usage.ru_utime+usage.ru_stime})
    thread=threading.Thread(target=monitor,daemon=True);thread.start()
    try:
        data=prepare(config,output)
        # Actual data identities join the same run config and W&B identity.
        data_reference={'preparation_sha256':digest(data.directory/'complete.json'),'sources':data.metadata['sources']}
        frozen['data_reference']=data_reference
        (output/'config.yaml').write_text(yaml.safe_dump(frozen,allow_unicode=True,sort_keys=True))
        tracked.config.update({'data_reference':data_reference},allow_val_change=True)
        prepare_priors(data,config,output)
        features=Features(data,output/'cache/priors');evaluation=Evaluation(data,config,output,tracked)
        contexts=config['contexts'];results={}
        for training in itertools.combinations(contexts,3):
            held=[c for c in contexts if c not in training]
            def inner(booster,iteration):
                for context in held:
                    results[(fit_id(training),context,iteration)]=evaluation.score(context,training,iteration,booster,features)
            fit(data,features,training,max(config['checkpoints']),config,output,tracked,inner)
        selections=select_rounds(contexts,config,results)
        write_json(output/'cache/inner-selection.json',selections)
        result['outer']={};result['baselines']={};result['inner_selection']=selections
        for context in contexts:
            training=[c for c in contexts if c!=context];limit=selections[context]['rounds']
            def outer(booster,iteration):
                metrics=evaluation.score(context,training,iteration,booster,features,keep=iteration==limit)
                if iteration==limit:result['outer'][context]=metrics
            fit(data,features,training,limit,config,output,tracked,outer)
            zero=evaluation.score(context,[],0,None,features,kind='zero')
            shared=evaluation.score(context,training,0,None,features,kind='shared')
            result['baselines'][context]={'zero_response':zero,'shared_response':shared}
            write_json(output/'metrics.json',result);report(output,result)
        final_rounds=int(np.median([v['rounds'] for v in selections.values()]))
        booster=fit(data,features,contexts,final_rounds,config,output,tracked,lambda model,iteration:None)
        result['final_rounds']=final_rounds
        result['official_export']=export(data,features,booster,config,output,tracked,submit=args.submit)
        result['status']='uploading_artifacts';result['elapsed_seconds']=time.monotonic()-start
        write_json(output/'metrics.json',result)
        artifact=wandb.Artifact(f'exp005-{output.name}',type='model',metadata={'git_commit':identity['git_commit'],'run_id':output.name})
        for file in [output/'config.yaml',output/'metrics.json',output/'cache/inner-selection.json']+list((output/'checkpoints').glob('*/round-*.ubj')):
            artifact.add_file(str(file),name=str(file.relative_to(output)))
        artifact.add_file(str(output/'predictions/official/export.json'),name='official-export.json')
        for file in (output/'predictions/official').glob('*-response.npz'):
            artifact.add_file(str(file),name='official/'+file.name)
        if tracked.offline:tracked.log_artifact(artifact);result['artifact_sync']='pending_offline_sync'
        else:
            published=tracked.log_artifact(artifact);published.wait();result['artifact']=published.qualified_name
        result['status']='completed'
        write_json(output/'metrics.json',result);report(output,result)
        tracked.summary.update({'status':'completed','outer_context_mean':float(np.mean([m['score'] for m in result['outer'].values()]))})
        stop.set();thread.join(timeout=2);tracked.finish(exit_code=0)
    except BaseException as error:
        result.update(status='failed',error=f'{type(error).__name__}: {error}')
        write_json(output/'metrics.json',result);report(output,result)
        traceback.print_exc();stop.set();thread.join(timeout=2);tracked.finish(exit_code=1)
        raise
    finally:stop.set();thread.join(timeout=2)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',default='configs/default.yaml')
    parser.add_argument('--run-id')
    parser.add_argument('--resume')
    parser.add_argument('--submit',action='store_true')
    run(parser.parse_args())
