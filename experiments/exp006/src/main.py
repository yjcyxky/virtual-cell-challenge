"""One run: whole-context CV, global checkpoint selection, full refit and submission."""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import threading
import time
import traceback

from runtime import configure_memory_policy, memory_policy
configure_memory_policy()  # Before NumPy/CUDA initialize; also covers direct invocation.

import numpy as np
import wandb
import yaml
from common import EXPERIMENT, ROOT, digest, event, write_json
from data import prepare
from features import Features, prepare_priors
from training import fit, fit_id, select_rounds
from evaluation import Evaluation
from submission import export


def git(*arguments):
    return subprocess.check_output(['git', *arguments], cwd=ROOT, text=True).strip()


def pipeline_identity():
    paths=['experiments/exp006/src','experiments/exp006/configs','experiments/exp006/pyproject.toml',
           'experiments/exp006/uv.lock','experiments/exp006/reproduce.sh','experiments/exp006/PLAN.md',
           'experiments/exp001-context-pair-xgb/src','experiments/exp002-response-transfer-validation/src','scripts/dossier']
    if git('diff','HEAD','--',*paths) or git('ls-files','--others','--exclude-standard','--',*paths):
        raise ValueError('formal_training_requires_committed_code_config_and_lock')
    return {'git_commit':git('rev-parse','HEAD'),'uv_lock_sha256':digest(EXPERIMENT/'uv.lock'),
            'code_tree':{path:git('rev-parse',f'HEAD:{path}') for path in paths},
            'environment':json.loads((EXPERIMENT/'.venv/experiment-environment.json').read_text()),
            'runtime':{p:importlib.metadata.version(p) for p in ['xgboost','cell-eval2','gpudge','numpy','pandas','anndata','wandb']},
            'memory_policy':memory_policy()}


def report(output, result):
    lines=[f'# exp006 结果\n\nRun `{output.name}`，状态：{result["status"]}。',
           f'W&B：{result.get("wandb_url") or "offline"}',
           '协议：五折按背景留一（每折四训一验），按五背景等权均值选一个统一轮数，全五背景从头重训，再提交 leaderboard。',
           '评分基线：官方 dispersed 构造、排除自身靶基因；协议 cell-eval2-dispersed-exclude-target-v1。',
           '本地分数用于模型选择，属于开发验证结果，不是独立测试成绩；本地原生面板分数不等同 A/B/C 榜单。']
    completed=result.get('validation',{})
    lines.append(f'已完成 checkpoint 背景评分：{sum(len(v) for v in completed.values())}/35。')
    if 'selection' in result:
        selection=result['selection']
        lines.append(f'统一选定轮数：{selection["rounds"]}；五背景等权平均 Overall：{selection["mean_scores"][selection["rounds"]]:.6f}。')
        table=['| 留出背景 | 轮数 | Overall | 零响应 | 同靶点转移 | 靶点数 | 训练中已见 |',
               '|---|---:|---:|---:|---:|---:|---:|']
        for context in selection['selection_contexts']:
            metric=completed[context][str(selection['rounds'])]
            baselines=result.get('baselines',{}).get(context,{})
            zero=baselines.get('zero_response',{}).get('score',float('nan'))
            shared=baselines.get('shared_response',{}).get('score',float('nan'))
            table.append(f'| {context} | {selection["rounds"]} | {metric["score"]:.6f} | {zero:.6f} | {shared:.6f} | {metric["targets"]} | {metric["seen_targets"]} |')
        lines.append('\n'.join(table))
        mean=selection['mean_scores'][selection['rounds']]
        baseline_results=result.get('baselines',{})
        if len(baseline_results)==5:
            zero=np.mean([v['zero_response']['score'] for v in baseline_results.values()])
            shared=np.mean([v['shared_response']['score'] for v in baseline_results.values()])
            lines.append(f'开发验证等权均值：模型 {mean:.6f}，零响应 {zero:.6f}，同靶点转移 {shared:.6f}；模型相对两者差值分别为 {mean-zero:+.6f}、{mean-shared:+.6f}。')
    official=result.get('official_export',{}).get('leaderboard')
    if official:
        lines.append(f'Leaderboard：entry `{official.get("entry_id")}`，状态 {official.get("status")}，Overall {official.get("score_avg")}，rank {official.get("rank")}。')
        lines.append('榜单反馈若用于后续改模，也属于开发反馈。此 run 不根据榜单反馈重新选轮数。')
    if result.get('artifact'):
        lines.append(f'W&B Artifact：`{result["artifact"]}`。')
    lines.append(f'W&B 日志同步：{result.get("wandb_sync","unknown")}；Artifact 同步：{result.get("artifact_sync","not yet uploaded")}。')
    if 'error' in result: lines.append('失败/阻塞：'+result['error'])
    lines.append('与 exp005 的差异：模型、数据与固定采样沿用，四背景留一选模替代嵌套三背景训练；评分基线改为官方支持的 dispersed/排除靶基因。旧 tile 基线分数不能用于本协议选模或作为同口径成绩比较。未复用旧权重或 run 缓存。')
    lines.append('限制：仅五个独立背景且曾用于项目开发；NTC bag 不是新背景。均值 residual 加 NTC 模板不能完整恢复扰动后的新状态与分布。未测基因不作零标签。')
    # Each run retains its results; this experiment-level report compares their status.
    reports=[]
    for metrics_file in sorted((EXPERIMENT/'outputs').glob('*/metrics.json')):
        if metrics_file.parent==output: continue
        prior=json.loads(metrics_file.read_text())
        reports.append(f'- `{metrics_file.parent.name}`：{prior["status"]}，来源结果 `{metrics_file.relative_to(EXPERIMENT)}`，W&B {prior.get("wandb_url") or "offline"}。评估有效性：{prior.get("evaluation_validity","见该 run 配置与结果")}。')
    if reports: lines.append('其他历史 runs：\n\n'+'\n'.join(reports))
    (EXPERIMENT/'REPORT.md').write_text('\n\n'.join(lines)+'\n')


def cross_validate(data, features, evaluation, config, output, tracked, result, persist):
    contexts=config['contexts']
    results={}
    result.setdefault('validation',{})
    result.setdefault('baselines',{})
    for context in contexts:
        training=[c for c in contexts if c!=context]
        result['status']='cross_validation'
        result['active_context']=context
        persist()
        def validate(booster, iteration):
            metric=evaluation.score(context,training,iteration,booster,features)
            results[(fit_id(training),context,iteration)]=metric
            result['validation'].setdefault(context,{})[str(iteration)]=metric
            persist()
        fit(data,features,training,max(config['checkpoints']),config,output,tracked,validate)
        result['baselines'][context]={
            'zero_response':evaluation.score(context,[],0,None,features,kind='zero'),
            'shared_response':evaluation.score(context,training,0,None,features,kind='shared')}
        persist()
    selection=select_rounds(contexts,config,results)
    write_json(output/'cache/selection.json',selection)
    result['selection']=selection
    result['final_rounds']=selection['rounds']
    result.pop('active_context',None)
    persist()
    tracked.log({'validation/selected_rounds':selection['rounds'],
                 'validation/selected_mean':selection['mean_scores'][selection['rounds']]})
    return selection['rounds']


def run(args):
    identity=pipeline_identity()
    if args.resume:
        output=EXPERIMENT/'outputs'/args.resume
        frozen=yaml.safe_load((output/'config.yaml').read_text())
        config=frozen['configuration']
        if {k:v for k,v in frozen['identity'].items() if k!='git_commit'}!={k:v for k,v in identity.items() if k!='git_commit'}:
            raise ValueError('resume_code_or_environment_changed')
        identity=frozen['identity']
    else:
        config=yaml.safe_load(Path(args.config).read_text())
        if args.submit: config['submit']=True
        if config['experiment_id']!='exp006': raise ValueError('wrong_experiment_configuration')
        if set(config['contexts'])!={'H1','K562','RPE1','HepG2','Jurkat'} or len(config['contexts'])!=5:
            raise ValueError('five_distinct_training_contexts_required')
        if config['checkpoints']!=sorted(set(config['checkpoints'])) or min(config['checkpoints'])<1:
            raise ValueError('positive_increasing_checkpoints_required')
        run_id=args.run_id or datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')+'-loco-s'+str(config['seed'])
        output=EXPERIMENT/'outputs'/run_id
        if output.exists(): raise ValueError('existing_run_requires_resume')
        output.mkdir(parents=True)
        frozen={'run_id':run_id,'experiment_id':'exp006','configuration':config,'identity':identity,
                'source_run':config.get('source_run'),'source_usage':config.get('source_usage'),
                'split':{'unit':'whole biological context','folds':{
                    c:{'train':[t for t in config['contexts'] if t!=c],'validation':[c]} for c in config['contexts']},
                    'selection':'equal-context mean Overall; earliest checkpoint on tie; development validation'},
                'created_at':datetime.now(timezone.utc).isoformat()}
        (output/'config.yaml').write_text(yaml.safe_dump(frozen,allow_unicode=True,sort_keys=True))
    if args.submit and not config.get('submit'):
        # Publishing previously prepared predictions does not change the model conditions.
        config['submit']=True
        (output/'config.yaml').write_text(yaml.safe_dump(frozen,allow_unicode=True,sort_keys=True))
    log=(output/'train.log').open('a',buffering=1)
    class Tee:
        def __init__(self,stream): self.stream=stream
        def write(self,text): self.stream.write(text); log.write(text)
        def flush(self): self.stream.flush(); log.flush()
    sys.stdout=Tee(sys.stdout); sys.stderr=Tee(sys.stderr)
    os.environ['WANDB_DIR']=str(output/'wandb')
    (output/'wandb').mkdir(exist_ok=True)
    options=dict(entity='yjcyxky',project='virtual-cell-challenge',group='exp006',id=output.name,
                 name=f'exp006-{output.name}',dir=str(output/'wandb'),config=frozen,resume='allow',allow_val_change=True)
    try: tracked=wandb.init(**options)
    except wandb.errors.Error: tracked=wandb.init(**options,mode='offline')
    result={'status':'preparing','run_id':output.name,'wandb_url':tracked.url,'identity':identity,
            'wandb_sync':'offline' if tracked.offline else 'online'}
    stop=threading.Event(); start=time.monotonic()
    def persist():
        write_json(output/'metrics.json',result)
        report(output,result)
    def monitor():
        while not stop.wait(30):
            usage=resource.getrusage(resource.RUSAGE_SELF)
            tracked.log({'resources/elapsed_seconds':time.monotonic()-start,'resources/max_rss_gib':usage.ru_maxrss/(1024**2),
                         'resources/cpu_seconds':usage.ru_utime+usage.ru_stime})
    thread=threading.Thread(target=monitor,daemon=True); thread.start()
    try:
        persist()
        event('run_start',run_id=output.name,resume=bool(args.resume),git_commit=identity['git_commit'])
        data=prepare(config,output)
        data_reference={'preparation_sha256':digest(data.directory/'complete.json'),'sources':data.metadata['sources']}
        frozen['data_reference']=data_reference
        (output/'config.yaml').write_text(yaml.safe_dump(frozen,allow_unicode=True,sort_keys=True))
        tracked.config.update({'data_reference':data_reference},allow_val_change=True)
        prepare_priors(data,config,output)
        features=Features(data,output/'cache/priors'); evaluation=Evaluation(data,config,output,tracked)
        final_rounds=cross_validate(data,features,evaluation,config,output,tracked,result,persist)
        result['status']='final_training'; persist()
        booster=fit(data,features,config['contexts'],final_rounds,config,output,tracked,lambda model,iteration:None)
        result['status']='export_and_submission'; persist()
        result['official_export']=export(data,features,booster,config,output,tracked,submit=config.get('submit',False))
        result['status']='uploading_artifacts'; result['elapsed_seconds']=time.monotonic()-start
        persist()
        artifact=wandb.Artifact(f'exp006-{output.name}',type='model',metadata={'git_commit':identity['git_commit']})
        paths=[output/'config.yaml',output/'metrics.json',output/'cache/selection.json']
        paths+=list((output/'checkpoints').glob('*/*.ubj'))+list((output/'checkpoints').glob('*/latest.pkl'))
        for pattern in ['**/metrics.json','**/scores.csv','**/aggregate.csv','**/target-support.parquet','**/predicted-response.npz']:
            paths+=list((output/'predictions').glob(pattern))
        paths+=list((output/'predictions/official').glob('*.json'))+list((output/'predictions/official').glob('*-response.npz'))
        for path in paths: artifact.add_file(str(path),name=str(path.relative_to(output)))
        if tracked.offline:
            tracked.log_artifact(artifact); result['artifact_sync']='pending_offline_sync'
        else:
            published=tracked.log_artifact(artifact); published.wait()
            result['artifact']=published.qualified_name; result['artifact_sync']='uploaded'
        result['status']='completed' if config.get('submit') else 'prepared'
        persist()
        tracked.summary.update({'status':result['status'],'validation/selected_rounds':final_rounds,
                                'validation/selected_mean':result['selection']['mean_scores'][final_rounds]})
        stop.set(); thread.join(timeout=2); tracked.finish(exit_code=0)
    except BaseException as error:
        result.update(status='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',error=f'{type(error).__name__}: {error}')
        persist(); traceback.print_exc()
        stop.set(); thread.join(timeout=2); tracked.finish(exit_code=1)
        raise
    finally:
        stop.set(); thread.join(timeout=2)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',default='configs/default.yaml')
    parser.add_argument('--run-id')
    parser.add_argument('--resume')
    parser.add_argument('--submit',action='store_true',help='Publish official predictions; default config already enables this.')
    run(parser.parse_args())
