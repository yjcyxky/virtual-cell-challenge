"""One run: inputs, four-context log2FC fit, H1 validation and result archival."""
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
configure_memory_policy()

import wandb
import yaml
from common import EXPERIMENT, ROOT, digest, event, write_json
from data import prepare
from features import Features, prepare_priors
from training import fit, fit_id, select_rounds
from evaluation import Evaluation


def git(*arguments):
    return subprocess.check_output(['git', *arguments], cwd=ROOT, text=True).strip()


def pipeline_identity():
    paths=['experiments/exp007/src','experiments/exp007/configs','experiments/exp007/pyproject.toml',
           'experiments/exp007/uv.lock','experiments/exp007/reproduce.sh','experiments/exp007/PLAN.md',
           'experiments/exp001-context-pair-xgb/src','scripts/dossier']
    if git('diff','HEAD','--',*paths) or git('ls-files','--others','--exclude-standard','--',*paths):
        raise ValueError('formal_training_requires_committed_code_config_and_lock')
    return {'git_commit':git('rev-parse','HEAD'),'uv_lock_sha256':digest(EXPERIMENT/'uv.lock'),
            'code_tree':{path:git('rev-parse',f'HEAD:{path}') for path in paths},
            'environment':json.loads((EXPERIMENT/'.venv/experiment-environment.json').read_text()),
            'runtime':{p:importlib.metadata.version(p) for p in ['xgboost','cell-eval2','gpudge','numpy','pandas','anndata','wandb']},
            'memory_policy':memory_policy()}


def report(output, result):
    lines=[f'# exp007 结果\n\nRun `{output.name}`，状态：{result["status"]}。',
           f'W&B：{result.get("wandb_url") or "offline"}',
           '目标：逐细胞 CPM 算术均值的 log2 fold change（epsilon=1e-9），加权平方误差。',
           '划分：K562 / RPE1 / HepG2 / Jurkat 训练，H1 唯一留出背景；一次拟合至 512 轮。',
           'H1 Overall 选择检查点，因此 H1 是开发验证，不是独立测试；不做五折、全背景重训或榜单提交。',
           '评分：固定官方六项指标、dispersed 基线、排除自身靶基因；本地原生面板分数不等同 A/B/C 榜单。']
    completed=result.get('validation',{}).get('H1',{})
    lines.append(f'已完成 H1 checkpoint 评分：{len(completed)}/7。')
    if completed:
        lines.append('\n'.join(['| 轮数 | Overall | 靶点数 | 训练中已见 |',
                                '|---:|---:|---:|---:|']+
                    [f'| {i} | {m["score"]:.6f} | {m["targets"]} | {m["seen_targets"]} |'
                     for i,m in sorted(completed.items(),key=lambda item:int(item[0]))]))
    baselines=result.get('baselines',{})
    for name,metric in baselines.items():lines.append(f'对照 {name}：Overall {metric["score"]:.6f}。')
    if 'selection' in result:
        selection=result['selection'];iteration=selection['rounds'];metric=completed[str(iteration)]
        lines.append(f'选定轮数：{iteration}；H1 Overall：{metric["score"]:.6f}。')
        for name,baseline in baselines.items():
            lines.append(f'相对 {name} 的 Overall 差值：{metric["score"]-baseline["score"]:+.6f}。')
        lines.append(f'选定模型：`checkpoints/{fit_id(selection["training_contexts"])}/round-{iteration:04d}.ubj`；预测：`predictions/{fit_id(selection["training_contexts"])}/H1/model-{iteration:04d}/`。')
    if result.get('artifact'):lines.append(f'W&B Artifact：`{result["artifact"]}`。')
    lines.append(f'W&B 日志同步：{result.get("wandb_sync","unknown")}；Artifact 同步：{result.get("artifact_sync","not yet uploaded")}。')
    if 'error' in result:lines.append('失败/阻塞：'+result['error'])
    lines.append('与 exp006：沿用特征、采样、树参数和官方评分；标签改为 log2FC，生成器相应校准非深度加权的逐细胞均值。只比较相同 H1 面板与评分协议，不能把单背景选模分数与五折均值直接比较。')
    lines.append('限制：H1 曾用于项目开发且本次用于选轮数；均值与 NTC 模板不能完整模拟新细胞状态。独立基因 LFC 经总量约束投影后会变化，保存逐靶点生成前后 LFC 误差诊断；低表达/零表达 LFC 可能较大，仍保留可测基因标签，不静默截断。NTC bag 不是新背景，未测基因不作零标签。')
    previous=[]
    for file in sorted((EXPERIMENT/'outputs').glob('*/metrics.json')):
        if file.parent==output:continue
        prior=json.loads(file.read_text())
        previous.append(f'- `{file.parent.name}`：{prior["status"]}，记录 `{file.relative_to(EXPERIMENT)}`。')
    if previous:lines.append('历史 runs：\n\n'+'\n'.join(previous))
    (EXPERIMENT/'REPORT.md').write_text('\n\n'.join(lines)+'\n')


def validate_h1(data,features,evaluation,config,output,tracked,result,persist):
    context=config['validation_context']
    training=[c for c in config['contexts'] if c!=context]
    results={}
    result.setdefault('validation',{}).setdefault(context,{})
    result['status']='h1_validation';persist()
    def validate(booster,iteration):
        metric=evaluation.score(context,training,iteration,booster,features,keep=True)
        results[(fit_id(training),context,iteration)]=metric
        result['validation'][context][str(iteration)]=metric
        persist()
    fit(data,features,training,max(config['checkpoints']),config,output,tracked,validate)
    result.setdefault('baselines',{})
    for name,kind,sources in [('zero_response','zero',[]),('shared_response','shared',training)]:
        result['baselines'][name]=evaluation.score(context,sources,0,None,features,kind=kind,keep=True)
        persist()
    selection=select_rounds(config['contexts'],config,results)
    write_json(output/'cache/selection.json',selection)
    result['selection']=selection;persist()
    tracked.log({'validation/selected_rounds':selection['rounds'],
                 'validation/selected_overall':selection['scores'][selection['rounds']]})
    return selection


def run(args):
    identity=pipeline_identity()
    if args.resume:
        output=EXPERIMENT/'outputs'/args.resume
        frozen=yaml.safe_load((output/'config.yaml').read_text());config=frozen['configuration']
        if {k:v for k,v in frozen['identity'].items() if k!='git_commit'}!={k:v for k,v in identity.items() if k!='git_commit'}:
            raise ValueError('resume_code_or_environment_changed')
        identity=frozen['identity']
    else:
        config=yaml.safe_load(Path(args.config).read_text())
        if config['experiment_id']!='exp007' or config['prediction_target']!='log2fc_cpm_mean':
            raise ValueError('wrong_experiment_configuration')
        if set(config['contexts'])!={'H1','K562','RPE1','HepG2','Jurkat'} or len(config['contexts'])!=5 or config['validation_context']!='H1':
            raise ValueError('H1_holdout_and_four_training_contexts_required')
        if config['lfc_epsilon']!=1e-9:raise ValueError('official_lfc_epsilon_required')
        if config['checkpoints']!=sorted(set(config['checkpoints'])) or min(config['checkpoints'])<1:
            raise ValueError('positive_increasing_checkpoints_required')
        run_id=args.run_id or datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')+'-h1-log2fc-s'+str(config['seed'])
        output=EXPERIMENT/'outputs'/run_id
        if output.exists():raise ValueError('existing_run_requires_resume')
        output.mkdir(parents=True)
        frozen={'run_id':run_id,'experiment_id':'exp007','configuration':config,'identity':identity,
                'source_run':config.get('source_run'),'source_usage':config.get('source_usage'),
                'split':{'unit':'whole biological context','train':[c for c in config['contexts'] if c!='H1'],
                         'validation':['H1'],'selection':'H1 Overall; earliest checkpoint on tie; development validation'},
                'created_at':datetime.now(timezone.utc).isoformat()}
        (output/'config.yaml').write_text(yaml.safe_dump(frozen,allow_unicode=True,sort_keys=True))
    log=(output/'train.log').open('a',buffering=1)
    class Tee:
        def __init__(self,stream):self.stream=stream
        def write(self,text):self.stream.write(text);log.write(text)
        def flush(self):self.stream.flush();log.flush()
    sys.stdout=Tee(sys.stdout);sys.stderr=Tee(sys.stderr)
    os.environ['WANDB_DIR']=str(output/'wandb');(output/'wandb').mkdir(exist_ok=True)
    options=dict(entity='yjcyxky',project='virtual-cell-challenge',group='exp007',id=output.name,
                 name=f'exp007-{output.name}',dir=str(output/'wandb'),config=frozen,resume='allow',allow_val_change=True)
    try:tracked=wandb.init(**options)
    except wandb.errors.Error:tracked=wandb.init(**options,mode='offline')
    result=json.loads((output/'metrics.json').read_text()) if (output/'metrics.json').exists() else {}
    result.update(status='preparing',run_id=output.name,wandb_url=tracked.url,identity=identity,
                  wandb_sync='offline' if tracked.offline else 'online')
    result.pop('error',None)
    stop=threading.Event();start=time.monotonic()
    def persist():
        write_json(output/'metrics.json',result);report(output,result)
    def monitor():
        while not stop.wait(30):
            usage=resource.getrusage(resource.RUSAGE_SELF)
            tracked.log({'resources/elapsed_seconds':time.monotonic()-start,'resources/max_rss_gib':usage.ru_maxrss/(1024**2),
                         'resources/cpu_seconds':usage.ru_utime+usage.ru_stime})
    thread=threading.Thread(target=monitor,daemon=True);thread.start()
    try:
        persist();event('run_start',run_id=output.name,resume=bool(args.resume),git_commit=identity['git_commit'])
        data=prepare(config,output)
        data_reference={'preparation_sha256':digest(data.directory/'complete.json'),'sources':data.metadata['sources']}
        frozen['data_reference']=data_reference
        (output/'config.yaml').write_text(yaml.safe_dump(frozen,allow_unicode=True,sort_keys=True))
        tracked.config.update({'data_reference':data_reference},allow_val_change=True)
        prepare_priors(data,config,output)
        features=Features(data,output/'cache/priors');evaluation=Evaluation(data,config,output,tracked)
        selection=validate_h1(data,features,evaluation,config,output,tracked,result,persist)
        result['status']='uploading_artifacts';result['elapsed_seconds']=time.monotonic()-start;persist()
        artifact=wandb.Artifact(f'exp007-{output.name}',type='model',metadata={'git_commit':identity['git_commit'],
                               'prediction_target':config['prediction_target'],'validation_context':'H1'})
        paths=[output/'config.yaml',output/'metrics.json',output/'cache/selection.json']
        paths+=list((output/'checkpoints').glob('*/*.ubj'))+list((output/'checkpoints').glob('*/latest.pkl'))
        for pattern in ['**/metrics.json','**/scores.csv','**/aggregate.csv','**/target-support.parquet','**/predicted-response.npz','**/generated-response.npz']:
            paths+=list((output/'predictions').glob(pattern))
        for path in paths:artifact.add_file(str(path),name=str(path.relative_to(output)))
        if tracked.offline:
            tracked.log_artifact(artifact);result['artifact_sync']='pending_offline_sync'
        else:
            published=tracked.log_artifact(artifact);published.wait()
            result['artifact']=published.qualified_name;result['artifact_sync']='uploaded'
        result['status']='completed';persist()
        tracked.summary.update({'status':result['status'],'validation/selected_rounds':selection['rounds'],
                                'validation/selected_overall':selection['scores'][selection['rounds']]})
        stop.set();thread.join(timeout=2);tracked.finish(exit_code=0)
    except BaseException as error:
        result.update(status='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',error=f'{type(error).__name__}: {error}')
        persist();traceback.print_exc()
        stop.set();thread.join(timeout=2);tracked.finish(exit_code=1)
        raise
    finally:
        stop.set();thread.join(timeout=2)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',default='configs/default.yaml')
    parser.add_argument('--run-id')
    parser.add_argument('--resume')
    run(parser.parse_args())
