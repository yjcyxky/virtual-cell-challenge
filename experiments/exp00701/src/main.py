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
from references import attach_reference
from responses import prepare_response_features


def git(*arguments):
    return subprocess.check_output(['git', *arguments], cwd=ROOT, text=True).strip()


def pipeline_identity():
    paths=['experiments/exp00701/src','experiments/exp00701/configs','experiments/exp00701/pyproject.toml',
           'experiments/exp00701/uv.lock','experiments/exp00701/reproduce.sh','experiments/exp00701/PLAN.md',
           'experiments/exp001-context-pair-xgb/src','scripts/dossier']
    if git('diff','HEAD','--',*paths) or git('ls-files','--others','--exclude-standard','--',*paths):
        raise ValueError('formal_training_requires_committed_code_config_and_lock')
    return {'git_commit':git('rev-parse','HEAD'),'uv_lock_sha256':digest(EXPERIMENT/'uv.lock'),
            'code_tree':{path:git('rev-parse',f'HEAD:{path}') for path in paths},
            'environment':json.loads((EXPERIMENT/'.venv/experiment-environment.json').read_text()),
            'runtime':{p:importlib.metadata.version(p) for p in ['xgboost','cell-eval2','gpudge','numpy','pandas','anndata','wandb']},
            'memory_policy':memory_policy()}


def report(output, result):
    reference=json.loads((EXPERIMENT/'configs/source-reference.json').read_text())
    source=Path(reference['source_run'])
    baseline=json.loads((source/'metrics.json').read_text())
    records=[('exp007 baseline',source.name,baseline)]
    for file in sorted((EXPERIMENT/'outputs').glob('*/metrics.json')):
        record=json.loads(file.read_text())
        cfg=yaml.safe_load((file.parent/'config.yaml').read_text())['configuration']
        if cfg.get('task')=='exp004_reevaluation':continue
        records.append((cfg['feature_mode'],file.parent.name,record))
    lines=['# exp00701 结果',
           'H1 唯一开发留出；四个训练背景。沿用 exp007 标签、采样、512 轮、七个检查点和官方六指标。',
           '对照 exp007 使用其已完成的固定历史 run，不重新训练或改动历史结果。每个新增特征组为独立 run。',
           '| 特征组 | Run | 状态 | 选定轮数 | H1 Overall |',
           '|---|---|---|---:|---:|']
    for arm,rid,r in records:
        selected=r.get('selection',{});iteration=selected.get('rounds')
        metric=r.get('validation',{}).get('H1',{}).get(str(iteration),{})
        score=metric.get('score')
        lines.append(f'| {arm} | {rid} | {r["status"]} | {iteration or "—"} | {score if score is not None else "—"} |')
    lines=lines[:3]+['\n'.join(lines[3:])]
    lines+=['','训练结束条件：三个新特征组各完成 512 轮与全部 H1 评估；不是以超过基线为完成条件。',
            'H1 用于选轮数和实验开发，不是独立测试。不进行全五背景训练或线上提交。',
            '每个训练样本的实测响应输入排除其接收 context，H1 永远不进入响应特征。模块坐标只从固定 Reactome 建立。',
            '连续响应使用全部合格来源细胞；DE 证据使用固定至多 400 个来源细胞与全部 NTC。非显著不当作零效应，缺测由支持数表达。']
    for arm,rid,r in records[1:]:
        lines+=['',f'## {arm}: {rid}',f'W&B: {r.get("wandb_url") or "offline"}',
                f'Artifact: {r.get("artifact","not yet uploaded")}; sync={r.get("artifact_sync","pending")}',
                f'已完成 H1 检查点: {len(r.get("validation",{}).get("H1",{}))}/7']
        if 'error' in r:lines.append('失败/阻塞: '+r['error'])
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
    source=Path(json.loads((EXPERIMENT/config['source_reference']).read_text())['source_run'])
    historical=json.loads((source/'metrics.json').read_text())
    result['baselines']=historical['baselines']
    result['baseline_source']={'run':str(source),'metrics_sha256':digest(source/'metrics.json')}
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
        if getattr(args,'reevaluate_exp004',False):
            from reevaluation import configuration
            config=configuration(args.feature_mode)
        else:config=yaml.safe_load(Path(args.config).read_text())
        if config['experiment_id']!='exp00701' or config['prediction_target']!='log2fc_cpm_mean':
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
        frozen={'run_id':run_id,'experiment_id':'exp00701','configuration':config,'identity':identity,
                'source_run':config.get('source_run'),'source_usage':config.get('source_usage'),
                'split':{'unit':'whole biological context','train':[c for c in config['contexts'] if c!='H1'],
                         'validation':['H1'],'selection':'H1 Overall; earliest checkpoint on tie; development validation'},
                'created_at':datetime.now(timezone.utc).isoformat()}
        if config.get('task')=='exp004_reevaluation':
            frozen['split']['selection']='Fixed original best checkpoint; exp004 input/reference NTC halves; no training or reselection'
        (output/'config.yaml').write_text(yaml.safe_dump(frozen,allow_unicode=True,sort_keys=True))
    log=(output/'train.log').open('a',buffering=1)
    class Tee:
        def __init__(self,stream):self.stream=stream
        def write(self,text):self.stream.write(text);log.write(text)
        def flush(self):self.stream.flush();log.flush()
    sys.stdout=Tee(sys.stdout);sys.stderr=Tee(sys.stderr)
    os.environ['WANDB_DIR']=str(output/'wandb');(output/'wandb').mkdir(exist_ok=True)
    options=dict(entity='yjcyxky',project='virtual-cell-challenge',group='exp00701',id=output.name,
                 name=f'exp00701-{output.name}',dir=str(output/'wandb'),config=frozen,resume='allow',allow_val_change=True)
    try:tracked=wandb.init(**options)
    except wandb.errors.Error:tracked=wandb.init(**options,mode='offline')
    result=json.loads((output/'metrics.json').read_text()) if (output/'metrics.json').exists() else {}
    result.update(status='preparing',run_id=output.name,wandb_url=tracked.url,identity=identity,
                  wandb_sync='offline' if tracked.offline else 'online')
    result.pop('error',None)
    stop=threading.Event();start=time.monotonic()
    def persist():
        write_json(output/'metrics.json',result)
        if config.get('task')=='exp004_reevaluation':
            from reevaluation import report as reevaluation_report
            reevaluation_report()
        else:report(output,result)
    def monitor():
        while not stop.wait(30):
            usage=resource.getrusage(resource.RUSAGE_SELF)
            tracked.log({'resources/elapsed_seconds':time.monotonic()-start,'resources/max_rss_gib':usage.ru_maxrss/(1024**2),
                         'resources/cpu_seconds':usage.ru_utime+usage.ru_stime})
    thread=threading.Thread(target=monitor,daemon=True);thread.start()
    try:
        persist();event('run_start',run_id=output.name,resume=bool(args.resume),git_commit=identity['git_commit'])
        if config.get('task')=='exp004_reevaluation':
            from reevaluation import evaluate_run
            selection=evaluate_run(config,output,tracked,result,persist)
        else:
            reference=attach_reference(config,output)
            data=prepare(config,output)
            data_reference={'preparation_sha256':digest(data.directory/'complete.json'),'sources':data.metadata['sources']}
            frozen['data_reference']=data_reference
            (output/'config.yaml').write_text(yaml.safe_dump(frozen,allow_unicode=True,sort_keys=True))
            tracked.config.update({'data_reference':data_reference},allow_val_change=True)
            prepare_priors(data,config,output)
            base_features=Features(data,output/'cache/priors')
            features=prepare_response_features(data,base_features,config,output,reference)
            frozen['response_features']=json.loads((output/'cache/response-features.json').read_text())
            (output/'config.yaml').write_text(yaml.safe_dump(frozen,allow_unicode=True,sort_keys=True))
            tracked.config.update({'response_features':frozen['response_features']},allow_val_change=True)
            evaluation=Evaluation(data,config,output,tracked)
            selection=validate_h1(data,features,evaluation,config,output,tracked,result,persist)
        result['status']='uploading_artifacts';result['elapsed_seconds']=time.monotonic()-start;persist()
        artifact=wandb.Artifact(f'exp00701-{output.name}',type='model',metadata={'git_commit':identity['git_commit'],
                               'prediction_target':config['prediction_target'],'validation_context':'H1'})
        if config.get('task')=='exp004_reevaluation':
            from reevaluation import artifact_files
            paths=artifact_files(output)
        else:
            paths=[output/'config.yaml',output/'metrics.json',output/'cache/selection.json',output/'cache/reference.json',output/'cache/response-features.json']
            paths+=list((output/'cache/response-evidence').glob('*'))
            if (output/'cache/module-basis.npy').exists():paths.append(output/'cache/module-basis.npy')
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
    parser.add_argument('--config',default='configs/continuous.yaml')
    parser.add_argument('--run-id')
    parser.add_argument('--resume')
    parser.add_argument('--reevaluate-exp004',action='store_true')
    parser.add_argument('--feature-mode',choices=['continuous','deg','modules'])
    run(parser.parse_args())
