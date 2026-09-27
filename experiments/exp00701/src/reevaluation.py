"""Fixed-checkpoint H1 reevaluation against exp004's immutable reference protocol."""
from dataclasses import replace
import gc
import json
from pathlib import Path
import shutil

import anndata as ad
import numpy as np
import pandas as pd
import xgboost as xgb
import yaml
from cell_eval2 import EvalConfig
from cell_eval2.real_bundle import read_real_bundle

from common import EXPERIMENT,digest,event,log_profile,seed,write_json
from data import Data,GeneIdentity,NTC
from evaluation import read_count_matrix,score_prediction,write_count_matrix
from features import Features
from generation import generate_counts
from responses import AugmentedFeatures,ResponseEvidence,TRAINING_CONTEXTS
from training import predict

PROTOCOL='exp004-h1-fixed-reference-separated-ntc-tiled-v1'
SPEC='configs/exp004-reference.json'


def configuration(mode):
    if mode not in ('continuous','deg','modules'):raise ValueError('feature_mode_required')
    spec=json.loads((EXPERIMENT/SPEC).read_text());source=spec['models'][mode]
    original=yaml.safe_load((Path(source['run'])/'config.yaml').read_text())['configuration']
    return {**original,'task':'exp004_reevaluation','evaluation_protocol':PROTOCOL,
            'source_run':source['run'],'source_usage':'Fixed original best checkpoint; inference and evaluation only; no fitting or reselection.',
            'source_experiment':'exp00701','source_code_commit':source['training_git_commit'],
            'reevaluation_reference':SPEC,'reevaluation_reference_sha256':digest(EXPERIMENT/SPEC),
            'reference_cells_per_target':128,'cells_per_prediction':128,
            'prediction_allocation':'Exact exp004 reference target/batch sizes; at most 128 per perturbation',
            'fixed_round':source['round'],'prediction_seeds':spec['prediction_seeds']}


def check_control_partition(inputs,reference):
    """Check raw identities, not merely rows in independently concatenated caches."""
    if not inputs.is_NTC.all() or not reference.is_NTC.all():raise ValueError('non_control_in_partition')
    if not inputs.half.eq(0).all() or not reference.half.eq(1).all():raise ValueError('wrong_control_half')
    for keys in [('input_sha256','row_index'),('source_batch','source_barcode')]:
        left=set(inputs[list(keys)].itertuples(index=False,name=None))
        right=set(reference[list(keys)].itertuples(index=False,name=None))
        if len(left)!=len(inputs) or len(right)!=len(reference):raise ValueError('duplicate_control_identity')
        if left&right:raise ValueError('input_reference_control_overlap')


def aligned_genes(genes,lookup,canonical):
    mapped=[canonical(g) for g in genes]
    if any(g not in lookup for g in mapped):raise ValueError('legacy_gene_unmapped')
    if len(set(mapped))!=len(mapped):raise ValueError('legacy_gene_collision')
    return np.array([lookup[g] for g in mapped]),mapped


def input_statistics(counts):
    """Existing model feature definitions, recomputed using only input-half counts."""
    total=np.zeros(counts.shape[1]);detected=total.copy();squares=total.copy();cpm=total.copy()
    for start in range(0,len(counts),512):
        block=np.asarray(counts[start:start+512],np.float64)
        depth=block.sum(1)
        if np.any(depth<=0):raise ValueError('empty_input_control')
        normalized=block/depth[:,None]
        total+=block.sum(0);detected+=(block>0).sum(0)
        squares+=np.square(normalized*50000).sum(0)
        cpm+=normalized.sum(0)*1e6
    features=np.stack([log_profile(total),detected/len(counts),np.log1p(squares/len(counts))]).astype(np.float32)
    return features[None],cpm/len(counts)


def prepare_inputs(spec,data,output):
    legacy=Path(spec['legacy_data']);reference=Path(spec['legacy_reference'])
    layout=json.loads((reference/'reference.json').read_text())['identity']
    genes=layout['genes'];obs=pd.read_parquet(reference/'reference-observations.parquet')
    old_genes=json.loads((legacy/'genes.json').read_text());old_lookup={g:i for i,g in enumerate(old_genes)}
    frames=[pd.read_parquet(legacy/f'panel-{i}/cells.parquet') for i in (0,1,2)]
    if any(not frame.context.eq('H1').all() for frame in frames):raise ValueError('legacy_H1_prefix_changed')
    cells=pd.concat(frames,ignore_index=True)
    reference_cells=cells.iloc[obs.source_cached_row.to_numpy()]
    expected_targets=np.where(reference_cells.is_NTC,NTC,reference_cells.target)
    if not np.array_equal(expected_targets,obs.target.to_numpy()):raise ValueError('legacy_reference_cell_order_changed')
    inputs=cells.loc[cells.is_NTC&cells.half.eq(0)].copy()
    reference_ntc=reference_cells.loc[reference_cells.is_NTC]
    check_control_partition(inputs,reference_ntc)
    if len(inputs)!=3072 or len(reference_ntc)!=3072 or len(obs)!=40756 or len(genes)!=18005:
        raise ValueError('legacy_H1_population_changed')
    if not inputs.panel_index.eq(0).all():raise ValueError('unexpected_input_control_panel')
    panel_axis=np.load(legacy/'panel-0/genes.npy');column_lookup={int(g):i for i,g in enumerate(panel_axis)}
    columns=np.array([column_lookup[old_lookup[g]] for g in genes])
    panel_counts=np.load(legacy/'panel-0/counts.npy',mmap_mode='r')
    counts=np.asarray(panel_counts[np.ix_(inputs.cache_row.to_numpy(),columns)])
    identity=GeneIdentity();axis,mapped=aligned_genes(genes,data.lookup,identity.canonical)
    targets=sorted(set(obs.target)-{NTC});model_targets=[identity.canonical(p) for p in targets]
    if len(set(model_targets))!=len(targets) or any(p not in data.lookup for p in model_targets):
        raise ValueError('legacy_target_identity_changed')
    features,baseline=input_statistics(counts)
    # Do not load H1 training statistics or responses at all in this inference path.
    data.contexts['H1']={'gene_indices':axis,'features':features,'baseline_cpm':baseline,'targets':np.array(model_targets)}
    cache=output/'cache';cache.mkdir(exist_ok=True)
    np.save(cache/'input-counts.npy',counts);inputs.to_parquet(cache/'input-cells.parquet',index=False)
    np.savez(cache/'input-statistics.npz',features=features,baseline_cpm=baseline)
    pd.DataFrame({'scoring_gene':genes,'model_gene':mapped}).to_csv(cache/'gene-alignment.csv',index=False)
    pd.DataFrame({'scoring_target':targets,'model_target':model_targets}).to_csv(cache/'target-alignment.csv',index=False)
    stamp={'input_controls':len(inputs),'reference_controls':len(reference_ntc),'control_identity_overlap':0,
           'identity_keys':[['input_sha256','row_index'],['source_batch','source_barcode']],
           'reference_cells':len(obs),'genes':len(genes),'targets':len(targets),
           'input_counts_sha256':digest(cache/'input-counts.npy'),'input_cells_sha256':digest(cache/'input-cells.parquet'),
           'input_statistics_sha256':digest(cache/'input-statistics.npz'),
           'gene_alignment_sha256':digest(cache/'gene-alignment.csv'),'target_alignment_sha256':digest(cache/'target-alignment.csv')}
    write_json(cache/'input-identity.json',stamp);event('isolated_inputs_ready',**stamp)
    return counts,inputs,obs,genes,targets,model_targets


def generated_blocks(counts,inputs,observations,targets,response,baseline,prediction_seed,epsilon):
    """Match exp004 target/batch cell counts; templates come only from half=0."""
    pools={b:np.flatnonzero(inputs.batch.to_numpy()==b) for b in inputs.batch.unique()}
    response_rows={p:i for i,p in enumerate(targets)}
    next_row=0
    for target,positions in observations.groupby('target',sort=False).indices.items():
        positions=np.asarray(positions)
        if not np.array_equal(positions,np.arange(next_row,next_row+len(positions))):raise ValueError('noncontiguous_reference_targets')
        batches=observations.iloc[positions].batch.to_numpy();chosen=np.empty(len(positions),np.int64)
        rng=np.random.default_rng(seed(prediction_seed,'H1',target,'exp004-templates'))
        for batch in sorted(set(batches)):
            if batch not in pools:raise ValueError('reference_batch_has_no_input_controls')
            where=np.flatnonzero(batches==batch);pool=pools[batch]
            chosen[where]=rng.choice(pool,len(where),replace=len(where)>len(pool))
        templates=counts[chosen]
        if target==NTC:generated=templates.astype(np.uint32)
        else:
            rounding=np.random.default_rng(seed(prediction_seed,'H1',target,'rounding'))
            generated=generate_counts(templates,baseline,response[response_rows[target]],rounding,epsilon)
        yield generated
        next_row+=len(positions)
    if next_row!=len(observations):raise ValueError('prediction_row_count_mismatch')


def evaluate_run(config,output,tracked,result,persist):
    spec_path=EXPERIMENT/config['reevaluation_reference']
    if digest(spec_path)!=config['reevaluation_reference_sha256']:raise ValueError('reevaluation_spec_changed')
    spec=json.loads(spec_path.read_text())
    for path,expected in spec['files'].items():
        file=Path(path)
        if file.stat().st_size!=expected['bytes'] or digest(file)!=expected['sha256']:raise ValueError(f'fixed_reevaluation_source_changed:{file}')
    source=Path(config['source_run']);model_spec=spec['models'][config['feature_mode']]
    if str(source)!=model_spec['run'] or config['fixed_round']!=model_spec['round']:raise ValueError('fixed_model_changed')
    write_json(output/'cache/fixed-sources.json',spec)
    data=Data(Path(spec['model_data']),contexts=TRAINING_CONTEXTS)
    counts,inputs,obs,genes,targets,model_targets=prepare_inputs(spec,data,output)
    base=Features(data,Path(spec['model_priors']));de=None;basis=None
    if config['feature_mode']!='continuous':
        de={}
        for context in TRAINING_CONTEXTS:
            with np.load(source/f'cache/response-evidence/{context}.npz') as record:de[context]={k:record[k] for k in record.files}
    if config['feature_mode']=='modules':basis=np.load(source/'cache/module-basis.npy')
    features=AugmentedFeatures(base,ResponseEvidence(data,config['feature_mode'],de,basis))
    checkpoint=Path(model_spec['checkpoint']);destination=output/'checkpoints'/checkpoint.name
    destination.parent.mkdir(exist_ok=True)
    if not destination.exists():shutil.copy2(checkpoint,destination)
    if digest(destination)!=spec['files'][str(checkpoint)]['sha256']:raise ValueError('copied_checkpoint_changed')
    booster=xgb.Booster();booster.load_model(destination)
    if booster.num_boosted_rounds()!=config['fixed_round']:raise ValueError('wrong_checkpoint_round')
    response=predict(booster,features,'H1',model_targets,data.contexts['H1']['gene_indices'])
    directory=output/'predictions';directory.mkdir(exist_ok=True)
    np.savez_compressed(directory/'predicted-response.npz',response=response,targets=np.array(targets),genes=np.array(genes))
    reference=Path(spec['legacy_reference']);private=output/'cache/official';private.mkdir(exist_ok=True)
    for name in ['reference-bundle','real-metric-cache']:
        if not (private/name).exists():shutil.copytree(reference/name,private/name)
    cfg=replace(EvalConfig.from_yaml(str(reference/'eval-config.yaml')),cache_real=str(private/'real-metric-cache'))
    cfg.to_yaml(str(private/'eval-config.yaml'))
    bundle=private/'reference-bundle'
    if read_real_bundle(bundle).manifest['rule_digest']!=spec['rule_digest']:raise ValueError('legacy_rule_changed')
    real=ad.AnnData(np.load(reference/'reference-counts.npy',mmap_mode='r'),obs=obs.copy(),var=pd.DataFrame(index=genes))
    write_json(output/'cache/reference.json',{'protocol':PROTOCOL,'spec_sha256':digest(spec_path),
               'legacy_reference':str(reference),'source_model':model_spec,'model_inference_only':True})
    result['status']='exp004_evaluation';result.setdefault('seed_scores',{});persist()
    for prediction_seed in config['prediction_seeds']:
        pred_dir=directory/f'seed-{prediction_seed}';done=pred_dir/'metrics.json'
        if done.exists():metric=json.loads(done.read_text())
        else:
            event('exp004_prediction',mode=config['feature_mode'],seed=prediction_seed,round=config['fixed_round'])
            matrix_path=pred_dir/'predicted-counts'
            if (matrix_path/'matrix.json').exists():
                metadata=json.loads((matrix_path/'matrix.json').read_text())
                if any(digest(matrix_path/name)!=value for name,value in metadata['files'].items()):
                    raise ValueError('cached_prediction_changed')
                matrix=read_count_matrix(matrix_path)
            else:
                blocks=generated_blocks(counts,inputs,obs,targets,response,data.contexts['H1']['baseline_cpm'],prediction_seed,config['lfc_epsilon'])
                matrix=write_count_matrix(matrix_path,blocks,real.shape,np.uint32)
            predicted_obs=obs.drop(columns='source_cached_row').copy();predicted_obs.index=[f'generated-{prediction_seed}-{i}' for i in range(len(obs))]
            prediction=ad.AnnData(matrix,obs=predicted_obs,var=real.var.copy())
            event('exp004_score_start',mode=config['feature_mode'],seed=prediction_seed)
            metric=score_prediction(prediction,real,cfg,bundle,pred_dir)
            metric.update(seed=prediction_seed,protocol=PROTOCOL,round=config['fixed_round'])
            write_json(done,metric);del prediction,matrix;gc.collect()
        if metric.get('protocol')!=PROTOCOL:raise ValueError('cached_reevaluation_protocol_changed')
        result['seed_scores'][str(prediction_seed)]=metric
        tracked.log({'exp004/seed':prediction_seed,'exp004/overall':metric['score'],**{f'exp004/{k}':v for k,v in metric['components'].items()}})
        event('exp004_score',mode=config['feature_mode'],**metric);persist()
    scores=list(result['seed_scores'].values())
    result['reevaluation']={'protocol':PROTOCOL,'round':config['fixed_round'],
        'mean_score':float(np.mean([s['score'] for s in scores])),'score_sd':float(np.std([s['score'] for s in scores],ddof=1)),
        'components':{k:float(np.mean([s['components'][k] for s in scores])) for k in scores[0]['components']},
        'selection':'Original best fixed before reevaluation; no new checkpoint selection',
        'scope':'exp004 local H1 reference; not leaderboard-equivalent'}
    selection={'rounds':config['fixed_round'],'scores':{config['fixed_round']:result['reevaluation']['mean_score']},'selection':result['reevaluation']['selection']}
    result['selection']=selection
    write_json(output/'cache/selection.json',selection);persist()
    return selection


def artifact_files(output):
    paths=[output/'config.yaml',output/'metrics.json',output/'cache/reference.json',output/'cache/selection.json',
           output/'cache/input-identity.json',output/'cache/gene-alignment.csv',output/'cache/target-alignment.csv',
           output/'cache/fixed-sources.json',output/'cache/official/eval-config.yaml']
    paths+=list((output/'cache/official/reference-bundle').glob('*'))
    paths+=list((output/'checkpoints').glob('*.ubj'))+list((output/'predictions').glob('*.npz'))
    for name in ['metrics.json','raw-metrics.parquet','aggregate.csv','scores.csv','run-meta.json']:
        paths+=list((output/'predictions').glob(f'seed-*/{name}'))
    paths+=list((output/'predictions').glob('seed-*/predicted-counts/matrix.json'))
    return paths


def report():
    marker='<!-- exp004-reevaluation -->';path=EXPERIMENT/'REPORT.md'
    existing=path.read_text().split(marker)[0].rstrip() if path.exists() else '# exp00701'
    lines=[marker,'## 固定最优模型按 exp004 重评',
           '用户要求保留三个原最优检查点（均 128 轮），仅重新推理和评估；不训练、不重新选轮数。新输入/评估条件各建立独立 run，并固定引用原训练 run。',
           '直接使用 exp004 原 H1 的 40,756 个参考细胞、18,005 基因、297 靶点和 baseline/replicate bundle。输入 NTC 是原 half=0 的 3,072 个；评分 NTC 是原 half=1 的另外 3,072 个，两种原始身份键均验证交集为零。',
           '重算输入半池的 NTC 特征与 CPM 基线，原四背景响应/DE/模块及模型权重固定。基因别名唯一映射后，评分端保留 exp004 原名称与顺序。沿用 exp00701 的 LFC 计数生成算法；模板仅来自输入半池，按 exp004 每靶点/批次参考数量分配。',
           '使用 exp004 原始 tiled baseline 与原 anchors、原配置（只改私有缓存路径），生成种子 101/202/303，汇总均值及样本标准差。这是旧最优 checkpoint 的协议敏感性检验，不是重新训练或独立测试，也不证明等同线上 r4 bundles。',
           '| 特征组 | 状态 | exp004 Overall 均值 ± SD | W&B |', '|---|---|---:|---|']
    for file in sorted((EXPERIMENT/'outputs').glob('*/metrics.json')):
        cfg=yaml.safe_load((file.parent/'config.yaml').read_text())['configuration']
        if cfg.get('task')!='exp004_reevaluation':continue
        result=json.loads(file.read_text());score=result.get('reevaluation',{})
        display=f'{score["mean_score"]:.6f} ± {score["score_sd"]:.6f}' if score else '待完成'
        lines.append(f'| {cfg["feature_mode"]} | {result["status"]} | {display} | [{file.parent.name}]({result.get("wandb_url", "")}) |')
    path.write_text(existing+'\n\n'+'\n\n'.join(lines[:6])+'\n\n'+'\n'.join(lines[6:])+'\n')
