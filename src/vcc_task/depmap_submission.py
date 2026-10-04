"""Export a frozen DepMap kNN fit, diagnose it, and resume one official entry."""
import argparse
import fcntl
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = ROOT/'experiments/depmap-response'
sys.path[:0] = [str(ROOT/'scripts'), str(ROOT/'scripts/dossier')]
from research import git, ref_error


def read_config(path):
    path = Path(path).resolve()
    config = json.loads(path.read_text())
    dag = json.loads((ROOT/'docs/research/experiment_dag.json').read_text())
    node = dag['nodes'][config['run_id']]
    stage = node['official_evaluation']
    if path != ROOT/stage['config_ref']['path'] or node['status'] != 'completed':
        raise ValueError('Export must use the registered config and completed fit')
    if config['comparison_id'] != stage['comparison_id']:
        raise ValueError('Official comparison differs from registration')
    training = node['expected_config']
    for key in ('seed','fit_scope','model','decoder','emitter','environment','prediction_cells','runtime'):
        if config[key] != training[key]:
            raise ValueError('Export changed frozen inference condition: '+key)
    if config['model']['kind'] != 'knn' or config['fit_scope']['outer_split'] not in ('S2-H1','S3-H1'):
        raise ValueError('This entry exports the requested H1 kNN fit only')
    snapshot = node['code_snapshot_commit']
    references = [node['config_ref'], node['metrics_ref'], config['checkpoint_ref'], config['local_response_ref'],
                  *config['official_input_refs'], *[dict(r, git_commit=snapshot) for r in node['code_refs']]]
    stage_refs = [stage['config_ref'], *stage['code_refs']]
    for item in references + stage_refs:
        if error := ref_error(ROOT, item):
            raise ValueError(error)
    for item in stage_refs:
        if hashlib.sha256(git(ROOT,'show','HEAD:'+item['path']).stdout).hexdigest() != item['sha256']:
            raise ValueError('Export code/config has not been committed: '+item['path'])
    committed = json.loads(git(ROOT,'show','HEAD:docs/research/experiment_dag.json').stdout)['nodes'][config['run_id']]['official_evaluation']
    if any(stage[key] != committed[key] for key in ('comparison_id','config_ref','code_refs')):
        raise ValueError('Export stage has not been committed')
    metrics = json.loads((ROOT/node['metrics_ref']['path']).read_text())
    if metrics['checkpoint_ref'] != config['checkpoint_ref'] or not metrics['evaluation_completed']:
        raise ValueError('Checkpoint is not the completed, locally evaluated model')
    return config, metrics, stage, path


def legacy_io():
    """Retain the tested VCC packaging, credential handling and entry-resume code."""
    sys.path.append(str(ROOT/'experiments/init-linear/src'))
    spec = importlib.util.spec_from_file_location('frozen_vcc_submission_io', ROOT/'experiments/init-linear/src/submission.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, module.helpers()


def generate_context(config, stored, targets, ntc, context, destination, checkpoint_ref):
    import numpy as np
    import pandas as pd
    from scipy import sparse
    from .common import ref, verified, write_json, stable_seed
    from .counts import profile
    from .count_store import CountWriter
    from .frozen_knn import predict_knn
    from .response_data import decode_composition
    from .population_emission import fit_population, draw_population, state_record
    from .official import annotated
    marker = destination/'generation.json'
    if marker.exists():
        result = json.loads(marker.read_text())
        for item in result['files']:
            verified(item)
        return result
    latent, coverage = predict_knn(stored, targets, config['model']['neighbors'])
    prediction = destination/'counts.h5ad'
    if prediction.exists():
        raise ValueError('Unsealed context prediction exists; inspect it before recovery')
    response = destination/'response.npz'
    np.savez_compressed(response, genes=np.asarray(ntc.var_names,dtype=str), targets=np.asarray(targets),
                        response=latent, global_supervision=stored['observed_readouts'])
    writer = CountWriter(prediction, ntc.var.copy())
    x = sparse.csr_matrix(ntc.X,dtype=np.float64)
    baseline = profile(x)
    obs, emission, repeat_blocks, repeat_labels, repeat_statistics = [], [], [], [], []
    repeat_targets = sorted(targets,key=lambda t:stable_seed(config['seed'],context,'repeat-target',t))[:config['diagnostics']['repeat_targets']]
    for i, target in enumerate(targets):
        own = ntc.var_names.get_loc(target)
        desired, detail = decode_composition(baseline, latent[i], own=own, **config['decoder'])
        fitted = fit_population(x, desired-baseline, **config['emitter'])
        seed = stable_seed(config['seed'],config['fit_scope']['outer_split'],context,target,'prediction')
        counts = draw_population(x,fitted,seed,config['prediction_cells'])
        writer.append(counts)
        if writer.nnz > config['max_nnz']:
            raise ValueError('Submission capacity exceeded even before context concatenation')
        obs.append(pd.DataFrame({'target_gene':target,'context':context},
                   index=[f'{config["run_id"]}-{context}-{target}-{r}' for r in range(config['prediction_cells'])]))
        detail.update(target=target, seed=seed, **state_record(fitted),
                      sample_bulk_rmse=float(np.sqrt(np.mean((profile(counts)-desired)**2))))
        emission.append(detail)
        if target in repeat_targets:
            means, variances = [], []
            for repeat in range(config['diagnostics']['generation_repeats']):
                repeated = draw_population(x,fitted,stable_seed(seed,'repeat',repeat),config['prediction_cells'])
                normalized = sparse.diags(10_000/np.asarray(repeated.sum(1)).ravel())@repeated.astype(float)
                mean = np.asarray(normalized.mean(0)).ravel()
                variance = np.maximum(np.asarray(normalized.power(2).sum(0)).ravel()-counts.shape[0]*mean**2,0)/(config['prediction_cells']-1)
                means.append(mean); variances.append(variance)
                repeat_blocks.append(repeated); repeat_labels.extend([f'{target}__repeat_{repeat}']*config['prediction_cells'])
            expected = np.mean(variances,axis=0)/config['prediction_cells']
            actual = np.var(means,axis=0,ddof=1)
            allowed = (expected>1e-10)&~np.isin(ntc.var_names,targets)
            repeat_statistics.append({'target':target,'repeats':config['diagnostics']['generation_repeats'],
                'valid_genes':int(allowed.sum()),'variance_ratio':float(actual[allowed].sum()/expected[allowed].sum())})
        if (i+1)%25 == 0:
            print(json.dumps({'stage':'generate','context':context,'targets':i+1,'nnz':writer.nnz}),flush=True)
    frame = pd.concat(obs)
    for column in frame:
        frame[column] = frame[column].astype('category')
    writer.finish(frame)
    repeats = destination/'independent-repeats.h5ad'
    annotated(sparse.vstack(repeat_blocks,format='csr'),ntc.var_names,repeat_labels,context+'-repeat').write_h5ad(repeats,compression='gzip')
    write_json(destination/'emission.json',{'records':emission,'independent_repeats':repeat_statistics,'source_coverage':coverage})
    result = {'prediction_ref':ref(prediction),'response_ref':ref(response),'checkpoint_ref':checkpoint_ref,
              'coverage':coverage,'independent_repeats':repeat_statistics,
              'files':[ref(p) for p in (prediction,response,repeats,destination/'emission.json')]}
    write_json(marker,result)
    return result


def assemble(config, directory, official, genes, targets, helper):
    import anndata as ad
    import pandas as pd
    from .common import ref, verified, write_json
    from .count_store import CountWriter
    marker = directory/'generation.json'
    if marker.exists():
        result = json.loads(marker.read_text()); verified(result['file_ref'])
        return result
    prediction = directory/'predictions.h5ad'
    if prediction.exists():
        raise ValueError('Unsealed submission prediction exists')
    writer = CountWriter(prediction,pd.DataFrame(index=pd.Index(genes,name='gene_name')))
    observations = []
    for context in ('A','B','C'):
        data = ad.read_h5ad(directory/context/'counts.h5ad',backed='r')
        try:
            matrix = data.X
            for start in range(0,len(data),config['prediction_cells']):
                writer.append(matrix[start:start+config['prediction_cells']])
            observations.append(data.obs.copy())
        finally:
            data.file.close()
    writer.finish(pd.concat(observations))
    audit = helper.audit_prediction(prediction,genes,targets,['A','B','C'],config['prediction_cells'],
                                    config['max_nnz'],config['max_counts_per_cell'])
    write_json(directory/'prediction-audit.json',audit)
    result = {'file_ref':ref(prediction),'audit':audit,'contexts':['A','B','C'],
              'checkpoint_ref':config['checkpoint_ref'],'fit_scope':config['fit_scope']}
    write_json(marker,result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--gate',action='store_true')
    parser.add_argument('--submit',action='store_true')
    args = parser.parse_args()
    config,metrics,stage,config_path = read_config(args.config)
    if args.gate:
        print('Frozen H1 kNN official export gate passed',flush=True); return
    import anndata as ad
    import numpy as np
    import torch
    import wandb
    import yaml
    from .common import ref, write_json
    from .frozen_knn import predict_knn
    from .official_diagnostics import diagnose_official
    for key,value in config['environment'].items():
        if os.environ.get(key) != value:
            raise ValueError('Frozen environment mismatch: '+key)
    output = EXPERIMENT/'outputs'/config['run_id']
    training = yaml.safe_load((output/'config.yaml').read_text())
    versions = {key:importlib.metadata.version(key) for key in training['runtime_versions']}
    if versions != training['runtime_versions'] or ref(EXPERIMENT/'uv.lock') != training['lock_ref']:
        raise ValueError('Export runtime differs from completed fit')
    if subprocess.check_output([config['vcc_cli'],'--version'],text=True).splitlines()[0] != 'vcc '+config['vcc_cli_version']:
        raise ValueError('Official CLI version changed')
    io,helper = legacy_io()
    official,manifest,genes,targets,identities = helper.official_inputs(
        {'official':config['official'],'cells_per_perturbation':config['prediction_cells']})
    if manifest['panel_id'] != config['panel_id']:
        raise ValueError('Official panel changed')
    targets = sorted(targets)
    stored = torch.load(ROOT/config['checkpoint_ref']['path'],map_location='cpu',weights_only=False)
    if stored['research'] != metrics['research'] or list(stored['axis']) != genes or stored['decoder'] != config['decoder']:
        raise ValueError('Frozen checkpoint identity/axis/decoder mismatch')
    with np.load(ROOT/config['local_response_ref']['path']) as original:
        replay,_ = predict_knn(stored,original['targets'].tolist(),config['model']['neighbors'])
        if not np.array_equal(replay,original['response']):
            raise ValueError('Exporter cannot exactly replay the locally evaluated responses')
        replay_cells = len(replay)
    directory = output/'predictions/official-abc-knn'
    directory.mkdir(parents=True,exist_ok=True)
    with (directory/'.export.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        binding = {'config_ref':ref(config_path),'checkpoint_ref':config['checkpoint_ref'],
                   'research':metrics['research'],'runtime_versions':versions,'lock_ref':training['lock_ref'],
                   'official_inputs':identities,'code_refs':stage['code_refs']}
        identity = directory/'export-identity.json'
        if identity.exists() and json.loads(identity.read_text())['binding'] != binding:
            raise ValueError('Export inputs/code/config changed on resume')
        if not identity.exists():
            write_json(identity,{'binding':binding,'export_git_commit':git(ROOT,'rev-parse','HEAD').stdout.decode().strip()})
        write_json(directory/'local-replay.json',{'exact':True,'targets':replay_cells,
                   'checkpoint_ref':config['checkpoint_ref'],'response_ref':config['local_response_ref']})
        tracked = wandb.init(entity='yjcyxky',project='virtual-cell-challenge',group='depmap-response',
                            id=config['run_id'],name=config['run_id'],resume='must',dir=str(output),
                            config={'official_export':json.loads(identity.read_text())})
        try:
            diagnostics = {}
            for context in ('A','B','C'):
                ntc = ad.read_h5ad(official/f'context_{context}.h5ad')
                if (ntc.var_names.tolist()!=genes or len(ntc)!=manifest['per_context'][context]['control_cells']
                        or set(ntc.obs.context.astype(str))!={context}
                        or set(ntc.obs.target_gene.astype(str))!={'non-targeting'}):
                    raise ValueError('Official NTC identity/axis mismatch')
                destination = directory/context; destination.mkdir(exist_ok=True)
                generated = generate_context(config,stored,targets,ntc,context,destination,config['checkpoint_ref'])
                diagnostics[context] = diagnose_official(destination/'counts.h5ad',ntc,destination/'diagnostics',
                    targets,config,config['checkpoint_ref'],generated['coverage'],stored['observed_readouts'])
                tracked.summary[f'official/knn/diagnostics/{context}'] = diagnostics[context]['supervision']
                del ntc
            result = assemble(config,directory,official,genes,targets,helper)
            decision = {'decision':'submit','reason':'User-requested frozen-model official measurement; prior local non-promotion remains unchanged.',
                        'hard_constraints_passed':True,'all_context_diagnostics_completed':True,
                        'prediction_ref':result['file_ref'],'diagnostic_refs':[
                            ref(directory/c/'diagnostics/completed.json') for c in ('A','B','C')],
                        'statistical_anomalies_policy':'Report all; no target filtering or generation changes after observing diagnostics.',
                        'local_metrics_ref':ref(output/'metrics.json'),'no_sota_claim':True}
            write_json(directory/'submission-decision.json',decision)
            io.package(config,directory,official)
            if args.submit:
                io.submit(config,directory,'knn',tracked)
                if not (directory/'artifact.json').exists():
                    artifact = wandb.Artifact(config['run_id']+'-official-knn',type='evaluation',metadata=binding)
                    for name in ('predictions.vcc','export-identity.json','local-replay.json','submission-decision.json',
                                 'official-status.json','official-summary.json','submission-entry.json','submission-file.json',
                                 'prediction-audit.json','generation.json','prep.json'):
                        artifact.add_file(str(directory/name),name=name)
                    for context in ('A','B','C'):
                        for name in ('completed.json','diagnostics.json','per-target.parquet','derived-moments.h5',
                                     'null.json','null-de.parquet','null-distributions.parquet','same-n-ntc.parquet','detected-cv.parquet','readout-strata.parquet'):
                            artifact.add_file(str(directory/context/'diagnostics'/name),name=f'{context}/{name}')
                        artifact.add_file(str(directory/context/'emission.json'),name=f'{context}/emission.json')
                    logged = tracked.log_artifact(artifact); logged.wait()
                    write_json(directory/'artifact.json',{'artifact':logged.qualified_name})
            tracked.summary['official/knn/status'] = 'published' if args.submit else 'prepared'
            tracked.finish()
        except BaseException:
            tracked.finish(exit_code=1)
            raise


if __name__ == '__main__':
    main()
