"""One complete full-panel reference/bundle batch; no predictive model fit."""
import argparse
from contextlib import ExitStack
import gc
import json
import os
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT/'scripts/dossier')]

import anndata as ad
from anndata.io import write_elem
import h5py
import numpy as np
import pandas as pd
from research import bind
from vcc_task.common import NTC, ref, verified, write_json
from vcc_task.count_store import copy_panel
from vcc_task.full_reference import prepare_full_context, external_prior
from vcc_task.bounded_bundle import bounded_bundle, mapped_reference
from vcc_task.prediction_diagnostics import diagnose_panel
from vcc_task.frozen_scoring import score_frozen
from vcc_task.run_context import RegisteredRun


def template_for(split, observations, width):
    total=np.zeros(width); mass=np.zeros(width)
    for context in split['training_contexts']:
        with np.load(verified(observations[context]['statistics_ref'])) as data:
            labels=data['labels']; positions=data['positions']; bulk=data['bulk']
            keep=np.isin(labels,split['training_targets'])
            delta=bulk[keep]-bulk[np.flatnonzero(labels==NTC)[0]]
            genes=pd.read_csv(ROOT/'docs/research/challenge_2026/official_gene_axis.csv').gene_name.to_numpy()[positions]
            valid=labels[keep,None]!=genes[None,:]
            mean=(delta*valid).sum(0)/np.maximum(valid.sum(0),1)
            available=valid.any(0)
            total[positions[available]]+=mean[available]
            mass[positions[available]]+=1
    return np.divide(total,mass,out=np.zeros_like(total),where=mass>0)


def panel(job,split,context,observations,checkpoint,banks,reference_stack):
    cfg=job.config; info=observations[context]
    directory=job.output/'predictions'/split['id']/context
    directory.mkdir(parents=True,exist_ok=True)
    marker=directory/'panel.json'
    reuse_key=split['id']+'/'+context
    if reuse_key in cfg.get('reuse',{}).get('panels',{}):
        source=cfg['reuse']['panels'][reuse_key]
        result=json.loads(verified(source).read_text())
        for item in result['files']+result['bundle']['files']:
            verified(item)
        if result['split']!=split['id'] or result['context']!=context:
            raise ValueError('Reused panel identity mismatch')
        if reuse_key==cfg['baseline_memory']['canonical_equivalence_panel']:
            check=result['bundle']['baseline_equivalence']
            if not check['canonical_full_checked'] or check['different_entries']!=0:
                raise ValueError('Canonical official baseline equivalence was not established')
        result=dict(result,reused_panel_ref=source)
        write_json(marker,result)
        print('REUSED COMPLETE PANEL',reuse_key,flush=True)
        return result
    if marker.exists():
        result=json.loads(marker.read_text())
        for item in result['files']:
            verified(item)
        return result
    targets=info['panels'][split['id']]
    real_path=directory/'real.h5ad'
    if not real_path.exists():
        if context not in banks:
            banks[context]=reference_stack.enter_context(mapped_reference(
                verified(info['bank_ref']),job.output/'cache/decoded-banks'/context))
        copy_panel(banks[context],targets,real_path,cfg['chunk_rows'])
    ntc=ad.read_h5ad(verified(info['input_ntc_ref']))
    full_width=len(pd.read_csv(verified(cfg['benchmark']['gene_axis'])))
    template=template_for(split,observations,full_width)[info['official_gene_positions']]
    counts=info['target_counts']
    seen=set(split['training_targets'])
    official=set(split['official_overlap_targets'][context])
    strata={
        'n_lt_100':[t for t in targets if counts[t]<100],
        'n_100_399':[t for t in targets if 100<=counts[t]<400],
        'n_ge_400':[t for t in targets if counts[t]>=400],
        'seen':[t for t in targets if t in seen],
        'unseen':[t for t in targets if t not in seen],
        'official':[t for t in targets if t in official]}
    # True-count oracle is an evaluator diagnostic, never a predictor artifact.
    diagnostic=diagnose_panel(real_path,ntc,directory/'real-copy-diagnostics',seed=cfg['seed'],
        checkpoint_ref=ref(checkpoint),expected_targets=targets,real=real_path,template=template,
        pathway_ref=cfg['pathway_ref'],covariance_genes=cfg['diagnostics']['covariance_genes'])
    runtime=cfg['runtime']
    status=bounded_bundle(str(real_path),directory/'bundle',f"full-{split['id']}-{context}",runtime,
        chunk_targets=cfg['baseline_memory']['chunk_targets'],
        verify_full=split['id']+'/'+context==cfg['baseline_memory']['canonical_equivalence_panel'])
    scores={}
    scores['real_copy']=score_frozen(real_path,real_path,status,directory/'real-copy-score',runtime,diagnostic,strata)
    shuffled=directory/'target-derangement.h5ad'
    if not shuffled.exists():
        shutil.copyfile(real_path,shuffled)
        data=ad.read_h5ad(real_path,backed='r'); obs=data.obs.copy(); data.file.close()
        ordered=sorted(targets); rename={t:ordered[(i+1)%len(ordered)] for i,t in enumerate(ordered)}
        rename[NTC]=NTC
        obs['target_gene']=obs.target_gene.astype(str).map(rename).astype('category')
        with h5py.File(shuffled,'r+') as handle:
            del handle['obs']; write_elem(handle,'obs',obs)
    shuffled_diagnostic=diagnose_panel(shuffled,ntc,directory/'derangement-diagnostics',seed=cfg['seed'],
        checkpoint_ref=ref(checkpoint),expected_targets=targets,real=real_path,template=template,
        pathway_ref=cfg['pathway_ref'],covariance_genes=cfg['diagnostics']['covariance_genes'])
    scores['target_derangement']=score_frozen(shuffled,real_path,status,directory/'derangement-score',runtime,
                                             shuffled_diagnostic,strata)
    finite=all(s['Overall'] is not None and s['normalized'] is not None and
               all(v is not None for v in s['normalized'].values()) for s in scores.values())
    result={'split':split['id'],'scenario':split['scenario'],'context':context,'targets':len(targets),
        'real_n_strata':{key:len(value) for key,value in strata.items()},
        'measured_genes':ntc.n_vars,'official_genes':full_width,
        'real_ref':ref(real_path),'input_ntc_ref':info['input_ntc_ref'],
        'bundle':status,'scores':scores,'six_normalized_available':finite,
        'files':[ref(real_path),ref(shuffled),ref(directory/'real-copy-diagnostics/diagnostics.json'),
                 ref(directory/'derangement-diagnostics/diagnostics.json'),
                 ref(directory/'real-copy-score/result.json'),ref(directory/'derangement-score/result.json')]}
    write_json(marker,result)
    print('FULL PANEL COMPLETE',split['id'],context,'normalized available',finite,flush=True)
    del ntc; gc.collect()
    return result


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--config',type=Path,required=True)
    cfg=json.loads(parser.parse_args().config.read_text())
    research=bind(ROOT,cfg['run_id'],cfg,resume=os.environ.get('VCC_RESEARCH_RESUME')=='1')
    job=RegisteredRun(cfg,research)
    try:
        prior=external_prior(job)
        observations={c:prepare_full_context(job,c) for c in cfg['contexts']}
        support=job.output/'cache/full-support.json'
        write_json(support,{'observations':observations,'prior':prior,'research':research})
        checkpoint=job.output/'checkpoints/evaluator-specification.json'
        write_json(checkpoint,{'research':research,'protocol':cfg['benchmark'],'predictor_fit':False})
        splits=json.loads(verified(cfg['benchmark']['split_manifest']).read_text())
        results={}
        with ExitStack() as reference_stack:
            banks={}
            for split in splits:
                if split['scenario'] not in cfg['scenarios']:
                    continue
                for context in split['evaluation_contexts']:
                    results[split['id']+'/'+context]=panel(job,split,context,observations,checkpoint,banks,reference_stack)
            banks.clear()
        if len(results)!=cfg['expected_panels']:
            raise ValueError('Incomplete full-fold evaluation support batch')
        manifest=job.output/'predictions/manifest.json'
        write_json(manifest,{key:ref(job.output/'predictions'/key/'panel.json') for key in results})
        job.complete({'support':{'fraction':sum(r['six_normalized_available'] for r in results.values())/len(results)},
            'panels':results,'support_ref':ref(support),'diagnostics_completed':True,
            'submission_decision':'not_applicable_evaluator_only'},checkpoint,manifest,[support])
    except BaseException:
        job.run.finish(exit_code=1); raise


if __name__=='__main__':
    main()
