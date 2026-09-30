"""Official scoring gated by a completed diagnostic tied to the same prediction."""
import json
from pathlib import Path

import numpy as np
import polars as pl
from cell_eval2 import compute_metrics, score_metrics
from cell_eval2.baseline import build_run_meta
from cell_eval2.competition import competition_members
from cell_eval2.run import metric_output_names
from challenge import scorer_config

from .capability import aggregate_with_unavailable, finite_number, reference_unavailable
from .common import ROOT, ref, verified, write_json


def score_frozen(prediction, real, bundle_status, directory, runtime, diagnostic, strata=None):
    directory=Path(directory); directory.mkdir(parents=True,exist_ok=True)
    marker=directory/'result.json'
    if marker.exists():
        result=json.loads(marker.read_text())
        for item in result['files']:
            verified(item)
        return result
    if (diagnostic.get('status')!='completed' or diagnostic.get('hard_constraints_passed') is not True
            or diagnostic['prediction_ref']!=ref(prediction)):
        raise ValueError('Formal scoring requires completed diagnostics for these frozen counts')
    cfg=scorer_config(outdir=str(directory),**runtime)
    raw=compute_metrics(str(prediction),str(real),config=cfg,write_de=True)
    raw.write_parquet(directory/'raw.parquet')
    names=metric_output_names(cfg)
    aggregate,errors=aggregate_with_unavailable(raw,names)
    aggregate.write_csv(directory/'aggregate.csv')
    meta=build_run_meta(cfg,str(real),str(prediction)); write_json(directory/'run_meta.json',meta)
    means=aggregate.filter(pl.col('statistic')=='mean').to_dicts()[0]
    result={'raw':{k:finite_number(v) for k,v in means.items() if k!='statistic'},
        'normalized':None,'Overall':None,'raw_aggregation_rejections':errors,
        'official_rejection':bundle_status.get('official_rejection'),
        'diagnostic_prediction_ref':diagnostic['prediction_ref']}
    if bundle_status['available'] and not errors:
        try:
            scored=score_metrics(aggregate,real_bundle=str(ROOT/bundle_status['bundle_directory']),user_meta=meta)
        except ValueError as exc:
            if not reference_unavailable(exc):
                raise
            result['official_rejection']=str(exc)
        else:
            scored.write_csv(directory/'scores.csv')
            result['normalized']={m:finite_number(scored.filter(pl.col('metric')==m)['from_replicate'].item())
                                  for m in competition_members()}
            result['Overall']=finite_number(scored.filter(pl.col('metric')=='avg_score')['from_replicate'].item())
            if all(v is not None for v in result['normalized'].values()) and not np.isclose(
                    result['Overall'],np.mean(list(result['normalized'].values()))):
                raise ValueError('Official six-member aggregation mismatch')
    result['raw_strata']={}
    for label,targets in (strata or {}).items():
        subset=raw.filter(pl.col('perturbation').is_in(targets))
        if not subset.height:
            result['raw_strata'][label]={'targets':0,'raw':None}; continue
        sub,issues=aggregate_with_unavailable(subset,names)
        means=sub.filter(pl.col('statistic')=='mean').to_dicts()[0]
        result['raw_strata'][label]={'targets':len(targets),'raw':{k:finite_number(v) for k,v in means.items()
            if k!='statistic'},'rejections':issues,'normalized':None,
            'scope':'raw stratification only; full-panel anchors are not reused as subgroup anchors'}
    result['files']=[ref(p) for p in directory.iterdir() if p.is_file() and p!=marker]
    write_json(marker,result)
    return result
