"""Validate fixed historical inputs and attach immutable run artifacts."""
import json
from pathlib import Path
import shutil
import yaml
from common import EXPERIMENT,digest,event,write_json


def attach_reference(config,output):
    reference=EXPERIMENT/config['source_reference']
    spec=json.loads(reference.read_text());source=Path(spec['source_run'])
    identity={'reference_sha256':digest(reference),'source_run':str(source)}
    done=output/'cache/reference.json'
    if done.exists():
        if json.loads(done.read_text())!=identity:raise ValueError('reference_changed')
    # Re-check the immutable reference on every resume, including large count files.
    for relative,expected in spec['files'].items():
        p=source/relative
        if p.stat().st_size!=expected['bytes'] or digest(p)!=expected['sha256']:
            raise ValueError(f'source_artifact_changed:{relative}')
    old=yaml.safe_load((source/'config.yaml').read_text())['configuration']
    changes={'experiment_id','source_run','source_experiment','source_code_commit','source_usage',
             'source_reference','feature_mode','module_dimensions','module_min_genes','module_max_genes'}
    if {k:v for k,v in config.items() if k not in changes}!={k:v for k,v in old.items() if k not in changes}:
        raise ValueError('baseline_configuration_mismatch')
    cache=output/'cache';cache.mkdir(exist_ok=True)
    for name in ['data','priors']:
        dest=cache/name;origin=source/'cache'/name
        if dest.exists():
            if not dest.is_symlink() or dest.resolve()!=origin.resolve():raise ValueError('reference_link_changed')
        else:dest.symlink_to(origin,target_is_directory=True)
    official=cache/'official/H1';official.mkdir(parents=True,exist_ok=True)
    origin=source/'cache/official/H1'
    ref=official/'reference'
    if ref.exists():
        if not ref.is_symlink() or ref.resolve()!=(origin/'reference').resolve():raise ValueError('reference_link_changed')
    else:ref.symlink_to(origin/'reference',target_is_directory=True)
    # Scorer-owned caches are private. Never let a scorer mutate historical files.
    for name in ['reference.json','baseline-protocol.json','bundle','real-cache']:
        dest=official/name;src=origin/name
        if dest.exists():continue
        if src.is_dir():shutil.copytree(src,dest)
        else:shutil.copy2(src,dest)
    write_json(done,identity)
    event('reference_validated',**identity)
    return spec
