"""Full frozen eligible panels, with bounded raw reads and no response filtering."""
import gc
import hashlib
import json
from pathlib import Path
import urllib.request

import anndata as ad
import numpy as np
import pandas as pd
from rna import RNAFile

from .common import ROOT, NTC, ref, verified, write_json, stable_seed
from .count_store import CountWriter
from .counts import thin_counts, validate_counts
from .observations import selected_panels


def external_prior(job):
    spec = job.config['external_prior']
    path = job.output/'cache'/spec['filename']
    if not path.exists():
        temporary = path.with_suffix('.download')
        with urllib.request.urlopen(spec['url'], timeout=120) as response, temporary.open('wb') as output:
            while data := response.read(8 << 20):
                output.write(data)
        temporary.replace(path)
    md5 = hashlib.md5()
    with path.open('rb') as handle:
        while data := handle.read(8 << 20):
            md5.update(data)
    if path.stat().st_size != spec['size'] or md5.hexdigest() != spec['md5']:
        raise ValueError('External DepMap release checksum mismatch')
    return dict(spec, file_ref=ref(path))


def prepare_full_context(job, context):
    cfg = job.config
    directory = job.output/'cache'/context
    marker = directory/'reference.json'
    if marker.exists():
        record = json.loads(marker.read_text())
        for item in record['files']:
            verified(item)
        return record
    directory.mkdir(parents=True, exist_ok=True)
    old = json.loads(verified(cfg['observation_refs'][context]).read_text())
    by_name = {Path(item['path']).name: item for item in old['files']}
    frame = pd.read_parquet(verified(by_name['cells.parquet']))
    if frame.physical_id.duplicated().any():
        raise ValueError('Duplicate physical cells in source observation')
    tasks = pd.read_csv(verified(cfg['tasks_ref']))
    splits = json.loads(verified(cfg['benchmark']['split_manifest']).read_text())
    panels = selected_panels(context, tasks, splits, 1_000_000, cfg['seed'])
    targets = set().union(*map(set, panels.values()))
    selected = []
    for target, group in frame.loc[frame.eligible & frame.target.isin(targets)].groupby('target', sort=True):
        rows = sorted(group.index, key=lambda i: stable_seed(cfg['seed'], context, target, frame.at[i,'physical_id']))
        selected.extend(rows[:cfg['reference_cells']])
    chosen = frame.loc[selected].copy()
    chosen['target_gene'] = chosen.target
    chosen['bank'] = 'perturbation'
    mapping = pd.read_csv(verified(cfg['mapping_ref']))
    mapping = mapping.loc[mapping.context.eq(context) & mapping.measured]
    old_bank = ad.read_h5ad(verified(by_name['bank-depth-view.h5ad']))
    ntc_input = old_bank[old_bank.obs.bank.eq('input')].copy()
    ntc_input.obs['target_gene'] = NTC
    ntc_path = directory/'input-ntc.h5ad'; ntc_input.write_h5ad(ntc_path, compression='gzip')
    score = old_bank[old_bank.obs.bank.eq('score')].copy()
    score.obs['target_gene'] = NTC
    if set(score.obs_names) & set(ntc_input.obs_names):
        raise ValueError('Input and scoring NTC overlap')
    writer = CountWriter(directory/'bank.h5ad', old_bank.var.copy())
    writer.append(score.X)
    observation_frames = [score.obs.copy()]
    source_refs = {item['path']: item for item in old['source_refs']}
    fraction = old['native_depth_thinning_probability']
    for relative, group in chosen.groupby('file', sort=False):
        source_ref = source_refs['data/raw/'+relative]
        path = verified(source_ref)
        columns = mapping.loc[mapping.file.eq(relative)].sort_values('official_position').source_position.to_numpy()
        with RNAFile(path) as source:
            index = np.full(source.shape[0], -1, dtype=np.int64)
            index[group.source_row] = group.index
            for start, raw in source.blocks(cfg['chunk_rows']):
                rows = index[start:start+raw.shape[0]]
                keep = rows >= 0
                if not keep.any():
                    continue
                measured = raw[keep][:, columns].tocsr()
                # Frozen block-derived random streams; only Bernoulli thinning.
                matrix = thin_counts(measured, fraction, stable_seed(cfg['seed'], context, relative, start, 'full-view'))
                validate_counts(matrix)
                writer.append(matrix)
                obs = chosen.loc[rows[keep]].copy().set_index('physical_id')
                observation_frames.append(obs)
        print(f'{context}: complete full reference source {relative}', flush=True)
    obs = pd.concat(observation_frames, sort=False)
    # Store only fields used for provenance/stratification; mixed missing metadata
    # columns from the old NTC bank are not part of the new matrix contract.
    fields = ['target_gene','target','bank','file','source_row','guide','batch']
    obs = obs[[field for field in fields if field in obs]]
    for field in ['target_gene','target','bank','file','guide','batch']:
        if field in obs:
            obs[field] = obs[field].astype(str).astype('category')
    bank_path = writer.finish(obs)
    counts = obs.loc[obs.target_gene.ne(NTC)].target_gene.value_counts()
    result = {'context':context, 'panels':panels, 'targets':len(targets),
        'target_counts':{str(k):int(v) for k,v in counts.items() if v},
        'real_cells':int(counts.sum()), 'n_ge_400':int(counts.ge(400).sum()),
        'measured_genes':old_bank.n_vars, 'official_gene_positions':old_bank.var.official_position.tolist(),
        'input_score_overlap':0, 'thinning_probability':fraction,
        'source_observation_ref':cfg['observation_refs'][context],
        'statistics_ref':by_name['statistics.npz'], 'dose_ref':by_name['knockdown-by-guide-batch.parquet'],
        'bank_ref':ref(bank_path), 'input_ntc_ref':ref(ntc_path),
        'files':[ref(bank_path),ref(ntc_path)]}
    if set(counts.index[counts.gt(0)]) != targets or counts.max() > cfg['reference_cells']:
        raise ValueError('Full eligible reference target coverage mismatch')
    write_json(marker,result)
    del old_bank, ntc_input, score, frame, chosen, obs
    gc.collect()
    return result
