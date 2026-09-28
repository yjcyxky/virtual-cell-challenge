"""Rebuild frozen physical splits; expose only allowed training pseudobulks."""
from pathlib import Path
import json
import sys
import gc
import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'scripts/dossier'))
from challenge import ChallengeIdentity, inspect_context, reference_indices, NTC
from rna import RNAFile, hash_file, value_hash


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    tmp.replace(path)


def ref(path):
    return {'path': str(Path(path).relative_to(ROOT)), 'sha256': hash_file(Path(path))}


def normalized(matrix, target_sum):
    matrix = matrix.astype(np.float64)
    total = np.asarray(matrix.sum(axis=1)).ravel()
    if (total <= 0).any():
        raise ValueError('zero measured library')
    matrix = sparse.diags(target_sum / total) @ matrix
    matrix.data = np.log1p(matrix.data)
    return matrix.tocsr()


def prepare(output, config):
    directory = output / 'cache'
    audit = json.loads((ROOT / config['benchmark']['data_audit']['path']).read_text())
    splits = json.loads((ROOT / config['benchmark']['split_manifest']['path']).read_text())
    split = next(s for s in splits if s['id'] == config['fit_scope']['outer_split'])
    if split['id'] != 'S2-H1':
        raise ValueError('this registered implementation supports S2-H1 only')
    axis = pd.read_csv(ROOT / config['benchmark']['gene_axis']['path']).gene_name.tolist()
    identity = ChallengeIdentity(axis, pd.read_csv(ROOT / 'data/raw/networks/hgnc_complete_set.txt', sep='\t', low_memory=False))
    tasks = pd.read_csv(ROOT / 'docs/research/challenge_2026/tasks.csv')
    source_hashes = {r['path']: r['sha256'] for r in audit['inputs']}
    prepared = {}
    for context in split['training_contexts'] + ['H1']:
        done = directory / f'{context}-prepared.json'
        if done.exists():
            result = json.loads(done.read_text())
            for record in result['files']:
                if hash_file(ROOT / record['path']) != record['sha256']:
                    raise ValueError('prepared cache checksum changed')
            prepared[context] = result
            continue
        inputs = []
        frame, mapping, summary = inspect_context(ROOT, context, identity, audit['config'], inputs)
        if summary != audit['contexts'][context] or any(source_hashes.get(r['path']) != r['sha256'] for r in inputs):
            raise ValueError(f'frozen data audit mismatch: {context}')
        frame.to_parquet(directory / f'{context}-identities.parquet', index=False)
        training_targets = (sorted(set(frame.loc[frame.eligible, 'target']) & set(split['training_targets']))
                            if context in split['training_contexts'] else [])
        labels = [NTC] + training_targets
        label_index = {name: i for i, name in enumerate(labels)}
        positions = np.asarray(summary['official_gene_positions'])
        sums = np.zeros((len(labels), len(positions)), dtype=np.float64)
        counts = np.zeros(len(labels), dtype=np.int64)
        input_blocks, input_obs, reference_blocks, reference_obs = [], [], [], []
        chosen = set()
        if context == 'H1':
            targets = split['evaluation_targets'][context]
            for target in targets:
                members = frame.loc[frame.eligible & frame.target.eq(target)]
                ids = reference_indices(members, audit['config']['split_seed'])
                task = tasks.loc[tasks.context.eq(context) & tasks.target.eq(target)].iloc[0]
                if value_hash(sorted(frame.loc[ids, 'physical_id'])) != task.reference_identity_sha256:
                    raise ValueError('reference identities differ from frozen task')
                chosen.update(ids)
            chosen.update(frame.index[frame.ntc_pool.eq('score')])
        for relative, members in frame.groupby('file', sort=False):
            feature_map = mapping.loc[mapping.file.eq(relative) & mapping.measured].sort_values('official_position')
            cols = feature_map.source_position.to_numpy()
            with RNAFile(ROOT / 'data/raw' / relative) as source:
                row_to_frame = np.full(source.shape[0], -1, dtype=np.int64)
                row_to_frame[members.source_row.to_numpy()] = members.index.to_numpy()
                for start, block in source.blocks(config['data']['chunk_rows']):
                    indices = row_to_frame[start:start + block.shape[0]]
                    keep = indices >= 0
                    if not keep.any():
                        continue
                    rows = frame.loc[indices[keep]]
                    block = block[keep][:, cols].tocsr()
                    inp = rows.ntc_pool.eq('input').to_numpy()
                    fit = (rows.eligible & rows.target.isin(training_targets)).to_numpy() | inp
                    if fit.any():
                        group = np.array([label_index[t] for t in rows.loc[fit, 'target']])
                        present, inverse = np.unique(group, return_inverse=True)
                        selector = sparse.csr_matrix((np.ones(len(group)), (inverse, np.arange(len(group)))), shape=(len(present), len(group)))
                        sums[present] += (selector @ normalized(block[fit], config['data']['target_sum'])).toarray()
                        counts += np.bincount(group, minlength=len(labels))
                    if inp.any():
                        input_blocks.append(block[inp].astype(np.float32))
                        input_obs.append(rows.loc[inp, ['physical_id', 'target', 'batch']])
                    take = rows.index.isin(chosen)
                    if take.any():
                        reference_blocks.append(block[take].astype(np.float32))
                        reference_obs.append(rows.loc[take, ['physical_id', 'target', 'batch']])
        if (counts <= 0).any():
            raise ValueError('empty training pseudobulk')
        mean = (sums / counts[:, None]).astype(np.float32)
        statistics = directory / f'{context}-statistics.npz'
        np.savez_compressed(statistics, labels=np.asarray(labels), positions=positions, mean=mean, counts=counts)
        obs = pd.concat(input_obs).set_index('physical_id').rename(columns={'target': 'target_gene'})
        ntc_path = directory / f'{context}-input.h5ad'
        var = pd.DataFrame(index=np.asarray(axis)[positions])
        ad.AnnData(sparse.vstack(input_blocks, format='csr'), obs=obs, var=var).write_h5ad(ntc_path, compression='lzf')
        files = [ref(statistics), ref(ntc_path), ref(directory / f'{context}-identities.parquet')]
        if context == 'H1':
            obs = pd.concat(reference_obs).set_index('physical_id').rename(columns={'target': 'target_gene'})
            real_path = directory / 'H1-reference.h5ad'
            real = ad.AnnData(sparse.vstack(reference_blocks, format='csr'), obs=obs, var=var)
            real.write_h5ad(real_path, compression='lzf')
            files.append(ref(real_path))
            del real
        result = {'context': context, 'training_tasks': len(labels)-1, 'training_cells': int(counts[1:].sum()),
                  'input_ntc': int(counts[0]), 'files': files, 'source_refs': inputs,
                  'evaluation_only_reference': context == 'H1'}
        write_json(done, result)
        prepared[context] = result
        print(f'Prepared {context}: {result["training_tasks"]} training tasks, {result["training_cells"]} training cells', flush=True)
        del frame, mapping, sums, mean, input_blocks, reference_blocks
        gc.collect()
    write_json(directory / 'preparation.json', prepared)
    return axis, split, prepared
