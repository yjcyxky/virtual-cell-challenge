"""Immutable count inputs, technical support, balanced sampling and fold identities."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts/dossier'))
from rna import RNAFile, hash_file as _hash_file, value_hash
sys.path.append(str(ROOT / 'experiments/exp001-context-pair-xgb/src'))
from features import canonical_map


def release_read_cache(path):
    """GB10 shares RAM with file cache; advise only this completed file's clean pages."""
    path = Path(path)
    if hasattr(os, 'posix_fadvise') and path.stat().st_size >= 256 << 20:
        with path.open('rb') as stream:
            os.posix_fadvise(stream.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)


def hash_file(path):
    result = _hash_file(Path(path))
    release_read_cache(path)
    return result


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    temporary.replace(path)


def reserved(target, percent=20):
    return int.from_bytes(hashlib.sha256(('module-cvae-v1|' + target).encode()).digest()[:8], 'little') % 100 < percent


def raw_path(panel):
    family, name = panel.split(':')
    if family == 'H1':
        return ROOT / f'data/raw/arc_vcc2025_h1/adata_{name}.h5ad'
    if family == 'replogle':
        return ROOT / f'data/raw/replogle2022/{name}_raw_singlecell_01.h5ad'
    if family == 'nadig':
        return ROOT / f'data/raw/nadig2025/GSE264667_{name}_raw_singlecell_01.h5ad'
    raise ValueError(panel)


def balanced_selection(frame, maximum, rng):
    """Keep every observed construct×batch; allocate remaining slots hierarchically."""
    groups = {k: g.index.to_numpy() for k, g in frame.groupby(['construct', 'batch'], sort=True)}
    pools = {k: list(rng.permutation(v)) for k, v in groups.items()}
    selected = [pool.pop() for pool in pools.values()]
    maximum = min(len(frame), max(maximum, len(selected)))
    while len(selected) < maximum:
        available = defaultdict(list)
        for (construct, batch), pool in pools.items():
            if pool:
                available[construct].append(batch)
        construct = rng.choice(sorted(available))
        batch = rng.choice(available[construct])
        selected.append(pools[(construct, batch)].pop())
    return frame.loc[sorted(selected)].copy()


def prepare(config, output):
    cache = output / 'cache' / 'data'
    completion = cache / 'complete.json'
    if config.get('cache_source'):
        cache = Path(config['cache_source']).resolve()
        report = json.loads((cache / 'complete.json').read_text())
        assert report['data_spec_hash'] == data_spec_hash(config), 'cache_data_conditions_differ'
        verify_cache(cache, report)
        return cache
    cache.mkdir(parents=True, exist_ok=True)
    if completion.exists():
        report = json.loads(completion.read_text())
        assert report['data_spec_hash'] == data_spec_hash(config)
        verify_cache(cache, report)
        return cache
    genes = pd.read_csv(ROOT / config['gene_axis']).iloc[:, 0].astype(str).tolist()
    axis_source = json.loads((ROOT / config['gene_axis']).with_name('SOURCE.json').read_text())
    expected_axis = next(x['sha256'] for x in axis_source['files'] if x['name'] == Path(config['gene_axis']).name)
    assert hash_file(ROOT / config['gene_axis']) == expected_axis
    gene_index = {g: i for i, g in enumerate(genes)}
    assert len(gene_index) == len(genes)
    hgnc_path = ROOT / 'data/raw/networks/hgnc_complete_set.txt'
    prior_sources = json.loads((hgnc_path.parent / 'SOURCE.json').read_text())
    hgnc_sha = next(x['sha256'] for x in prior_sources['files'] if x['name'] == hgnc_path.name)
    assert hash_file(hgnc_path) == hgnc_sha
    aliases = canonical_map(pd.read_csv(hgnc_path, sep='\t', low_memory=False), genes)
    collection = ROOT / config['collection']
    collection_report = json.loads((collection / 'report.json').read_text())
    reports = {c['panel_id']: c for c in collection_report['contexts']}
    inputs = [{'path': config['gene_axis'], 'sha256': hash_file(ROOT / config['gene_axis'])},
              {'path': str(hgnc_path.relative_to(ROOT)), 'sha256': hgnc_sha},
              {'path': str((collection / 'report.json').relative_to(ROOT)), 'sha256': hash_file(collection / 'report.json')}]
    rng = np.random.default_rng(config['data_seed'])
    excluded, factors, panels = [], [], []
    for number, panel in enumerate(config['panels']):
        source_report = reports[panel]
        assert source_report['source_metadata']['intervention_mechanism'] == 'CRISPRi'
        assert not source_report['source_metadata']['additional_experimental_treatment']
        folder = collection / source_report['context_id']
        directory = cache / f'panel-{number}'
        directory.mkdir(exist_ok=True)
        for name in ['cell-splits.parquet', 'tasks.parquet', 'native-gene-mapping.parquet']:
            path = folder / name
            expected = source_report['artifacts'][name]
            assert hash_file(path) == expected, str(path)
            inputs.append({'path': str(path.relative_to(ROOT)), 'sha256': expected})
        cells = pd.read_parquet(folder / 'cell-splits.parquet')
        tasks = pd.read_parquet(folder / 'tasks.parquet')
        mapping = pd.read_parquet(folder / 'native-gene-mapping.parquet')
        source = raw_path(panel)
        assert cells.input_sha256.nunique() == 1
        expected = cells.input_sha256.iloc[0]
        print(json.dumps({'stage': 'input_hash', 'panel': panel, 'bytes': source.stat().st_size}), flush=True)
        assert hash_file(source) == expected, str(source)
        inputs.append({'path': str(source.relative_to(ROOT)), 'sha256': expected})
        with RNAFile(source) as raw:
            assert len(mapping) == raw.shape[1]
            native_names = raw.var.index.astype(str).to_numpy() if panel.startswith('H1:') else raw.var.gene_name.astype(str).to_numpy()
            assert np.array_equal(native_names, mapping.source_gene.astype(str).to_numpy()), 'source_gene_axis_mismatch'
            assert np.array_equal(raw.obs.index.astype(str).to_numpy()[cells.row_index.to_numpy()], cells.source_barcode.astype(str).to_numpy()), 'cell_identity_mismatch'
            safe = mapping.mapped_symbol.notna() & ~mapping.many_to_one_mapping & ~mapping.symbol_vs_ensembl.eq('conflict')
            safe &= mapping.mapped_symbol.isin(aliases)
            positions = np.flatnonzero(safe)
            destination = mapping.loc[safe, 'mapped_symbol'].map(aliases).to_numpy(np.int64)
            assert len(set(destination)) == len(destination)
            np.save(directory / 'genes.npy', destination)
            cells['canonical_source_target'] = cells.task_uid.map(tasks.set_index('task_uid').canonical_target)
            cells['target'] = cells.canonical_source_target.map(lambda t: genes[aliases[t]] if t in aliases else None)
            cells.loc[cells.is_NTC, 'target'] = '__NTC__'
            cells['context'] = source_report['source_metadata']['cell_line']
            cells['batch'] = cells.source_batch.astype(str)
            guide_column = 'guide_id' if panel.startswith('H1:') else 'sgID_AB'
            cells['construct'] = raw.obs[guide_column].astype(str).to_numpy()[cells.row_index.to_numpy()]
            raw_batch = 'batch' if panel.startswith('H1:') else 'gem_group'
            assert np.array_equal(raw.obs[raw_batch].astype(str).to_numpy()[cells.row_index.to_numpy()], cells.batch.to_numpy())
            for column in ['UMI_count', 'core_adjusted_UMI_count', 'core_scale_factor', 'mitopercent', 'z_gemgroup_UMI']:
                if column in raw.obs:
                    cells['source_' + column] = raw.obs[column].to_numpy()[cells.row_index.to_numpy()]
            cells['half'] = cells[f"half_{config['control_split_seed']}"]
            # H1 NTC copies are exact historical duplicates. Only Training provides them.
            controls = cells.loc[cells.is_NTC].copy()
            ntc_support = controls.groupby(['batch', 'half']).size().unstack(fill_value=0)
            good_batches = ntc_support.index[(ntc_support.get(0, 0) >= config['minimum_controls_per_half_batch']) &
                                             (ntc_support.get(1, 0) >= config['minimum_controls_per_half_batch'])]
            eligible = cells.loc[~cells.is_NTC & cells.target.isin(gene_index) & cells.batch.isin(good_batches)].copy()
            n_target = eligible.groupby('target').size()
            valid_targets = n_target.index[n_target >= config['minimum_target_cells']]
            for target, group in cells.loc[~cells.is_NTC].groupby('target', dropna=False):
                if target not in valid_targets:
                    excluded.append({'panel': panel, 'target': str(target), 'original_cells': len(group),
                                     'supported_cells': int(n_target.get(target, 0)),
                                     'reason': 'unmapped_target' if target not in gene_index else 'insufficient_common_support'})
            eligible = eligible.loc[eligible.target.isin(valid_targets)]
            selected = []
            for target, group in eligible.groupby('target', sort=True):
                chosen = balanced_selection(group, config['maximum_cells_per_target'], rng)
                selected.append(chosen)
                constructs = group.construct.nunique()
                for (construct, batch), layer in group.groupby(['construct', 'batch'], sort=True):
                    kept = chosen.loc[chosen.construct.eq(construct) & chosen.batch.eq(batch)]
                    mass = 1.0 / constructs / group.loc[group.construct.eq(construct), 'batch'].nunique()
                    factors.append({'panel': panel, 'context': layer.context.iloc[0], 'target': target,
                                    'construct': construct, 'batch': batch, 'source_cells': len(layer),
                                    'cached_cells': len(kept), 'balanced_mass': mass,
                                    'original_mass': len(layer) / len(group), 'cell_training_mass': mass / len(kept),
                                    'biological_replicate_identified': False})
            if panel not in ['H1:Validation', 'H1:Test']:
                for _, group in controls.loc[controls.batch.isin(good_batches)].groupby(['batch', 'half'], sort=True):
                    indices = rng.choice(group.index, min(len(group), config['maximum_controls_per_half_batch']), replace=False)
                    selected.append(group.loc[indices])
            chosen = pd.concat(selected).sort_values('row_index').reset_index(drop=True)
            chosen['cache_row'] = np.arange(len(chosen))
            chosen['panel_index'] = number
            chosen['reserved'] = [False if t == '__NTC__' else reserved(t, config['unseen_target_percent']) for t in chosen.target]
            counts = np.lib.format.open_memmap(directory / 'counts.npy', mode='w+', dtype=np.uint16,
                                               shape=(len(chosen), len(destination)))
            totals = np.empty(len(chosen), np.float32)
            rows = chosen.row_index.to_numpy()
            for start, block in raw.blocks(chunk=1024):
                lo, hi = np.searchsorted(rows, [start, start + block.shape[0]])
                if lo == hi:
                    continue
                data = block[rows[lo:hi] - start]
                assert np.all(np.isfinite(data.data)) and np.all(data.data >= 0) and np.all(data.data == np.floor(data.data))
                assert data.data.max(initial=0) <= np.iinfo(np.uint16).max, 'uint16_cache_would_overflow'
                totals[lo:hi] = np.asarray(data.sum(1)).ravel()
                counts[lo:hi] = data[:, positions].toarray().astype(np.uint16)
            counts.flush()
            del counts
            chosen['source_total_counts'] = totals
            chosen.to_parquet(directory / 'cells.parquet', index=False)
            panels.append({'panel': panel, 'index': number, 'context': chosen.context.iloc[0],
                           'source_metadata': source_report['source_metadata'], 'selected_cells': len(chosen),
                           'genes': len(destination), 'raw_input': str(source.relative_to(ROOT)),
                           'raw_sha256': expected, 'eligible_batches': len(good_batches),
                           'excluded_batches': len(ntc_support) - len(good_batches)})
            print(json.dumps({'stage': 'counts_cached', **panels[-1]}), flush=True)
        release_read_cache(source)
    pd.DataFrame(factors).to_parquet(cache / 'factor-weights.parquet', index=False)
    pd.DataFrame(excluded).to_parquet(cache / 'excluded.parquet', index=False)
    write_json(cache / 'genes.json', genes)
    write_json(cache / 'inputs.json', inputs)
    hashes = {str(p.relative_to(cache)): hash_file(p) for p in sorted(cache.rglob('*')) if p.is_file()}
    write_json(completion, {'data_spec_hash': data_spec_hash(config), 'panels': panels, 'genes': len(genes),
                            'artifacts': hashes, 'inputs': inputs})
    return cache


def data_spec_hash(config):
    fields = ['collection', 'gene_axis', 'panels', 'data_seed', 'control_split_seed', 'minimum_target_cells',
              'minimum_controls_per_half_batch', 'maximum_cells_per_target', 'maximum_controls_per_half_batch', 'unseen_target_percent']
    return value_hash({k: config[k] for k in fields})


def verify_cache(cache, report):
    for name, expected in report['artifacts'].items():
        assert hash_file(cache / name) == expected, 'cache_changed: ' + name
    for item in report['inputs']:
        assert hash_file(ROOT / item['path']) == item['sha256'], 'source_changed: ' + item['path']


class CountData:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.report = json.loads((self.directory / 'complete.json').read_text())
        self.genes = json.loads((self.directory / 'genes.json').read_text())
        self.gene_index = {g: i for i, g in enumerate(self.genes)}
        self.counts, self.axes, frames, self.masks, self.panel_context = {}, {}, [], {}, {}
        for panel in self.report['panels']:
            i = panel['index']; folder = self.directory / f'panel-{i}'
            self.counts[i] = np.load(folder / 'counts.npy', mmap_mode='r')
            self.axes[i] = np.load(folder / 'genes.npy')
            self.panel_context[i] = panel['context']
            cells = pd.read_parquet(folder / 'cells.parquet')
            frames.append(cells)
            mask = np.zeros(len(self.genes), bool); mask[self.axes[i]] = True
            if panel['context'] in self.masks:
                self.masks[panel['context']] &= mask
            else:
                self.masks[panel['context']] = mask
        self.cells = pd.concat(frames, ignore_index=True)
        self.common = np.logical_and.reduce(list(self.masks.values()))
        self.readout_support = []
        for panel in self.report['panels']:
            usable = self.masks[panel['context']]
            unsupported = [self.genes[g] for g in self.axes[panel['index']] if not usable[g]]
            self.readout_support.append({'panel': panel['panel'], 'context': panel['context'],
                                         'safe_cached_genes': len(self.axes[panel['index']]),
                                         'common_safe_context_genes': int(usable.sum()),
                                         'unmatched_safe_readouts_excluded': unsupported,
                                         'reason': 'intersection across source panels and feature NTC; excluded is not measured zero'})
        self.factors = pd.read_parquet(self.directory / 'factor-weights.parquet')
        self.weights = {}
        if len(self.factors):
            columns = ['context', 'target', 'construct', 'batch', 'balanced_mass', 'original_mass']
            for context, target, construct, batch, balanced, original in self.factors[columns].itertuples(index=False, name=None):
                pair = (construct, batch)
                maps = self.weights.setdefault((context, target), ({}, {}))
                assert pair not in maps[0], 'duplicated_factor_layer'
                maps[0][pair], maps[1][pair] = float(balanced), float(original)
        self.panel_indices = self.cells.panel_index.to_numpy(np.int16)
        self.cache_rows = self.cells.cache_row.to_numpy(np.int64)
        observed = self.cells.loc[~self.cells.is_NTC, ['context', 'target', 'construct', 'batch']]
        row_ids = observed.index.to_numpy()
        self.tasks = {key: row_ids[pos] for key, pos in observed.groupby(['context', 'target'], sort=True).indices.items()}
        controls = self.cells.loc[self.cells.is_NTC, ['context', 'batch', 'half']]
        control_ids = controls.index.to_numpy()
        self.controls = {(c, b, int(h)): control_ids[pos] for (c, b, h), pos in
                         controls.groupby(['context', 'batch', 'half'], sort=True).indices.items()}
        self.groups = {key: {} for key in self.tasks}
        for (context, target, construct, batch), pos in observed.groupby(['context', 'target', 'construct', 'batch'], sort=True).indices.items():
            self.groups[(context, target)][(construct, batch)] = row_ids[pos]

    def read(self, indices):
        indices = np.asarray(indices, dtype=np.int64)
        panels = self.panel_indices[indices]
        result = np.zeros((len(indices), len(self.genes)), np.float32)
        for panel in np.unique(panels):
            positions = np.flatnonzero(panels == panel)
            local = self.cache_rows[indices[positions]]
            measured = self.masks[self.panel_context[panel]][self.axes[panel]]
            result[np.ix_(positions, self.axes[panel])] = self.counts[panel][local] * measured
        return result

    def control(self, context, batch, half):
        return self.read(self.controls[(context, batch, half)])

    def task_weights(self, context, target, original=False):
        values = self.weights[(context, target)][int(original)]
        assert abs(sum(values.values()) - 1) < 1e-6
        return values


def logcp(counts, mask=None):
    counts = np.asarray(counts, np.float32)
    if mask is not None:
        counts = counts * mask
    return np.log1p(counts * (10000.0 / np.maximum(counts.sum(-1, keepdims=True), 1)))


class BalancedSampler:
    """Serializable per-background queues; each draw balances context/target/construct/batch."""
    def __init__(self, data, contexts, seed, reserved_percent):
        self.data, self.contexts = data, list(contexts)
        self.rng = np.random.default_rng(seed)
        self.targets = {c: sorted(t for cc, t in data.tasks if cc == c and not reserved(t, reserved_percent)) for c in contexts}
        assert all(self.targets.values())
        self.queues = {c: [] for c in contexts}
        self.seen = {c: set() for c in contexts}
        self.context_queue = []

    def sample(self, tasks, cells):
        if not self.context_queue:
            self.context_queue = list(self.rng.permutation(self.contexts))
        context = self.context_queue.pop()
        result = []
        for _ in range(tasks):
            if not self.queues[context]:
                self.queues[context] = list(self.rng.permutation(self.targets[context]))
            target = self.queues[context].pop(); self.seen[context].add(target)
            groups = self.data.groups[(context, target)]
            construct = self.rng.choice(sorted({g for g, b in groups}))
            batch = self.rng.choice(sorted(b for g, b in groups if g == construct))
            indices = self.rng.choice(groups[(construct, batch)], cells, replace=True)
            result.append((context, target, batch, indices))
        return result

    def state_dict(self):
        return {'rng': self.rng.bit_generator.state, 'queues': self.queues, 'seen': self.seen, 'context_queue': self.context_queue}

    def load_state_dict(self, state):
        self.rng.bit_generator.state = state['rng']; self.queues = state['queues']
        self.seen = state['seen']; self.context_queue = state['context_queue']
