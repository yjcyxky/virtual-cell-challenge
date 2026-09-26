"""Whole-context, full-cell training statistics and immutable evaluation populations."""
from __future__ import annotations
from collections import Counter, defaultdict
from functools import lru_cache
import json
import sys
import h5py
import numpy as np
import pandas as pd
from scipy import sparse
from common import ROOT, digest, event, log_profile, seed, stratified_sample, write_json

sys.path.insert(0, str(ROOT / 'scripts/dossier'))
from rna import read_frame, read_array

PANELS = {
    'H1': ['arc_vcc2025_h1/adata_Training.h5ad', 'arc_vcc2025_h1/adata_Validation.h5ad', 'arc_vcc2025_h1/adata_Test.h5ad'],
    'K562': ['replogle2022/K562_gwps_raw_singlecell_01.h5ad'],
    'RPE1': ['replogle2022/rpe1_raw_singlecell_01.h5ad'],
    'HepG2': ['nadig2025/GSE264667_hepg2_raw_singlecell_01.h5ad'],
    'Jurkat': ['nadig2025/GSE264667_jurkat_raw_singlecell_01.h5ad'],
    'A': ['arc_vcc2026_controls/context_A.h5ad'],
    'B': ['arc_vcc2026_controls/context_B.h5ad'],
    'C': ['arc_vcc2026_controls/context_C.h5ad'],
}
NTC = 'non-targeting'

class GeneIdentity:
    def __init__(self):
        self.table = pd.read_csv(ROOT / 'data/raw/networks/hgnc_complete_set.txt', sep='\t', low_memory=False)
        self.table = self.table.loc[self.table.status.eq('Approved')]
        self.approved = set(self.table.symbol)
        self.aliases = defaultdict(set)
        for row in self.table.to_dict('records'):
            for field in ['symbol', 'alias_symbol', 'prev_symbol', 'ensembl_gene_id']:
                for item in str(row.get(field, '')).split('|'):
                    if item and item != 'nan': self.aliases[item].add(row['symbol'])
        self.aliases['ENSG00000215271'].add('HOMEZ')

    def canonical(self, value):
        value = str(value).strip()
        # The challenge axis contains BOTH TIAF1 and MYO18A. Modern HGNC aliases
        # merge them, but collapsing two independently supplied columns is invalid.
        # Keep the literal historical task slot; conflicting source ENSG rows are
        # excluded rather than silently reassigned to the other column.
        if value == 'TIAF1': return value
        if value in self.approved: return value
        if value.startswith('ENSG'): value = value.split('.')[0]
        choices = self.aliases.get(value, set())
        return next(iter(choices)) if len(choices) == 1 else None

def dense_block(handle, start, stop):
    x = handle['X']
    if isinstance(x, h5py.Dataset): return np.asarray(x[start:stop])
    if x.attrs.get('encoding-type') != 'csr_matrix': raise ValueError('source_must_be_dense_or_csr')
    ptr = x['indptr'][start:stop + 1]
    lo, hi = int(ptr[0]), int(ptr[-1])
    return sparse.csr_matrix((x['data'][lo:hi], x['indices'][lo:hi], ptr - lo),
                             shape=(stop-start, int(x.attrs['shape'][1]))).toarray()

def inspect_panel(path, context, identity):
    with h5py.File(path) as h:
        obs, var = read_frame(h['obs']), read_frame(h['var'])
    official = context in ['A', 'B', 'C']
    is_h1 = context == 'H1'
    targets = obs['target_gene' if is_h1 or official else 'gene'].astype(str)
    canonical = targets.map(lambda t: NTC if t == NTC else identity.canonical(t))
    guide = obs['ntc_id' if official else 'guide_id' if is_h1 else 'sgID_AB'].astype(str)
    batch = pd.Series('official', index=obs.index) if official else obs['batch' if is_h1 else 'gem_group'].astype(str)
    frame = pd.DataFrame({'target': canonical.to_numpy(), 'original_target': targets.to_numpy(),
                          'guide': guide.to_numpy(), 'batch': batch.to_numpy(),
                          'barcode': obs.index.astype(str), 'source_row': np.arange(len(obs))})
    names = var.index.astype(str).to_numpy() if is_h1 or official else var.gene_name.astype(str).to_numpy()
    ids = var.gene_id.astype(str).to_numpy() if 'gene_id' in var else var.index.astype(str).to_numpy()
    resolved = [identity.canonical(g) for g in names]
    counts = Counter(g for g in resolved if g is not None)
    rows = []
    for i, (name, ensg, gene) in enumerate(zip(names, ids, resolved)):
        via_id = identity.canonical(ensg)
        conflict = gene is not None and via_id is not None and gene != via_id
        # Literal official genes always remain available for baseline-conditioned inference.
        if official and gene is None: gene = name
        valid = gene is not None and counts.get(gene, 1) == 1 and not conflict
        if official: valid = gene is not None and counts.get(gene, 1) == 1
        rows.append({'source_position': i, 'source_gene': name, 'ensembl': ensg,
                     'gene': gene, 'valid': valid, 'id_conflict': conflict})
    return frame, pd.DataFrame(rows)

def prepare(config, output):
    directory = output / 'cache/data'; directory.mkdir(parents=True, exist_ok=True)
    done = directory / 'complete.json'
    if done.exists(): return Data(directory)
    sources = []
    for relative in ['networks/hgnc_complete_set.txt', 'arc_vcc2026_controls/gene_names.csv', 'arc_vcc2026_controls/pert_counts.csv']:
        path = ROOT/'data/raw'/relative
        manifest = json.loads((path.parent/'SOURCE.json').read_text())
        entry = next(item for item in manifest['files'] if item['name'] == path.name)
        actual = digest(path)
        if actual != entry['sha256']: raise ValueError(f'input_version_mismatch:{relative}')
        sources.append({'path':str(path.resolve()), 'sha256':actual})
    identity = GeneIdentity()
    panels = {}; gene_set = set()
    for context, names in PANELS.items():
        panels[context] = []
        for name in names:
            path = ROOT / 'data/raw' / name
            source = json.loads((path.parent / 'SOURCE.json').read_text())
            entry = next(x for x in source['files'] if x['name'] == path.name)
            event('input_validation', context=context, file=name)
            actual = digest(path)
            if actual != entry['sha256']: raise ValueError(f'input_version_mismatch:{name}')
            sources.append({'path': str(path.resolve()), 'sha256': actual})
            obs, mapping = inspect_panel(path, context, identity)
            gene_set.update(mapping.loc[mapping.valid, 'gene'])
            gene_set.update(t for t in obs.target.dropna() if t != NTC)
            panels[context].append((path, obs, mapping))
    official_genes = pd.read_csv(ROOT/'data/raw/arc_vcc2026_controls/gene_names.csv').iloc[:,0].astype(str).tolist()
    official_internal = [identity.canonical(g) or g for g in official_genes]
    if len(set(official_internal)) != len(official_internal): raise ValueError('official_axis_canonical_collision')
    genes = official_internal + sorted(gene_set - set(official_internal))
    lookup = {g:i for i,g in enumerate(genes)}
    for context in PANELS:
        prepare_context(context, panels[context], genes, lookup, config, directory)
    report = {'genes': genes, 'official_genes': official_genes, 'official_internal': official_internal,
              'official_targets': pd.read_csv(ROOT/'data/raw/arc_vcc2026_controls/pert_counts.csv').target_gene.tolist(),
              'sources': sources, 'contexts': list(PANELS), 'data_seed': config['seed'],
              'protocol': 'whole contexts; all NTC; all admitted cells for supervision; fixed <=400 real cells per target for evaluation; no target intersection filter'}
    write_json(done, report)
    return Data(directory)

def prepare_context(context, panels, genes, lookup, config, directory):
    folder = directory / context; folder.mkdir(exist_ok=True)
    if (folder/'complete.json').exists(): return
    native = sorted(set.intersection(*[set(mapping.loc[mapping.valid,'gene']) for _,_,mapping in panels]))
    frames = []
    ntc_ids = None
    for number, (path, obs, mapping) in enumerate(panels):
        mapping.to_parquet(folder/f'gene-mapping-{number}.parquet', index=False)
        obs = obs.copy(); obs['panel'] = number
        if context == 'H1':
            controls = obs.loc[obs.target.eq(NTC)]
            ids = set(zip(controls.batch, controls.barcode))
            if ntc_ids is None: ntc_ids = ids
            elif ids != ntc_ids: raise ValueError('H1_duplicate_NTC_identity_changed')
            if number: obs = obs.loc[~obs.target.eq(NTC)]
        frames.append(obs)
    if context == 'H1':
        verify_duplicate_controls(panels, native)
    frame = pd.concat(frames, ignore_index=True)
    counts = frame.target.value_counts()
    targets = sorted(t for t,n in counts.items() if t != NTC and n >= config['minimum_target_cells'])
    frame.loc[~frame.target.isin(targets+[NTC])].to_parquet(folder/'excluded-cells.parquet', index=False)
    frame = frame.loc[frame.target.isin(targets+[NTC])].reset_index(drop=True)
    if not frame.target.eq(NTC).any(): raise ValueError('no_context_NTC')
    shape = (len(frame), len(native))
    matrix = np.lib.format.open_memmap(folder/'counts.npy', mode='w+', dtype=np.uint16, shape=shape)
    target_index = {t:i for i,t in enumerate(targets + [NTC])}
    sums = np.zeros((len(target_index), len(native)), dtype=np.float64)
    target_codes = frame.target.map(target_index).to_numpy()
    for number, (path, _, mapping) in enumerate(panels):
        by_gene = mapping.loc[mapping.valid].set_index('gene').source_position.to_dict()
        positions = np.array([by_gene[g] for g in native])
        destination = np.flatnonzero(frame.panel.eq(number))
        rows = frame.loc[destination,'source_row'].to_numpy()
        with h5py.File(path) as h:
            total = len(read_array(h['obs'][h['obs'].attrs.get('_index', '_index')]))
            for start in range(0, total, 512):
                lo, hi = np.searchsorted(rows,[start, min(start+512,total)])
                if lo == hi: continue
                values = dense_block(h, start, min(start+512,total))[rows[lo:hi]-start][:,positions]
                if not np.isfinite(values).all() or (values<0).any() or not np.array_equal(values,np.floor(values)):
                    raise ValueError('raw_integer_counts_required')
                if values.max(initial=0)>65535: raise ValueError('source_count_exceeds_uint16')
                if np.any(values.sum(1)<=0): raise ValueError('empty_cell_after_gene_identity_filter')
                dst = destination[lo:hi]; matrix[dst] = values.astype(np.uint16)
                np.add.at(sums, target_codes[dst], values)
                if start % (512*100) == 0: event('prepare_cells', context=context, source=number, row=start, total=total)
    matrix.flush()
    frame.to_parquet(folder/'cells.parquet', index=False)
    controls = np.flatnonzero(frame.target.eq(NTC))
    baseline = log_profile(sums[-1])
    response = log_profile(sums[:-1]) - baseline if targets else np.empty((0,len(native)),dtype=np.float32)
    np.savez(folder/'statistics.npz', baseline=baseline, response=response, sums=sums,
             targets=np.asarray(targets), gene_indices=np.array([lookup[g] for g in native]),
             control_rows=controls, target_cell_counts=np.array([counts[t] for t in targets]))
    # Full-pool baseline remains fixed; only input statistics receive bag augmentation.
    strata = (frame.loc[controls,'batch']+'|'+frame.loc[controls,'guide']).to_numpy()
    bags = [controls[stratified_sample(strata, config['ntc_bag_cells'], np.random.default_rng(seed(config['seed'],context,'bag',i)))]
            for i in range(config['ntc_augmentation_bags'])]
    features = []
    for rows in [controls] + bags:
        total = np.zeros(len(native)); detected = np.zeros(len(native)); squares = np.zeros(len(native))
        for start in range(0,len(rows),512):
            cells = np.asarray(matrix[rows[start:start+512]],dtype=np.float64)
            total += cells.sum(0); detected += (cells>0).sum(0)
            norm = cells / cells.sum(1,keepdims=True)*50000
            squares += (norm**2).sum(0)
        features.append(np.stack([log_profile(total),detected/len(rows),np.log1p(squares/len(rows))]).astype(np.float32))
    np.save(folder/'context-features.npy',np.asarray(features))
    evaluation = list(controls)
    for target in targets:
        rows = np.flatnonzero(frame.target.eq(target))
        strata = (frame.loc[rows,'batch']+'|'+frame.loc[rows,'guide']).to_numpy()
        selected = stratified_sample(strata,min(config['reference_cells_per_target'],len(rows)),np.random.default_rng(seed(config['seed'],context,target,'reference')))
        evaluation.extend(rows[selected])
    np.save(folder/'evaluation-rows.npy',np.asarray(evaluation))
    write_json(folder/'complete.json',{'context':context,'cells':len(frame),'controls':len(controls),
               'targets':len(targets),'genes':len(native),'native_genes':native,'shape':shape,
               'evaluation_cells':len(evaluation),'source_files':[str(p.resolve()) for p,_,_ in panels]})
    del matrix

def verify_duplicate_controls(panels, native):
    """Require exact repeated NTC counts, aligned by batch/barcode and native gene."""
    original = None
    for path, obs, mapping in panels:
        controls = obs.loc[obs.target.eq(NTC)].sort_values(['batch','barcode'])
        by_gene = mapping.loc[mapping.valid].set_index('gene').source_position.to_dict()
        columns = np.array([by_gene[g] for g in native])
        # Controls are modest (38k); read in raw row order to preserve sequential I/O.
        values = np.empty((len(controls), len(native)), dtype=np.uint16)
        order = np.argsort(controls.source_row.to_numpy())
        rows = controls.source_row.to_numpy()[order]
        with h5py.File(path) as handle:
            for start in range(0, len(obs), 512):
                lo, hi = np.searchsorted(rows, [start, start+512])
                if lo == hi: continue
                block = dense_block(handle, start, min(start+512,len(obs)))[rows[lo:hi]-start][:,columns]
                if (block < 0).any() or block.max(initial=0)>65535 or not np.array_equal(block,np.floor(block)):
                    raise ValueError('H1_duplicate_controls_not_uint16_counts')
                values[order[lo:hi]] = block
        if original is None: original = values
        elif not np.array_equal(original,values): raise ValueError('H1_duplicate_NTC_expression_changed')

class Data:
    def __init__(self,directory):
        self.directory = directory
        self.metadata = json.loads((directory/'complete.json').read_text())
        self.genes = self.metadata['genes']; self.lookup = {g:i for i,g in enumerate(self.genes)}
        self.contexts = {}
        for context in self.metadata['contexts']:
            folder = directory/context
            with np.load(folder/'statistics.npz') as z: stats={k:z[k] for k in z.files}
            stats['features'] = np.load(folder/'context-features.npy')
            stats['folder'] = folder
            self.contexts[context]=stats

    def counts(self,context): return np.load(self.contexts[context]['folder']/'counts.npy',mmap_mode='r')
    @lru_cache(maxsize=1)
    def cells(self,context): return pd.read_parquet(self.contexts[context]['folder']/'cells.parquet')
