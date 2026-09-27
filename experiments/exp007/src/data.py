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

class RawReader:
    """Reuse HDF5/CSR read buffers: fresh large allocations stall under UMA pressure."""
    def __init__(self,handle,capacity=512):
        self.x=handle['X'];x=self.x
        dense=isinstance(x,h5py.Dataset)
        if not dense and x.attrs.get('encoding-type')!='csr_matrix':
            raise ValueError('source_must_be_dense_or_csr')
        columns=x.shape[1] if dense else int(x.attrs['shape'][1])
        self.values=np.empty((capacity,columns),dtype=x.dtype if dense else x['data'].dtype)
        if not dense:
            self.data=np.empty(capacity*columns,x['data'].dtype)
            self.indices=np.empty(capacity*columns,x['indices'].dtype)

    def read(self,start,stop):
        x=self.x;out=self.values[:stop-start]
        if isinstance(x,h5py.Dataset):
            x.read_direct(out,np.s_[start:stop,:])
        else:
            ptr=x['indptr'][start:stop+1];lo,hi=int(ptr[0]),int(ptr[-1]);size=hi-lo
            if size:
                x['data'].read_direct(self.data,np.s_[lo:hi],np.s_[:size])
                x['indices'].read_direct(self.indices,np.s_[lo:hi],np.s_[:size])
            sparse.csr_matrix((self.data[:size],self.indices[:size],ptr-lo),shape=out.shape).toarray(out=out)
        return out

class CountBlock:
    """Validate and aggregate integer counts without per-batch large allocations."""
    def __init__(self,columns,dtype,capacity=512):
        self.columns=np.asarray(columns)
        shape=(capacity,len(columns))
        self.canonical=np.empty(shape,dtype)
        self.selected=np.empty(shape,dtype)
        self.ordered=np.empty(shape,dtype)
        self.floor=np.empty(shape,dtype)
        self.mask=np.empty(shape,bool)
        self.counts=np.empty(shape,np.uint16)
        self.profile=np.empty(len(columns),np.float64)
        self.normalized=np.empty(shape,np.float64)

    def select(self,raw,rows):
        if self.columns.min(initial=0)<0 or self.columns.max(initial=0)>=raw.shape[1]:
            raise ValueError('invalid_source_gene_positions')
        np.take(raw,self.columns,axis=1,out=self.canonical[:len(raw)],mode='clip')
        if np.any(rows<0) or np.any(rows>=len(raw)):raise ValueError('invalid_source_cell_positions')
        values=self.selected[:len(rows)]
        np.take(self.canonical[:len(raw)],rows,axis=0,out=values,mode='clip')
        lower,upper=values.min(initial=0),values.max(initial=0)
        if not 0<=lower<=upper<=65535:raise ValueError('source_counts_not_finite_nonnegative_uint16')
        if np.issubdtype(values.dtype,np.floating):
            np.floor(values,out=self.floor[:len(rows)])
            np.equal(values,self.floor[:len(rows)],out=self.mask[:len(rows)])
            if not self.mask[:len(rows)].all():raise ValueError('raw_integer_counts_required')
        if np.any(values.sum(1)<=0):raise ValueError('empty_cell_after_gene_identity_filter')
        np.copyto(self.counts[:len(rows)],values,casting='unsafe')
        return self.counts[:len(rows)]

    def accumulate(self,sums,codes,cpm_sums):
        order=np.argsort(codes,kind='stable')
        groups,starts=np.unique(codes[order],return_index=True)
        ends=np.r_[starts[1:],len(codes)]
        np.take(self.selected[:len(codes)],order,axis=0,out=self.ordered[:len(codes)],mode='clip')
        ordered=self.ordered[:len(codes)]
        np.divide(ordered,ordered.sum(1,dtype=np.float64)[:,None],out=self.normalized[:len(codes)])
        self.normalized[:len(codes)]*=1e6
        for group,lo,hi in zip(groups,starts,ends):
            np.sum(self.ordered[lo:hi],axis=0,dtype=np.float64,out=self.profile)
            np.add(sums[group],self.profile,out=sums[group])
            np.sum(self.normalized[lo:hi],axis=0,out=self.profile)
            np.add(cpm_sums[group],self.profile,out=cpm_sums[group])

def inspect_panel(path, context, identity):
    with h5py.File(path) as h:
        obs, var = read_frame(h['obs']), read_frame(h['var'])
    is_h1 = context == 'H1'
    targets = obs['target_gene' if is_h1 else 'gene'].astype(str)
    canonical = targets.map(lambda t: NTC if t == NTC else identity.canonical(t))
    guide = obs['guide_id' if is_h1 else 'sgID_AB'].astype(str)
    batch = obs['batch' if is_h1 else 'gem_group'].astype(str)
    frame = pd.DataFrame({'target': canonical.to_numpy(), 'original_target': targets.to_numpy(),
                          'guide': guide.to_numpy(), 'batch': batch.to_numpy(),
                          'barcode': obs.index.astype(str), 'source_row': np.arange(len(obs))})
    names = var.index.astype(str).to_numpy() if is_h1 else var.gene_name.astype(str).to_numpy()
    ids = var.gene_id.astype(str).to_numpy() if 'gene_id' in var else var.index.astype(str).to_numpy()
    resolved = [identity.canonical(g) for g in names]
    counts = Counter(g for g in resolved if g is not None)
    rows = []
    for i, (name, ensg, gene) in enumerate(zip(names, ids, resolved)):
        via_id = identity.canonical(ensg)
        conflict = gene is not None and via_id is not None and gene != via_id
        valid = gene is not None and counts.get(gene, 1) == 1 and not conflict
        rows.append({'source_position': i, 'source_gene': name, 'ensembl': ensg,
                     'gene': gene, 'valid': valid, 'id_conflict': conflict})
    return frame, pd.DataFrame(rows)

def prepare(config, output):
    directory = output / 'cache/data'; directory.mkdir(parents=True, exist_ok=True)
    done = directory / 'complete.json'
    if done.exists(): return Data(directory)
    sources = []
    for relative in ['networks/hgnc_complete_set.txt', 'arc_vcc2026_controls/gene_names.csv']:
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
              'sources': sources, 'contexts': list(PANELS), 'data_seed': config['seed'],
              'prediction_target':config['prediction_target'],'lfc_epsilon':config['lfc_epsilon'],
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
    target_index = {t:i for i,t in enumerate(targets + [NTC])}
    sums = np.zeros((len(target_index), len(native)), dtype=np.float64)
    cpm_sums = np.zeros_like(sums)
    target_codes = frame.target.map(target_index).to_numpy()
    # Sequential writes avoid a writable mmap fault for every new output page.
    written=0
    with (folder/'counts.npy').open('wb') as stream:
        np.lib.format.write_array_header_2_0(stream,{'descr':np.dtype(np.uint16).str,'fortran_order':False,'shape':shape})
        for number, (path, _, mapping) in enumerate(panels):
            by_gene = mapping.loc[mapping.valid].set_index('gene').source_position.to_dict()
            positions = np.array([by_gene[g] for g in native])
            destination = np.flatnonzero(frame.panel.eq(number))
            rows = frame.loc[destination,'source_row'].to_numpy()
            with h5py.File(path) as h:
                reader=RawReader(h);block=CountBlock(positions,reader.values.dtype)
                total = len(read_array(h['obs'][h['obs'].attrs.get('_index', '_index')]))
                for start in range(0, total, 512):
                    lo, hi = np.searchsorted(rows,[start, min(start+512,total)])
                    if lo == hi: continue
                    values=block.select(reader.read(start,min(start+512,total)),rows[lo:hi]-start)
                    dst=destination[lo:hi]
                    if dst[0]!=written or dst[-1]!=written+len(dst)-1:raise ValueError('nonsequential_preparation')
                    stream.write(memoryview(values).cast('B'));written+=len(values)
                    block.accumulate(sums,target_codes[dst],cpm_sums)
                    if start % (512*100) == 0: event('prepare_cells', context=context, source=number, row=start, total=total)
    if written!=len(frame):raise ValueError('incomplete_context_counts')
    matrix=np.load(folder/'counts.npy',mmap_mode='r')
    frame.to_parquet(folder/'cells.parquet', index=False)
    controls = np.flatnonzero(frame.target.eq(NTC))
    baseline = log_profile(sums[-1])
    group_sizes=np.bincount(target_codes,minlength=len(target_index))
    mean_cpm=cpm_sums/group_sizes[:,None]
    epsilon=config['lfc_epsilon']
    response=np.log2((mean_cpm[:-1]+epsilon)/(mean_cpm[-1]+epsilon)).astype(np.float32)
    np.savez(folder/'statistics.npz', baseline=baseline, response=response, sums=sums,
             baseline_cpm=mean_cpm[-1],mean_cpm=mean_cpm,
             targets=np.asarray(targets), gene_indices=np.array([lookup[g] for g in native]),
             control_rows=controls, target_cell_counts=np.array([counts[t] for t in targets]))
    # Full-pool baseline remains fixed; only input statistics receive bag augmentation.
    strata = (frame.loc[controls,'batch']+'|'+frame.loc[controls,'guide']).to_numpy()
    bags = [controls[stratified_sample(strata, config['ntc_bag_cells'], np.random.default_rng(seed(config['seed'],context,'bag',i)))]
            for i in range(config['ntc_augmentation_bags'])]
    features = []
    cells=np.empty((512,len(native)),np.float64);norm=np.empty_like(cells)
    mask=np.empty(cells.shape,bool);buffer=np.empty(cells.shape,np.uint16)
    profile=np.empty(len(native),np.float64);depths=np.empty(512,np.float64)
    for rows in [controls] + bags:
        total = np.zeros(len(native)); detected = np.zeros(len(native)); squares = np.zeros(len(native))
        for start in range(0,len(rows),512):
            selected=rows[start:start+512];n=len(selected)
            np.take(matrix,selected,axis=0,out=buffer[:n],mode='clip')
            np.copyto(cells[:n],buffer[:n])
            np.sum(cells[:n],axis=0,out=profile);total+=profile
            np.greater(cells[:n],0,out=mask[:n]);np.sum(mask[:n],axis=0,dtype=np.float64,out=profile);detected+=profile
            np.sum(cells[:n],axis=1,out=depths[:n])
            np.divide(cells[:n],depths[:n,None],out=norm[:n]);np.multiply(norm[:n],50000,out=norm[:n])
            np.square(norm[:n],out=norm[:n]);np.sum(norm[:n],axis=0,out=profile);squares+=profile
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
            reader=RawReader(handle);block=CountBlock(columns,reader.values.dtype)
            for start in range(0, len(obs), 512):
                lo, hi = np.searchsorted(rows, [start, start+512])
                if lo == hi: continue
                values[order[lo:hi]]=block.select(reader.read(start,min(start+512,len(obs))),rows[lo:hi]-start)
        if original is None: original = values
        else:
            for start in range(0,len(values),512):
                n=min(512,len(values)-start)
                np.equal(original[start:start+n],values[start:start+n],out=block.mask[:n])
                if not block.mask[:n].all():raise ValueError('H1_duplicate_NTC_expression_changed')

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
