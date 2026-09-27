"""Static biological priors and baseline-only context/readout pair features."""
import importlib.util
import json
import sys
import numpy as np
import pandas as pd
from scipy import sparse
from common import ROOT, digest, write_json
from data import GeneIdentity

def prepare_priors(data,config,output):
    directory=output/'cache/priors'; directory.mkdir(parents=True,exist_ok=True)
    if (directory/'complete.json').exists(): return
    # Reuse the existing fixed network parsing and spectral embedding implementation.
    sys.path.insert(0,str(ROOT/'experiments/exp001-context-pair-xgb/src'))
    spec=importlib.util.spec_from_file_location('exp001_prior_features',ROOT/'experiments/exp001-context-pair-xgb/src/features.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    source=ROOT/'data/raw/networks'
    manifest=json.loads((source/'SOURCE.json').read_text())
    for entry in manifest['files']:
        if digest(source/entry['name']) != entry['sha256']: raise ValueError('prior_source_changed')
    identity=GeneIdentity()
    def aligned_aliases(hgnc,genes):
        index={g:i for i,g in enumerate(genes)}
        return {key:index[value] for key in set(identity.aliases)|set(genes)
                if (value:=identity.canonical(key)) in index}
    module.canonical_map=aligned_aliases
    module.build_priors(source,data.genes,config,directory)
    path=ROOT/'data/raw/collectri/human-interactions.tsv'
    if digest(path)!=config['collectri_sha256']: raise ValueError('collectri_version_mismatch')
    frame=pd.read_csv(path,sep='\t')
    summary={}
    for field,name in [('is_stimulation','activation'),('is_inhibition','repression')]:
        pairs=set()
        for row in frame.loc[frame[field].eq(True)&frame.is_directed.eq(True)].itertuples():
            source=identity.canonical(row.source_genesymbol)
            target=identity.canonical(row.target_genesymbol)
            if source in data.lookup and target in data.lookup:
                pairs.add((data.lookup[source],data.lookup[target]))
        rr,cc=zip(*sorted(pairs)) if pairs else ([],[])
        matrix=sparse.csr_matrix((np.ones(len(rr),np.float32),(rr,cc)),shape=(len(data.genes),len(data.genes)))
        sparse.save_npz(directory/f'{name}.npz',matrix)
        summary[name]=len(pairs)
    summary.update({'collectri_sha256':config['collectri_sha256'],
                    'complexes':'unresolved multi-protein source names are excluded, never split into artificial single-TF edges',
                    'sign_conflicts':'activation and repression retained as separate flags',
                    'priors_contain_no_experiment_response_labels':True})
    write_json(directory/'complete.json',summary)

class Features:
    def __init__(self,data,directory):
        self.data=data
        self.embedding=np.load(directory/'embedding.npy')
        self.functional=sparse.load_npz(directory/'functional.npz')
        self.physical=sparse.load_npz(directory/'physical.npz')
        self.pathways=sparse.load_npz(directory/'pathways.npz')
        self.shared=(self.pathways@self.pathways.T).tocsr()
        self.pathway_degree=np.asarray(self.pathways.sum(1)).ravel()
        self.activation=sparse.load_npz(directory/'activation.npz')
        self.repression=sparse.load_npz(directory/'repression.npz')
        self.tf_degree=np.asarray((self.activation+self.repression).sum(1)).ravel()
        self.degree=np.asarray(self.functional.sum(1)).ravel()
        self.states={};self.global_states={};self.activities={}
        for context,stats in data.contexts.items():
            feature=np.full((len(stats['features']),3,len(data.genes)),np.nan,dtype=np.float32)
            feature[:,:,stats['gene_indices']]=stats['features']
            self.states[context]=feature
            baseline=np.nan_to_num(feature[:,0])
            self.global_states[context]=baseline@self.embedding/max(1,len(stats['gene_indices']))
            self.activities[context]=np.asarray((self.activation-self.repression)@baseline.T).T/np.maximum(1,self.tf_degree)

    def matrix(self,context,target,readout,bag=0):
        target=np.asarray(target,dtype=np.int32);readout=np.asarray(readout,dtype=np.int32)
        state=self.states[context][bag]
        functional=np.asarray(self.functional[target,readout]).ravel()
        physical=np.asarray(self.physical[target,readout]).ravel()
        shared=np.asarray(self.shared[target,readout]).ravel()
        activation=np.asarray(self.activation[target,readout]).ravel()
        repression=np.asarray(self.repression[target,readout]).ravel()
        union=self.pathway_degree[target]+self.pathway_degree[readout]-shared
        pairs=np.column_stack([functional,physical,shared/np.maximum(1,union),activation,repression,
               functional*state[0,target],functional*state[0,readout],activation*state[0,target],
               repression*state[0,target],np.log1p(self.degree[target]),np.log1p(self.degree[readout]),
               self.activities[context][bag,target],self.activities[context][bag,readout],
               (target==readout).astype(np.float32),np.isfinite(state[0,target]),np.isfinite(state[0,readout])])
        return np.concatenate([state[:,target].T,state[:,readout].T,self.embedding[target],self.embedding[readout],
                               pairs,np.broadcast_to(self.global_states[context][bag],(len(target),self.embedding.shape[1]))],axis=1).astype(np.float32)
