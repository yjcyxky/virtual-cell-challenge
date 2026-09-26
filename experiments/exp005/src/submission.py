"""Official A/B/C prediction export, validation and optional resumable submission."""
import importlib.util
import json
import sys
import anndata as ad
import h5py
import numpy as np
import pandas as pd
from scipy import sparse
from common import ROOT,digest,event,write_json
from data import GeneIdentity
from generation import Generator
from training import predict

def helpers():
    sys.path.insert(0,str(ROOT/'experiments/exp002-response-transfer-validation/src'))
    sys.path.insert(0,str(ROOT/'scripts/dossier'))
    spec=importlib.util.spec_from_file_location('exp002_submission',ROOT/'experiments/exp002-response-transfer-validation/src/vcc_submission.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

def export(data,features,booster,config,output,tracked,submit=False):
    directory=output/'predictions/official';directory.mkdir(parents=True,exist_ok=True)
    completed=directory/'export.json';legacy=helpers()
    official=ROOT/'data/raw/arc_vcc2026_controls'
    genes=data.metadata['official_genes'];targets=data.metadata['official_targets'];contexts=['A','B','C']
    identity=GeneIdentity();internal=[identity.canonical(t) for t in targets]
    if any(t not in data.lookup for t in internal):raise ValueError('official_target_without_gene_features')
    size=config['cells_per_prediction'];rows=len(contexts)*len(targets)*size
    final=directory/'predictions.h5ad'
    if not completed.exists():
        obs=pd.DataFrame({'context':np.repeat(contexts,len(targets)*size),
                          'target_gene':np.tile(np.repeat(targets,size),len(contexts))},
                         index=[f'exp005-{i}' for i in range(rows)])
        for column in obs:obs[column]=pd.Categorical(obs[column])
        temp=directory/'predictions.partial.h5ad'
        ad.AnnData(sparse.csr_matrix((rows,len(genes)),dtype=np.int32),obs=obs,
                   var=pd.DataFrame(index=pd.Index(genes,name='gene_name'))).write_h5ad(temp)
        pointer=0
        with h5py.File(temp,'r+') as h:
            del h['X'];matrix=h.create_group('X')
            matrix.attrs.update({'encoding-type':'csr_matrix','encoding-version':'0.1.0','shape':[rows,len(genes)]})
            for name in ['data','indices']:
                matrix.create_dataset(name,(0,),maxshape=(None,),chunks=(1048576,),dtype='int32',compression='gzip',compression_opts=1)
            matrix.create_dataset('indptr',(rows+1,),dtype='int64')
            for context in contexts:
                stats=data.contexts[context];axis=stats['gene_indices']
                positions={int(g):i for i,g in enumerate(axis)}
                if not all(i in positions for i in range(len(genes))):raise ValueError('official_control_axis_incomplete')
                reorder=np.array([positions[i] for i in range(len(genes))])
                response=predict(booster,features,context,internal,axis)
                np.savez_compressed(directory/f'{context}-response.npz',targets=np.array(targets),response=response[:,reorder])
                generator=Generator(data,context,config)
                for index,(target,canonical) in enumerate(zip(targets,internal)):
                    counts=generator.generate(canonical,response[index])[:,reorder]
                    legacy.append_sparse(matrix,counts.astype(np.int32),pointer);pointer+=size
                    if len(matrix['data'])>4750000000:raise ValueError('official_prediction_density_limit')
                    if index%25==0:event('official_export',context=context,target_index=index)
                del generator
        if pointer!=rows:raise ValueError('incomplete_official_export')
        temp.replace(final)
        audit=legacy.audit_prediction(final,genes,targets,contexts,size,4750000000,1000000)
        write_json(directory/'prediction-audit.json',audit)
        packaged=legacy.package(config,directory,official)
        write_json(completed,{'prediction':{'path':str(final),'sha256':digest(final)},'package':packaged,'audit':audit})
    result=json.loads(completed.read_text())
    if submit:
        result['leaderboard']=legacy.submit_and_score({**config,'model_name':f'exp005-{output.name}',
                    'submission_description':'Nested whole-context XGBoost residual response; NTC-conditioned STRING, Reactome and CollecTRI features.'},directory,tracked)
    return result
