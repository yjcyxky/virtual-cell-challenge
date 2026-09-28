"""Response evidence from other training contexts; held-out labels never enter features."""
import gc
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from common import digest,event,write_json


TRAINING_CONTEXTS=('HepG2','Jurkat','K562','RPE1')

def prepare_de(data,config,output,reference):
    """Use fixed real DE tables; compute missing RPE1 from fixed evaluation cells."""
    directory=output/'cache/response-evidence';directory.mkdir(exist_ok=True)
    for context in TRAINING_CONTEXTS:
        stats=data.contexts[context];path=directory/f'{context}.npz';stamp=directory/f'{context}.json'
        if stamp.exists():
            if digest(path)!=json.loads(stamp.read_text())['sha256']:raise ValueError('de_evidence_cache_changed')
            continue
        if context in reference['de_sources']:
            source=reference['de_sources'][context];table=Path(source['path'])
            if digest(table)!=source['sha256']:raise ValueError('source_de_changed')
            provenance=source
        else:
            from cell_eval2.de_compute import compute_de
            import anndata as ad
            from data import NTC
            table=directory/f'{context}-computed.parquet'
            if not table.exists():
                ids=np.load(stats['folder']/'evaluation-rows.npy');cells=data.cells(context)
                counts=data.counts(context)
                blocks=[sparse.csr_matrix(counts[ids[i:i+512]]) for i in range(0,len(ids),512)]
                matrix=sparse.vstack(blocks,format='csr');del blocks,counts;gc.collect()
                obs=pd.DataFrame({'target':cells.loc[ids,'target'].to_numpy()},index=[f'{context}-{i}' for i in ids])
                real=ad.AnnData(matrix,obs=obs,var=pd.DataFrame(index=np.asarray(data.genes)[stats['gene_indices']]))
                event('source_de_start',context=context,cells=real.n_obs,genes=real.n_vars)
                result=compute_de(real,backend=config['official_de_backend'],groupby='target',reference=NTC,
                                  mean_calc='arithmetic',epsilon=config['lfc_epsilon'],input_type='counts',
                                  target_sum=1e6,clip_value=None,filter_gene_min_cpm_cell=5.,
                                  fdr_scope='per_pert',threads=config['threads'],device=config['official_device'])
                result.write_parquet(table);del result,real,matrix;gc.collect()
            provenance={'computed_from':str(stats['folder']),'sha256':digest(table),
                        'method':'pinned cell-eval2 gpudge, arithmetic CPM, per-pert BH, NTC CPM>5, fixed <=400 cells'}
        targets=list(stats['targets'].astype(str));genes=list(np.asarray(data.genes)[stats['gene_indices']])
        shape=(len(targets),len(genes));sign=np.zeros(shape,np.int8);evidence=np.zeros(shape,np.float32)
        tested=np.zeros(shape,bool);cells=np.zeros(len(targets),np.int32)
        for batch in pq.ParquetFile(table).iter_batches(batch_size=250000,columns=['target','feature','p_adj','log2_fold_change','target_ncells']):
            frame=batch.to_pandas();ti=pd.Categorical(frame.target,categories=targets).codes;gi=pd.Categorical(frame.feature,categories=genes).codes
            if (ti<0).any() or (gi<0).any():raise ValueError('de_axis_mismatch')
            q=frame.p_adj.to_numpy();lfc=frame.log2_fold_change.to_numpy();valid=np.isfinite(q)&np.isfinite(lfc)&frame.target.ne(frame.feature).to_numpy()
            tested[ti,gi]=valid
            sign[ti,gi]=np.where(valid&(q<.05),np.sign(lfc),0).astype(np.int8)
            evidence[ti,gi]=np.where(valid,np.minimum(12.,-np.log10(np.maximum(q,1e-12))),0)
            cells[ti]=frame.target_ncells.to_numpy()
        np.savez_compressed(path,sign=sign,evidence=evidence,tested=tested,cells=cells)
        write_json(stamp,{'sha256':digest(path),'source':provenance,'targets':len(targets),'genes':len(genes)})
        if context not in reference['de_sources']:table.unlink()
        event('source_de_ready',context=context,significant=int(np.count_nonzero(sign)))
    result={}
    for context in TRAINING_CONTEXTS:
        with np.load(directory/f'{context}.npz') as record:
            result[context]={k:record[k] for k in record.files}
    return result


def module_basis(pathways,config):
    """Fixed pathway coordinates; no response labels determine the coordinate system."""
    size=np.asarray(pathways.sum(0)).ravel()
    keep=(size>=config['module_min_genes'])&(size<=config['module_max_genes'])
    matrix=pathways[:,keep]@sparse.diags(1/np.sqrt(size[keep]))
    dimensions=config['module_dimensions']
    if min(matrix.shape)<=dimensions:raise ValueError('insufficient_pathway_module_rank')
    basis=TruncatedSVD(n_components=dimensions,n_iter=7,random_state=config['seed']).fit_transform(matrix).astype(np.float32)
    # Unit coordinates with deterministic component signs.
    basis/=np.maximum(np.linalg.norm(basis,axis=0),1e-12)
    for j in range(dimensions):
        basis[:,j]*=np.sign(basis[np.argmax(np.abs(basis[:,j])),j])
    return basis


class ResponseEvidence:
    def __init__(self,data,mode,de=None,basis=None,training_contexts=TRAINING_CONTEXTS):
        if mode not in ('continuous','deg','modules'):raise ValueError('unknown_feature_mode')
        if 'H1' in training_contexts:raise ValueError('H1_response_features_forbidden')
        self.data,self.mode,self.de,self.basis=data,mode,de,basis
        self.sources=tuple(sorted(training_contexts));self.axes={};self.programs={}
        for context in self.sources:
            stats=data.contexts[context]
            pi=np.full(len(data.genes),-1,np.int32);gi=pi.copy()
            pi[[data.lookup[str(t)] for t in stats['targets']]]=np.arange(len(stats['targets']))
            gi[stats['gene_indices']]=np.arange(len(stats['gene_indices']))
            self.axes[context]=(pi,gi)
            if mode=='modules':
                sign=de[context]['sign'];coords=basis[stats['gene_indices']];program=[]
                for direction in [1,-1]:
                    mask=sparse.csr_matrix(sign==direction,dtype=np.float32);count=np.asarray(mask.sum(1)).ravel()
                    projection=np.asarray(mask@coords)/np.maximum(np.sqrt(count[:,None]),1.)
                    program.extend([projection,np.log1p(count[:,None])])
                self.programs[context]=np.concatenate(program,axis=1).astype(np.float32)

    def matrix(self,context,p,g):
        p=np.asarray(p,np.int32);g=np.asarray(g,np.int32);n=len(p)
        count=np.zeros(n,np.float32);target_count=count.copy();total=count.copy();squares=count.copy();absolute=count.copy()
        low=np.full(n,np.inf,np.float32);high=np.full(n,-np.inf,np.float32)
        has_de=self.mode!='continuous'
        if has_de:
            tested=count.copy();up=count.copy();down=count.copy();evidence=count.copy();max_evidence=count.copy();ncells=count.copy()
        if self.mode=='modules':program=np.zeros((n,2*self.basis.shape[1]+2),np.float32)
        for source in self.sources:
            if source==context:continue
            pi,gi=self.axes[source];rows=pi[p];columns=gi[g];target_ok=rows>=0;valid=target_ok&(columns>=0)
            target_count+=target_ok
            ids=np.flatnonzero(valid);rr=rows[ids];cc=columns[ids]
            values=self.data.contexts[source]['response'][rr,cc]
            count[ids]+=1;total[ids]+=values;squares[ids]+=values*values;absolute[ids]+=np.abs(values)
            low[ids]=np.minimum(low[ids],values);high[ids]=np.maximum(high[ids],values)
            if has_de:
                record=self.de[source];valid_de=record['tested'][rr,cc];di=ids[valid_de];dr=rr[valid_de];dc=cc[valid_de]
                signs=record['sign'][dr,dc];strength=record['evidence'][dr,dc]
                tested[di]+=1;up[di]+=signs>0;down[di]+=signs<0
                evidence[di]+=strength;max_evidence[di]=np.maximum(max_evidence[di],strength)
                ncells[di]+=record['cells'][dr]
            if self.mode=='modules':program[target_ok]+=self.programs[source][rows[target_ok]]
        denominator=np.maximum(count,1);mean=total/denominator
        columns=[mean,np.sqrt(np.maximum(squares/denominator-mean*mean,0)),
                 np.where(count>0,low,0),np.where(count>0,high,0),absolute/denominator,count,target_count]
        if has_de:
            denom=np.maximum(tested,1)
            columns.extend([tested,up,down,up/denom,down/denom,(up-down)/denom,
                            ((up>0)&(down>0)).astype(np.float32),evidence/denom,max_evidence,np.log1p(ncells/denom)])
        blocks=[np.column_stack(columns).astype(np.float32)]
        if self.mode=='modules':
            program/=np.maximum(target_count[:,None],1)
            d=self.basis.shape[1];readout=self.basis[g]
            blocks.extend([program,readout,program[:,:d]*readout,program[:,d+1:2*d+1]*readout])
        result=np.concatenate(blocks,axis=1)
        if not np.isfinite(result).all():raise ValueError('nonfinite_response_feature')
        return result


class AugmentedFeatures:
    def __init__(self,base,evidence):self.base,self.evidence,self.data=base,evidence,base.data
    def matrix(self,context,target,readout,bag=0):
        return np.concatenate([self.base.matrix(context,target,readout,bag),self.evidence.matrix(context,target,readout)],axis=1)


def prepare_response_features(data,base,config,output,reference):
    mode=config['feature_mode'];de=prepare_de(data,config,output,reference) if mode!='continuous' else None
    basis=None
    if mode=='modules':
        path=output/'cache/module-basis.npy'
        if path.exists():basis=np.load(path)
        else:
            basis=module_basis(base.pathways,config);np.save(path,basis)
    features=AugmentedFeatures(base,ResponseEvidence(data,mode,de,basis))
    manifest={'mode':mode,'source_contexts':list(TRAINING_CONTEXTS),
              'exclusion':'Each recipient context is excluded from all of its response evidence; H1 never a source.',
              'continuous_statistics':'Full-cell source log2FC: mean/std/min/max/mean_abs/pair support/target support.',
              'de_statistics':'Fixed <=400 source cells, NTC CPM>5, BH<.05: tested/up/down counts, fractions, net direction, conflicts, bounded -log10 q and cell support.',
              'modules':'Fixed Reactome SVD coordinates of source up/down signatures; no label-fitted basis.',
              'feature_dimensions':int(features.matrix('H1',[0],[0]).shape[1]),
              'basis_sha256':digest(output/'cache/module-basis.npy') if basis is not None else None}
    path=output/'cache/response-features.json'
    if path.exists() and json.loads(path.read_text())!=manifest:raise ValueError('response_feature_identity_changed')
    write_json(path,manifest);event('response_features_ready',**manifest)
    return features
