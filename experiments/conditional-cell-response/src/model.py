"""NTC-conditioned count VAE with a source response offset and joint response loss."""
import runtime
import gc
import json
import numpy as np
import pandas as pd
import anndata as ad
import torch
from torch import nn
from scipy import sparse
from data import ROOT, NTC, write_json, ref
from vcc_mechanism.learning import initialize, optimizer_for, save_state, restore_state, response_loss
from vcc_mechanism.cell_inputs import load_moments, cell_pool
from vcc_mechanism.inputs import release_file_cache


def nb_nll(counts, mean, phi):
    mean=mean.clamp_min(1e-8); theta=phi.reciprocal()
    return -(torch.lgamma(counts+theta)-torch.lgamma(theta)-torch.lgamma(counts+1)
             +theta*(torch.log(theta)-torch.log(theta+mean))
             +counts*(torch.log(mean)-torch.log(theta+mean))).mean()


class CellVAE(nn.Module):
    def __init__(self, base, phi, ntc_log, shared, spec):
        super().__init__()
        for name,array in [('base',base),('phi',phi),('ntc_log',ntc_log),('shared',shared)]:
            self.register_buffer(name,torch.as_tensor(array,dtype=torch.float32))
        self.register_buffer('context_steps',torch.zeros(len(base)-1,dtype=torch.long))
        h,rank,zdim=spec['hidden'],spec['rank'],spec['latent']
        self.embedding=nn.Embedding(len(shared),spec['target_dimensions'],padding_idx=0)
        self.ntc=nn.Sequential(nn.Linear(ntc_log.shape[1],h),nn.SiLU(),nn.Linear(h,spec['ntc_dimensions']))
        condition=spec['ntc_dimensions']+spec['target_dimensions']
        self.posterior=nn.Sequential(nn.Linear(ntc_log.shape[1]+condition,h),nn.SiLU(),nn.Linear(h,2*zdim))
        self.mean_head=nn.Sequential(nn.Linear(condition,h),nn.SiLU(),nn.Linear(h,rank))
        self.residual=nn.Sequential(nn.Linear(zdim+condition,h),nn.SiLU(),nn.Linear(h,rank))
        self.mean_genes=nn.Parameter(torch.randn(base.shape[1],rank)*spec['output_initial_std'])
        self.residual_genes=nn.Parameter(torch.randn(base.shape[1],rank)*spec['output_initial_std'])
        self.log_dispersion=nn.Parameter(torch.zeros(base.shape[1]))
        nn.init.zeros_(self.mean_head[-1].weight); nn.init.zeros_(self.mean_head[-1].bias)
        self.spec=spec

    def condition(self,targets,context):
        state=self.ntc(self.ntc_log[context]/self.spec['ntc_scale'])
        return torch.cat([self.embedding(targets),state.expand(len(targets),-1)],1)

    def posterior_sample(self,common_log,targets,context):
        condition=self.condition(targets,context)
        mu,logvar=self.posterior(torch.cat([(common_log-self.ntc_log[context])/self.spec['ntc_scale'],condition],1)).chunk(2,1)
        logvar=logvar.clamp(self.spec['posterior_logvar_min'],self.spec['posterior_logvar_max'])
        z=mu+torch.exp(.5*logvar)*torch.randn_like(mu)
        kl=.5*(mu.square()+logvar.exp()-1-logvar).sum(1).mean()
        return z,condition,kl

    def decode(self,z,condition,targets,context,positions,libraries,learned_mask=None):
        mean=(self.mean_head(condition) @ self.mean_genes[positions].T)*(targets!=0)[:,None]
        residual=(self.residual(torch.cat([z,condition],1))-self.residual(torch.cat([torch.zeros_like(z),condition],1))) @ self.residual_genes[positions].T
        log_phi=self.log_dispersion[positions].clamp(self.spec['dispersion_log_min'],self.spec['dispersion_log_max'])
        if learned_mask is not None:
            # No untrained free output coordinates may affect generation.
            mean=mean*learned_mask; residual=residual*learned_mask; log_phi=log_phi*learned_mask
        response=self.shared[targets[:,None],positions[None,:]]+mean
        logits=torch.log(self.base[context,positions]+self.spec['pseudocount'])+(response+residual).clamp(-self.spec['log_shift_clip'],self.spec['log_shift_clip'])
        expected=libraries[:,None]*torch.softmax(logits,dim=1)
        phi=(self.phi[context,positions]*log_phi.exp()).clamp(self.spec['dispersion_min'],self.spec['dispersion_max'])
        return expected,phi,response


def encoder_positions(statistics):
    """NTC input support only; does not shrink the likelihood or output gene axes."""
    if list(statistics[-1]['labels']) != [NTC]:
        raise ValueError('Heldout context representation may contain only input NTC')
    common=np.array(sorted(set.intersection(*(set(s['positions']) for s in statistics))),dtype=np.int64)
    if not len(common): raise ValueError('Empty common NTC encoder support')
    return common


def build(output,genes,split,config):
    spec=config['cvae']; contexts=split['training_contexts']+['H1']; n=len(genes)
    statistics=[dict(np.load(output/f'cache/{c}-statistics.npz')) for c in contexts]
    moments=load_moments(config,contexts,statistics)
    targets=[NTC]+sorted(set.union(*(set(s['labels'][1:]) for s in statistics[:-1])))
    target_index={t:i for i,t in enumerate(targets)}
    common=encoder_positions(statistics)
    base=np.zeros((len(contexts),n),dtype=np.float32); phi=np.zeros_like(base)
    ntc_log=np.zeros((len(contexts),len(common)),dtype=np.float32)
    shared=np.zeros((len(targets),n),dtype=np.float32); mass=np.zeros_like(shared)
    training=[]
    for ci,(s,m) in enumerate(zip(statistics,moments,strict=True)):
        pos=s['positions']; base[ci,pos]=m['mean'][0]
        raw_phi=np.maximum(m['variance'][0]-m['mean'][0]*config['data']['target_sum']*m['inverse_library'][0],0)/np.maximum(m['mean'][0]**2,1e-12)
        phi[ci,pos]=np.clip(raw_phi,spec['dispersion_min'],spec['dispersion_max'])
        ntc_log[ci]=s['mean'][0,np.searchsorted(pos,common)]
        if ci==len(contexts)-1: continue
        ids=np.array([target_index[t] for t in s['labels']])
        lfc=np.log((m['mean']+spec['pseudocount'])/(m['mean'][0]+spec['pseudocount'])).astype(np.float32)
        shared[np.ix_(ids[1:],pos)]+=lfc[1:]*s['counts'][1:,None]
        mass[np.ix_(ids[1:],pos)]+=s['counts'][1:,None]
        training.append({'positions':torch.as_tensor(pos,device=spec['device']),
                         'targets':torch.as_tensor(ids,device=spec['device']),
                         'lfc':torch.as_tensor(lfc,device=spec['device']),
                         'common_local':np.searchsorted(pos,common)})
    np.divide(shared,mass,out=shared,where=mass>0)
    union=np.any(mass[1:]>0,axis=0); del mass,moments
    net=CellVAE(base,phi,ntc_log,shared,spec).to(spec['device'])
    return net,training,statistics,targets,union,common


def fit(output,genes,split,config,run):
    spec=config['cvae']; rng=initialize(config)
    net,training,statistics,targets,union,common=build(output,genes,split,config)
    pools,records=cell_pool(output,genes,split['training_contexts'],statistics[:-1],config)
    release_file_cache(output)
    optimizer=optimizer_for(net,spec); checkpoint=output/'checkpoints/cvae.pt'
    step=restore_state(checkpoint,net,optimizer,rng,config) if checkpoint.exists() else 0
    while step<spec['steps']:
        context=int(rng.integers(len(training))); data=training[context]; pool=pools[context]
        labels=rng.integers(1,len(pool['groups']),size=spec['batch_cells'])
        labels[rng.random(spec['batch_cells'])<spec['ntc_fraction']]=0
        rows=np.array([rng.choice(pool['groups'][i]) for i in labels])
        counts=torch.as_tensor(np.asarray(pool['matrix'][rows],dtype=np.float32),device=spec['device'])
        libraries=counts.sum(1)
        if (libraries<=0).any(): raise ValueError('Empty likelihood library')
        common_log=torch.log1p(counts[:,data['common_local']]*config['data']['target_sum']/libraries[:,None])
        ids=data['targets'][labels]
        z,condition,kl=net.posterior_sample(common_log,ids,context)
        expected,phi,response=net.decode(z,condition,ids,context,data['positions'],libraries)
        nll=nb_nll(counts,expected,phi)
        auxiliary=response_loss(response,data['lfc'][labels],spec)
        beta=spec['kl_weight']*min(1,(step+1)/spec['kl_warmup_steps'])
        loss=nll+spec['auxiliary_weight']*auxiliary+beta*kl
        if not torch.isfinite(loss): raise ValueError('Non-finite VAE loss')
        optimizer.zero_grad(set_to_none=True); loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(),spec['gradient_clip']); optimizer.step(); step+=1
        net.context_steps[context]+=1
        if step % spec['checkpoint_every']==0 or step==spec['steps']:
            save_state(checkpoint,net,optimizer,rng,step,config)
            values={'train/step':step,'train/loss':float(loss.detach()),'train/nb_nll':float(nll.detach()),
                    'train/response_auxiliary':float(auxiliary.detach()),'train/kl':float(kl.detach()),'train/kl_weight':beta}
            run.log(values)
            print(f'CVAE step {step}/{spec["steps"]}: NLL={float(nll.detach()):.5f} auxiliary={float(auxiliary.detach()):.5f} KL={float(kl.detach()):.5f}',flush=True)
    audit={'steps_completed':step,'parameters':sum(p.numel() for p in net.parameters()),
           'context_steps':dict(zip(split['training_contexts'],net.context_steps.cpu().tolist())),
           'likelihood_cells':sum(r['cells'] for r in records),'pool_records':records,
           'training_contexts':split['training_contexts'],'training_targets':len(targets)-1,
           'training_measured_union':int(union.sum()),'posterior_encoder_genes':len(common),
           'full_population_response_supervision':True,'checkpoint_selection':'fixed_last',
           'heldout_response_used':False,'inference_latent':'standard normal prior, never posterior',
           'no_longitudinal_cell_pairs':True}
    write_json(output/'cache/fit.json',audit)
    del optimizer,pools,training; gc.collect(); torch.cuda.empty_cache(); release_file_cache(output)
    return checkpoint,dict(network=net.eval(),targets=targets,genes=genes,union=union,spec=spec),audit


def draw(model,ntc,positions,target,seed,n_cells,config):
    net=model['network']; spec=model['spec']; device=net.base.device
    target_index={t:i for i,t in enumerate(model['targets'])}
    if target not in target_index:
        # No learned identity for a globally unseen target: explicit zero-response
        # NTC resampling, never an arbitrary embedding or a false efficacy claim.
        from evaluation import generate_counts
        return generate_counts(ntc,np.zeros(len(positions)),seed,n_cells,config['data']['target_sum'])
    rng=np.random.default_rng(seed)
    rows=rng.integers(0,ntc.shape[0],n_cells)
    libraries=np.asarray(ntc[rows].sum(1)).ravel().astype(np.float32)
    z=torch.as_tensor(rng.normal(size=(n_cells,spec['latent'])),device=device,dtype=torch.float32)
    ids=torch.full((n_cells,),target_index[target],device=device,dtype=torch.long)
    pos=torch.as_tensor(positions,device=device)
    with torch.no_grad():
        expected,phi,_=net.decode(z,net.condition(ids,4),ids,4,pos,torch.as_tensor(libraries,device=device),
                                  torch.as_tensor(model['union'][positions],device=device))
        expected=expected.cpu().numpy().astype(np.float64); phi=phi.cpu().numpy().astype(np.float64)
    rates=rng.gamma(1/phi,expected*phi)
    counts=rng.poisson(rates).astype(np.float32)
    if not np.isfinite(counts).all() or (counts.sum(1)<=0).any() or (counts.sum(1)>1e6).any():
        raise ValueError('Generated count bounds violated')
    return sparse.csr_matrix(counts)


def generate(output,model,genes,split,config):
    directory=output/'predictions'; directory.mkdir(exist_ok=True); manifest=directory/'manifest.json'
    if manifest.exists():
        result=json.loads(manifest.read_text())
        if any(ref(ROOT/r['path'])!=r for r in result['files']): raise ValueError('Prediction changed')
        return result
    ntc=ad.read_h5ad(output/'cache/H1-input.h5ad'); positions=np.load(output/'cache/H1-statistics.npz')['positions']
    targets=split['evaluation_targets']['H1']; blocks=[]
    for i,target in enumerate(targets):
        local=draw(model,ntc.X,positions,target,config['seed']+i,config['generation']['cells_per_target'],config)
        blocks.append(sparse.csr_matrix((local.data,positions[local.indices],local.indptr),shape=(local.shape[0],len(genes))))
    labels=np.repeat(targets,config['generation']['cells_per_target'])
    prediction=ad.AnnData(sparse.vstack(blocks,format='csr'),obs=pd.DataFrame({'target_gene':labels},index=[f'linear-{i}' for i in range(len(labels))]),var=pd.DataFrame(index=genes))
    path=directory/'linear.h5ad'; prediction.write_h5ad(path,compression='lzf')
    result={'files':[ref(path)],'official_gene_axis':config['benchmark']['gene_axis'],'targets':targets,
            'cells_per_target':config['generation']['cells_per_target'],
            'unmeasured_input_positions':np.flatnonzero(~np.isin(np.arange(len(genes)),positions)).tolist(),
            'missing_input_completion':'zero baseline and zero response; not observed truth',
            'inference':'standard normal latent, NTC library resampling, learned Gamma-Poisson decoder'}
    write_json(manifest,result)
    print(f'Generated CVAE counts: {prediction.shape}',flush=True)
    return result


def diagnostic_counts(output,model,ntc,config):
    positions=np.load(output/'cache/H1-statistics.npz')['positions']
    return draw(model,ntc.X,positions,NTC,config['diagnostics']['seed'],config['generation']['cells_per_target'],config)


def predict(*args,**kwargs):
    raise ValueError('Cell VAE uses its explicit generative interface')
