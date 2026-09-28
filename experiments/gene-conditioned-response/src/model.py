"""Functional readout decoder with NTC conditioning and source residual anchoring."""
import runtime
import gc
import numpy as np
import torch
from torch import nn
from scipy import sparse
from data import write_json
from representation import pathway_incidence
from vcc_mechanism.learning import initialize, optimizer_for, save_state, restore_state, response_loss


class GeneDecoder(nn.Module):
    def __init__(self, features, baselines, masks, target_positions, contexts, spec):
        super().__init__()
        # Last feature/position row is the explicit missing-annotation fallback.
        self.register_buffer('features', torch.as_tensor(features, dtype=torch.float32))
        self.register_buffer('baselines', torch.as_tensor(baselines, dtype=torch.float32))
        self.register_buffer('masks', torch.as_tensor(masks, dtype=torch.float32))
        self.register_buffer('target_positions', torch.as_tensor(target_positions))
        self.register_buffer('contexts', torch.as_tensor(contexts, dtype=torch.float32))
        h, rank = spec['hidden'], spec['rank']
        self.target = nn.Sequential(nn.Linear(features.shape[1]+contexts.shape[1]+2, h), nn.SiLU(), nn.Linear(h, rank))
        self.gene = nn.Sequential(nn.Linear(features.shape[1]+2, h), nn.SiLU(), nn.Linear(h, rank))
        self.identity = nn.Embedding(len(target_positions), rank, padding_idx=0)
        nn.init.normal_(self.identity.weight, std=spec['target_identity_initial_std'])
        with torch.no_grad():
            self.identity.weight[0].zero_()
        self.scale = spec['response_scale']/np.sqrt(rank)

    def forward(self, targets, genes, context, unseen_position=None):
        tp = self.target_positions[targets] if unseen_position is None else torch.full_like(targets, unseen_position)
        query = self.target(torch.cat([self.features[tp], self.contexts[context].expand(len(tp), -1),
                                      self.baselines[context, tp, None], self.masks[context, tp, None]], dim=1))
        query = query + self.identity(targets)
        key = self.gene(torch.cat([self.features[genes], self.baselines[context, genes, None],
                                  self.masks[context, genes, None]], dim=1))
        gate = targets != 0 if unseen_position is None else 1
        return self.scale*(query*key).sum(1)*gate


def build(output, genes, split, config):
    spec = config['decoder']
    names = split['training_contexts'] + ['H1']
    statistics = [dict(np.load(output/f'cache/{c}-statistics.npz')) for c in names]
    n = len(genes); lookup = {g:i for i,g in enumerate(genes)}
    targets = ['non-targeting'] + sorted(set.union(*(set(s['labels'][1:]) for s in statistics[:-1])))
    target_index = {g:i for i,g in enumerate(targets)}
    baselines = np.zeros((len(names), n+1), dtype=np.float32); masks = np.zeros_like(baselines)
    for i,s in enumerate(statistics):
        baselines[i,s['positions']] = s['mean'][0]/spec['ntc_scale']; masks[i,s['positions']] = 1
    positions, modules, incidence = pathway_incidence(genes, range(n), config['functional'])
    degree = np.asarray(incidence.sum(1)).ravel(); size = np.asarray(incidence.sum(0)).ravel()
    normalized = sparse.diags(1/np.sqrt(degree)) @ incidence @ sparse.diags(1/np.sqrt(size))
    rng = np.random.default_rng(config['functional']['seed'])
    sketch = normalized @ rng.normal(size=(len(modules), spec['functional_dimensions']))
    sketch /= np.maximum(np.linalg.norm(sketch, axis=1, keepdims=True), 1e-12)
    function = np.zeros((n+1,spec['functional_dimensions']),dtype=np.float32); function[positions]=sketch
    annotated = np.zeros((n+1,1),dtype=np.float32); annotated[positions] = 1
    # This transductive gene state includes only permitted input NTC from H1.
    features = np.concatenate([function, annotated, baselines.T, masks.T],axis=1)
    common = sorted(set.intersection(*(set(s['positions']) for s in statistics[:-1])))
    projection = rng.normal(size=(len(common),spec['context_dimensions']))/np.sqrt(len(common))
    contexts = baselines[:,common] @ projection
    target_positions = np.asarray([lookup.get(t,n) for t in targets])
    net = GeneDecoder(features,baselines,masks,target_positions,contexts,spec).to(spec['device'])
    training=[]
    for s in statistics[:-1]:
        training.append({'positions':torch.as_tensor(s['positions'],device=spec['device']),
                         'targets':torch.as_tensor([target_index[t] for t in s['labels'][1:]],device=spec['device']),
                         'delta':torch.as_tensor(s['mean'][1:]-s['mean'][0],device=spec['device']),
                         'counts':s['counts'][1:], 'labels':s['labels'][1:]})
    union=set.union(*(set(s['positions']) for s in statistics[:-1]))
    audit={'functional_genes':len(positions),'functional_modules':len(modules),'functional_edges':incidence.nnz,
           'functional_genes_without_training_measurement':len(set(positions)-union),
           'unannotated_official_genes':n-len(positions),'training_measured_union':len(union),
           'target_count':len(targets)-1,'source_refs':config['functional'],
           'heldout_NTC_only_transductive_gene_state':True,'heldout_response_used':False,
           'missing_annotation':'zero functional vector plus explicit flag and NTC state',
           'unseen_target':'zero learned ID embedding; functional and NTC query remain active'}
    write_json(output/'cache/representation.json',audit)
    return net,training,targets,audit


def fit(output,genes,split,config,run):
    spec=config['decoder']; rng=initialize(config)
    net,training,targets,audit=build(output,genes,split,config)
    optimizer=optimizer_for(net,spec); checkpoint=output/'checkpoints/decoder.pt'
    step=restore_state(checkpoint,net,optimizer,rng,config) if checkpoint.exists() else 0
    while step < spec['steps']:
        context=int(rng.integers(len(training))); data=training[context]
        rows=rng.integers(len(data['targets']),size=spec['batch_examples'])
        cols=rng.integers(len(data['positions']),size=spec['batch_examples'])
        prediction=net(data['targets'][rows],data['positions'][cols],context)
        truth=data['delta'][rows,cols]
        loss=response_loss(prediction,truth,spec)
        if not torch.isfinite(loss): raise ValueError('Non-finite decoder loss')
        optimizer.zero_grad(set_to_none=True); loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(),spec['gradient_clip']); optimizer.step(); step+=1
        if step % spec['checkpoint_every']==0 or step==spec['steps']:
            save_state(checkpoint,net,optimizer,rng,step,config)
            raw_mse=float((prediction.detach()-truth).square().mean())
            run.log({'train/step':step,'train/loss':float(loss.detach()),'train/batch_delta_mse':raw_mse})
            print(f'Decoder step {step}/{spec["steps"]}: joint={float(loss.detach()):.5f} delta_mse={raw_mse:.7f}',flush=True)
    audit.update(steps_completed=step,parameters=sum(p.numel() for p in net.parameters()),
                 checkpoint_selection='fixed_last',training_contexts=split['training_contexts'])
    write_json(output/'cache/fit.json',audit)
    del optimizer; gc.collect(); torch.cuda.empty_cache()
    return checkpoint,dict(network=net.eval(),training=training,targets=targets,genes=genes,spec=spec),audit


def predict(model,stats,targets,arm,source=None):
    if arm!='linear': raise ValueError('Unregistered candidate arm')
    net=model['network']; spec=model['spec']; device=net.features.device
    n=len(model['genes']); target_index={t:i for i,t in enumerate(model['targets'])}
    gene_index={g:i for i,g in enumerate(model['genes'])}
    result=np.zeros((len(targets),n),dtype=np.float32)
    with torch.no_grad():
        for i,target in enumerate(targets):
            tid=target_index.get(target)
            unseen_position=gene_index.get(target,n) if tid is None else None
            tid=0 if tid is None else tid
            pred=[]
            for start in range(0,n,spec['prediction_chunk']):
                g=torch.arange(start,min(n,start+spec['prediction_chunk']),device=device)
                pred.append(net(torch.full_like(g,tid),g,4,unseen_position).cpu().numpy())
            value=np.concatenate(pred).astype(np.float64)
            residual=np.zeros(n); mass=np.zeros(n)
            for context,data in enumerate(model['training']):
                matched=np.flatnonzero(data['labels']==target)
                if not len(matched): continue
                row=int(matched[0]); pos=data['positions']; weight=float(data['counts'][row])
                source_prediction=net(torch.full_like(pos,tid),pos,context).cpu().numpy()
                residual[pos.cpu().numpy()]+=weight*(data['delta'][row].cpu().numpy()-source_prediction)
                mass[pos.cpu().numpy()]+=weight
            value+=np.divide(residual,mass,out=np.zeros_like(residual),where=mass>0)
            result[i]=value
    if not np.isfinite(result).all(): raise ValueError('Non-finite response')
    return result


def generate(output,model,genes,split,config):
    from evaluation import generate as frozen_generate
    return frozen_generate(output,model,genes,split,config)


def diagnostic_counts(output,model,ntc,config):
    from evaluation import generate_counts
    return generate_counts(ntc.X,np.zeros(ntc.n_vars),config['diagnostics']['seed'],
                           config['generation']['cells_per_target'],config['data']['target_sum'])
