"""Amplitude projection and soft, training-only within-context retrieval."""
import numpy as np
import torch
from torch.nn import functional as F
from model import masked_logcp
from objectives import TaskObjective, COMPONENTS


def amplitude_loss(predicted, observed, scale, weight):
    signal = (observed.square()*weight).mean(-1)
    reliability = (observed.square()/(observed.square()+scale.square())).mean(-1).detach()
    # Signed projection is differentiable at a zero response, unlike norm matching.
    projection = (predicted*observed*weight).mean(-1)/(signal+1e-4)
    return reliability * F.huber_loss(projection, torch.ones_like(projection), reduction='none')


def specificity_loss(predicted, observed, references, scale, mask, temperature=.2):
    mass = mask.sum(-1).clamp_min(1)
    def normalized(x):
        return x / ((x.square()*mask).sum(-1)/mass + .05**2).sqrt()[...,None]
    pn = normalized(predicted); rn = normalized(observed)
    ref = references / ((references.square()*mask[:,None]).sum(-1)/mass[:,None]+.05**2).sqrt()[...,None]
    truth = ((rn[:,None]*ref*mask[:,None]).sum(-1)/mass[:,None]/temperature).softmax(-1).detach()
    logits = (pn[:,None]*ref*mask[:,None]).sum(-1)/mass[:,None]/temperature
    reliability = (observed.square()/(observed.square()+scale.square())).mean(-1).detach()
    return reliability * F.kl_div(logits.log_softmax(-1), truth, reduction='none').sum(-1)


class ResponseObjective(TaskObjective):
    def __init__(self, data, view, contexts, config, cache):
        extra = config['response_auxiliary']
        self.components = COMPONENTS + (() if extra=='none' else (extra+'_loss',))
        self.loss_terms = ('elbo', *self.components)
        super().__init__(data, view, contexts, config, cache)
        self.by_context = {c: np.array([i for i,(ctx,t) in enumerate(self.keys) if ctx==c]) for c in contexts}
        self.common_index = {data.genes[g]:i for i,g in enumerate(self.axis)}

    def additional_losses(self, target, control, observed, keys, rng):
        kind = self.config['response_auxiliary']
        if kind == 'none': return {}
        delta = masked_logcp(target,torch.ones_like(target))-masked_logcp(control,torch.ones_like(control))
        if kind == 'amplitude':
            return {'amplitude_loss':amplitude_loss(delta, observed['delta'], observed['response_scale'], observed['response_weight'])}
        references=[];masks=[]
        for key in keys:
            own=self.index[key]; candidates=self.by_context[key[0]]; candidates=candidates[candidates!=own]
            n=self.config['retrieval_candidates']-1
            chosen=rng.choice(candidates,n,replace=len(candidates)<n) if len(candidates) else np.full(n,own)
            indices=np.r_[own,chosen]
            references.append(np.asarray(self.arrays['delta'][indices]))
            mask=np.ones(len(self.axis),np.float32)
            for idx in indices:
                t=self.keys[int(idx)][1]
                if t in self.common_index:mask[self.common_index[t]]=0
            masks.append(mask)
        refs=torch.as_tensor(np.asarray(references),device=target.device)
        mask=torch.as_tensor(np.asarray(masks),device=target.device)
        return {'specificity_loss':specificity_loss(delta,observed['delta'],refs,observed['response_scale'],mask,self.config['retrieval_temperature'])}
