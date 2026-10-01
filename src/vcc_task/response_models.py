"""Numerical masked Ridge and a small MLP, with resumable full training state."""
import numpy as np
import torch
from torch import nn

from .response_data import feature_rows


def save_state(path,state):
    temporary=path.with_suffix('.partial.pt')
    torch.save(state,temporary); temporary.replace(path)


def centered(value,own):
    present=own>=0
    rows=torch.where(present)[0]
    summed=value.sum(1)
    summed=summed.index_add(0,rows,-value[rows,own[rows]])
    mean=summed/(value.shape[1]-present.to(value.dtype))
    result=value-mean[:,None]
    # Avoid an in-place edit of the autograd graph's matrix product.
    mask=torch.ones_like(result)
    mask[rows,own[rows]]=0
    return result*mask


def tensors(tasks,lookup,features,device,intercept=False):
    out=[]
    for task in tasks:
        x=feature_rows(task['labels'],lookup,features)
        if intercept:
            x=np.column_stack((x,np.ones(len(x),dtype=np.float32)))
        own=torch.as_tensor(task['own'],device=device,dtype=torch.long)
        row_weight=1/(len(tasks)*len(x)*(len(task['positions'])-(own>=0).float()))
        out.append({'x':torch.as_tensor(x,device=device),
            'y':torch.as_tensor(task['values'],device=device),
            'positions':torch.as_tensor(task['positions'],device=device,dtype=torch.long),
            'own':own,'weight':row_weight})
    return out


def fit_ridge(tasks,lookup,features,width,cfg,path,tracking):
    device=cfg['device']; packs=tensors(tasks,lookup,features,device,intercept=True)
    shape=(features.shape[1]+1,width)
    regularization=cfg['alpha']/(sum(len(t['labels']) for t in tasks)*width)
    penalty=torch.full((shape[0],1),regularization,device=device)
    penalty[-1]=cfg['intercept_jitter']
    def multiply(weights):
        result=weights*penalty
        for pack in packs:
            predicted=centered(pack['x']@weights[:,pack['positions']],pack['own'])
            gradient=pack['x'].T@(predicted*pack['weight'][:,None])
            result.index_add_(1,pack['positions'],gradient)
        return result
    rhs=torch.zeros(shape,device=device)
    for pack in packs:
        rhs.index_add_(1,pack['positions'],pack['x'].T@(pack['y']*pack['weight'][:,None]))
    if path.exists():
        state=torch.load(path,map_location=device,weights_only=False)
        w,r,p=state['weights'],state['residual'],state['direction']; step=state['step']
        if state['complete']:
            return state
    else:
        w=torch.zeros(shape,device=device); r=rhs.clone(); p=r.clone(); step=0
    rhs_norm=torch.linalg.norm(rhs)
    rr=torch.sum(r*r)
    with torch.no_grad():
        while step<cfg['maximum_iterations']:
            relative=float(torch.sqrt(rr)/rhs_norm)
            if relative<=cfg['relative_tolerance']:
                break
            ap=multiply(p); denominator=torch.sum(p*ap)
            if denominator<=0 or not torch.isfinite(denominator):
                raise ValueError('Ridge operator lost positive definiteness')
            alpha=rr/denominator
            w=w+alpha*p; r=r-alpha*ap
            rr_new=torch.sum(r*r); p=r+(rr_new/rr)*p; rr=rr_new; step+=1
            if step%cfg['checkpoint_every']==0:
                save_state(path,{'weights':w,'residual':r,'direction':p,'step':step,'complete':False})
                tracking.log({'training/step':step,'training/relative_residual':float(torch.sqrt(rr)/rhs_norm)})
        # Recompute the true normal-equation residual, rather than trusting CG's
        # recursively accumulated residual after finite precision iterations.
        actual=float(torch.linalg.norm(rhs-multiply(w))/rhs_norm)
        state={'weights':w,'residual':r,'direction':p,'step':step,'complete':True,
               'actual_relative_residual':actual,'converged':actual<=cfg['relative_tolerance'],
               'regularization':regularization}
        save_state(path,state)
    return state


class ResponseMLP(nn.Module):
    def __init__(self,input_width,output_width,hidden):
        super().__init__()
        layers=[]; previous=input_width
        for width in hidden:
            layers.extend([nn.Linear(previous,width),nn.ReLU()]); previous=width
        layers.append(nn.Linear(previous,output_width))
        self.network=nn.Sequential(*layers)

    def forward(self,x):
        return self.network(x)


def fit_mlp(tasks,lookup,features,width,cfg,path,tracking,seed):
    device=cfg['device']; torch.manual_seed(seed); rng=np.random.default_rng(seed)
    model=ResponseMLP(features.shape[1],width,cfg['hidden']).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=cfg['epochs'])
    packs=tensors(tasks,lookup,features,device)
    contexts=np.concatenate([np.full(len(p['x']),i) for i,p in enumerate(packs)])
    local=np.concatenate([np.arange(len(p['x'])) for p in packs]); total=len(local)
    start_epoch=0
    if path.exists():
        state=torch.load(path,map_location=device,weights_only=False)
        model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler']); rng.bit_generator.state=state['numpy_rng']
        torch.set_rng_state(state['torch_rng'].cpu()); torch.cuda.set_rng_state_all([v.cpu() for v in state['cuda_rng']])
        start_epoch=state['epoch']
        if state['complete']:
            model.eval(); return model,state
    for epoch in range(start_epoch,cfg['epochs']):
        model.train(); order=rng.permutation(total); epoch_loss=0.
        for start in range(0,total,cfg['batch_size']):
            selected=order[start:start+cfg['batch_size']]
            optimizer.zero_grad(set_to_none=True); loss=torch.zeros((),device=device)
            for context in np.unique(contexts[selected]):
                pack=packs[context]
                row=torch.as_tensor(local[selected[contexts[selected]==context]],device=device)
                predicted=model(pack['x'][row])[:,pack['positions']]
                error=centered(predicted,pack['own'][row])-pack['y'][row]
                loss=loss+torch.sum(error**2*pack['weight'][row,None])*total/len(selected)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_clip'])
            optimizer.step(); epoch_loss+=float(loss.detach())*len(selected)/total
        scheduler.step()
        if (epoch+1)%cfg['checkpoint_every']==0 or epoch+1==cfg['epochs']:
            state={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
                'numpy_rng':rng.bit_generator.state,'torch_rng':torch.get_rng_state(),
                'cuda_rng':torch.cuda.get_rng_state_all(),'epoch':epoch+1,'complete':epoch+1==cfg['epochs'],
                'training_loss':epoch_loss}
            save_state(path,state)
            tracking.log({'training/epoch':epoch+1,'training/loss':epoch_loss})
            print(f'MLP epoch {epoch+1}/{cfg["epochs"]} loss {epoch_loss:.7g}',flush=True)
    model.eval()
    return model,state
