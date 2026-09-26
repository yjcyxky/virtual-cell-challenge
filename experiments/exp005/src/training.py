"""Deterministic context-weighted XGBoost and complete same-version resumption."""
import random
import numpy as np
import xgboost as xgb
from common import digest,event,restore_training_state,save_training_state,seed,write_json

class Rows(xgb.DataIter):
    def __init__(self,data,features,contexts,config):
        super().__init__(release_data=True)
        self.data,self.features,self.contexts,self.config=data,features,sorted(contexts),config
        self.reset()

    def reset(self): self.iterator=self.batches()

    def batches(self):
        for context in self.contexts:
            stats=self.data.contexts[context];targets=stats['targets'];native=stats['gene_indices']
            bags=len(stats['features']);readouts=min(len(native),self.config['training_readouts_per_target'])
            # Equal context, then target, then bag/readout. Absolute scale fixed across fits.
            weight=1e6/(len(self.contexts)*len(targets)*bags*readouts)
            for bag in range(bags):
                for start in range(0,len(targets),32):
                    ps,gs,ys=[],[],[]
                    for row in range(start,min(start+32,len(targets))):
                        target=str(targets[row])
                        rng=np.random.default_rng(seed(self.config['seed'],context,target,bag,'readouts'))
                        chosen=np.sort(rng.choice(len(native),readouts,replace=False))
                        ps.extend([self.data.lookup[target]]*len(chosen));gs.extend(native[chosen]);ys.extend(stats['response'][row,chosen])
                    matrix=self.features.matrix(context,ps,gs,bag)
                    yield matrix,np.asarray(ys,np.float32),np.full(len(ys),weight,np.float32)

    def next(self,input_data):
        try: matrix,labels,weights=next(self.iterator)
        except StopIteration: return False
        input_data(data=matrix,label=labels,weight=weights)
        return True

def fit_id(contexts): return '+'.join(sorted(contexts))

def predict(booster,features,context,targets,genes):
    result=np.empty((len(targets),len(genes)),np.float32)
    for start in range(0,len(targets),8):
        selected=targets[start:start+8]
        p=np.repeat([features.data.lookup[str(t)] for t in selected],len(genes))
        g=np.tile(genes,len(selected))
        result[start:start+len(selected)]=booster.inplace_predict(features.matrix(context,p,g)).reshape(len(selected),len(genes))
    return result

def fit(data,features,contexts,limit,config,output,tracked,evaluate):
    name=fit_id(contexts);directory=output/'checkpoints'/name;directory.mkdir(parents=True,exist_ok=True)
    identity={'training_contexts':sorted(contexts),'configuration':config,
              'data_identity':digest(data.directory/'complete.json'),'xgboost':xgb.__version__}
    latest=directory/'latest.pkl';last=0
    parameters={**config['model'],'seed':seed(config['seed'],name,'xgb')}
    if latest.exists():
        booster,last=restore_training_state(latest,identity);booster.set_param(parameters)
    else:
        np.random.seed(parameters['seed']);random.seed(parameters['seed']);booster=None
    rounds=[i for i in config['checkpoints'] if i<=limit]
    if limit not in rounds: rounds.append(limit)
    # Pending evaluation of an already-written snapshot resumes before further updates.
    for iteration in rounds:
        snapshot=directory/f'round-{iteration:04d}.ubj'
        if iteration<=last:
            loaded=xgb.Booster();loaded.load_model(snapshot)
            evaluate(loaded,iteration);continue
        event('quantile_matrix',training_contexts=sorted(contexts),resume_round=last)
        train=xgb.QuantileDMatrix(Rows(data,features,contexts,config),max_bin=config['model']['max_bin'],nthread=config['threads'])
        if booster is None: booster=xgb.Booster(parameters,[train])
        event('training',training_contexts=sorted(contexts),from_round=last,to_round=iteration)
        for step in range(last,iteration): booster.update(train,step)
        loss=booster.eval(train,name='train',iteration=iteration)
        del train
        import gc;gc.collect()
        booster.save_model(snapshot)
        save_training_state(latest,booster,iteration,identity)
        write_json(directory/'state.json',{'iteration':iteration,'training_contexts':sorted(contexts),'loss':loss,
                   'snapshot':snapshot.name,'checkpoint_sha256':digest(snapshot),'training_seed':parameters['seed']})
        event('checkpoint',training_contexts=sorted(contexts),round=iteration,training_loss=loss)
        tracked.log({f'train/{name}/round':iteration,f'train/{name}/rmse':float(loss.rsplit(':',1)[1])})
        evaluate(booster,iteration)
        last=iteration
    return booster

def select_rounds(contexts,config,results):
    selections={}
    for outer in contexts:
        remaining=[c for c in contexts if c!=outer]
        scores={}
        for iteration in config['checkpoints']:
            values=[results[(fit_id([c for c in remaining if c!=validation]),validation,iteration)]['score'] for validation in remaining]
            scores[iteration]=float(np.mean(values))
        selected=max(scores,key=lambda r:(scores[r],-r))
        selections[outer]={'rounds':selected,'inner_scores':scores,'selection_contexts':remaining}
    return selections
