"""Serial full-run ablation matrix; resume the same run, never skip a failure."""
import fcntl
import json
from pathlib import Path
import subprocess
import yaml

EXPERIMENT=Path(__file__).resolve().parents[1]

def main():
    config=yaml.safe_load((EXPERIMENT/'configs/matrix.yaml').read_text())
    with open('/tmp/vcc-exp004-matrix.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        completed=[]
        for seed in config['seeds']:
            for arm in config['arms']:
                run_id=f"{config['run_prefix']}-{arm}-s{seed}"
                output=EXPERIMENT/'outputs'/run_id
                print(json.dumps({'stage':'matrix_run','run_id':run_id,'completed_runs':len(completed),'planned_runs':len(config['seeds'])*len(config['arms'])}),flush=True)
                # The run entry validates code/config/environment on every resume,
                # including completed runs; completion metadata does not bypass checks.
                command=[str(EXPERIMENT/'reproduce.sh'),'--run-id',run_id,'--config',str(EXPERIMENT/'configs'/f'{arm}.yaml'),'--seed',str(seed)]
                subprocess.run(command,cwd=EXPERIMENT,check=True)
                completion=json.loads((output/'complete.json').read_text())
                metrics=json.loads((output/'metrics.json').read_text())
                assert completion['status']=='completed' and metrics['models_trained']==5
                completed.append({'run_id':run_id,'arm':arm,'seed':seed,'mean_validation_score':metrics['mean_validation_score'],'folds':metrics['folds']})
        # Cross-run aggregate is a result of the final run, not a new stage/run.
        destination=output/'matrix-comparison.json'
        destination.write_text(json.dumps({'runs':completed,'status':'all_planned_runs_completed','selection_scope':'retrospective five-fold development evaluation; no official submissions'},indent=2)+'\n')
        print(json.dumps({'stage':'matrix_complete','runs':len(completed),'models':5*len(completed),'comparison':str(destination)}),flush=True)

if __name__=='__main__':main()
