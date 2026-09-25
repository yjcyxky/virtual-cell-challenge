"""Exp004 entry using the shared audited LODO workflow with explicit factories."""
import argparse
from pathlib import Path
import sys
from types import SimpleNamespace

EXPERIMENT = Path(__file__).resolve().parents[1]
ROOT = EXPERIMENT.parents[1]
sys.path.insert(0, str(ROOT/'experiments/exp003-context-module-cvae/src'))
from main import main, finalize_run
from shared_model import construct
from response_objective import ResponseObjective

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--config',default=str(EXPERIMENT/'configs/conditional.yaml'))
    parser.add_argument('--seed',type=int)
    parser.add_argument('--cache-source')
    parser.add_argument('--finalize-only',action='store_true')
    parser.add_argument('--repair-validation-reason')
    parser.set_defaults(variant=None,restart_from_run=None)
    args = parser.parse_args()
    if args.finalize_only:
        assert not any(getattr(args, name) is not None for name in ['seed', 'cache_source', 'repair_validation_reason'])
        finalize_run(args.run_id, EXPERIMENT)
    else:
        main(args,EXPERIMENT,SimpleNamespace(model_factory=construct,objective_type=ResponseObjective))
