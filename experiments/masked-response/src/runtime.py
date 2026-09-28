"""Reuse frozen preprocessing/scoring providers without copying their implementations.

The local model.py implements the predictor interface used by the existing
evaluator. Providers are appended, so that evaluation imports this model, not
the historical ridge model. All provider files are covered by DAG code_refs.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
PROVIDERS = ROOT / 'experiments/init-linear/src'
sys.path.append(str(PROVIDERS))
sys.path.insert(0, str(ROOT / 'scripts'))
