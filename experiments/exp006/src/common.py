"""Atomic run artifacts, stable random streams, and bounded file hashing."""
from __future__ import annotations
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import pickle
import random
import numpy as np

EXPERIMENT = Path(__file__).resolve().parents[1]
ROOT = EXPERIMENT.parents[1]

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while block := f.read(16 << 20):
            h.update(block)
        if hasattr(os, 'posix_fadvise'):
            os.posix_fadvise(f.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
    return h.hexdigest()

def seed(*parts):
    return int.from_bytes(hashlib.sha256('|'.join(map(str, parts)).encode()).digest()[:4], 'little')

def atomic_bytes(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    with temp.open('wb') as f:
        f.write(data); f.flush(); os.fsync(f.fileno())
    temp.replace(path)
    fd = os.open(path.parent, os.O_DIRECTORY)
    try: os.fsync(fd)
    finally: os.close(fd)

def write_json(path, value):
    atomic_bytes(path, (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + '\n').encode())

def event(stage, **values):
    print(json.dumps({'time': datetime.now(timezone.utc).isoformat(), 'stage': stage, **values}, ensure_ascii=False, allow_nan=False), flush=True)

def save_training_state(path, booster, iteration, identity):
    # A pickle of Booster is XGBoost's full memory snapshot, not the UBJ inference export.
    payload = {'booster': booster, 'iteration': iteration, 'identity': identity,
               'numpy_random_state': np.random.get_state(), 'python_random_state': random.getstate()}
    atomic_bytes(path, pickle.dumps(payload, protocol=5))

def restore_training_state(path, identity):
    with Path(path).open('rb') as f: state = pickle.load(f)
    if state['identity'] != identity: raise ValueError('training_state_identity_mismatch')
    np.random.set_state(state['numpy_random_state']); random.setstate(state['python_random_state'])
    return state['booster'], state['iteration']

def log_profile(counts):
    values = np.asarray(counts, dtype=np.float64)
    totals = values.sum(axis=-1, keepdims=True)
    if np.any(totals <= 0): raise ValueError('empty_expression_profile')
    return np.log1p(50000.0 * values / totals).astype(np.float32)

def stratified_sample(strata, n, rng):
    """Proportional randomized rounding; no mandatory slot per sparse stratum."""
    groups, inverse, sizes = np.unique(strata, return_inverse=True, return_counts=True)
    expected = n * sizes / sizes.sum()
    allocation = np.floor(expected).astype(int)
    extra = n - allocation.sum()
    if extra:
        fraction = expected - allocation
        chosen = rng.choice(len(groups), extra, replace=False, p=fraction / fraction.sum())
        allocation[chosen] += 1
    indices = []
    for group in np.flatnonzero(allocation):
        pool = np.flatnonzero(inverse == group)
        indices.extend(rng.choice(pool, allocation[group], replace=allocation[group] > len(pool)))
    return np.asarray(indices, dtype=np.int64)
