"""Resource contention delays evaluation instead of silently changing its backend."""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import evaluation


def test_cuda_context_oom_waits_then_recovers(monkeypatch):
    import cupy
    calls=[]
    def available():
        calls.append(1)
        if len(calls)==1:raise cupy.cuda.runtime.CUDARuntimeError(2)
        return (100<<30,120<<30)
    sleeps=[]
    monkeypatch.setattr(cupy.cuda.runtime,'memGetInfo',available)
    monkeypatch.setattr(evaluation.time,'sleep',lambda seconds:sleeps.append(seconds))
    original=Path.read_text
    monkeypatch.setattr(Path,'read_text',lambda path,*a,**k:'MemAvailable: 104857600 kB' if str(path)=='/proc/meminfo' else original(path,*a,**k))
    real=SimpleNamespace(X=np.zeros((10,8),np.uint16),n_vars=8,obs=pd.DataFrame({'target':['one']*10}))
    evaluation.wait_for_memory(real,'toy','cuda',anchor=True)
    assert sleeps==[30] and len(calls)==2
