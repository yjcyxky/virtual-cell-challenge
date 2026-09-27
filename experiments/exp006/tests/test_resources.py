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


def test_base_launcher_disables_thp_for_its_child():
    import os,subprocess
    # Explicitly re-enable THP in a disposable process so inherited settings cannot
    # make this regression test pass without exercising the actual base launcher.
    child="import ctypes; assert ctypes.CDLL(None).prctl(42,0,0,0,0)==1"
    probe=("import ctypes,sys; ctypes.CDLL(None).prctl(41,0,0,0,0); "
           "from runtime import run_in_base; sys.exit(run_in_base([sys.executable,'-c',"+repr(child)+"]))")
    environment={**os.environ,'PYTHONPATH':str(Path(__file__).resolve().parents[1]/'src')}
    checked=subprocess.run([sys.executable,'-c',probe],env=environment,text=True,capture_output=True)
    assert checked.returncode==0,checked.stderr
