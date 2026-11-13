"""Meaningful numerical and causality checks; uses the supplied bank data."""
import json
from pathlib import Path
import numpy as np
from scipy.optimize._numdiff import approx_derivative
from models import CATALOG,TAU,run_model,caviar_objective
from run_all import ROOT,tables

def main():
    y=np.load(ROOT/'data/bank_returns.npz',allow_pickle=False)['y']
    checks=[]
    # Changing the return being predicted or later observations must not change q_t.
    a=y[:2006].copy();changed=a.copy();changed[2005]+=15
    for model in (m for m in CATALOG if not m.startswith('ml_')):
        q,_,_=run_model(model,a);r,_,_=run_model(model,changed)
        assert np.array_equal(q,r),model+' leaks contemporaneous/future returns'
        checks.append(model+': forecast-before-observe invariance passed')
    for asym in (False,True):
        p=np.array([-.2,.8,-.1,-.15]) if asym else np.array([-.2,.8,-.1])
        v,g=caviar_objective(p,y[500:2000,0],.01,-4.,asym)
        num=approx_derivative(lambda z:caviar_objective(z,y[500:2000,0],.01,-4.,asym)[0],p).reshape(-1)
        assert np.allclose(g,num,atol=1e-5), (g,num)
    checks.append('CAViaR analytical subgradients agree with numerical derivatives away from kinks')
    print(json.dumps(checks,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
