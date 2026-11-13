"""Numerical, distributional and information-set tests on actual bank data."""
import json
from pathlib import Path
import numpy as np
from scipy.optimize import linprog
from scipy import sparse
from ml_models import ML_CATALOG,run_ml,features,kernel_fit,forest_quantiles,TAU,nystrom_map
ROOT=Path(__file__).resolve().parents[1]

def main():
    y=np.load(ROOT/'data/bank_returns.npz',allow_pickle=False)['y'];checks=[]
    a=y[:2006,:2].copy();changed=a.copy();changed[2002:]+=10
    x,s=features(a);xx,ss=features(changed)
    assert np.array_equal(x[:2003],xx[:2003]) and np.array_equal(s[:2003],ss[:2003])
    for model in ML_CATALOG:
        q,_,_=run_ml(model,a);r,_,_=run_ml(model,changed)
        assert np.array_equal(q[:3],r[:3]),model
        checks.append(model+': current/future-return perturbation does not alter past forecasts')
    # Kernel primal/dual objectives, including the L1 penalty and intercept.
    X=x[500:2000,0];scale=np.maximum(X.std(0),1e-6);X=(X-X.mean(0))/scale
    Z=nystrom_map(X,X,20261113);target=a[500:2000,0];n,p=Z.shape
    eye=sparse.eye(n);A=sparse.hstack([np.ones((n,1)),Z,-Z,eye,-eye],format='csc')
    for tau in TAU:
        costs=np.r_[0.,np.full(2*p,.5),np.full(n,tau),np.full(n,1-tau)]
        sol=linprog(costs,A_eq=A,b_eq=target,bounds=[(None,None)]+[(0,None)]*(2*p+2*n),method='highs')
        bias,coef=kernel_fit(Z,target,tau);e=target-bias-np.einsum('ij,j->i',Z,coef)
        obj=np.maximum(tau*e,(tau-1)*e).sum()+.5*abs(coef).sum()
        assert sol.success and abs(obj-sol.fun)<1e-6,(obj,sol.fun)
    checks.append('Nyström kernel L1-quantile primal and dual objectives agree within 1e-6')
    from sklearn.ensemble import RandomForestRegressor
    f=RandomForestRegressor(n_estimators=3,min_samples_leaf=1000,bootstrap=False,random_state=1,n_jobs=1).fit(X,target)
    q=forest_quantiles(f,X,target,X[:2])
    assert np.array_equal(q,np.tile(np.quantile(target,TAU,method='inverted_cdf'),(2,1)))
    checks.append('Single-leaf quantile forest equals the empirical inverse CDF, not tree means')
    import torch
    from ml_networks import BankNet
    torch.set_num_threads(1);torch.manual_seed(1)
    tx=torch.randn(4,2,28);ts=torch.randn(4,2,20,2)
    for name in ['ml_mlp','ml_lstm','ml_gru','ml_tcn','ml_transformer']:
        net=BankNet(name,2,28);v=net(tx,ts);tx2=tx.clone();ts2=ts.clone();tx2[:,1]+=5;ts2[:,1]+=5
        w=net(tx2,ts2);assert torch.equal(v[:,0],w[:,0]),name
        net.zero_grad();v[:,0].sum().backward()
        assert all(torch.count_nonzero(p.grad[1])==0 for p in net.parameters() if p.grad is not None),name
    checks.append('All five bank-vectorized neural architectures have isolated predictions and parameter gradients')
    print(json.dumps(checks,indent=2))
if __name__=='__main__':main()
