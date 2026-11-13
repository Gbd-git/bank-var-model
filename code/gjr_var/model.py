"""Rolling constant-mean GJR-GARCH(1,1), Hansen standardized skew-t.
Extracted from experiment 50; default numerical behavior preserved.
"""
import warnings
import numpy as np
from scipy.special import gammaln
from scipy.stats import norm, t as student
from arch import arch_model
TAU=np.array([.01,.05])
def quantiles(nu,lam):
    nu=np.asarray(nu)[:,None];lam=np.asarray(lam)[:,None];p=np.broadcast_to(TAU,nu.shape[:1]+(2,))
    lc=gammaln((nu+1)/2)-gammaln(nu/2)-.5*np.log(np.pi*(nu-2))
    a=4*lam*np.exp(lc)*(nu-2)/(nu-1);b=np.sqrt(1+3*lam**2-a**2);left=p<(1-lam)/2
    prob=np.where(left,p/(1-lam),.5+(p-(1-lam)/2)/(1+lam))
    v=student.ppf(prob,nu)*(1+np.where(left,-1,1)*lam)*np.sqrt(1-2/nu)
    return (v-a)/b

class RollingGJR:
    def __init__(self,n,window=1500,warmup=2000,refit_every=5):
        if not (window >= 50 and warmup >= window and refit_every >= 1):
            raise ValueError("Require warmup >= window >= 50 and refit_every >= 1")
        self.window=window;self.warmup=warmup;self.refit_every=refit_every
        self.n=n;self.history=[];self.count=0;self.params=None;self.h=np.ones(n)
        self.refits=0;self.failures=[];self.retries=0;self.refit_positions=[]
    def refit_if_due(self):
        if self.count<self.warmup or (self.count-self.warmup)%self.refit_every:return False
        y=np.asarray(self.history[-self.window:]);pars=[];var=[]
        for k in range(self.n):
            am=arch_model(y[:,k],mean='Constant',vol='GARCH',p=1,o=1,q=1,dist='skewt',rescale=False)
            with warnings.catch_warnings(record=True):
                res=am.fit(disp='off',show_warning=False,starting_values=None if self.params is None else self.params[k],options={'maxiter':500})
                if res.convergence_flag!=0:
                    self.retries+=1;res=am.fit(disp='off',show_warning=False,options={'maxiter':1500})
            if res.convergence_flag!=0 or not np.isfinite(res.params).all():
                self.failures.append(dict(position=self.count,bank=k,status=int(res.convergence_flag)))
                if self.params is None:raise RuntimeError('Initial GJR fit failed')
                pars.append(self.params[k]);var.append(self.h[k])
            else:
                pars.append(res.params.to_numpy());var.append(float(res.forecast(horizon=1,reindex=False).variance.iloc[-1,0]))
        self.params=np.asarray(pars);self.h=np.asarray(var);self.refits+=1;self.refit_positions.append(self.count);return True
    def forecast(self,date):
        if self.params is None:return np.sqrt(self.h)[:,None]*norm.ppf(TAU)
        return self.params[:,0,None]+np.sqrt(self.h)[:,None]*quantiles(self.params[:,5],self.params[:,6])
    def update(self,date,y,x):
        if self.params is None:self.h=.94*self.h+.06*y*y
        else:
            mu,o,a,g,b=self.params[:,:5].T;eps=y-mu;self.h=np.maximum(o+(a+g*(eps<0))*eps**2+b*self.h,1e-8)
        self.history.append(y.copy());self.count+=1
        if len(self.history)>self.window:self.history.pop(0)
    def summary(self):return dict(updates=self.count,parameters=7*self.n,training_history=len(self.history),refits=self.refits,refit_positions=self.refit_positions,retries=self.retries,failures=self.failures)
