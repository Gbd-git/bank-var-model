"""Causal rolling one-observation-ahead VaR models; log-percent returns.

Distribution fitting and empirical tail fitting use only the trailing window.
All models refit on the same schedule. Recursive states update after forecasting.
"""
import warnings
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.stats import norm, t as student, genpareto
from numba import njit
from arch import arch_model
from arch.univariate.distribution import SkewStudent
from gjr_var.model import RollingGJR

TAU = np.array([.01, .05])
CATALOG = {
 'hs': ('01_historical', 'Historical Simulation'),
 'normal': ('02_parametric', 'Normal'),
 'student': ('02_parametric', 'Student-t'),
 'mc_normal': ('03_monte_carlo', 'Monte Carlo Normal'),
 'mc_student': ('03_monte_carlo', 'Monte Carlo Student-t'),
 'garch_normal': ('04_conditional_volatility', 'GARCH-Normal'),
 'garch_t': ('04_conditional_volatility', 'GARCH-t'),
 'garch_skewt': ('04_conditional_volatility', 'GARCH-skewt'),
 'gjr_normal': ('04_conditional_volatility', 'GJR-GARCH-Normal'),
 'gjr_t': ('04_conditional_volatility', 'GJR-GARCH-t'),
 'gjr_skewt': ('04_conditional_volatility', 'GJR-GARCH-skewt'),
 'pot_gpd': ('05_evt_semiparametric', 'POT-GPD'),
 'garch_t_evt': ('05_evt_semiparametric', 'GARCH-t-EVT'),
 'garch_t_fhs': ('05_evt_semiparametric', 'GARCH-t-FHS'),
 'caviar_sav': ('06_conditional_quantile', 'CAViaR-SAV'),
 'caviar_as': ('06_conditional_quantile', 'CAViaR-AS'),
 'quantile_regression': ('06_conditional_quantile', 'Linear Quantile Regression'),
}
from ml_models import ML_CATALOG
CATALOG.update(ML_CATALOG)

def model_relative_path(model):
    """Result path relative to the project/output root."""
    category=CATALOG[model][0]
    prefix='' if model in ML_CATALOG else '经典计量经济学模型/'
    return f'{prefix}{category}/{model}'


def pot_quantile(y):
    """POT on losses -y, 90th percentile threshold, fixed-location GPD MLE."""
    loss=-np.asarray(y); u=np.quantile(loss,.9); ex=loss[loss>u]-u
    if len(ex)<50: raise ValueError('Fewer than 50 threshold exceedances')
    xi,_,scale=genpareto.fit(ex,floc=0)
    if not np.isfinite([xi,scale]).all() or scale<=0: raise ValueError('Invalid GPD fit')
    q=-(u+genpareto.ppf(1-TAU/(len(ex)/len(y)),xi,loc=0,scale=scale))
    if not np.isfinite(q).all(): raise ValueError('Invalid GPD quantiles')
    return q,dict(shape=float(xi),scale=float(scale),threshold=float(u),exceedances=len(ex))

@njit(cache=True)
def caviar_objective(p,y,tau,q0,asym):
    q=q0; d=np.zeros(len(p)); g=np.zeros(len(p)); value=0.
    v=np.zeros(len(p));v[0]=1.
    for t in range(len(y)):
        if t>=25:
            e=y[t]-q; value+=max(tau*e,(tau-1)*e)
            multiplier=(1. if e<0 else 0.)-tau
            for j in range(len(p)):g[j]+=multiplier*d[j]
        v[1]=q
        if asym: v[2]=max(y[t],0.);v[3]=max(-y[t],0.)
        else: v[2]=abs(y[t])
        q=0.
        for j in range(len(p)):
            q+=p[j]*v[j]
            d[j]=v[j]+p[1]*d[j]
    return value/(len(y)-25),g/(len(y)-25)

@njit(cache=True)
def caviar_next(p,y,q0,asym):
    q=q0
    for v in y:
        q=p[0]+p[1]*q+p[2]*(max(v,0.) if asym else abs(v))+(p[3]*max(-v,0.) if asym else 0.)
    return q

def qr_features(y):
    """Feature at t depends on y[t-1] and trailing 20 observed returns."""
    out=np.zeros((len(y)+1,4));out[:,0]=1
    for t in range(20,len(y)+1):
        out[t,1]=y[t-1];out[t,2]=abs(y[t-1]);out[t,3]=np.std(y[t-20:t],ddof=1)
    return out

def fit_quantile_regression(X,y,tau):
    """Exact LP dual: max y'u subject to X'u=0, tau-1 <= u <= tau.

    HiGHS equality marginals for the minimization of -y'u equal -beta.
    Four equalities replace the primal's window-length equality system.
    """
    r=linprog(-y,A_eq=X.T,b_eq=np.zeros(X.shape[1]),bounds=[(tau-1,tau)]*len(y),method='highs')
    if not r.success:raise ValueError('Quantile dual LP failed: '+r.message)
    return -r.eqlin.marginals

def run_model(model,y,warmup=2000,window=1500,refit_every=5,seed=20261113,mc_draws=20000):
    if model in ML_CATALOG:
        from ml_models import run_ml
        return run_ml(model,y,warmup,window,refit_every,seed)
    if warmup<window or window<50 or refit_every<1: raise ValueError('Invalid window/schedule')
    n,k=y.shape;pred=np.empty((n-warmup,k,2));raw=pred.copy()
    events=[]; fits=[];crossings=0
    # Preserve the original baseline implementation exactly.
    if model=='gjr_skewt':
        m=RollingGJR(k,window,warmup,refit_every)
        for t in range(n):
            m.refit_if_due()
            if t>=warmup:pred[t-warmup]=m.forecast(None)
            m.update(None,y[t],None)
        return pred,pred.copy(),dict(events=m.failures,baseline_state=m.summary(),crossings=0)
    is_garch=model.startswith(('garch_','gjr_'))
    if is_garch:
        o=int(model.startswith('gjr_'));dist='t' if model.endswith(('evt','fhs')) else model.split('_')[-1]
        pars=[None]*k; h=np.ones(k); zq=np.zeros((k,2))
    qrX=[qr_features(y[:,b]) for b in range(k)] if model=='quantile_regression' else None
    coefficients={}; q=np.zeros((k,2));rng=np.random.default_rng(seed)
    for t in range(warmup,n):
        if (t-warmup)%refit_every==0:
            for b in range(k):
                a=y[t-window:t,b]
                try:
                    if is_garch:
                        am=arch_model(a,mean='Constant',vol='GARCH',p=1,o=o,q=1,dist=dist,rescale=False)
                        with warnings.catch_warnings(record=True):
                            r=am.fit(disp='off',show_warning=False,starting_values=pars[b],options={'maxiter':500})
                            if r.convergence_flag!=0:
                                events.append(dict(t=t,bank=b,kind='optimizer_retry',status=int(r.convergence_flag)))
                                r=am.fit(disp='off',show_warning=False,options={'maxiter':1500})
                        if r.convergence_flag!=0 or not np.isfinite(r.params).all():raise ValueError('ARCH fit did not converge')
                        p=r.params.to_numpy(); newh=float(r.forecast(horizon=1,reindex=False).variance.iloc[-1,0])
                        if model.endswith(('evt','fhs')):
                            z=np.asarray(r.std_resid);z=z[np.isfinite(z)]
                            if model.endswith('evt'):
                                zqb,info=pot_quantile(z);fits.append(dict(t=t,bank=b,**info))
                            else:zqb=np.quantile(z,TAU)
                        elif dist=='normal':zqb=norm.ppf(TAU)
                        elif dist=='t':zqb=student.ppf(TAU,p[-1])*np.sqrt((p[-1]-2)/p[-1])
                        else:zqb=SkewStudent().ppf(TAU,p[-2:])
                        pars[b]=p;h[b]=newh;zq[b]=zqb
                    elif model=='hs':q[b]=np.quantile(a,TAU)
                    elif model in ('normal','mc_normal'):
                        mu=a.mean();sd=a.std(ddof=0)
                        z=norm.ppf(TAU) if model=='normal' else np.quantile(rng.standard_normal(mc_draws),TAU)
                        q[b]=mu+sd*z
                    elif model in ('student','mc_student'):
                        # Location-scale t (scale is not standard deviation).
                        nu,mu,scale=student.fit(a)
                        if scale<=0 or not np.isfinite([nu,mu,scale]).all():raise ValueError('Invalid t fit')
                        z=student.ppf(TAU,nu) if model=='student' else np.quantile(rng.standard_t(nu,mc_draws),TAU)
                        q[b]=mu+scale*z
                    elif model=='pot_gpd':
                        q[b],info=pot_quantile(a);fits.append(dict(t=t,bank=b,**info))
                    elif model.startswith('caviar_'):
                        asym=model.endswith('as');dim=4 if asym else 3
                        for j,tau in enumerate(TAU):
                            q0=float(np.quantile(a[:250],tau))
                            starts=[np.array([q0*.1,.9,-.05,-.1][:dim]),np.array([q0*.3,.7,-.1,-.2][:dim])]
                            if (b,j) in coefficients:starts.append(coefficients[b,j])
                            candidates=[]
                            for start in starts:
                                r=minimize(caviar_objective,start,args=(a,float(tau),q0,asym),jac=True,method='L-BFGS-B',
                                    bounds=[(-20,20),(0,.995)]+[(-5,5)]*(dim-2),options={'maxiter':300,'ftol':1e-10,'maxls':40})
                                if np.isfinite(r.fun):candidates.append(r)
                            if not candidates:raise ValueError('All CAViaR starts failed')
                            best=min(candidates,key=lambda r:r.fun)
                            if not best.success:events.append(dict(t=t,bank=b,tau=float(tau),kind='finite_nonconverged_caviar',message=str(best.message)))
                            coefficients[b,j]=best.x
                            q[b,j]=caviar_next(best.x,a,q0,asym)
                    elif model=='quantile_regression':
                        X=qrX[b][t-window:t]
                        for j,tau in enumerate(TAU):
                            coefficients[b,j]=fit_quantile_regression(X,a,tau)
                except (ValueError,RuntimeError,FloatingPointError) as e:
                    if t==warmup:raise RuntimeError(f'Initial fit failed {model} bank {b}') from e
                    events.append(dict(t=t,bank=b,kind='carry_previous_fit',message=str(e)))
        if is_garch:
            for b in range(k):q[b]=pars[b][0]+np.sqrt(h[b])*zq[b]
        elif model=='quantile_regression':
            for b in range(k):
                for j in range(2):q[b,j]=qrX[b][t]@coefficients[b,j]
        raw[t-warmup]=q
        crossings+=int(np.sum(q[:,0]>q[:,1]))
        pred[t-warmup]=np.sort(q,axis=1)  # documented monotone rearrangement
        if not np.isfinite(pred[t-warmup]).all():raise RuntimeError('Nonfinite forecast')
        if is_garch:
            for b in range(k):
                p=pars[b];e=y[t,b]-p[0]
                h[b]=max(p[1]+(p[2]+(p[3]*(e<0) if o else 0))*e*e+p[3+o]*h[b],1e-8)
        elif model.startswith('caviar_'):
            for b in range(k):
                for j in range(2):q[b,j]=caviar_next(coefficients[b,j],y[t:t+1,b],q[b,j],model.endswith('as'))
    return pred,raw,dict(events=events,tail_fits=fits,crossings=crossings)
