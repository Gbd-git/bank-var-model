"""Reproduce forecasts and tables. Run from any working directory."""
import argparse, json, time, hashlib, platform
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd
from scipy.special import xlogy
from scipy.stats import chi2
from models import CATALOG, TAU, run_model, model_relative_path
ROOT=Path(__file__).resolve().parents[1]

def coverage_tests(hit,tau):
    h=np.asarray(hit,dtype=int);n=len(h);x=int(h.sum());p=x/n
    ll0=xlogy(x,tau)+xlogy(n-x,1-tau)
    ll1=xlogy(x,p)+xlogy(n-x,1-p)
    uc=max(0.,float(2*(ll1-ll0)))
    c=np.bincount(2*h[:-1]+h[1:],minlength=4).reshape(2,2)
    row=c.sum(axis=1);pt=c[:,1].sum()/c.sum()
    null=xlogy(c[:,1].sum(),pt)+xlogy(c[:,0].sum(),1-pt)
    alt=0.
    for i in range(2):
        if row[i]:
            pi=c[i,1]/row[i];alt+=xlogy(c[i,1],pi)+xlogy(c[i,0],1-pi)
    ind=max(0.,float(2*(alt-null)))
    return dict(kupiec_p=float(chi2.sf(uc,1)),independence_p=float(chi2.sf(ind,1)),conditional_coverage_p=float(chi2.sf(uc+ind,2)),transition_11=int(c[1,1]))

def tables(y,q,banks,model):
    err=y[...,None]-q;loss=np.maximum(TAU*err,(TAU-1)*err);hit=y[...,None]<q
    rows=[]
    for b,bank in enumerate(banks):
        for j,tau in enumerate(TAU):
            rows.append(dict(model=model,bank=str(bank),confidence=1-tau,n=len(y),violations=int(hit[:,b,j].sum()),
                violation_rate=float(hit[:,b,j].mean()),pinball_loss=float(loss[:,b,j].mean()),
                **coverage_tests(hit[:,b,j],tau)))
    aggregate=[dict(model=model,category=CATALOG[model][0],confidence=1-tau,n_dates=len(y),n_banks=len(banks),
        violation_rate=float(hit[:,:,j].mean()),pinball_loss=float(loss[:,:,j].mean())) for j,tau in enumerate(TAU)]
    return pd.DataFrame(rows),aggregate

def task(model,warmup,window,refit,limit,output,seed,draws,workers=1):
    d=np.load(ROOT/'data/bank_returns.npz',allow_pickle=False);y=d['y'];banks=d['banks'];dates=d['dates']
    if limit:y=y[:warmup+limit];dates=dates[:warmup+limit]
    t=time.perf_counter();q,raw,diagnostics=run_model(model,y,warmup,window,refit,seed,draws);elapsed=time.perf_counter()-t
    folder=Path(output)/model_relative_path(model);folder.mkdir(parents=True,exist_ok=True)
    np.save(folder/'predictions.npy',q);np.save(folder/'raw_predictions.npy',raw)
    metrics,agg=tables(y[warmup:],q,banks,model);metrics.to_csv(folder/'metrics_by_bank.csv',index=False)
    idx=pd.MultiIndex.from_product([dates[warmup:].astype(str),banks.astype(str)],names=['date','bank'])
    pd.DataFrame({'return':y[warmup:].reshape(-1),'q_01':q[:,:,0].reshape(-1),'q_05':q[:,:,1].reshape(-1)},index=idx).to_csv(folder/'predictions.csv')
    summary=dict(model=model,label=CATALOG[model][1],window=window,warmup=warmup,refit_every=refit,
        dates=len(q),banks=len(banks),start=str(dates[warmup]),end=str(dates[-1]),seed=seed,mc_draws=draws,
        elapsed_seconds=elapsed,compute_workers=workers,metrics=agg,diagnostics=diagnostics,
        data_sha256=hashlib.sha256((ROOT/'data/bank_returns.npz').read_bytes()).hexdigest(),python=platform.python_version())
    (folder/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    for a in agg:a['elapsed_seconds']=elapsed
    print(f'{model}: {elapsed:.1f}s, loss {[a["pinball_loss"] for a in agg]}, events {len(diagnostics["events"])}',flush=True)
    return agg

def main():
    p=argparse.ArgumentParser();p.add_argument('--models',nargs='+',choices=list(CATALOG),default=list(CATALOG))
    p.add_argument('--workers',type=int,default=1);p.add_argument('--limit',type=int);p.add_argument('--warmup',type=int,default=2000)
    p.add_argument('--window',type=int,default=1500);p.add_argument('--refit-every',type=int,default=5)
    p.add_argument('--seed',type=int,default=20261113);p.add_argument('--mc-draws',type=int,default=20000)
    p.add_argument('--output',type=Path,default=ROOT);a=p.parse_args();rows=[]
    if a.workers<1 or a.mc_draws<100 or (a.limit is not None and a.limit<1):p.error('Positive workers/limit and mc-draws >=100 required')
    with np.load(ROOT/'data/bank_returns.npz',allow_pickle=False) as data:
        y=data['y'];dates=data['dates'].astype('datetime64[D]');banks=data['banks']
        if not (50<=a.window<=a.warmup<len(y)) or a.refit_every<1:p.error('Require 50 <= window <= warmup < observations and positive refit interval')
        if y.shape!=(len(dates),len(banks)) or not np.isfinite(y).all():p.error('Nonfinite or misaligned inputs')
        if len(set(banks))!=len(banks) or not np.all(np.diff(dates)>np.timedelta64(0,'D')):p.error('Bank names must be unique and dates increasing')
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        jobs=[pool.submit(task,m,a.warmup,a.window,a.refit_every,a.limit,str(a.output),a.seed,a.mc_draws,a.workers) for m in a.models]
        for job in as_completed(jobs):rows+=job.result()
    pd.DataFrame(rows).sort_values(['confidence','pinball_loss']).to_csv(a.output/'run_comparison.csv',index=False)
if __name__=='__main__':main()
