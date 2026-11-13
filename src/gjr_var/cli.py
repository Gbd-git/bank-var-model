"""Run from repository root: python -m gjr_var.cli."""
import argparse
import json
import hashlib
import time
from pathlib import Path
import numpy as np
import pandas as pd
from .model import RollingGJR, TAU

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=Path('data/bank_returns.npz'))
    p.add_argument('--output',type=Path,default=Path('outputs/default'))
    p.add_argument('--window',type=int,default=1500)
    p.add_argument('--warmup',type=int,default=2000)
    p.add_argument('--refit-every',type=int,default=5)
    p.add_argument('--limit',type=int,help='Number of evaluation dates; omit for full run')
    p.add_argument('--reference',type=Path,help='Optional npy prediction array for full default reproduction')
    a=p.parse_args()
    if a.data.suffix=='.csv':
        df=pd.read_csv(a.data,index_col=0,float_precision='round_trip')
        y=df.to_numpy(float);dates=df.index.to_numpy(str);banks=df.columns.to_numpy(str)
    else:
        with np.load(a.data,allow_pickle=False) as z:
            y=z['y'];dates=z['dates'].astype(str);banks=z['banks'].astype(str)
    if y.ndim!=2 or y.shape!=(len(dates),len(banks)) or not np.isfinite(y).all():
        p.error('Expected finite date-by-bank returns')
    if len(set(banks))!=len(banks) or not (np.diff(dates.astype('datetime64[D]'))>np.timedelta64(0,'D')).all():
        p.error('Unique bank names and strictly increasing dates required')
    if not (50<=a.window<=a.warmup<len(y)) or a.refit_every<1 or (a.limit is not None and a.limit<1):
        p.error('Require 50 <= window <= warmup < observations; positive refit-every and limit')
    end=len(y) if a.limit is None else min(len(y),a.warmup+a.limit)
    a.output.mkdir(parents=True,exist_ok=True)
    model=RollingGJR(len(banks),a.window,a.warmup,a.refit_every)
    predictions=[];states=[];started=time.perf_counter()
    for t in range(end):
        model.refit_if_due()
        q=model.forecast(dates[t])
        if not np.isfinite(q).all() or (q[:,0]>q[:,1]).any():
            raise FloatingPointError('Invalid forecast')
        if t>=a.warmup:
            predictions.append(q.copy())
            states.append(np.column_stack([model.params,model.h]).copy())
        # Current outcome is revealed only after issuing the forecast.
        model.update(dates[t],y[t],None)
        if (t+1)%500==0: print(f'Processed {t+1}/{end}',flush=True)
    elapsed=time.perf_counter()-started
    q=np.asarray(predictions);actual=y[a.warmup:end];evaluation_dates=dates[a.warmup:end]
    loss=np.maximum(TAU*(actual[:,:,None]-q),(TAU-1)*(actual[:,:,None]-q))
    hits=actual[:,:,None]<q
    np.save(a.output/'predictions.npy',q)
    np.savez_compressed(a.output/'forecast_states.npz',states=np.asarray(states),dates=evaluation_dates,banks=banks,
        columns=np.array(['mu','omega','alpha','gamma','beta','eta','lambda','variance']))
    rows=[];metrics=[]
    for b,bank in enumerate(banks):
        for j,tau in enumerate(TAU):
            metrics.append(dict(bank=bank,tail_probability=tau,pinball_loss=float(loss[:,b,j].mean()),
                                exceedance_rate=float(hits[:,b,j].mean()),exceedances=int(hits[:,b,j].sum())))
            rows.append(pd.DataFrame(dict(date=evaluation_dates,bank=bank,tail_probability=tau,
                confidence=1-tau,return_log_pct=actual[:,b],return_quantile=q[:,b,j],
                var_loss_log_pct=-q[:,b,j],exceeded=hits[:,b,j],pinball_loss=loss[:,b,j])))
    pd.concat(rows,ignore_index=True).to_csv(a.output/'predictions.csv',index=False)
    pd.DataFrame(metrics).to_csv(a.output/'metrics_by_bank.csv',index=False)
    summary=dict(window=a.window,warmup=a.warmup,refit_every=a.refit_every,observations=len(q),banks=len(banks),
        start=evaluation_dates[0],end=evaluation_dates[-1],tail_probabilities=TAU.tolist(),
        pinball_loss=loss.mean(axis=(0,1)).tolist(),exceedance_rate=hits.mean(axis=(0,1)).tolist(),
        elapsed_seconds_including_warmup=elapsed,data_sha256=hashlib.sha256(a.data.read_bytes()).hexdigest(),
        state=model.summary())
    if a.reference:
        ref=np.load(a.reference,allow_pickle=False)
        if ref.shape!=q.shape: raise ValueError('Reference shape mismatch; use full default run')
        summary['reference_max_abs_error']=float(np.max(np.abs(ref-q)))
        summary['reference_allclose']=bool(np.allclose(ref,q,rtol=1e-7,atol=1e-8))
        if not summary['reference_allclose']: raise AssertionError('Reference mismatch')
    (a.output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k!='state'},ensure_ascii=False,indent=2))

if __name__=='__main__': main()
