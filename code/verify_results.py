"""Independently recompute every saved metric and validate prediction alignment."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from models import CATALOG,TAU,model_relative_path
ROOT=Path(__file__).resolve().parents[1]

def main():
    d=np.load(ROOT/'data/bank_returns.npz',allow_pickle=False);checks=[]
    for m,(cat,_) in CATALOG.items():
        f=ROOT/model_relative_path(m);s=json.loads((f/'summary.json').read_text());q=np.load(f/'predictions.npy')
        raw=np.load(f/'raw_predictions.npy');y=d['y'][s['warmup']:];metric=pd.read_csv(f/'metrics_by_bank.csv')
        assert q.shape==(*y.shape,2) and np.isfinite(q).all()
        assert np.array_equal(q,np.sort(raw,axis=2))
        for b,bank in enumerate(d['banks']):
            for j,tau in enumerate(TAU):
                r=metric[(metric.bank==bank)&np.isclose(metric.confidence,1-tau)].iloc[0]
                u=y[:,b]-q[:,b,j];loss=u*(tau-(u<0))
                assert abs(loss.mean()-r.pinball_loss)<1e-12
                assert np.count_nonzero(u<0)==r.violations
                assert abs(np.mean(u<0)-r.violation_rate)<1e-12
        csv=pd.read_csv(f/'predictions.csv')
        assert len(csv)==y.size
        assert np.allclose(csv['return'],y.reshape(-1),rtol=0,atol=1e-12)
        for j,col in enumerate(['q_01','q_05']):assert np.allclose(csv[col],q[:,:,j].reshape(-1),rtol=0,atol=1e-12)
        assert list(csv.date[::14])==list(d['dates'][s['warmup']:].astype(str))
        checks.append(dict(model=m,shape=list(q.shape),finite=True,metrics_recomputed=True,csv_aligned=True))
    print(json.dumps(checks,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
