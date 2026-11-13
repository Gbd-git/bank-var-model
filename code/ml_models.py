"""Eight reproducible quantile predictors. No bank shares trainable parameters.

Neural bank-specific networks are evaluated in a vectorized bank dimension.
All input/target scaling is frozen using only the initial training window.
"""
import numpy as np
from scipy.optimize import linprog
from scipy.spatial.distance import cdist
from scipy.linalg import eigh
TAU=np.array([.01,.05])
ML_CATALOG={
 'ml_kernel':('machine_learning/01_kernel','Nystrom-RBF Quantile'),
 'ml_qrf':('machine_learning/02_trees','Quantile Regression Forest'),
 'ml_hgbr':('machine_learning/02_trees','Quantile Histogram Boosting'),
 'ml_mlp':('machine_learning/03_feedforward','Quantile MLP'),
 'ml_lstm':('machine_learning/04_sequence','Quantile LSTM'),
 'ml_gru':('machine_learning/04_sequence','Quantile GRU'),
 'ml_tcn':('machine_learning/04_sequence','Quantile TCN'),
 'ml_transformer':('machine_learning/04_sequence','Quantile Transformer'),
}

def features(y):
    """Row t uses y[:t] only. Last 20 returns + past-window summaries."""
    n,k=y.shape;table=np.zeros((n,k,28));seq=np.zeros((n,k,20,2))
    for t in range(60,n):
        hist=y[t-20:t].T
        seq[t,:,:,0]=hist;seq[t,:,:,1]=abs(hist)
        table[t,:,:20]=hist[:,::-1]
        pos=20
        for w in (5,20,60):
            table[t,:,pos]=y[t-w:t].mean(axis=0);table[t,:,pos+1]=y[t-w:t].std(axis=0,ddof=1);pos+=2
        table[t,:,26]=np.square(np.minimum(y[t-20:t],0)).mean(axis=0)
        table[t,:,27]=abs(y[t-1])
    return table,seq

def kernel_fit(Z,y,alpha,penalty=.5):
    """Exact L1-regularized quantile LP on the explicit kernel features.

    min sum pinball(y-b-Zw) + penalty*||w||_1 (intercept unpenalized).
    Dual: sum u=0, |Z.T u|<=penalty, alpha-1<=u<=alpha.
    """
    p=Z.shape[1]
    r=linprog(-y,A_ub=np.r_[Z.T,-Z.T],b_ub=np.full(2*p,penalty),
        A_eq=np.ones((1,len(y))),b_eq=[0.],bounds=[(alpha-1,alpha)]*len(y),method='highs')
    if not r.success:raise RuntimeError(r.message)
    coef=r.ineqlin.marginals[p:]-r.ineqlin.marginals[:p]
    return float(-r.eqlin.marginals[0]),coef

def nystrom_map(train,X,seed,components=32,gamma=1/28):
    """K(X,C) K(C,C)^(-1/2), landmarks chosen only from initial training rows.

    cdist/einsum avoid tiny BLAS matrix products in platform-specific backends.
    """
    ids=np.random.RandomState(seed).permutation(len(train))[:components];centers=train[ids]
    W=np.exp(-gamma*cdist(centers,centers,'sqeuclidean'))
    vals,U=eigh(W);normalizer=np.einsum('ik,k,jk->ij',U,1/np.sqrt(np.maximum(vals,1e-12)),U)
    K=np.exp(-gamma*cdist(X,centers,'sqeuclidean'))
    Z=np.einsum('ij,jk->ik',K,normalizer)
    if not np.isfinite(Z).all():raise RuntimeError('Invalid kernel features')
    return Z

def forest_quantiles(forest,X,y,query):
    """Leaf-weighted empirical CDF; NOT quantiles of tree mean predictions."""
    leaves=forest.apply(X);new=forest.apply(query);order=np.argsort(y);ys=y[order]
    out=[]
    for leaf in new:
        same=leaves==leaf[None,:]
        weight=(same/np.maximum(same.sum(axis=0,keepdims=True),1)).mean(axis=1)
        cum=np.cumsum(weight[order]);cum/=cum[-1]
        # Floating cumulative sums can fall a few ulps below an exact mass boundary.
        out.append(ys[np.minimum(np.searchsorted(cum,TAU-1e-12),len(ys)-1)])
    return np.asarray(out)

def run_ml(model,y,warmup=2000,window=1500,refit_every=5,seed=20261113):
    if not (60<=warmup-window and 50<=window and warmup<len(y) and refit_every>=1):
        raise ValueError('ML requires >=60 historical feature rows before its initial label window')
    from threadpoolctl import threadpool_limits
    threadpool_limits(1)
    X,S=features(y);n,k=y.shape
    initial=slice(warmup-window,warmup)
    xm=X[initial].mean(axis=0);xs=np.maximum(X[initial].std(axis=0),1e-6)
    ym=y[initial].mean(axis=0);ys=np.maximum(y[initial].std(axis=0),1e-6)
    X=(X-xm)/xs;target=(y-ym)/ys
    # Sequence channels retain signed returns and magnitudes, per bank scale.
    S=S/ys[None,:,None,None]
    raw=np.zeros((n-warmup,k,2));events=[]
    config=dict(seed=seed,feature_lookback=60,sequence_length=20,scaling='frozen_initial_training_window',
        fitting_scope='independent_bank_parameters',refit_every=refit_every,threads=1)
    if model in ('ml_kernel','ml_qrf','ml_hgbr'):
        from sklearn.ensemble import RandomForestRegressor,HistGradientBoostingRegressor
        Z=[]
        if model=='ml_kernel':
            config.update(components=32,gamma=1/28,l1_penalty=.5)
            for b in range(k):
                Z.append(nystrom_map(X[initial,b],X[:,b],seed+b))
        elif model=='ml_qrf':config.update(trees=32,max_depth=5,min_samples_leaf=50,max_features=.5,bootstrap=False,forest_workers=4,cdf_boundary_tolerance=1e-12)
        else:config.update(iterations=40,max_leaf_nodes=7,min_samples_leaf=40,max_bins=32,learning_rate=.05,l2_regularization=1.,early_stopping=False)
        for t in range(warmup,n,refit_every):
            end=min(t+refit_every,n);sl=slice(t-window,t)
            for b in range(k):
                a=target[sl,b]
                if model=='ml_kernel':
                    qb=np.empty((end-t,2))
                    for j,tau in enumerate(TAU):
                        bias,coef=kernel_fit(Z[b][sl],a,tau)
                        qb[:,j]=bias+np.einsum('ij,j->i',Z[b][t:end],coef)
                elif model=='ml_qrf':
                    m=RandomForestRegressor(n_estimators=32,max_depth=5,min_samples_leaf=50,max_features=.5,
                        bootstrap=False,n_jobs=4,random_state=seed+b)
                    m.fit(X[sl,b],a);qb=forest_quantiles(m,X[sl,b],a,X[t:end,b])
                else:
                    qb=np.empty((end-t,2))
                    for j,tau in enumerate(TAU):
                        m=HistGradientBoostingRegressor(loss='quantile',quantile=float(tau),max_iter=40,max_leaf_nodes=7,
                            min_samples_leaf=40,max_bins=32,learning_rate=.05,l2_regularization=1.,early_stopping=False,random_state=seed+b)
                        m.fit(X[sl,b],a);qb[:,j]=m.predict(X[t:end,b])
                raw[t-warmup:end-warmup,b]=ym[b]+ys[b]*qb
            if (t-warmup)%500==0:print(model,'forecasted through',end,flush=True)
    else:
        import torch
        from ml_networks import BankNet
        torch.set_num_threads(1);torch.manual_seed(seed);torch.use_deterministic_algorithms(True)
        net=BankNet(model,k,X.shape[-1],hidden=12)
        # Zero output weights and empirical lower-tail intercepts at initialization.
        with torch.no_grad():
            net.out.weight.zero_();net.out.bias.copy_(torch.tensor(np.quantile(target[initial],TAU,axis=0).T,dtype=torch.float32))
        optimizer=torch.optim.Adam(net.parameters(),lr=.001,weight_decay=1e-5)
        rng=np.random.default_rng(seed);tx=torch.tensor(X,dtype=torch.float32);ts=torch.tensor(S,dtype=torch.float32)
        ty=torch.tensor(target,dtype=torch.float32);tau=torch.tensor(TAU,dtype=torch.float32)
        config.update(hidden=12,optimizer='Adam',learning_rate=.001,weight_decay=1e-5,
            initial_steps=200,refresh_steps=12,batch_size=128,gradient_clip=5.,device='cpu',
            training='warm_start_parameters_and_optimizer_with_trailing_window_minibatches',
            parameters=sum(p.numel() for p in net.parameters()))
        train_log=[]
        for t in range(warmup,n,refit_every):
            steps=200 if t==warmup else 12
            net.train()
            for _ in range(steps):
                idx=rng.integers(t-window,t,size=128)
                q=net(tx[idx],ts[idx]);e=ty[idx,:,None]-q
                # Sum independent per-bank losses; no cross-bank features or parameters.
                loss=torch.maximum(tau*e,(tau-1)*e).mean(dim=(0,2)).sum()
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite neural objective')
                optimizer.zero_grad();loss.backward()
                # Clip each bank separately: global clipping would couple banks.
                net.clip_grad(5.);optimizer.step()
            end=min(t+refit_every,n);net.eval()
            with torch.no_grad():
                q=net(tx[t:end],ts[t:end]).numpy()
            raw[t-warmup:end-warmup]=ym[None,:,None]+ys[None,:,None]*q
            train_log.append(dict(position=t,steps=steps,last_minibatch_loss=float(loss.detach())))
            if (t-warmup)%500==0:print(model,'forecasted through',end,flush=True)
        config['training_log']=train_log
    if not np.isfinite(raw).all():raise RuntimeError('Nonfinite predictions')
    return np.sort(raw,axis=-1),raw,dict(events=events,crossings=int((raw[:,:,0]>raw[:,:,1]).sum()),configuration=config)
