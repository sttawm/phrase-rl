import numpy as np, pandas as pd
RNG=np.random.default_rng(20260903)

def cell_diffs(df, feat, outcome='pct', lo=False, hi=True):
    """One paired diff per task: episode-weighted mean(outcome | feat==hi) - mean(outcome | feat==lo).
       df must already be restricted to one (kind x band) stratum."""
    out={}
    for t,g in df.groupby('task'):
        a=g[g[feat]==hi]; b=g[g[feat]==lo]
        if len(a)==0 or len(b)==0: continue
        ma=np.average(a[outcome],weights=a['n']); mb=np.average(b[outcome],weights=b['n'])
        out[t]=ma-mb
    return pd.Series(out,dtype=float)

def boot(vals, B=20000, seed=7):
    v=np.asarray(vals,dtype=float)
    if len(v)==0: return (np.nan,np.nan,np.nan)
    r=np.random.default_rng(seed)
    idx=r.integers(0,len(v),size=(B,len(v)))
    m=v[idx].mean(axis=1)
    return v.mean(), np.percentile(m,2.5), np.percentile(m,97.5)

def perm(df, feat, outcome='pct', B=5000, seed=11, stat='mean', lo=False, hi=True):
    """Permute the feature label WITHIN each task (n stays glued to the row)."""
    r=np.random.default_rng(seed)
    groups=[]
    for t,g in df.groupby('task'):
        f=g[feat].values; 
        if f.sum()==0 or (~f.astype(bool)).sum()==0:
            if not ((f==hi).any() and (f==lo).any()): continue
        y=g[outcome].values.astype(float); w=g['n'].values.astype(float)
        lab=(g[feat].values==hi)
        if lab.all() or (~lab).all(): continue
        groups.append((y,w,lab))
    if not groups: return np.nan,np.nan,np.nan
    def agg(labs):
        ds=[]
        for (y,w,_),l in zip(groups,labs):
            ds.append(np.average(y[l],weights=w[l])-np.average(y[~l],weights=w[~l]))
        return np.mean(ds) if stat=='mean' else np.median(ds)
    obs=agg([l for _,_,l in groups])
    null=np.empty(B)
    for b in range(B):
        labs=[]
        for (y,w,l) in groups:
            p=r.permutation(l)
            while p.all() or (~p).all(): p=r.permutation(l)
            labs.append(p)
        null[b]=agg(labs)
    p=( (np.abs(null)>=abs(obs)).sum()+1 )/(B+1)
    return obs, null.std(), p

def report(df, feat, label, B=20000, Bp=3000, seed=7, lo=False, hi=True, outcome='pct'):
    v=cell_diffs(df,feat,outcome=outcome,lo=lo,hi=hi)
    m,l,u=boot(v,B=B,seed=seed)
    o,sd,p=perm(df,feat,outcome=outcome,B=Bp,seed=seed+1,lo=lo,hi=hi)
    nhi=int((df[feat]==hi).sum()); nlo=int((df[feat]==lo).sum())
    return dict(label=label, tasks_vary=len(v), tasks_cell=df.task.nunique(),
                rows_hi=nhi, rows_lo=nlo, eps_hi=int(df.loc[df[feat]==hi,'n'].sum()),
                eps_lo=int(df.loc[df[feat]==lo,'n'].sum()),
                mean=m, lo=l, hi=u, median=float(np.median(v)) if len(v) else np.nan,
                npos=int((v>0).sum()), nneg=int((v<0).sum()),
                null_sd=sd, perm_p=p)

def fmt(r):
    return (f"{r['label']:34s} {r['mean']:+7.2f} [{r['lo']:+6.2f},{r['hi']:+6.2f}] "
            f"med {r['median']:+6.2f} {r['npos']:2d}+/{r['nneg']:2d}- "
            f"tasks {r['tasks_vary']:2d}/{r['tasks_cell']:2d} rows {r['rows_lo']:4d}v{r['rows_hi']:4d} "
            f"nullSD {r['null_sd']:5.2f} p={r['perm_p']:.4f}")
