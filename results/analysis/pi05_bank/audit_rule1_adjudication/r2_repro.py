import sys; sys.path.insert(0,'/private/tmp/claude-501/-Users-sttawm-dev-robotics-phrase-rl/5032c440-d6d5-4ef4-9205-a261e39bc67c/scratchpad/adj')
from feat import build
from est import *
import numpy as np, pandas as pd
BANDS=['floor(<20)','mid(20-70)','high(>70)']

def withincell_pooled(df, feat, restrict_bands=None, seed=7, B=20000):
    """Rulebook's stated estimator: diff inside each (task x kind) cell, average the task's cells,
       then average over tasks."""
    per_task={}
    for t,g in df.groupby('task'):
        ds=[]
        for k,gg in g.groupby('kind'):
            a=gg[gg[feat]]; b=gg[~gg[feat]]
            if len(a)==0 or len(b)==0: continue
            ds.append(np.average(a.pct,weights=a.n)-np.average(b.pct,weights=b.n))
        if ds: per_task[t]=np.mean(ds)
    v=pd.Series(per_task,dtype=float)
    m,l,u=boot(v,B=B,seed=seed)
    return m,l,u,len(v),float(np.median(v)),int((v>0).sum()),int((v<0).sum())

for mode,tag in [('head','HEAD-NOUN-ONLY (mine, A, C)'),('loose','ALL-CONTENT-TOKEN (rulebook / B-ish)')]:
    d,H=build(mode)
    un=d[d.kind!='oracle_board']
    print(f"\n########## FEATURE = {tag} ##########")
    print("-- within-cell-averaged pooled (the rulebook's stated estimator), unsearched only --")
    m,l,u,nt,md,p,ng=withincell_pooled(un,'keep_all')
    print(f"  pooled  {m:+.2f} [{l:+.2f},{u:+.2f}]  tasks={nt} med={md:+.2f} {p}+/{ng}-")
    for b in BANDS:
        m,l,u,nt,md,p,ng=withincell_pooled(un[un.canon_band==b],'keep_all')
        print(f"  {b:11s} {m:+.2f} [{l:+.2f},{u:+.2f}]  tasks={nt} med={md:+.2f} {p}+/{ng}-")
    print("-- within-family HIGH BAND (the claimed replication) --")
    for k in ['adversarial','natural']:
        g=un[(un.kind==k)&(un.canon_band=='high(>70)')]
        r=report(g,'keep_all',f'  {k} high'); print("  "+fmt(r))
    print("-- within-family POOLED over bands --")
    for k in ['adversarial','natural']:
        g=un[un.kind==k]
        r=report(g,'keep_all',f'  {k} pooled'); print("  "+fmt(r))
