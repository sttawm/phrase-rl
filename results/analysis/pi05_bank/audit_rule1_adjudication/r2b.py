import sys; sys.path.insert(0,'/private/tmp/claude-501/-Users-sttawm-dev-robotics-phrase-rl/5032c440-d6d5-4ef4-9205-a261e39bc67c/scratchpad/adj')
from feat import build
from est import *
import numpy as np, pandas as pd
sys.path.insert(0,'.')
exec(open('/private/tmp/claude-501/-Users-sttawm-dev-robotics-phrase-rl/5032c440-d6d5-4ef4-9205-a261e39bc67c/scratchpad/adj/r2_repro.py').read().split('for mode,tag')[0].split('BANDS=')[0]) if False else None
BANDS=['floor(<20)','mid(20-70)','high(>70)']
def withincell_pooled(df, feat, seed=7, B=20000):
    per_task={}
    for t,g in df.groupby('task'):
        ds=[]
        for k,gg in g.groupby('kind'):
            a=gg[gg[feat]]; b=gg[~gg[feat]]
            if len(a)==0 or len(b)==0: continue
            ds.append(np.average(a.pct,weights=a.n)-np.average(b.pct,weights=b.n))
        if ds: per_task[t]=np.mean(ds)
    v=pd.Series(per_task,dtype=float); m,l,u=boot(v,B=B,seed=seed)
    return m,l,u,len(v),float(np.median(v)),int((v>0).sum()),int((v<0).sum())

d,H=build('loose_nocol')
print("head-noun sets (loose_nocol) sample:")
for c in list(sorted(H))[:6]: print("  ",H[c][0],"|",c)
un=d[d.kind!='oracle_board']
print("\nRULEBOOK TARGET: pooled +3.19 [+0.65,+6.10]; floor +0.36; mid +0.01; high +10.34 [+3.51,+18.52]")
m,l,u,nt,md,p,ng=withincell_pooled(un,'keep_all'); print(f"  MINE pooled  {m:+.2f} [{l:+.2f},{u:+.2f}] tasks={nt}")
for b in BANDS:
    m,l,u,nt,md,p,ng=withincell_pooled(un[un.canon_band==b],'keep_all')
    print(f"  MINE {b:11s} {m:+.2f} [{l:+.2f},{u:+.2f}] tasks={nt} {p}+/{ng}-")
print("\nRULEBOOK TARGET replication: adversarial-high +11.19 [+4.13,+19.78] (16 tasks); natural-high +3.43 [-0.15,+7.24] (12 tasks)")
for k in ['adversarial','natural']:
    g=un[(un.kind==k)&(un.canon_band=='high(>70)')]
    print("  MINE "+fmt(report(g,'keep_all',f'{k} high')))
