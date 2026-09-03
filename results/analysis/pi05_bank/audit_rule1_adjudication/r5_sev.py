import sys; sys.path.insert(0,'/private/tmp/claude-501/-Users-sttawm-dev-robotics-phrase-rl/5032c440-d6d5-4ef4-9205-a261e39bc67c/scratchpad/adj')
from est import *
import numpy as np, pandas as pd
d=pd.read_parquet('/private/tmp/claude-501/-Users-sttawm-dev-robotics-phrase-rl/5032c440-d6d5-4ef4-9205-a261e39bc67c/scratchpad/adj/d.parquet')
un=d[d.kind!='oracle_board'].copy()
un['viol']=~un.keep_all

print("=== SUBSTITUTION SIZE (novel content words occupying the missing noun's slot), violation rows ===")
for k in ['natural','adversarial']:
    g=un[(un.kind==k)&un.viol]
    print(f" {k:12s} n={len(g):4d}  n_novel_sub dist: "+
          "  ".join(f"{v}:{c}" for v,c in sorted(g.n_novel_sub.value_counts().items())[:8])+
          f"   median={g.n_novel_sub.median():.0f} mean={g.n_novel_sub.mean():.2f}")
    gh=g[g.canon_band=='high(>70)']
    print(f"   high band only n={len(gh)} median={gh.n_novel_sub.median():.0f} "
          f"  P(1-word swap)={100*(gh.n_novel_sub<=1).mean():.0f}%  P(>=3 words)={100*(gh.n_novel_sub>=3).mean():.0f}%")

print("\n=== MILD (1 novel word in the slot) vs AGGRESSIVE (>=2) renaming, vs KEEP ===")
print("    contrast is KEEP minus that violation subtype, within kind x band, task-clustered")
for k in ['adversarial','natural']:
    for b in ['high(>70)','mid(20-70)','floor(<20)']:
        g=un[(un.kind==k)&(un.canon_band==b)]
        for lab,sub in [('keep vs MILD(<=1)', g[g.keep_all | (g.viol&(g.n_novel_sub<=1))]),
                        ('keep vs AGGR(>=2)', g[g.keep_all | (g.viol&(g.n_novel_sub>=2))]),
                        ('keep vs AGGR(>=3)', g[g.keep_all | (g.viol&(g.n_novel_sub>=3))])]:
            if (~sub.keep_all).sum()<3: 
                print(f"  {k:11s} {b:10s} {lab:18s}  -- only {(~sub.keep_all).sum()} viol rows, skip")
                continue
            print(f"  {k:11s} {b:10s} "+fmt(report(sub,'keep_all',lab,B=8000,Bp=2000)))
    print()

print("=== dose-response on within-(task x kind) demeaned pct, high band ===")
for k in ['adversarial','natural']:
    g=un[(un.kind==k)&(un.canon_band=='high(>70)')].copy()
    g['dm']=g.pct-g.groupby('task').pct.transform(lambda s: s.mean())
    bins=[(0,0,'keep'),(1,1,'1 novel'),(2,2,'2 novel'),(3,4,'3-4'),(5,99,'5+')]
    print(f" {k}:")
    for lo,hi,lab in bins:
        s=g[(g.n_novel_sub>=lo)&(g.n_novel_sub<=hi)]
        if len(s)==0: continue
        print(f"   {lab:8s} n={len(s):4d}  demeaned pct {s.dm.mean():+6.2f}  raw {s.pct.mean():5.1f}  collapse<=-40 {100*(s.delta_vs_canon<=-40).mean():5.1f}%")
