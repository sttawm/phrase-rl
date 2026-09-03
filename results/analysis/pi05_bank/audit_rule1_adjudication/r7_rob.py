import sys; sys.path.insert(0,'/private/tmp/claude-501/-Users-sttawm-dev-robotics-phrase-rl/5032c440-d6d5-4ef4-9205-a261e39bc67c/scratchpad/adj')
from feat import toks, norm
from est import *
import numpy as np, pandas as pd
d=pd.read_parquet('/private/tmp/claude-501/-Users-sttawm-dev-robotics-phrase-rl/5032c440-d6d5-4ef4-9205-a261e39bc67c/scratchpad/adj/d.parquet')
un=d[d.kind!='oracle_board'].copy()
ah=un[(un.kind=='adversarial')&(un.canon_band=='high(>70)')]

print("=== per-task contributions, adversarial x high (16 tasks) ===")
v=cell_diffs(ah,'keep_all').sort_values()
for t,x in v.items():
    g=ah[ah.task==t]; nv=int((~g.keep_all).sum())
    print(f"  {t:14s} diff {x:+7.2f}  canon {g.canon_pct.iloc[0]:5.1f}  viol_rows {nv:2d}/{len(g)}  viol_eps {int(g.loc[~g.keep_all,'n'].sum()):3d}  '{g.canonical.iloc[0][:52]}'")
print(f"  MEAN {v.mean():+.2f}   MEDIAN {v.median():+.2f}   sign {int((v>0).sum())}+/{int((v<0).sum())}-")
from math import comb
k=int((v>0).sum()); nn=int((v!=0).sum())
p2=sum(comb(nn,i) for i in range(k,nn+1))/2**nn*2
print(f"  exact two-sided sign test on {nn} non-zero tasks: p={min(1,p2):.3f}")
print("\n  leave-one-task-out mean:")
for t in v.index:
    print(f"    drop {t:14s} -> {v.drop(t).mean():+6.2f}")
print(f"  10% trimmed mean {float(np.mean(np.sort(v.values)[1:-1])):+.2f};  drop 3 most influential -> {v.drop(v.abs().sort_values().index[-3:]).mean():+.2f}")

print("\n=== the libero_90:10 question (rulebook says 18 of 32 high-band collapses are this one task) ===")
hi=un[un.canon_band=='high(>70)'].copy(); hi['collapse']=(hi.delta_vs_canon<=-40)
print("  collapses by task:", hi[hi.collapse].task.value_counts().to_dict())
for lab,g in [('adv-high ALL 16 tasks',ah),('adv-high minus libero_90:10',ah[ah.task!='libero_90:10'])]:
    print("  "+fmt(report(g,'keep_all',lab,B=8000,Bp=2000)))
for lab,g in [('collapse adv-high ALL',hi[hi.kind=='adversarial']),('collapse adv-high minus :10',hi[(hi.kind=='adversarial')&(hi.task!='libero_90:10')])]:
    gg=g.copy(); gg['drop_all']=~gg.keep_all; gg['collapse']=gg.collapse.astype(float)
    print("  "+fmt(report(gg,'drop_all',lab,outcome='collapse',B=8000,Bp=2000)))

print("\n=== does renaming survive controls for OVERALL distortion, INSIDE adversarial-high? ===")
# non-noun oov: oov words minus any token that is a canonical head noun or appears in the noun slot
ah2=ah.copy()
ah2['nn_oov']=ah2.n_oov
for q,lab in [(0,'lower half'),(1,'upper half')]:
    sub=[]
    for t,g in ah2.groupby('task'):
        med=g.nn_oov.median()
        sub.append(g[g.nn_oov<=med] if q==0 else g[g.nn_oov>med])
    s=pd.concat(sub)
    print(f"  distortion {lab:11s} "+fmt(report(s,'keep_all','keep-viol',B=8000,Bp=2000)))
# placebo: among noun-KEEPING adversarial-high rows, does distortion predict pct?
kp=ah2[ah2.keep_all].copy()
kp['dm']=kp.pct-kp.groupby('task').pct.transform('mean'); kp['dq']=kp.n_oov-kp.groupby('task').n_oov.transform('mean')
b=np.polyfit(kp.dq,kp.dm,1)[0]
print(f"  PLACEBO among keep-rows only: slope of demeaned pct on demeaned n_oov = {b:+.3f} pp/word (n={len(kp)})")
kw=ah2[ah2.keep_all].copy(); kw['dw']=kw.n_words-kw.groupby('task').n_words.transform('mean')
print(f"  PLACEBO among keep-rows only: slope on demeaned n_words = {np.polyfit(kw.dw,kp.dm,1)[0]:+.3f} pp/word")

print("\n=== TARGET (manipulated object) vs LANDMARK noun ===")
for k in ['adversarial','natural']:
    for b in ['high(>70)','mid(20-70)','floor(<20)']:
        g=un[(un.kind==k)&(un.canon_band==b)].copy()
        tg=g[g.keep_all | (g.miss_target)]
        lm=g[g.keep_all | (g.miss_landmark & ~g.miss_target)]
        for lab,s in [('TARGET missing',tg),('LANDMARK-only missing',lm)]:
            nv=int((~s.keep_all).sum())
            if nv<3: print(f"  {k:11s} {b:10s} {lab:22s} -- {nv} viol rows, UNMEASURED"); continue
            print(f"  {k:11s} {b:10s} "+fmt(report(s,'keep_all',lab,B=8000,Bp=2000)))
    print()
