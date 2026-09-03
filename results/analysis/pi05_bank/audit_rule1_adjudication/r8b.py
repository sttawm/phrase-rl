import sys; sys.path.insert(0,'/private/tmp/claude-501/-Users-sttawm-dev-robotics-phrase-rl/5032c440-d6d5-4ef4-9205-a261e39bc67c/scratchpad/adj')
from est import *
import numpy as np, pandas as pd
d=pd.read_parquet('/private/tmp/claude-501/-Users-sttawm-dev-robotics-phrase-rl/5032c440-d6d5-4ef4-9205-a261e39bc67c/scratchpad/adj/d.parquet')
un=d[d.kind!='oracle_board'].copy(); un['viol']=(~un.keep_all).astype(float)
hi=un[un.canon_band=='high(>70)'].copy(); hi['collapse']=(hi.delta_vs_canon<=-40).astype(float)

print("=== RAW 14.3%-vs-6.4% headline: driven by libero_90:10? ===")
for lab,g in [('all 16 high tasks',hi),('minus libero_90:10',hi[hi.task!='libero_90:10'])]:
    k=g[g.keep_all]; v=g[~g.keep_all]
    print(f"  {lab:20s} POOLED keep {100*k.collapse.mean():5.1f}% ({len(k)}) vs viol {100*v.collapse.mean():5.1f}% ({len(v)})")
    for kind in ['natural','adversarial']:
        gg=g[g.kind==kind]; a=gg[gg.keep_all]; b=gg[~gg.keep_all]
        print(f"      {kind:11s} keep {100*a.collapse.mean():5.1f}% ({len(a):3d}) vs viol {100*b.collapse.mean():5.1f}% ({len(b):3d})")

print("\n=== adjusted beta_viol: within (task x kind) episode-weighted demeaning + n_words + n_oov ===")
def prep(df):
    Xs=[];Ys=[];Ws=[];Ts=[]
    for (t,k),g in df.groupby(['task','kind']):
        if g.viol.nunique()<2: continue
        w=g.n.values.astype(float); dm=lambda a:(a-np.average(a,weights=w))
        Xs.append(np.c_[dm(g.viol.values),dm(g.n_words.values.astype(float)),dm(g.n_oov.values.astype(float))])
        Ys.append(dm(g.pct.values)); Ws.append(w); Ts.append(np.array([t]*len(g)))
    return np.vstack(Xs),np.concatenate(Ys),np.concatenate(Ws),np.concatenate(Ts)
def wls(X,Y,W):
    s=np.sqrt(W)[:,None]; b,*_=np.linalg.lstsq(X*s,Y*np.sqrt(W),rcond=None); return b
for lab,g in [('adversarial high',hi[hi.kind=='adversarial']),('natural high',hi[hi.kind=='natural']),
              ('adversarial ALL',un[un.kind=='adversarial']),('natural ALL',un[un.kind=='natural']),
              ('nat+adv high',hi)]:
    X,Y,W,T=prep(g); b=wls(X,Y,W); tasks=np.unique(T); r=np.random.default_rng(3)
    idx={t:np.where(T==t)[0] for t in tasks}; bs=[]
    for _ in range(4000):
        sel=np.concatenate([idx[t] for t in r.choice(tasks,len(tasks),replace=True)])
        try: bs.append(wls(X[sel],Y[sel],W[sel])[0])
        except Exception: pass
    bs=np.array(bs)
    print(f"  {lab:18s} beta_viol {b[0]:+7.2f} [{np.percentile(bs,2.5):+6.2f},{np.percentile(bs,97.5):+6.2f}]"
          f"  beta_n_words {b[1]:+6.3f}  beta_n_oov {b[2]:+6.3f}")

print("\n=== length-caliper matched, adversarial high (|dn_words|<=3, greedy 1:1 within task) ===")
rows=[]
for t,g in hi[hi.kind=='adversarial'].groupby('task'):
    K=g[g.keep_all]; V=g[~g.keep_all]; used=set(); ds=[]
    for v in V.itertuples():
        c=sorted([(abs(k.n_words-v.n_words),k.Index,k.pct) for k in K.itertuples() if k.Index not in used and abs(k.n_words-v.n_words)<=3])
        if c: used.add(c[0][1]); ds.append(c[0][2]-v.pct)
    if ds: rows.append((t,float(np.mean(ds)),len(ds)))
vv=pd.Series({t:x for t,x,_ in rows}); m,l,u=boot(vv,B=20000,seed=5)
print(f"  pairs={sum(n for _,_,n in rows)} tasks={len(vv)}  keep-minus-viol {m:+.2f} [{l:+.2f},{u:+.2f}] median {vv.median():+.2f}")

print("\n=== what does a MILD rename cost? ===")
nh=hi[hi.kind=='natural']; k=nh[nh.keep_all]; v=nh[~nh.keep_all]
print(f"  natural high keep n={len(k)} mean pct {k.pct.mean():.1f} delta {np.average(k.delta_vs_canon,weights=k.n):+.2f}")
print(f"  natural high viol n={len(v)} mean pct {v.pct.mean():.1f} delta {np.average(v.delta_vs_canon,weights=v.n):+.2f}  collapses {int(v.collapse.sum())}/{len(v)}")
for kk in ['natural','adversarial']:
    a=un[(un.kind==kk)&(un.viol==1)]; b=un[(un.kind==kk)&(un.viol==0)]
    print(f"  {kk:11s} ALL bands: collapse<=-40  renamed {100*(a.delta_vs_canon<=-40).mean():.1f}% (n={len(a)}) vs kept {100*(b.delta_vs_canon<=-40).mean():.1f}% (n={len(b)})")

print("\n=== P(pct==0) and thresholds, adversarial high (auditor C: 9.1% vs 9.1% matched) ===")
g=hi[hi.kind=='adversarial']
print(f"  pct==0 : keep {100*(g[g.keep_all].pct==0).mean():.1f}% ({len(g[g.keep_all])}) vs viol {100*(g[~g.keep_all].pct==0).mean():.1f}% ({len(g[~g.keep_all])})")
for thr in [-20,-40,-50,-60]:
    print(f"  delta<={thr:4d}: keep {100*(g[g.keep_all].delta_vs_canon<=thr).mean():5.1f}% vs viol {100*(g[~g.keep_all].delta_vs_canon<=thr).mean():5.1f}%")
