import sys; sys.path.insert(0,'/private/tmp/claude-501/-Users-sttawm-dev-robotics-phrase-rl/5032c440-d6d5-4ef4-9205-a261e39bc67c/scratchpad/adj')
from est import *
import numpy as np, pandas as pd
d=pd.read_parquet('/private/tmp/claude-501/-Users-sttawm-dev-robotics-phrase-rl/5032c440-d6d5-4ef4-9205-a261e39bc67c/scratchpad/adj/d.parquet')
un=d[d.kind!='oracle_board'].copy()
hi=un[un.canon_band=='high(>70)']

print("=== LANDMARK-only violations, high band pooled over kinds ===")
g=hi.copy(); s=g[g.keep_all | (g.miss_landmark & ~g.miss_target)]
print("  "+fmt(report(s,'keep_all','landmark-only high nat+adv',B=8000,Bp=2000)))
s2=un[un.keep_all | (un.miss_landmark & ~un.miss_target)]
print("  "+fmt(report(s2,'keep_all','landmark-only ALL bands nat+adv',B=8000,Bp=2000)))

print("\n=== permutation on the MEDIAN task effect (heavy-tail robustness), adversarial high ===")
ah=hi[hi.kind=='adversarial']
o,sd,p=perm(ah,'keep_all',B=5000,seed=21,stat='median')
print(f"  median task effect {o:+.2f}, null sd {sd:.2f}, p={p:.4f}")
o,sd,p=perm(hi,'keep_all',B=5000,seed=22,stat='median')
print(f"  nat+adv high median {o:+.2f}, null sd {sd:.2f}, p={p:.4f}")

print("\n=== THE PROPOSED NARROWED FEATURE ===")
print("  bad = the TARGET head noun is replaced by a multi-word description (>=2 novel content words in its slot)")
un2=un.copy()
un2['narrow_ok'] = ~(un2.miss_target & (un2.n_novel_sub>=2))
for k in ['natural','adversarial']:
    print(f"  prevalence {k:12s}: violates narrow rule {100*(~un2[un2.kind==k].narrow_ok).mean():5.1f}%   "
          f"violates rule-1-as-written {100*(~un2[un2.kind==k].keep_all).mean():5.1f}%")
for k in ['adversarial','natural']:
    for b in ['high(>70)','mid(20-70)','floor(<20)']:
        gg=un2[(un2.kind==k)&(un2.canon_band==b)]
        nv=int((~gg.narrow_ok).sum())
        if nv<3: print(f"  {k:11s} {b:10s} -- {nv} viol rows, unmeasured"); continue
        print(f"  {k:11s} {b:10s} "+fmt(report(gg,'narrow_ok','ok - periphrastic-target',B=8000,Bp=2000)))
print("  "+fmt(report(un2[un2.canon_band=='high(>70)'],'narrow_ok','nat+adv HIGH',B=8000,Bp=2000)))
print("  "+fmt(report(un2,'narrow_ok','nat+adv POOLED',B=8000,Bp=2000)))
hh=un2[un2.canon_band=='high(>70)'].copy(); hh['collapse']=(hh.delta_vs_canon<=-40).astype(float); hh['bad']=~hh.narrow_ok
print("  collapse "+fmt(report(hh,'bad','nat+adv HIGH collapse[bad-ok]',outcome='collapse',B=8000,Bp=2000)))
print("  collapse "+fmt(report(hh[hh.kind=='adversarial'],'bad','adversarial HIGH collapse',outcome='collapse',B=8000,Bp=2000)))
