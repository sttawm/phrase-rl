import sys; sys.path.insert(0,'/private/tmp/claude-501/-Users-sttawm-dev-robotics-phrase-rl/5032c440-d6d5-4ef4-9205-a261e39bc67c/scratchpad/adj')
from feat import build
from est import *
import numpy as np, pandas as pd

for mode in ['loose_nocol','head']:
    d,H=build(mode)
    un=d[d.kind!='oracle_board'].copy()
    print(f"\n######## FEATURE={mode} : TAIL RISK ########")
    for thr in [-40,-50]:
        un['collapse']=(un.delta_vs_canon<=thr).astype(float)
        hi=un[un.canon_band=='high(>70)']
        # raw rates, rulebook framing (pooled over kinds)
        k=hi[hi.keep_all]; v=hi[~hi.keep_all]
        print(f"\n thr={thr}pp  HIGH BAND raw pooled-over-kind: keep {100*k.collapse.mean():.1f}% ({len(k)} rows) "
              f"vs viol {100*v.collapse.mean():.1f}% ({len(v)} rows)  [rulebook says 6.4% vs 14.3% at -40]")
        for kind in ['natural','adversarial']:
            g=hi[hi.kind==kind]; a=g[g.keep_all]; b=g[~g.keep_all]
            print(f"   raw {kind:11s}: keep {100*a.collapse.mean():5.1f}% ({len(a):3d}) vs viol {100*b.collapse.mean():5.1f}% ({len(b):3d})")
        # task-clustered: P(collapse | viol) - P(collapse | keep), so POSITIVE = keeping helps
        for lab,g in [('nat+adv high',hi),('adversarial high',hi[hi.kind=='adversarial']),
                      ('natural high',hi[hi.kind=='natural']),
                      ('nat+adv ALL bands',un),('adversarial ALL',un[un.kind=='adversarial']),
                      ('natural ALL',un[un.kind=='natural'])]:
            gg=g.copy(); gg['drop_all']=~gg.keep_all
            r=report(gg,'drop_all',f'   collapse[viol-keep] {lab}',outcome='collapse',B=20000,Bp=3000)
            print("   "+fmt(r).replace('  ',' ',0))
