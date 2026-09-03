import sys; sys.path.insert(0,'/private/tmp/claude-501/-Users-sttawm-dev-robotics-phrase-rl/5032c440-d6d5-4ef4-9205-a261e39bc67c/scratchpad/adj')
from feat import build
from est import *
import numpy as np, pandas as pd
pd.set_option('display.width',200)

d,H=build('head')
print("=== PREVALENCE: rows that actually vary on 'any head noun missing' ===")
tab=[]
for (k,b),g in d.groupby(['kind','canon_band']):
    v=g.groupby('task')['keep_all'].nunique()
    tab.append(dict(kind=k,band=b,rows=len(g),viol=int((~g.keep_all).sum()),
                    pct_viol=round(100*(~g.keep_all).mean(),1),
                    tasks=g.task.nunique(), tasks_vary=int((v>1).sum()),
                    eps_keep=int(g.loc[g.keep_all,'n'].sum()), eps_viol=int(g.loc[~g.keep_all,'n'].sum())))
print(pd.DataFrame(tab).sort_values(['kind','band']).to_string(index=False))
print()
print("Overall P(violation) by kind:", d.groupby('kind').keep_all.apply(lambda s: round(100*(~s).mean(),1)).to_dict())

print("\n=== KEEP-ALL minus VIOLATION, within kind x band, task-clustered, episode-weighted ===")
BANDS=['floor(<20)','mid(20-70)','high(>70)']
for k in ['natural','adversarial','oracle_board']:
    for b in BANDS:
        g=d[(d.kind==k)&(d.canon_band==b)]
        print(fmt(report(g,'keep_all',f'{k:12s} {b}')))
    g=d[d.kind==k]
    print(fmt(report(g,'keep_all',f'{k:12s} ALL BANDS')))
    print()

print("=== pooled over kinds (rulebook's headline framing) ===")
un=d[d.kind!='oracle_board']
for b in BANDS:
    print(fmt(report(un[un.canon_band==b],'keep_all',f'nat+adv {b}')))
print(fmt(report(un,'keep_all','nat+adv POOLED (all bands)')))
print(fmt(report(d,'keep_all','all3kinds POOLED')))
