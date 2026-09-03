import sys; sys.path.insert(0,'/private/tmp/claude-501/-Users-sttawm-dev-robotics-phrase-rl/5032c440-d6d5-4ef4-9205-a261e39bc67c/scratchpad/adj')
from feat import build, toks, norm, heads, np_spans, POSNL
from est import *
import numpy as np, pandas as pd
from collections import defaultdict

d,H=build('head')

# --- corpus-derived generic vocabulary: tokens used by phrases of >=45 of the 55 tasks ---
tasks_with=defaultdict(set)
for r in d.itertuples():
    for w in set(norm(x) for x in toks(r.phrase)): tasks_with[w].add(r.task)
GEN={w for w,ts in tasks_with.items() if len(ts)>=45}
print("GENERIC VOCAB (>=45/55 tasks), n=%d:"%len(GEN)); print(" ",sorted(GEN))
OBJ={norm(h) for c in H for h in H[c][0]}
print("object head nouns wrongly inside generic vocab:", sorted(OBJ & GEN))

def classify(canon, phrase):
    """For each missing head noun, decide REPLACED vs DELETED by inspecting the phrase span
       between the flanking canonical anchors that survived."""
    ct=[norm(w) for w in toks(canon)]; pt=[norm(w) for w in toks(phrase)]
    pset=set(pt); hs,tgt=H[canon]
    hsn=[norm(h) for h in hs]
    canon_content=set(ct)
    # anchors: canonical tokens (not the missing nouns) that appear in the phrase, matched in order
    res=[]
    for h in hsn:
        if h in pset: continue
        i=ct.index(h)
        prev=[w for w in ct[:i] if w in pset and w not in hsn]
        nxt =[w for w in ct[i+1:] if w in pset and w not in hsn]
        # locate span in phrase
        lo=0
        if prev:
            try: lo=pt.index(prev[-1])+1
            except ValueError: lo=0
        hiI=len(pt)
        if nxt:
            cands=[j for j,w in enumerate(pt) if w==nxt[0] and j>=lo]
            if cands: hiI=cands[0]
        span=pt[lo:hiI]
        novel=[w for w in span if w not in canon_content and w not in GEN]
        res.append(dict(noun=h, span_len=len(span), n_novel=len(novel),
                        cls='REPLACED' if novel else 'DELETED',
                        role='target' if h==norm(tgt) else 'landmark', novel=novel))
    return res

rows=[]
for r in d.itertuples():
    cs=classify(r.canonical,r.phrase)
    if not cs: rows.append(dict(cls='KEEP',n_novel_sub=0,max_span=0,role_miss=''))
    else:
        cls='REPLACED' if all(c['cls']=='REPLACED' for c in cs) else ('DELETED' if all(c['cls']=='DELETED' for c in cs) else 'MIXED')
        rows.append(dict(cls=cls,n_novel_sub=sum(c['n_novel'] for c in cs),
                         max_span=max(c['span_len'] for c in cs),
                         role_miss='+'.join(sorted({c['role'] for c in cs}))))
d=pd.concat([d.reset_index(drop=True),pd.DataFrame(rows)],axis=1)
d.to_parquet('/private/tmp/claude-501/-Users-sttawm-dev-robotics-phrase-rl/5032c440-d6d5-4ef4-9205-a261e39bc67c/scratchpad/adj/d.parquet')
print("\nROW CLASSES (unsearched only):")
un=d[d.kind!='oracle_board']
print(pd.crosstab([un.kind,un.canon_band],un.cls).to_string())

print("\nEXAMPLES of DELETED rows (high band):")
for r in un[(un.cls=='DELETED')&(un.canon_band=='high(>70)')].head(8).itertuples():
    print(f"  [{r.kind[:4]}] {r.pct:5.1f} vs {r.canon_pct:5.1f} | {r.phrase[:110]}")
print("\nEXAMPLES of REPLACED rows (adversarial high):")
for r in un[(un.cls=='REPLACED')&(un.canon_band=='high(>70)')&(un.kind=='adversarial')].head(6).itertuples():
    print(f"  {r.pct:5.1f} vs {r.canon_pct:5.1f} | {r.phrase[:110]}")
