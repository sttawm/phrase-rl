"""Adjudicator's own head-noun feature. Derived from the 48 canonical strings only."""
import re, pandas as pd, numpy as np

CSV='/Users/sttawm/dev/robotics/phrase-rl/results/analysis/pi05_bank/distill_evidence_ood/phrases.csv'

def toks(s):
    s=str(s).lower().replace("'s"," ").replace("'"," ")
    return [t for t in re.split(r'[^a-z0-9]+',s) if t]

def norm(w):
    if len(w)>4 and w.endswith('ies'): return w[:-3]+'y'
    if len(w)>4 and w[-3:] in ('ses','xes','zes'): return w[:-2]
    if len(w)>5 and w[-4:] in ('ches','shes'): return w[:-2]
    if len(w)>3 and w.endswith('s') and not w.endswith('ss'): return w[:-1]
    return w

# --- derived from the canonicals themselves ---
VERBS   = {'close','open','pick','place','put','stack','turn'}   # position-0 tokens across the 48
PARTS   = {'up','on','off'}                                       # position-1 non-determiner tokens
PREPS   = {'of','on','in','at','to','under'}                      # tokens immediately before 'the'
POSNL   = {'left','right','middle','back','front','top','bottom'} # prenominal minimal-pair alternants
COLOUR  = {'black','white','red','yellow'}                        # prenominal colour alternants
DET     = {'the'}
PRO     = {'it','them'}
BREAK   = VERBS | PREPS | DET | PRO | PARTS

def np_spans(tk):
    """NP spans opened by a determiner; 'and' closes a span only before a determiner or verb."""
    spans, i = [], 0
    while i < len(tk):
        if tk[i] in DET:
            j = i+1; span=[]
            while j < len(tk):
                w = tk[j]
                if w == 'and':
                    if j+1 < len(tk) and (tk[j+1] in DET or tk[j+1] in VERBS): break
                    span.append(w); j+=1; continue
                if w in BREAK: break
                span.append(w); j+=1
            spans.append(span); i=j
        else:
            i+=1
    return spans

def heads(canon, compound_mode='head'):
    """Return (ordered head-noun list, target head or None)."""
    tk = toks(canon); hs=[]
    for sp in np_spans(tk):
        content=[w for w in sp if w not in POSNL and w!='and']
        if not content: continue
        if compound_mode=='head': hs.append(content[-1])          # head-final NP rule
        elif compound_mode=='loose_nocol': hs.extend([w for w in content if w not in COLOUR])
        else: hs.extend(content)                                   # loose: all content tokens
    seen=[]; 
    for h in hs:
        if h not in seen: seen.append(h)
    tgt = hs[0] if hs else None
    return seen, tgt

def build(compound_mode='head'):
    d = pd.read_csv(CSV)
    d = d[d.kind!='original'].copy()          # canonical itself: zero variation
    H = {c: heads(c, compound_mode) for c in d.canonical.unique()}
    rows=[]
    for r in d.itertuples():
        hs, tgt = H[r.canonical]
        pt = set(norm(w) for w in toks(r.phrase))
        miss = [h for h in hs if norm(h) not in pt]
        rows.append(dict(n_heads=len(hs), n_miss=len(miss),
                         keep_all=len(miss)==0,
                         keep_target = (tgt is None) or (norm(tgt) in pt),
                         miss_target = (tgt is not None) and (norm(tgt) not in pt),
                         miss_landmark = any(h!=tgt for h in miss)))
    return pd.concat([d.reset_index(drop=True), pd.DataFrame(rows)], axis=1), H
