import sys; sys.path.insert(0,'/private/tmp/claude-501/-Users-sttawm-dev-robotics-phrase-rl/5032c440-d6d5-4ef4-9205-a261e39bc67c/scratchpad/adj')
from feat import build, toks, norm
import numpy as np, pandas as pd
d=pd.read_parquet('/private/tmp/claude-501/-Users-sttawm-dev-robotics-phrase-rl/5032c440-d6d5-4ef4-9205-a261e39bc67c/scratchpad/adj/d.parquet')
un=d[d.kind!='oracle_board']
g=un[(un.kind=='adversarial')&(un.canon_band=='high(>70)')&(~un.keep_all)]
print("ADVERSARIAL HIGH violation rows sorted by n_novel_sub (span size vs whole-phrase size):")
for r in g.sort_values('n_novel_sub').itertuples():
    print(f"  novel={r.n_novel_sub:2d} span={r.max_span:2d} nw={r.n_words:2d} pct={r.pct:5.1f} canon={r.canon_pct:5.1f} | {r.phrase[:96]}")
print("\nNATURAL HIGH violation rows:")
for r in un[(un.kind=='natural')&(un.canon_band=='high(>70)')&(~un.keep_all)].sort_values('n_novel_sub').itertuples():
    print(f"  novel={r.n_novel_sub:2d} span={r.max_span:2d} nw={r.n_words:2d} pct={r.pct:5.1f} canon={r.canon_pct:5.1f} | {r.phrase[:96]}")
