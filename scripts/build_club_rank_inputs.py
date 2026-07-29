#!/usr/bin/env python3
"""Build inputs for the high-fidelity club-phrase ranking (user 2026-07-28):
rank every >=16-episode club instruction at the F=4 x C=20 cell (grip 92.4% /
c4b 90.9% coarse sign accuracy).

From data/contexts_club.parquet (16,956 rows: 4 frames x 4,239 episodes x 213
instructions), per bridge task:
  - phrase panel  = the task's club instructions (results/analysis/club_rank_phrases.parquet)
  - context set   = C=20 seeded-sampled episodes of that task x their 4 frames
                    (data/contexts_club_rank.parquet) — COMMON per task, so all
                    phrases of a task are scored on identical contexts (CRN).
Then scripts/fine_exam_score.py scores the panel on these contexts unchanged.
"""
import numpy as np
import pandas as pd

C = 20
RNG = np.random.default_rng(7)

ctx = pd.read_parquet("data/contexts_club.parquet")
print(f"club contexts: {len(ctx)} rows, {ctx.task.nunique()} tasks, "
      f"{ctx.instruction.nunique()} instructions")

panel, picked = [], []
for task, sub in ctx.groupby("task"):
    for ins in sorted(sub.instruction.unique()):
        panel.append({"task": task, "phrase": ins})
    eps = sorted(sub.episode_index.unique())
    take = list(RNG.choice(eps, size=min(C, len(eps)), replace=False))
    pick = sub[sub.episode_index.isin(take)]
    picked.append(pick)
    print(f"  {task}: {len(sub.instruction.unique())} phrases, "
          f"{len(take)} ctx episodes ({len(pick)} frame-contexts)")

pd.DataFrame(panel).to_parquet("results/analysis/club_rank_phrases.parquet", index=False)
out = pd.concat(picked, ignore_index=True)
out.to_parquet("data/contexts_club_rank.parquet", index=False)
print(f"-> {len(panel)} panel phrases; {len(out)} frame-contexts")
