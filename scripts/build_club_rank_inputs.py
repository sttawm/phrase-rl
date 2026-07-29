#!/usr/bin/env python3
"""Build inputs for the high-fidelity club-phrase ranking (user 2026-07-28):
score every >=16-episode club instruction at the F=4 x C=20 cell (>=90% coarse
sign-accuracy regime) over ITS OWN club episodes — the same same-instruction
context estimator the v7e/v7f reward uses, at higher C.

Club contexts (data/contexts_club.parquet) carry no task labels (training bridge
episodes are instruction-labeled only), so each instruction is its own scoring
group: fine_exam_score.py's task column := the instruction string, and the
phrase panel for that group is the single instruction itself.
"""
import numpy as np
import pandas as pd

C = 20
RNG = np.random.default_rng(7)

ctx = pd.read_parquet("data/contexts_club.parquet")
print(f"club contexts: {len(ctx)} rows, {ctx.instruction.nunique()} instructions")

panel, picked = [], []
for ins, sub in ctx.groupby("instruction"):
    eps = sorted(sub.episode_index.unique())
    take = list(RNG.choice(eps, size=min(C, len(eps)), replace=False))
    pick = sub[sub.episode_index.isin(take)].copy()
    pick["task"] = ins
    picked.append(pick)
    panel.append({"task": ins, "phrase": ins, "n_club_eps": len(eps), "eff_C": len(take)})

pd.DataFrame(panel).to_parquet("results/analysis/club_rank_phrases.parquet", index=False)
out = pd.concat(picked, ignore_index=True)
out.to_parquet("data/contexts_club_rank.parquet", index=False)
cs = pd.DataFrame(panel).eff_C
print(f"-> {len(panel)} instructions; {len(out)} frame-contexts; "
      f"eff_C: min {cs.min()} / median {int(cs.median())} / >=20: {(cs >= 20).sum()}")
