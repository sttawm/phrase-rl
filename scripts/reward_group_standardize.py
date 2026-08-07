#!/usr/bin/env python3
"""Can a GRPO group standardize its own reward, with no per-task intercept?

A group is K candidates for ONE context, so the group's own mean/sd of each
channel is an in-situ estimate of the task's location and scale. If K=16 is
enough, we need no stored per-task parameters at all: the intercept cancels
under GRPO's mean-centering, and the scale comes from the group.

Simulated by drawing K phrases per task from the fine-exam panel, standardizing
inside that draw only, and measuring within-group Pearson against true success
(the quantity GRPO advantages are proportional to).

  .venv/bin/python scripts/reward_group_standardize.py
"""
import json

import numpy as np
import pandas as pd

F_CAP, B = 4, 400
feats = pd.concat([pd.read_parquet("results/analysis/fine_exam_features_native.parquet"),
                   pd.read_parquet("results/analysis/fine_exam_features_oov.parquet")],
                  ignore_index=True)
pan = pd.read_parquet("results/analysis/fine_exam_phrases.parquet")
rows = []
for (t, p), g in feats.groupby(["task", "phrase"]):
    per_ep = g.sort_values("t").groupby("episode_index").head(F_CAP).groupby("episode_index").agg(
        z=("z_row", "mean"), gr=("grip_row", "mean"))
    rows.append({"task": t, "phrase": p, "z": per_ep.z.mean(), "negg": -per_ep.gr.mean()})
X = pd.DataFrame(rows).merge(pan[["task", "phrase", "gt_success"]], on=["task", "phrase"])

rng = np.random.default_rng(5)
SCHEMES = ("rank01 blend (current)", "group-standardized linear", "group-standardized sigmoid")
print(f"{'K':>4}  " + "  ".join(f"{s:>26}" for s in SCHEMES))
out = {}
for K in (4, 8, 16, 32):
    acc = {s: [] for s in SCHEMES}
    for _ in range(B):
        for t, g in X.groupby("task"):
            if len(g) < K:
                continue
            d = g.sample(K, random_state=int(rng.integers(1 << 30)))
            y = d.gt_success.values
            if y.std() < 1e-9:
                continue
            zr = d.z.rank(pct=True).values
            gr = d.negg.rank(pct=True).values
            acc[SCHEMES[0]].append(np.corrcoef(0.25 * zr + 0.75 * gr, y)[0, 1])
            zs = (d.z.values - d.z.values.mean()) / (d.z.values.std() + 1e-9)
            gs = (d.negg.values - d.negg.values.mean()) / (d.negg.values.std() + 1e-9)
            lin = 0.3055 * zs + 0.4295 * gs
            acc[SCHEMES[1]].append(np.corrcoef(lin, y)[0, 1])
            acc[SCHEMES[2]].append(np.corrcoef(1 / (1 + np.exp(-(0.2212 + lin))), y)[0, 1])
    means = {s: float(np.nanmean(acc[s])) for s in SCHEMES}
    out[K] = means
    print(f"{K:>4}  " + "  ".join(f"{means[s]:26.3f}" for s in SCHEMES))

print("\n(within-group Pearson vs true success; GRPO advantages are proportional to this)")
json.dump(out, open("results/analysis/reward_group_standardize.json", "w"), indent=1)
print("-> results/analysis/reward_group_standardize.json")
