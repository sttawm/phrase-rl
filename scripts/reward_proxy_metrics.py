#!/usr/bin/env python3
"""Which candidate reward best serves RL — judged by the metric GRPO actually uses.

R2 and MAE describe how well a model predicts a success RATE. But GRPO never
consumes an absolute rate: it mean-centres rewards inside a group of candidates
for the same context, so only the WITHIN-TASK shape matters. The decision metric
is therefore within-task rank correlation with true success, plus how much of the
true advantage spread a reward recovers.

Reported per candidate reward, averaged over the 8 tasks:
  spearman   within-task rank correlation with ground-truth success
  pearson    within-task linear correlation (magnitude-sensitive, as GRPO is)
  top1_gain  true success of the phrase this reward ranks first, minus the
             task's mean phrase -- what you actually gain by picking with it
  oracle     the same for a perfect ranker (ceiling)

  .venv/bin/python scripts/reward_proxy_metrics.py
"""
import json

import numpy as np
import pandas as pd

F_CAP = 4
feats = pd.concat([pd.read_parquet("results/analysis/fine_exam_features_native.parquet"),
                   pd.read_parquet("results/analysis/fine_exam_features_oov.parquet")],
                  ignore_index=True)
pan = pd.read_parquet("results/analysis/fine_exam_phrases.parquet")

rows = []
for (t, p), g in feats.groupby(["task", "phrase"]):
    per_ep = g.sort_values("t").groupby("episode_index").head(F_CAP).groupby("episode_index").agg(
        z=("z_row", "mean"), gr=("grip_row", "mean"))
    rows.append({"task": t, "phrase": p, "z": per_ep.z.mean(), "g": per_ep.gr.mean()})
X = pd.DataFrame(rows).merge(pan[["task", "phrase", "gt_success", "gt_n"]], on=["task", "phrase"])
X["negg"] = -X.g
X["zs"] = X.groupby("task").z.transform(lambda s: (s - s.mean()) / (s.std() + 1e-9))
X["gs"] = X.groupby("task").negg.transform(lambda s: (s - s.mean()) / (s.std() + 1e-9))
X["zr"] = X.groupby("task").z.rank(pct=True)
X["gr_"] = X.groupby("task").negg.rank(pct=True)

# calibrated success-rate reward (the relative fit, standardized channels)
X["calib"] = 1 / (1 + np.exp(-(0.2212 + 0.3055 * X.zs + 0.4295 * X.gs)))

CANDS = {
    "grip only (rank)": X.gr_,
    "ensemble only (rank)": X.zr,
    "c4b (0.25 z + 0.75 g, ranks)": 0.25 * X.zr + 0.75 * X.gr_,
    "grip only (standardized)": X.gs,
    "c4b (standardized channels)": 0.25 * X.zs + 0.75 * X.gs,
    "CALIBRATED success-rate": X.calib,
}

print(f"{'reward':32s} {'spearman':>9} {'pearson':>8} {'top1 gain':>10} {'of oracle':>10}")
out = {}
for name, col in CANDS.items():
    X["_r"] = col
    sp, pe, t1, orc = [], [], [], []
    for t, g in X.groupby("task"):
        if len(g) < 4:
            continue
        sp.append(g._r.corr(g.gt_success, method="spearman"))
        pe.append(g._r.corr(g.gt_success))
        mean_s = g.gt_success.mean()
        t1.append(g.loc[g._r.idxmax()].gt_success - mean_s)
        orc.append(g.gt_success.max() - mean_s)
    sp, pe, t1, orc = map(np.array, (sp, pe, t1, orc))
    frac = 100 * t1.sum() / orc.sum()
    out[name] = {"spearman": float(sp.mean()), "pearson": float(pe.mean()),
                 "top1_gain_pp": float(t1.mean()), "pct_of_oracle": float(frac)}
    print(f"{name:32s} {sp.mean():9.3f} {pe.mean():8.3f} {t1.mean():+9.1f}pp {frac:9.0f}%")

print(f"\noracle (perfect ranker) top-1 gain: {orc.mean():+.1f}pp above task mean")
print(f"random pick: +0.0pp by construction")
json.dump(out, open("results/analysis/reward_proxy_metrics.json", "w"), indent=1)
print("-> results/analysis/reward_proxy_metrics.json")
