#!/usr/bin/env python3
"""One global monotone fit across all tasks, on WITHIN-TASK RELATIVE success.

Each task's phrases are rescaled to [0,1] between that task's worst and best
observed phrase, so a single function serves every task with no per-task
intercept to estimate.

Two normalizations, because min/max are single phrases measured with only 36
rollouts (SE ~8pp) and therefore noisy anchors:
  minmax : (y - min) / (max - min)          -- as proposed
  p10p90 : robust variant clipped to the 10th/90th percentile anchors

The honest comparison for these R2 values is NOT the task-intercept model's
overall R2 (0.803), which gets 0.695 free from task identity alone. It is that
model's WITHIN-TASK share, computed here on the same footing.

  .venv/bin/python scripts/fit_success_relative.py
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

# within-task normalizations of the TARGET
def norm_minmax(s):
    lo, hi = s.min(), s.max()
    return (s - lo) / (hi - lo) if hi > lo else s * 0 + 0.5

def norm_p10p90(s):
    lo, hi = s.quantile(0.10), s.quantile(0.90)
    return ((s - lo) / (hi - lo)).clip(0, 1) if hi > lo else s * 0 + 0.5

X["rel_minmax"] = X.groupby("task").gt_success.transform(norm_minmax)
X["rel_p1090"] = X.groupby("task").gt_success.transform(norm_p10p90)
# within-task normalizations of the PREDICTORS (so one global slope is meaningful)
X["zr"] = X.groupby("task").z.rank(pct=True)
X["gr_"] = X.groupby("task").negg.rank(pct=True)
X["c4b"] = 0.25 * X.zr + 0.75 * X.gr_
X["zs"] = X.groupby("task").z.transform(lambda s: (s - s.mean()) / (s.std() + 1e-9))
X["gs"] = X.groupby("task").negg.transform(lambda s: (s - s.mean()) / (s.std() + 1e-9))

print(f"{len(X)} phrases, {X.task.nunique()} tasks, 36 rollouts each")
print("per-task observed range (min -> max success):")
for t, g in X.groupby("task"):
    print(f"   {t.replace('widowx_', '')[:26]:28s} {g.gt_success.min():5.1f} -> {g.gt_success.max():5.1f}"
          f"   span {g.gt_success.max() - g.gt_success.min():5.1f}pp")


def irls(D, y, w, iters=80):
    b = np.zeros(D.shape[1])
    for _ in range(iters):
        eta = np.clip(D @ b, -30, 30)
        p = 1 / (1 + np.exp(-eta))
        W = w * p * (1 - p) + 1e-9
        zed = eta + (w * y - w * p) / W
        nb = np.linalg.solve((D * W[:, None]).T @ D + 1e-8 * np.eye(D.shape[1]),
                             (D * W[:, None]).T @ zed)
        if np.max(np.abs(nb - b)) < 1e-10:
            return nb
        b = nb
    return b


def fit(name, target, cols):
    D = np.column_stack([np.ones(len(X))] + [X[c].values for c in cols])
    y = X[target].values
    b = irls(D, y, np.full(len(X), 36.0))
    p = 1 / (1 + np.exp(-np.clip(D @ b, -30, 30)))
    r2 = 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    # convert error back to absolute pp using each task's own span
    span = X.groupby("task").gt_success.transform(lambda s: s.max() - s.min()).values
    mae_pp = np.abs((y - p) * span).mean()
    print(f"--- {name}")
    for c, v in zip(["intercept"] + cols, b):
        print(f"      {c:10s} {v:+.4f}")
    print(f"      R2(relative) {r2:.3f} | corr {np.corrcoef(p, y)[0, 1]:.3f} | "
          f"MAE back in absolute terms {mae_pp:.1f}pp")
    return b, r2


print()
res = {}
for tgt in ("rel_minmax", "rel_p1090"):
    for cols, nm in [(["c4b"], "c4b rank"), (["zr", "gr_"], "z-rank + grip-rank"),
                     (["zs", "gs"], "z-score + grip-score")]:
        b, r2 = fit(f"{tgt}  ~  {nm}", tgt, cols)
        res[f"{tgt}|{nm}"] = {"coef": list(map(float, b)), "r2": float(r2)}

# honest reference: within-task share of the task-intercept model
tm = X.groupby("task").gt_success.transform("mean")
within_tot = ((X.gt_success - tm) ** 2).sum()
D = np.column_stack([pd.get_dummies(X.task).values.astype(float), X.z, X.negg])
b = irls(D, (X.gt_success / 100).values, X.gt_n.values.astype(float))
pred = 100 / (1 + np.exp(-np.clip(D @ b, -30, 30)))
within_res = ((X.gt_success - pred) ** 2).sum()
print(f"\nreference — task-intercept model's WITHIN-TASK R2: {1 - within_res / within_tot:.3f}")
print("(its headline 0.803 counts task identity, which the relative fit deliberately removes)")

json.dump(res, open("results/analysis/success_relative_fit.json", "w"), indent=1)
print("-> results/analysis/success_relative_fit.json")
