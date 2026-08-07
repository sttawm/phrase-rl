#!/usr/bin/env python3
"""Fit estimated rollout success rate as a monotone function of the proxy reward.

Data: the fine-exam panel — 254 phrases across 8 tasks, each with 36 ground-truth
simulator rollouts and per-(episode, frame) reward channels (z = learned verifier
logit, g = gripper error).

Models (binomial GLM, IRLS, weights = 36 trials/phrase). Monotonicity is by
construction: a logistic link applied to a linear score, with the fitted signs
reported so any non-monotone fit would be visible rather than hidden.

  A  pooled          logit(p) = a + b*z + c*(-g)          # no task knowledge
  B  task intercepts logit(p) = a_task + b*z + c*(-g)     # within-task shape
  C  rank-only       logit(p) = a_task + b*rank01(blend)  # what the search sees

  .venv/bin/python scripts/fit_success_calibration.py
"""
import json

import numpy as np
import pandas as pd

F_CAP = 4  # match the production F=4 aggregation

feats = pd.concat([pd.read_parquet("results/analysis/fine_exam_features_native.parquet"),
                   pd.read_parquet("results/analysis/fine_exam_features_oov.parquet")],
                  ignore_index=True)
pan = pd.read_parquet("results/analysis/fine_exam_phrases.parquet")

# per-phrase aggregation: mean over episodes of the per-episode mean over <=F frames
rows = []
for (t, p), g in feats.groupby(["task", "phrase"]):
    per_ep = g.sort_values("t").groupby("episode_index").head(F_CAP).groupby("episode_index").agg(
        z=("z_row", "mean"), gr=("grip_row", "mean"))
    rows.append({"task": t, "phrase": p, "z": per_ep.z.mean(), "g": per_ep.gr.mean(),
                 "n_ep": len(per_ep)})
X = pd.DataFrame(rows).merge(pan[["task", "phrase", "gt_success", "gt_n"]], on=["task", "phrase"])
X["y"] = X.gt_success / 100.0
X["negg"] = -X.g
print(f"fit set: {len(X)} phrases x {int(X.gt_n.iloc[0])} rollouts, {X.task.nunique()} tasks")
print(f"success spread: {X.gt_success.min():.0f}-{X.gt_success.max():.0f}%, "
      f"sd {X.gt_success.std():.1f}pp")

# variance decomposition: how much is task vs phrasing?
tm = X.groupby("task").gt_success.transform("mean")
ss_tot = ((X.gt_success - X.gt_success.mean()) ** 2).sum()
ss_task = ((tm - X.gt_success.mean()) ** 2).sum()
print(f"variance explained by TASK identity alone: {100 * ss_task / ss_tot:.1f}%  "
      f"(phrasing has the remaining {100 * (1 - ss_task / ss_tot):.1f}%)\n")


def irls(D, y, n, iters=60):
    """binomial IRLS; D includes intercept column(s)."""
    b = np.zeros(D.shape[1])
    for _ in range(iters):
        eta = D @ b
        p = 1 / (1 + np.exp(-np.clip(eta, -30, 30)))
        W = n * p * (1 - p) + 1e-9
        zed = eta + (n * y - n * p) / W
        b_new = np.linalg.solve((D * W[:, None]).T @ D + 1e-8 * np.eye(D.shape[1]),
                                (D * W[:, None]).T @ zed)
        if np.max(np.abs(b_new - b)) < 1e-9:
            b = b_new
            break
        b = b_new
    return b


def report(name, D, cols, X):
    b = irls(D, X.y.values, X.gt_n.values)
    p = 1 / (1 + np.exp(-np.clip(D @ b, -30, 30)))
    resid = X.gt_success / 100 - p
    ss_res = (resid ** 2).sum()
    ss_tot = ((X.y - X.y.mean()) ** 2).sum()
    mae = 100 * np.abs(resid).mean()
    print(f"--- {name}")
    for c, v in zip(cols, b):
        if not c.startswith("task["):
            print(f"      {c:12s} {v:+.4f}")
    print(f"      R2 {1 - ss_res / ss_tot:.3f} | MAE {mae:.1f}pp | "
          f"corr(pred, actual) {np.corrcoef(p, X.y)[0, 1]:.3f}")
    return b, p


# A: pooled
DA = np.column_stack([np.ones(len(X)), X.z, X.negg])
bA, pA = report("A pooled (no task knowledge)", DA, ["intercept", "z", "-grip"], X)

# B: task intercepts
T = pd.get_dummies(X.task, prefix="task").values.astype(float)
DB = np.column_stack([T, X.z, X.negg])
colsB = [f"task[{i}]" for i in range(T.shape[1])] + ["z", "-grip"]
bB, pB = report("B + task intercepts", DB, colsB, X)

# C: within-task rank of the c4b blend (what the search actually ranks on)
X["r_z"] = X.groupby("task").z.rank(pct=True)
X["r_g"] = X.groupby("task").negg.rank(pct=True)
X["blend"] = 0.25 * X.r_z + 0.75 * X.r_g
DC = np.column_stack([T, X.blend])
bC, pC = report("C task intercepts + c4b rank", DC, [f"task[{i}]" for i in range(T.shape[1])] + ["c4b_rank"], X)

print("\ncalibration of model B (predicted vs actual, quintiles of prediction):")
X["pred"] = 100 * pB
X["q"] = pd.qcut(X.pred, 5, labels=False, duplicates="drop")
for q, g in X.groupby("q"):
    print(f"   Q{int(q) + 1}  predicted {g.pred.mean():5.1f}%   actual {g.gt_success.mean():5.1f}%   n={len(g)}")

out = {"n_phrases": int(len(X)), "tasks": int(X.task.nunique()),
       "var_explained_by_task_pct": round(100 * ss_task / ss_tot, 1),
       "pooled": {"intercept": bA[0], "z": bA[1], "neg_grip": bA[2]},
       "task_intercepts": {"z": bB[-2], "neg_grip": bB[-1]},
       "c4b_rank": {"coef": bC[-1]}}
json.dump(out, open("results/analysis/success_calibration.json", "w"), indent=1, default=float)
print("\n-> results/analysis/success_calibration.json")
