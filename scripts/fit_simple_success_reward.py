#!/usr/bin/env python3
"""The simple success-rate reward: ONE global formula, raw channels, no per-task
fitting, no standardization, no ranking.

    estimated success = sigmoid(8.12 + 0.445*z + 11.32*(-grip_err))

The constant 8.12 is the mean of the fitted per-task intercepts. It sets where
the sigmoid sits; GRPO subtracts the group mean anyway, so it only has to be
in the right neighbourhood, not exact. Getting it wrong saturates the sigmoid
(a guess of -1.0 predicted 0% for every phrase), so it is fitted here, not
assumed.

  .venv/bin/python scripts/fit_simple_success_reward.py
"""
import numpy as np, pandas as pd
F_CAP = 4
feats = pd.concat([pd.read_parquet("results/analysis/fine_exam_features_native.parquet"),
                   pd.read_parquet("results/analysis/fine_exam_features_oov.parquet")], ignore_index=True)
pan = pd.read_parquet("results/analysis/fine_exam_phrases.parquet")
rows = []
for (t, p), g in feats.groupby(["task", "phrase"]):
    pe = g.sort_values("t").groupby("episode_index").head(F_CAP).groupby("episode_index").agg(
        z=("z_row","mean"), gr=("grip_row","mean"))
    rows.append({"task": t, "phrase": p, "z": pe.z.mean(), "negg": -pe.gr.mean()})
X = pd.DataFrame(rows).merge(pan[["task","phrase","gt_success","gt_n"]], on=["task","phrase"])
print("channel scales:")
print(f"  verifier z : mean {X.z.mean():+.3f}  sd {X.z.std():.3f}  range {X.z.min():+.2f}..{X.z.max():+.2f}")
print(f"  -grip      : mean {X.negg.mean():+.4f}  sd {X.negg.std():.4f}  range {X.negg.min():+.4f}..{X.negg.max():+.4f}")

def irls(D, y, w, it=80):
    b = np.zeros(D.shape[1])
    for _ in range(it):
        eta = np.clip(D@b, -30, 30); p = 1/(1+np.exp(-eta)); W = w*p*(1-p)+1e-9
        z = eta + (w*y - w*p)/W
        nb = np.linalg.solve((D*W[:,None]).T@D + 1e-8*np.eye(D.shape[1]), (D*W[:,None]).T@z)
        if np.max(np.abs(nb-b)) < 1e-10: return nb
        b = nb
    return b
T = pd.get_dummies(X.task).values.astype(float)
b = irls(np.column_stack([T, X.z, X.negg]), (X.gt_success/100).values, X.gt_n.values.astype(float))
ints = b[:T.shape[1]]
print(f"\nfitted task intercepts: {np.round(ints,2)}")
print(f"  mean {ints.mean():+.3f}  -> this is the C to use")
C = float(ints.mean())
X["pred"] = 100/(1+np.exp(-np.clip(C + b[-2]*X.z + b[-1]*X.negg, -30, 30)))
print(f"\nSIMPLE formula: success = sigmoid({C:+.2f} + {b[-2]:.3f}*z + {b[-1]:.2f}*(-grip))")
print(f"  predicted range {X.pred.min():.0f}-{X.pred.max():.0f}%  (actual {X.gt_success.min():.0f}-{X.gt_success.max():.0f}%)")
print(f"  corr with actual {X.pred.corr(X.gt_success):.3f} | mean |error| {(X.pred-X.gt_success).abs().mean():.1f}pp")

rng = np.random.default_rng(5); K, B = 16, 400
acc = {"SIMPLE (raw, one constant)": [], "group-standardized": [], "rank01 c4b (current)": []}
for _ in range(B):
    for t, g in X.groupby("task"):
        if len(g) < K: continue
        d = g.sample(K, random_state=int(rng.integers(1<<30))); y = d.gt_success.values
        if y.std() < 1e-9: continue
        acc["SIMPLE (raw, one constant)"].append(np.corrcoef(
            1/(1+np.exp(-(C + b[-2]*d.z.values + b[-1]*d.negg.values))), y)[0,1])
        zs=(d.z.values-d.z.values.mean())/(d.z.values.std()+1e-9); gs=(d.negg.values-d.negg.values.mean())/(d.negg.values.std()+1e-9)
        acc["group-standardized"].append(np.corrcoef(0.3055*zs+0.4295*gs, y)[0,1])
        acc["rank01 c4b (current)"].append(np.corrcoef(0.25*d.z.rank(pct=True).values+0.75*d.negg.rank(pct=True).values, y)[0,1])
print(f"\nwithin-group Pearson (K={K}) — what GRPO advantages track:")
for k,v in acc.items(): print(f"  {k:32s} {np.nanmean(v):.3f}")
import numpy as np, pandas as pd
F_CAP = 4
feats = pd.concat([pd.read_parquet("results/analysis/fine_exam_features_native.parquet"),
                   pd.read_parquet("results/analysis/fine_exam_features_oov.parquet")], ignore_index=True)
pan = pd.read_parquet("results/analysis/fine_exam_phrases.parquet")
rows = []
for (t, p), g in feats.groupby(["task", "phrase"]):
    pe = g.sort_values("t").groupby("episode_index").head(F_CAP).groupby("episode_index").agg(
        z=("z_row","mean"), gr=("grip_row","mean"))
    rows.append({"task": t, "phrase": p, "z": pe.z.mean(), "negg": -pe.gr.mean()})
X = pd.DataFrame(rows).merge(pan[["task","phrase","gt_success","gt_n"]], on=["task","phrase"])
print("channel scales:")
print(f"  verifier z : mean {X.z.mean():+.3f}  sd {X.z.std():.3f}  range {X.z.min():+.2f}..{X.z.max():+.2f}")
print(f"  -grip      : mean {X.negg.mean():+.4f}  sd {X.negg.std():.4f}  range {X.negg.min():+.4f}..{X.negg.max():+.4f}")

def irls(D, y, w, it=80):
    b = np.zeros(D.shape[1])
    for _ in range(it):
        eta = np.clip(D@b, -30, 30); p = 1/(1+np.exp(-eta)); W = w*p*(1-p)+1e-9
        z = eta + (w*y - w*p)/W
        nb = np.linalg.solve((D*W[:,None]).T@D + 1e-8*np.eye(D.shape[1]), (D*W[:,None]).T@z)
        if np.max(np.abs(nb-b)) < 1e-10: return nb
        b = nb
    return b
T = pd.get_dummies(X.task).values.astype(float)
b = irls(np.column_stack([T, X.z, X.negg]), (X.gt_success/100).values, X.gt_n.values.astype(float))
ints = b[:T.shape[1]]
print(f"\nfitted task intercepts: {np.round(ints,2)}")
print(f"  mean {ints.mean():+.3f}  -> this is the C to use")
C = float(ints.mean())
X["pred"] = 100/(1+np.exp(-np.clip(C + b[-2]*X.z + b[-1]*X.negg, -30, 30)))
print(f"\nSIMPLE formula: success = sigmoid({C:+.2f} + {b[-2]:.3f}*z + {b[-1]:.2f}*(-grip))")
print(f"  predicted range {X.pred.min():.0f}-{X.pred.max():.0f}%  (actual {X.gt_success.min():.0f}-{X.gt_success.max():.0f}%)")
print(f"  corr with actual {X.pred.corr(X.gt_success):.3f} | mean |error| {(X.pred-X.gt_success).abs().mean():.1f}pp")

rng = np.random.default_rng(5); K, B = 16, 400
acc = {"SIMPLE (raw, one constant)": [], "group-standardized": [], "rank01 c4b (current)": []}
for _ in range(B):
    for t, g in X.groupby("task"):
        if len(g) < K: continue
        d = g.sample(K, random_state=int(rng.integers(1<<30))); y = d.gt_success.values
        if y.std() < 1e-9: continue
        acc["SIMPLE (raw, one constant)"].append(np.corrcoef(
            1/(1+np.exp(-(C + b[-2]*d.z.values + b[-1]*d.negg.values))), y)[0,1])
        zs=(d.z.values-d.z.values.mean())/(d.z.values.std()+1e-9); gs=(d.negg.values-d.negg.values.mean())/(d.negg.values.std()+1e-9)
        acc["group-standardized"].append(np.corrcoef(0.3055*zs+0.4295*gs, y)[0,1])
        acc["rank01 c4b (current)"].append(np.corrcoef(0.25*d.z.rank(pct=True).values+0.75*d.negg.rank(pct=True).values, y)[0,1])
print(f"\nwithin-group Pearson (K={K}) — what GRPO advantages track:")
for k,v in acc.items(): print(f"  {k:32s} {np.nanmean(v):.3f}")
