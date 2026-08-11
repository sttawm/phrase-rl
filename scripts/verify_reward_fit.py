#!/usr/bin/env python3
"""Re-verify the proxy reward against rollout ground truth on the full 434-phrase set.

Ground truth: results/analysis/fine_exam_phrases.parquet (254 close paraphrases)
+ sim_rollouts_natadv_{0,1}of2.parquet (180 natural/adversarial), all at n=36
rollouts (layouts 0-17 x 2). Proxy channels come from the scored bank
(bank_scores_*.parquet: z = verifier-ensemble logit, grip = gripper error,
F x C = 64 grade). The 180 nat/adv phrases were rolled AFTER the logistic was
fitted, so they are a genuine held-out test of the frozen coefficients
(success = sigmoid(8.123 + 0.4445*z + 11.3193*(-grip)), fit on the 254).

Caveat when comparing to the paper's 79.6% fine-pair number: fine_grid_all.json
was computed from the exam FEATURE extractions (fine_exam_features_*.parquet)
at matched budgets, not from the bank's single-draw scores — the two are not
apples-to-apples. See results/analysis/reward_verify_434.json for this
script's outputs.
"""
import glob
import itertools
import json

import numpy as np
import pandas as pd

FROZEN = dict(b0=8.123, bz=0.4445, bg=11.3193)


def load():
    fe = pd.read_parquet("results/analysis/fine_exam_phrases.parquet")[
        ["task", "phrase", "gt_success"]]
    fe["src"] = "fine_exam"
    na = pd.concat(
        [pd.read_parquet(f) for f in
         ("results/analysis/sim_rollouts_natadv_0of2.parquet",
          "results/analysis/sim_rollouts_natadv_1of2.parquet")],
        ignore_index=True)[["task", "phrase", "gt_success", "arm"]]
    na = na.rename(columns={"arm": "src"})
    gt = pd.concat([fe, na], ignore_index=True).drop_duplicates(["task", "phrase"])
    sc = pd.concat(
        [pd.read_parquet(f) for f in sorted(glob.glob("results/analysis/bank_scores_*.parquet"))],
        ignore_index=True).dropna(subset=["z", "grip"]).drop_duplicates(["task", "phrase"])
    d = gt.merge(sc[["task", "phrase", "z", "grip"]], on=["task", "phrase"], how="inner")
    d["y"] = d.gt_success / 100.0
    d["negg"] = -d.grip
    d["logit_old"] = FROZEN["bz"] * d.z + FROZEN["bg"] * d.negg
    print(f"gt {len(gt)} phrases, {len(d)} with bank proxy channels "
          f"({int((d.src == 'fine_exam').sum())} fine_exam, "
          f"{int((d.src != 'fine_exam').sum())} nat/adv)")
    return d


def spearman(a, b):
    return pd.Series(list(a)).corr(pd.Series(list(b)), method="spearman")


def within_task(frame, col, method):
    vals = [g[col].corr(g.y, method=method)
            for _, g in frame.groupby("task") if len(g) >= 5 and g.y.std() > 0]
    return float(np.mean(vals)), len(vals)


def pairacc(frame, col, lo, hi):
    ok = tot = 0
    for _, g in frame.groupby("task"):
        for (_, a), (_, b) in itertools.combinations(g.iterrows(), 2):
            gap = abs(a.gt_success - b.gt_success)
            if lo <= gap < hi:
                tot += 1
                ok += (a[col] - b[col]) * (a.gt_success - b.gt_success) > 0
    return (100 * ok / tot if tot else float("nan")), tot


def fit_logistic(frame, iters=200):
    """Newton fit of per-phrase success fractions on (1, z, -grip)."""
    X = np.column_stack([np.ones(len(frame)), frame.z.values, frame.negg.values])
    y = frame.y.values
    b = np.zeros(3)
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ b))
        g = X.T @ (y - p) / len(frame)
        H = (X * (p * (1 - p))[:, None]).T @ X / len(frame) + 1e-9 * np.eye(3)
        step = np.linalg.solve(H, g)
        b += step
        if np.abs(step).max() < 1e-10:
            break
    return b


def main():
    d = load()
    out = {"n": len(d), "frozen": FROZEN}
    subsets = [("ALL", d), ("fine_exam", d[d.src == "fine_exam"]),
               ("nat/adv", d[d.src != "fine_exam"])]

    print("\n=== within-task correlation with n=36 rollout success ===")
    out["within_task"] = {}
    for col, lab in [("negg", "gripper_only"), ("z", "verifier_only"),
                     ("logit_old", "calibrated_frozen")]:
        for name, sub in subsets:
            p, nt = within_task(sub, col, "pearson")
            s, _ = within_task(sub, col, "spearman")
            out["within_task"][f"{lab}|{name}"] = dict(pearson=round(p, 3),
                                                       spearman=round(s, 3), tasks=nt)
            print(f"  {lab:18} {name:10} Pearson {p:6.3f}  Spearman {s:6.3f}  ({nt} tasks)")

    print("\n=== pairwise ordering accuracy, within task, by true-success gap ===")
    out["pairwise"] = {}
    for col, lab in [("negg", "gripper_only"), ("logit_old", "calibrated_frozen")]:
        for name, sub in subsets:
            row = {}
            for lo, hi, blab in [(5, 10, "5-10pp"), (10, 15, "10-15pp"), (15, 1000, "15+pp")]:
                acc, n = pairacc(sub, col, lo, hi)
                row[blab] = dict(acc=round(acc, 1), pairs=n)
            out["pairwise"][f"{lab}|{name}"] = row
            print(f"  {lab:18} {name:10} " + "  ".join(
                f"{k} {v['acc']:5.1f} ({v['pairs']})" for k, v in row.items()))

    print("\n=== held-out test of the FROZEN fit (never saw nat/adv) ===")
    sub = d[d.src != "fine_exam"]
    p_old = 1 / (1 + np.exp(-(FROZEN["b0"] + sub.logit_old)))
    out["frozen_heldout_natadv"] = dict(
        pearson=round(float(np.corrcoef(p_old, sub.y)[0, 1]), 3),
        spearman=round(float(spearman(p_old, sub.y)), 3),
        mean_predicted_pct=round(float(p_old.mean() * 100), 1),
        mean_measured_pct=round(float(sub.y.mean() * 100), 1))
    print(f"  {out['frozen_heldout_natadv']}")

    print("\n=== refit on all phrases (candidate updated coefficients) ===")
    b = fit_logistic(d)
    d["logit_new"] = b[1] * d.z + b[2] * d.negg
    zsd = float(d.groupby("task").z.std().median())
    gsd = float(d.groupby("task").grip.std().median())
    out["refit"] = dict(b0=round(b[0], 3), bz=round(b[1], 4), bg=round(b[2], 4),
                        std_contrib_z=round(abs(b[1]) * zsd, 3),
                        std_contrib_grip=round(abs(b[2]) * gsd, 3))
    print(f"  success = sigmoid({b[0]:.3f} + {b[1]:.4f}*z + {b[2]:.4f}*(-grip))")
    print(f"  standardized contributions: z {out['refit']['std_contrib_z']}  "
          f"grip {out['refit']['std_contrib_grip']}")

    old_s, new_s = [], []
    for t in d.task.unique():
        te = d[d.task == t]
        if len(te) < 8 or te.y.std() == 0:
            continue
        bb = fit_logistic(d[d.task != t])
        old_s.append(spearman(te.logit_old, te.y))
        new_s.append(spearman(bb[1] * te.z + bb[2] * te.negg, te.y))
    out["cv_spearman"] = dict(frozen=round(float(np.mean(old_s)), 3),
                              refit_heldout=round(float(np.mean(new_s)), 3),
                              tasks=len(new_s))
    print(f"  task-held-out CV Spearman: frozen {out['cv_spearman']['frozen']}"
          f" -> refit {out['cv_spearman']['refit_heldout']} ({len(new_s)} tasks)")

    for name, sub in subsets[1:]:
        p_new = 1 / (1 + np.exp(-(b[0] + b[1] * sub.z + b[2] * sub.negg)))
        out[f"refit_calibration_{name.replace('/', '')}"] = dict(
            mean_predicted_pct=round(float(p_new.mean() * 100), 1),
            mean_measured_pct=round(float(sub.y.mean() * 100), 1))
        print(f"  refit calibration {name}: predicted "
              f"{p_new.mean() * 100:.1f}% vs measured {sub.y.mean() * 100:.1f}%")

    with open("results/analysis/reward_verify_434.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nwrote results/analysis/reward_verify_434.json")


if __name__ == "__main__":
    main()
