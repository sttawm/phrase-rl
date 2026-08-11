#!/usr/bin/env python3
"""Fit the proxy blend to PRESERVE RANKINGS, not to predict success rates.

Data: the 434-phrase ground-truth set (n=36 rollouts each) joined to the bank's
proxy channels (z = verifier-ensemble logit, grip = gripper error, F x C = 64).
Within each task, channels are rank-transformed to [0,1]; the candidate reward
is r(alpha) = alpha*rank(z) + (1-alpha)*rank(-grip). alpha is chosen to
maximize within-task pairwise ordering agreement on CONFIDENT pairs (binomial
ordering-confidence >= 0.8, the fine-exam standard from make_fine_grid.py),
with task-held-out cross-validation so the reported number is honest.

  .venv/bin/python scripts/fit_rank_reward.py
"""
import glob
import itertools
import json
import math

import numpy as np
import pandas as pd

GT_N = 36
CONF = 0.8


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
    d["z01"] = d.groupby("task").z.rank(pct=True)
    d["g01"] = d.groupby("task").grip.rank(pct=True, ascending=False)  # low grip = good
    d["logit_old"] = 0.4445 * d.z + 11.3193 * (-d.grip)
    return d


def pairs_conf(d):
    """(task, i, j, gap, bucket) for confident within-task pairs; i = truly better."""
    out = []
    for t, g in d.groupby("task"):
        for (i, a), (j, b) in itertools.combinations(g.iterrows(), 2):
            gap = abs(a.gt_success - b.gt_success)
            se = 100 * math.sqrt(a.gt_success / 100 * (1 - a.gt_success / 100) / GT_N
                                 + b.gt_success / 100 * (1 - b.gt_success / 100) / GT_N)
            conf = 0.5 * (1 + math.erf((gap / se) / math.sqrt(2))) if se > 0 else 1.0
            if gap < 5 or conf < CONF:
                continue
            bucket = "fine_5-10" if gap < 10 else ("med_10-15" if gap < 15 else "far_15+")
            out.append((t, i, j, gap, bucket) if a.gt_success > b.gt_success
                       else (t, j, i, gap, bucket))
    return out


def agreement(d, prs, col, bucket=None):
    ok = tot = 0
    v = d[col]
    for t, i, j, gap, b in prs:
        if bucket and b != bucket:
            continue
        tot += 1
        ok += v[i] > v[j]
    return (100 * ok / tot if tot else float("nan")), tot


def main():
    d = load()
    prs = pairs_conf(d)
    n_by = {b: sum(p[4] == b for p in prs) for b in ("fine_5-10", "med_10-15", "far_15+")}
    print(f"{len(d)} phrases, {len(prs)} confident pairs (>=5pp gap, ordering-conf>={CONF}): {n_by}")

    alphas = np.round(np.arange(0, 1.0001, 0.05), 2)
    print("\nalpha sweep, r = alpha*rank(z) + (1-alpha)*rank(-grip), agreement on ALL confident pairs:")
    for a in alphas:
        d["_r"] = a * d.z01 + (1 - a) * d.g01
        acc, _ = agreement(d, prs, "_r")
        bar = "#" * int((acc - 50) / 1.2)
        print(f"  alpha={a:4.2f}  {acc:5.1f}  {bar}")

    # task-held-out: pick alpha on 14 tasks, score the 15th
    print("\ntask-held-out CV (alpha chosen per fold on the other tasks):")
    cv_ok = cv_tot = 0
    picked = []
    for t in d.task.unique():
        tr_prs = [p for p in prs if p[0] != t]
        te_prs = [p for p in prs if p[0] == t]
        if not te_prs:
            continue
        best_a, best = None, -1
        for a in alphas:
            d["_r"] = a * d.z01 + (1 - a) * d.g01
            acc, _ = agreement(d, tr_prs, "_r")
            if acc > best:
                best, best_a = acc, a
        picked.append(best_a)
        d["_r"] = best_a * d.z01 + (1 - best_a) * d.g01
        acc, n = agreement(d, te_prs, "_r")
        cv_ok += acc / 100 * n
        cv_tot += n
    cv_acc = 100 * cv_ok / cv_tot
    print(f"  held-out agreement {cv_acc:.1f}  (alphas picked per fold: "
          f"{sorted(set(picked))})")

    print(f"\n{'reward':30} {'all>=5pp':>9} {'fine_5-10':>10} {'med_10-15':>10} {'far_15+':>9}")
    out = {"n_pairs": n_by, "cv_rankfit_agreement": round(cv_acc, 1),
           "cv_alphas": sorted(set(float(a) for a in picked)), "rows": {}}
    cands = [("g01", "gripper only (rank)"), ("z01", "verifier only (rank)"),
             ("logit_old", "frozen calibrated logit")]
    d["c4b"] = 0.25 * d.z01 + 0.75 * d.g01
    cands.append(("c4b", "c4b 0.25z+0.75g (ranks)"))
    a_star = max(alphas, key=lambda a: agreement(
        d.assign(_r=a * d.z01 + (1 - a) * d.g01), prs, "_r")[0])
    d["rankfit"] = a_star * d.z01 + (1 - a_star) * d.g01
    cands.append(("rankfit", f"rank-fit alpha={a_star} (in-sample)"))
    for col, lab in cands:
        row = [agreement(d, prs, col)[0]] + [
            agreement(d, prs, col, b)[0] for b in ("fine_5-10", "med_10-15", "far_15+")]
        out["rows"][lab] = [round(x, 1) for x in row]
        print(f"{lab:30} " + " ".join(f"{x:9.1f}" for x in row))

    with open("results/analysis/rank_reward_fit.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nwrote results/analysis/rank_reward_fit.json")


if __name__ == "__main__":
    main()
