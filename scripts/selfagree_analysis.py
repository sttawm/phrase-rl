#!/usr/bin/env python3
"""Draw-to-draw self-agreement of the proxy on the 434-phrase gt set.

Draw 1 = the bank (seed 7, earliest-F frames per episode).
Draw 2 = bank_scores_selfagree_draw2.parquet (seed 8007, rng-sampled frames).
Decode/flow noise is IDENTICAL between draws (pinned to the verifier ensemble
seed by design), so disagreement measures pure context-selection sensitivity:
the share of the proxy's error vs ground truth that more/wider contexts could
in principle remove. Tasks whose entire pool fits the budget redraw identically
(ctx_sig proves it) and are reported separately -- for them the deployed
measurement has zero resampling freedom.

  .venv/bin/python scripts/selfagree_analysis.py
"""
import itertools
import json
import math

import numpy as np
import pandas as pd

GT_N, CONF = 36, 0.8


def spearman(a, b):
    return pd.Series(list(a)).corr(pd.Series(list(b)), method="spearman")


def load_gt():
    fe = pd.read_parquet("results/analysis/fine_exam_phrases.parquet")[
        ["task", "phrase", "gt_success"]]
    na = pd.concat(
        [pd.read_parquet(f) for f in
         ("results/analysis/sim_rollouts_natadv_0of2.parquet",
          "results/analysis/sim_rollouts_natadv_1of2.parquet")],
        ignore_index=True)[["task", "phrase", "gt_success"]]
    return pd.concat([fe, na], ignore_index=True).drop_duplicates(["task", "phrase"])


def main():
    d1 = pd.read_parquet("results/analysis/bank_scores_sim_0of1.parquet") \
        .dropna(subset=["z", "grip"]).drop_duplicates(["task", "phrase"])
    d2 = pd.read_parquet("results/analysis/bank_scores_selfagree_draw2.parquet") \
        .dropna(subset=["z", "grip"]).drop_duplicates(["task", "phrase"])
    d = d1.merge(d2, on=["task", "phrase"], suffixes=("_1", "_2")) \
          .merge(load_gt(), on=["task", "phrase"])
    for s in ("_1", "_2"):
        d[f"logit{s}"] = 0.4445 * d[f"z{s}"] + 11.3193 * (-d[f"grip{s}"])
    print(f"{len(d)} phrases matched across draws, {d.task.nunique()} tasks")

    ident = d.groupby("task").apply(
        lambda g: bool(np.allclose(g.z_1, g.z_2) and np.allclose(g.grip_1, g.grip_2)),
        include_groups=False)
    print("\nidentical-redraw tasks (no resampling freedom at this budget):")
    for t in ident[ident].index:
        print(f"  {t}  (n_ctx {int(d[d.task == t].n_ctx_1.iloc[0])})")
    live = d[~d.task.isin(ident[ident].index)].copy()
    print(f"live comparison: {len(live)} phrases on {live.task.nunique()} tasks")

    print("\nwithin-task draw-to-draw correlation (live tasks):")
    out = {"n_matched": len(d), "identical_tasks": sorted(ident[ident].index),
           "n_live": len(live)}
    for col in ("z", "grip", "logit"):
        sp = [spearman(g[f"{col}_1"], g[f"{col}_2"])
              for _, g in live.groupby("task") if len(g) >= 5]
        out[f"selfcorr_{col}"] = round(float(np.mean(sp)), 3)
        print(f"  {col:6} Spearman {np.mean(sp):.3f}")

    # pairwise ordering agreement, confident-gt pairs (the fit_rank_reward standard)
    def confident_pairs(frame):
        prs = []
        for t, g in frame.groupby("task"):
            for (i, a), (j, b) in itertools.combinations(g.iterrows(), 2):
                gap = abs(a.gt_success - b.gt_success)
                se = 100 * math.sqrt(a.gt_success / 100 * (1 - a.gt_success / 100) / GT_N
                                     + b.gt_success / 100 * (1 - b.gt_success / 100) / GT_N)
                conf = 0.5 * (1 + math.erf((gap / se) / math.sqrt(2))) if se > 0 else 1.0
                if gap >= 5 and conf >= CONF:
                    prs.append((i, j))
        return prs

    prs = confident_pairs(live)
    res = {}
    for name, ca, cb in [("draw1 vs gt", "logit_1", None), ("draw2 vs gt", "logit_2", None),
                         ("draw1 vs draw2", "logit_1", "logit_2")]:
        ok = 0
        for i, j in prs:
            a, b = live.loc[i], live.loc[j]
            if cb is None:
                ok += (a[ca] - b[ca]) * (a.gt_success - b.gt_success) > 0
            else:
                ok += (a[ca] - b[ca]) * (a[cb] - b[cb]) > 0
        res[name] = round(100 * ok / len(prs), 1)
    out["pairwise_confident"] = res
    out["n_pairs"] = len(prs)
    print(f"\npairwise on {len(prs)} confident pairs (live tasks):")
    for k, v in res.items():
        print(f"  {k:16} {v:.1f}%")

    with open("results/analysis/selfagree_434.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nwrote results/analysis/selfagree_434.json")


if __name__ == "__main__":
    main()
