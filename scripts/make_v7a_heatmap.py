#!/usr/bin/env python3
"""Per-task x checkpoint heatmap of the complete tagged v7a matrix, both
conditions — the view that exposes what pooled curves hide (zero-sum task
trades under a pinned pooled number).

Reads all v7_dev8_tagged*.jsonl (polish) and v7_dev8adv_tagged*.jsonl (repair),
merged by step. Cell = task success %, columns = checkpoints, bottom row =
pooled. Reference column shows each task's Original/Adversarial arm value from
the rolled val8 reference leg.
"""
import glob
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REF = pd.read_parquet("results/sealed/val8_reference_x12.parquet")


def ref_col(arm):
    pt = REF[REF.arm == arm].groupby("task").success.mean() * 100
    return pt


def rows(pattern):
    by_step = {}
    for f in sorted(glob.glob(pattern)):
        for l in open(f):
            r = json.loads(l)
            by_step[r["step"]] = r
    return dict(sorted(by_step.items()))


def panel(ax, data, ref_arm, title):
    steps = list(data)
    tasks = sorted(data[steps[0]]["per_task"])
    ref = ref_col(ref_arm)
    M = np.array([[data[s]["per_task"][t] for s in steps] for t in tasks])
    M = np.column_stack([M, [ref.get(t, np.nan) for t in tasks]])
    pooled = [data[s]["pooled"] for s in steps] + [round(float(ref.mean()), 1)]
    M = np.vstack([M, pooled])
    im = ax.imshow(M, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center", fontsize=7,
                    fontweight="bold" if i == M.shape[0] - 1 else "normal")
    ax.set_xticks(range(len(steps) + 1))
    ax.set_xticklabels([str(s) for s in steps] + ["REF"], fontsize=7)
    labels = [t.replace("widowx_", "").replace("_clean", "*") for t in tasks] + ["POOLED"]
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("v7a checkpoint step", fontsize=8)
    return im


pol = rows("results/analysis/v7_dev8_tagged*.jsonl")
rep = rows("results/analysis/v7_dev8adv_tagged*.jsonl")
fig, (a1, a2) = plt.subplots(2, 1, figsize=(13.5, 9.5))
panel(a1, pol, "val8_orig", "POLISH: rewriting original instructions (tagged eval, greedy, n=192/cell-col)")
im = panel(a2, rep, "val8_adv", "REPAIR: rewriting adversarial instructions (tagged eval, greedy, n=192/cell-col)")
fig.colorbar(im, ax=[a1, a2], label="success %", shrink=0.6)
fig.suptitle("v7a per-task dynamics — what the pooled curves hide (* = out-of-vocab task; REF = no-rewrite arm)",
             fontsize=11)
fig.savefig("results/charts/v7a_heatmap.png", dpi=150, bbox_inches="tight", pad_inches=0.3)
print("chart -> results/charts/v7a_heatmap.png")
