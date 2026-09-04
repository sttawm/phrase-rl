#!/usr/bin/env python3
"""results/charts/a34_natural_board.png -- the A34 natural condition.

Rows: the four books + the un-rephrased baseline. Columns: the three appliers.
Grey = still rolling. Computed live from the d34* leg results, pooled per arm on
the judged A33 image-conditioned natural set (186 phrases, 24 layouts x 1 rep).
"""
import collections
import glob
import pathlib
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Rectangle

R = pathlib.Path(__file__).resolve().parents[1]
JOBS = R / "results/rules_runs/r1_sim/jobs"

ROWS = [("t", "train-only book"), ("s", "rollout-only book"),
        ("b", "combined book"), ("sc", "no-rules scaffold")]
APPS = [("cl", "claude"), ("ge", "gemini"), ("qw", "qwen")]

groups = collections.defaultdict(list)
for f in glob.glob(str(JOBS / "d34*.result.parquet")):
    groups[pathlib.Path(f).name.split("_")[0]].append(f)

def cell(stem):
    fs = groups.get(stem, [])
    if len(fs) < 12:
        return None, len(fs)
    d = pd.concat([pd.read_parquet(x) for x in fs]).dropna(subset=["gt_success"])
    return d.gt_success.mean(), len(d)

base_v, base_n = cell("d34basen")
A19 = {"t": (29.0, 28.6, 30.3), "s": (32.4, 32.7, 31.7),
       "b": (29.1, 29.1, 31.7), "sc": (30.2, 31.1, 30.0)}

fig, ax = plt.subplots(figsize=(9.6, 4.6))
ax.set_xlim(-0.15, 3.9); ax.set_ylim(-1.15, len(ROWS) + 0.75); ax.axis("off")

def shade(v):
    t = max(0.0, min(1.0, (v - 24) / 12))
    return (0.42 - 0.22 * t, 0.58 + 0.20 * t, 0.52 + 0.10 * t)

for ci, (_, an) in enumerate(APPS):
    ax.text(ci + 0.5, len(ROWS) + 0.28, an, ha="center", fontsize=10.5, color="#3d4a46")
ax.text(3.45, len(ROWS) + 0.28, "A19 set", ha="center", fontsize=9, color="#8a8578", style="italic")

for ri, (rk, rlabel) in enumerate(ROWS):
    y = len(ROWS) - 1 - ri
    ax.text(-0.22, y + 0.5, rlabel, ha="right", va="center", fontsize=10)
    for ci, (ak, _) in enumerate(APPS):
        v, n = cell(f"d34{rk}{ak}n")
        if v is None:
            ax.add_patch(Rectangle((ci + 0.04, y + 0.06), 0.92, 0.88,
                                   facecolor="#d9d9d4", edgecolor="#bcbcb5", lw=0.8))
            ax.text(ci + 0.5, y + 0.5, f"{n}/12", ha="center", va="center",
                    fontsize=8.5, color="#82827b")
        else:
            ax.add_patch(Rectangle((ci + 0.04, y + 0.06), 0.92, 0.88,
                                   facecolor=shade(v), edgecolor="#38443f", lw=0.9))
            ax.text(ci + 0.5, y + 0.56, f"{v:.1f}", ha="center", va="center",
                    fontsize=13, fontweight="bold", color="white")
            if base_v:
                ax.text(ci + 0.5, y + 0.22, f"{v - base_v:+.1f} vs base",
                        ha="center", va="center", fontsize=7.5, color="#eaf2ee")
    old = A19[rk]
    ax.text(3.45, y + 0.5, " / ".join(f"{x:.1f}" for x in old),
            ha="center", va="center", fontsize=8.5, color="#8a8578")

bl = f"{base_v:.1f}" if base_v else f"rolling {base_n}/12"
ax.add_patch(Rectangle((0.04, -0.92), 2.92, 0.62, facecolor="#efefe9", edgecolor="#cfcfc7"))
ax.text(1.5, -0.61, f"un-rephrased baseline (same 186 phrases):  {bl}",
        ha="center", va="center", fontsize=10, color="#4a5450")
fig.text(0.5, 0.965, "Sealed natural condition, wider-register set (A34)",
         ha="center", fontsize=13)
fig.text(0.5, 0.925, "186 phrases from 3 authors, image-conditioned  ·  24 layouts  ·  grey = legs still rolling",
         ha="center", fontsize=8.5, color="#6b7370")
fig.tight_layout(rect=[0, 0.01, 1, 0.90])
out = R / "results/charts/a34_natural_board.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("wrote", out)
