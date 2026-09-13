#!/usr/bin/env python3
"""results/charts/orig_appendix_dashboard.png — the Originals column, appendix
edition. Same idiom as rules_dashboard v2 (pastels, values in bars,
in-finetune/out-of-finetune/both, rulebook-N). CLEAN-TRACE data throughout:
book cells from results/analysis/a40_clean_orig_cells.json, scaffold cells
from the A38 scc legs, baseline = anchors originals 36.1. PAPER=1 drops the
title and writes *_paper.png.
"""
import json
import os
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = pathlib.Path(__file__).resolve().parents[1]
A40 = json.loads((R / "results/analysis/a40_clean_orig_cells.json").read_text())
SC = {"claude": 31.8, "gemini": 31.1, "qwen": 28.5}   # A38 scc cells (PREREG)
BASE = 36.1

APS = ["claude", "gemini", "qwen"]
DIETS = [("s", "out-of-\nfinetune"), ("b", "both"), ("t", "in-\nfinetune")]
DRAW_LBL = ["rulebook-1", "rulebook-2", "rulebook-3"]
BK = {("s", 1): "s", ("s", 2): "s2", ("s", 3): "s3", ("b", 1): "b",
      ("b", 2): "b2", ("b", 3): "b3", ("t", 1): "t", ("t", 2): "t2",
      ("t", 3): "t3"}
C_BASE, C_RULES = "#D9DEE6", "#3F6B52"
EDGE, INK, RED, GREY = "#6E7B8B", "#2d3748", "#C0504D", "#7A8698"


def cell(diet, dr, ap):
    return A40.get(f"{BK[(diet, dr)]}_{ap}")


def bar_annot(ax, x, v, fc, dy=0.75):
    txt = "white" if fc == C_RULES else INK
    ax.text(x, v - dy, f"{v:.1f}", ha="center", va="top", fontsize=6.0,
            fontweight="bold", color=txt, zorder=6,
            bbox=dict(boxstyle="square,pad=0.10", fc=fc, ec="none"))


fig, axes = plt.subplots(1, 1, figsize=(4.6, 3.4))
axes = [axes]

# Row 1: by applier (mean over draws)
ax = axes[0]
x, ticks, tlabels, gticks = 0.0, [], [], []
for ap in APS:
    gxs = []
    ax.bar(x, SC[ap], 0.62, color=C_BASE, edgecolor=EDGE, lw=0.8)
    bar_annot(ax, x, SC[ap], C_BASE)
    ticks.append(x); tlabels.append("no\nrules"); gxs.append(x); x += 0.82
    for diet, dlabel in DIETS:
        vals = [cell(diet, dr, ap) for dr in (1, 2, 3) if cell(diet, dr, ap)]
        m = sum(vals) / len(vals)
        ax.bar(x, m, 0.62, color=C_RULES, edgecolor="none")
        bar_annot(ax, x, m, C_RULES)
        ticks.append(x); tlabels.append(dlabel); gxs.append(x); x += 0.82
    gticks.append(sum(gxs) / len(gxs))
    ax.axvline(x - 0.31, color="#e2e8f0", lw=1.0, zorder=0)
    x += 0.42
ax.axhline(BASE, color=RED, lw=1.2, ls=(0, (4, 3)), zorder=1)
ax.text(1.005, BASE, f"{BASE:.1f}", fontsize=5.8, color=RED, ha="left",
        va="center", transform=ax.get_yaxis_transform())
ax.set_xticks(ticks); ax.set_xticklabels(tlabels, fontsize=5.2, color="#4a5568")
for gx, ap in zip(gticks, ["Claude", "Gemini", "Qwen"]):
    ax.text(gx, 20.1, ap, ha="center", fontsize=8.0, fontweight="bold",
            clip_on=False)
ax.set_ylim(24, 44)
ax.set_xlim(-0.6, x - 0.55)
ax.set_ylabel("success %", fontsize=8.0)

for ax in axes:
    ax.tick_params(axis="x", length=0)
    ax.grid(axis="y", alpha=0.16)
    ax.spines[["top", "right"]].set_visible(False)

handles = [plt.Rectangle((0, 0), 1, 1, fc=C_BASE, ec=EDGE),
           plt.Rectangle((0, 0), 1, 1, fc=C_RULES, ec="none"),
           plt.Line2D([0], [0], color=RED, lw=1.2, ls=(0, (4, 3))),
           plt.Line2D([0], [0], color=GREY, lw=1.2, ls=(0, (2, 2)))]
fig.legend(handles[:3], ["no-rules rephraser (scaffold)", "rulebook cell",
                     "no rephraser (canonical instructions)"],
           fontsize=5.8, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 0.0),
           frameon=False, columnspacing=0.9, handlelength=1.1)
PAPER = os.environ.get("PAPER") == "1"
if not PAPER:
    fig.suptitle("Originals — clean-trace cells (A40/A38), appendix companion "
                 "to the main dashboard", fontsize=12, y=0.99)
fig.tight_layout(rect=(0, 0.08, 1, 0.95 if not PAPER else 1.0))
out = R / ("results/charts/orig_appendix_dashboard_paper.png" if PAPER
           else "results/charts/orig_appendix_dashboard.png")
fig.savefig(out, dpi=145, bbox_inches="tight", pad_inches=0.22)
print("chart ->", out)
