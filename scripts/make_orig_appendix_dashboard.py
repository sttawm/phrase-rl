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
    ax.text(x, v - dy, f"{v:.1f}", ha="center", va="top", fontsize=7.2,
            fontweight="bold", color=txt, zorder=6,
            bbox=dict(boxstyle="square,pad=0.10", fc=fc, ec="none"))


fig, axes = plt.subplots(2, 1, figsize=(8.6, 8.8))

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
        ax.bar(x, m, 0.62, color=C_RULES, edgecolor=EDGE, lw=0.8)
        bar_annot(ax, x, m, C_RULES)
        ticks.append(x); tlabels.append(dlabel); gxs.append(x); x += 0.82
    gticks.append(sum(gxs) / len(gxs))
    ax.axvline(x - 0.31, color="#e2e8f0", lw=1.0, zorder=0)
    x += 0.42
ax.axhline(BASE, color=RED, lw=1.2, ls=(0, (4, 3)), zorder=1)
ax.text(1.005, BASE, f"{BASE:.1f}", fontsize=6.8, color=RED, ha="left",
        va="center", transform=ax.get_yaxis_transform())
ax.set_xticks(ticks); ax.set_xticklabels(tlabels, fontsize=6.2, color="#4a5568")
for gx, ap in zip(gticks, ["Claude", "Gemini", "Qwen"]):
    ax.text(gx, 22.2, ap, ha="center", fontsize=10.5, fontweight="bold",
            clip_on=False)
ax.set_ylim(24, 44)
ax.set_xlim(-0.6, x - 0.55)
ax.set_ylabel("success %\n(by applier)", fontsize=10.5)

# Row 2: by draw (mean over appliers)
ax = axes[1]
x, ticks, tlabels, gticks = 0.0, [], [], []
for diet, dlabel in DIETS:
    gxs = []
    for dr in (1, 2, 3):
        vals = [cell(diet, dr, ap) for ap in APS if cell(diet, dr, ap)]
        m = sum(vals) / len(vals)
        ax.bar(x, m, 0.62, color=C_RULES, edgecolor=EDGE, lw=0.8)
        bar_annot(ax, x, m, C_RULES)
        ticks.append(x); tlabels.append(DRAW_LBL[dr - 1]); gxs.append(x)
        x += 0.82
    gticks.append(sum(gxs) / len(gxs))
    ax.axvline(x - 0.31, color="#e2e8f0", lw=1.0, zorder=0)
    x += 0.42
sc_mean = sum(SC.values()) / 3
ax.axhline(sc_mean, color=GREY, lw=1.2, ls=(0, (2, 2)), zorder=1)
ax.text(1.005, sc_mean, f"{sc_mean:.1f}", fontsize=6.8, color=GREY, ha="left",
        va="center", transform=ax.get_yaxis_transform())
ax.axhline(BASE, color=RED, lw=1.2, ls=(0, (4, 3)), zorder=1)
ax.text(1.005, BASE, f"{BASE:.1f}", fontsize=6.8, color=RED, ha="left",
        va="center", transform=ax.get_yaxis_transform())
ax.set_xticks(ticks); ax.set_xticklabels(tlabels, fontsize=6.0, color="#4a5568")
for gx, (diet, dlabel) in zip(gticks, DIETS):
    ax.text(gx, 22.2, dlabel.replace("\n", ""), ha="center", fontsize=10.5,
            fontweight="bold", clip_on=False)
ax.set_ylim(24, 44)
ax.set_xlim(-0.6, x - 0.55)
ax.set_ylabel("success %\n(by rulebook draw)", fontsize=10.5)

for ax in axes:
    ax.tick_params(axis="x", length=0)
    ax.grid(axis="y", alpha=0.16)
    ax.spines[["top", "right"]].set_visible(False)

handles = [plt.Rectangle((0, 0), 1, 1, fc=C_BASE, ec=EDGE),
           plt.Rectangle((0, 0), 1, 1, fc=C_RULES, ec=EDGE),
           plt.Line2D([0], [0], color=RED, lw=1.2, ls=(0, (4, 3))),
           plt.Line2D([0], [0], color=GREY, lw=1.2, ls=(0, (2, 2)))]
fig.legend(handles, ["no-rules rephraser (scaffold)", "rulebook cell",
                     "no rephraser (canonical instructions)", "no rules (mean)"],
           fontsize=8.4, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 0.0))
PAPER = os.environ.get("PAPER") == "1"
if not PAPER:
    fig.suptitle("Originals — clean-trace cells (A40/A38), appendix companion "
                 "to the main dashboard", fontsize=12, y=0.99)
fig.tight_layout(rect=(0, 0.05, 1, 0.97 if not PAPER else 1.0))
out = R / ("results/charts/orig_appendix_dashboard_paper.png" if PAPER
           else "results/charts/orig_appendix_dashboard.png")
fig.savefig(out, dpi=145, bbox_inches="tight", pad_inches=0.22)
print("chart ->", out)
