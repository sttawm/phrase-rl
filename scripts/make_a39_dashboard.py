#!/usr/bin/env python3
"""results/charts/a39_dashboard.png — human-naturals results, dashboard idiom.

Same design language as make_rules_dashboard.py (approved 2026-09-12): value
inside each bar in a bar-colored text box, dashed red no-rephraser (raw
human) baseline labeled outside the right edge, grey = no-rules scaffold,
blue = rulebook cells, no deltas. Rows: by applier (mean over draws) · by
rulebook draw (mean over appliers) · by register (pooled books, register-
matched raw baselines). PAPER=1 drops the title/footer and writes
*_paper.png.
"""
import json
import os
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = pathlib.Path(__file__).resolve().parents[1]
C = json.loads((R / "results/analysis/a39_human_cells.json").read_text())
PAPER = os.environ.get("PAPER") == "1"

APS = ["claude", "gemini", "qwen"]
DIETS = [("s", "rollout\nonly"), ("b", "rollout\n+ train"), ("t", "train\nonly")]
DRAWS = {"s": ["s", "s2", "s3"], "b": ["b", "b2", "b3"], "t": ["t", "t2", "t3"]}
RAW = C["raw_human"]
C_BASE, C_RULES = "#8b96a5", "#a3bffa"


def cell(book, ap, key="pooled"):
    r = C.get(f"{book}|{ap}")
    return r[key] if r else None


def bar_annot(ax, x, v, fc, dy=0.55):
    ax.text(x, v - dy, f"{v:.1f}", ha="center", va="top", fontsize=7.4,
            fontweight="bold", color="#1a202c", zorder=6,
            bbox=dict(boxstyle="square,pad=0.10", fc=fc, ec="none"))


def baseline(ax, y, color="#c53030", ls=(0, (4, 3))):
    ax.axhline(y, color=color, lw=1.2, ls=ls, zorder=1)
    ax.text(1.005, y, f"{y:.1f}", fontsize=6.8, color=color, ha="left",
            va="center", transform=ax.get_yaxis_transform())


fig, axes = plt.subplots(3, 1, figsize=(8.6, 12.6))

# ---- Row 1: by applier (mean over draws) ----
ax = axes[0]
x, ticks, tlabels, gticks = 0.0, [], [], []
for ap in APS:
    gxs = []
    sc = cell("sc", ap)
    ax.bar(x, sc, 0.62, color=C_BASE, edgecolor="#4a5568", lw=0.8)
    bar_annot(ax, x, sc, C_BASE)
    ticks.append(x); tlabels.append("no\nrules"); gxs.append(x); x += 0.82
    for diet, dlabel in DIETS:
        vals = [cell(b, ap) for b in DRAWS[diet] if cell(b, ap) is not None]
        m = sum(vals) / len(vals)
        ax.bar(x, m, 0.62, color=C_RULES, edgecolor="#4a5568", lw=0.8)
        bar_annot(ax, x, m, C_RULES)
        ticks.append(x); tlabels.append(dlabel); gxs.append(x); x += 0.82
    gticks.append(sum(gxs) / len(gxs))
    ax.axvline(x - 0.31, color="#e2e8f0", lw=1.0, zorder=0)
    x += 0.42
baseline(ax, RAW["pooled"])
ax.set_xticks(ticks); ax.set_xticklabels(tlabels, fontsize=6.4, color="#4a5568")
for gx, ap in zip(gticks, ["Claude", "Gemini", "Qwen"]):
    ax.text(gx, 19.35, ap, ha="center", fontsize=10.5, fontweight="bold", clip_on=False)
ax.set_ylim(21, 33); ax.set_xlim(-0.6, x - 0.55)
ax.set_ylabel("success %\n(by applier)", fontsize=10.5)

# ---- Row 2: by rulebook draw (mean over appliers) ----
ax = axes[1]
x, ticks, tlabels, gticks = 0.0, [], [], []
for diet, dlabel in DIETS:
    gxs = []
    for i, bk in enumerate(DRAWS[diet]):
        vals = [cell(bk, ap) for ap in APS if cell(bk, ap) is not None]
        m = sum(vals) / len(vals)
        ax.bar(x, m, 0.62, color=C_RULES, edgecolor="#4a5568", lw=0.8)
        bar_annot(ax, x, m, C_RULES)
        ticks.append(x); tlabels.append(f"r{i+1}"); gxs.append(x); x += 0.82
    gticks.append(sum(gxs) / len(gxs))
    ax.axvline(x - 0.31, color="#e2e8f0", lw=1.0, zorder=0)
    x += 0.42
sc_mean = sum(cell("sc", ap) for ap in APS) / 3
ax.axhline(sc_mean, color="#4a5568", lw=1.2, ls=(0, (2, 2)), zorder=1)
ax.text(1.005, sc_mean, f"{sc_mean:.1f}", fontsize=6.8, color="#4a5568",
        ha="left", va="bottom", transform=ax.get_yaxis_transform())
baseline(ax, RAW["pooled"])
ax.set_xticks(ticks); ax.set_xticklabels(tlabels, fontsize=7.2, color="#4a5568")
for gx, (diet, dlabel) in zip(gticks, DIETS):
    ax.text(gx, 19.35, dlabel.replace("\n", " "), ha="center", fontsize=10.5,
            fontweight="bold", clip_on=False)
ax.set_ylim(21, 33); ax.set_xlim(-0.6, x - 0.55)
ax.set_ylabel("success %\n(by rulebook draw)", fontsize=10.5)

# ---- Row 3: by register (books pooled over draws + appliers) ----
ax = axes[2]
x, ticks, tlabels, gticks = 0.0, [], [], []
for reg in ["adult", "kid", "robot"]:
    gxs = []
    sc = sum(cell("sc", ap, reg) for ap in APS) / 3
    ax.bar(x, sc, 0.62, color=C_BASE, edgecolor="#4a5568", lw=0.8)
    bar_annot(ax, x, sc, C_BASE)
    ticks.append(x); tlabels.append("no\nrules"); gxs.append(x); x += 0.82
    for diet, dlabel in DIETS:
        vals = [cell(b, ap, reg) for b in DRAWS[diet] for ap in APS
                if cell(b, ap, reg) is not None]
        m = sum(vals) / len(vals)
        ax.bar(x, m, 0.62, color=C_RULES, edgecolor="#4a5568", lw=0.8)
        bar_annot(ax, x, m, C_RULES)
        ticks.append(x); tlabels.append(dlabel); gxs.append(x); x += 0.82
    gticks.append(sum(gxs) / len(gxs))
    ax.axvline(x - 0.31, color="#e2e8f0", lw=1.0, zorder=0)
    # register-matched raw baseline segment under this cluster
    ax.hlines(RAW[reg], gxs[0] - 0.5, gxs[-1] + 0.5, color="#c53030",
              lw=1.2, ls=(0, (4, 3)), zorder=1)
    ax.text(gxs[-1] + 0.55, RAW[reg], f"{RAW[reg]:.1f}", fontsize=6.4,
            color="#c53030", va="center")
    x += 0.42
ax.set_xticks(ticks); ax.set_xticklabels(tlabels, fontsize=6.4, color="#4a5568")
for gx, reg in zip(gticks, ["Adult", "Kid", "Robot"]):
    ax.text(gx, 19.35, reg, ha="center", fontsize=10.5, fontweight="bold",
            clip_on=False)
ax.set_ylim(21, 33); ax.set_xlim(-0.6, x - 0.55)
ax.set_ylabel("success %\n(by register)", fontsize=10.5)

for ax in axes:
    ax.tick_params(axis="x", length=0)
    ax.grid(axis="y", alpha=0.16)
    ax.spines[["top", "right"]].set_visible(False)

handles = [plt.Rectangle((0, 0), 1, 1, color=C_BASE),
           plt.Rectangle((0, 0), 1, 1, color=C_RULES),
           plt.Line2D([0], [0], color="#c53030", lw=1.2, ls=(0, (4, 3))),
           plt.Line2D([0], [0], color="#4a5568", lw=1.2, ls=(0, (2, 2)))]
fig.legend(handles, ["no-rules rephraser (scaffold)", "rulebook cell",
                     "raw human phrases (no rephraser)", "no rules (mean)"],
           fontsize=8.4, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 0.0))
if not PAPER:
    fig.suptitle("Human naturals (A39) — base-weighted, 363 phrases, 12 sealed tasks\n"
                 "rows: by applier (mean over draws) · by rulebook draw (mean over "
                 "appliers) · by register (books pooled)", fontsize=12, y=0.995)
fig.tight_layout(rect=(0, 0.045, 1, 0.975 if not PAPER else 1.0))
out = R / ("results/charts/a39_dashboard_paper.png" if PAPER
           else "results/charts/a39_dashboard.png")
fig.savefig(out, dpi=145, bbox_inches="tight", pad_inches=0.22)
print("chart ->", out)
