#!/usr/bin/env python3
"""results/charts/training_flow_loop.png -- the automated rules loop, teaser-strip
format. Replaces the hand-drawn training_flow_compact.png; this one is scripted
so the figure can never drift from the pipeline again.

What changed vs the old figure: (1) the input is the orig/natural/adversarial
evaluation set, not a single instruction; (2) the rephraser is conditioned on
the current rulebook AND a scene trace; (3) scored rewrites accumulate into an
evidence bank across iterations; (4) an LLM judge audits rule adherence and its
report feeds the distiller alongside the evidence.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle

REPO = Path(__file__).resolve().parent.parent

# palette lifted from the original figure
SLATE = "#5b6b7a"
PUR, PURF = "#7c5cd6", "#ece7fa"
BLU, BLUF = "#3a6db5", "#e8f0fb"
TEA, TEAF = "#2f9e8f", "#d9f4ec"
ORA, ORAF = "#d9822b", "#fdeeda"
GRN, GRNF = "#3f9d55", "#e2f6e5"
INK = "#22304a"

fig, ax = plt.subplots(figsize=(16.0, 7.4), dpi=150)
ax.set_xlim(0, 24); ax.set_ylim(0, 11); ax.axis("off")


def panel(x0, y0, x1, y1, color):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                                boxstyle="round,pad=0.12,rounding_size=0.35",
                                fc=color, ec="none", zorder=0))


def box(x0, y0, x1, y1, edge, fill, title, sub=None, tsize=15, ssize=11,
        lw=2.4, zorder=3):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                                boxstyle="round,pad=0.10,rounding_size=0.28",
                                fc=fill, ec=edge, lw=lw, zorder=zorder))
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    if sub:
        ax.text(cx, cy + 0.32, title, ha="center", va="center", fontsize=tsize,
                fontweight="bold", color=INK, zorder=zorder + 1)
        ax.text(cx, cy - 0.42, sub, ha="center", va="center", fontsize=ssize,
                color=edge, zorder=zorder + 1)
    else:
        ax.text(cx, cy, title, ha="center", va="center", fontsize=tsize,
                fontweight="bold", color=INK, zorder=zorder + 1)


def chip(x0, y0, x1, y1, edge, fill, text, size=11.5, style="normal", lw=1.8):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                                boxstyle="round,pad=0.08,rounding_size=0.30",
                                fc=fill, ec=edge, lw=lw, zorder=3))
    ax.text((x0 + x1) / 2, (y0 + y1) / 2, text, ha="center", va="center",
            fontsize=size, color=INK, style=style, zorder=4)


def arrow(p, q, rad=0.0, color=SLATE, lw=2.2, ls="-", ms=20, z=2):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=ms,
                                 lw=lw, color=color, linestyle=ls, zorder=z,
                                 shrinkA=2, shrinkB=2,
                                 connectionstyle=f"arc3,rad={rad}"))


# ------------------------------------------------------------------ panels
panel(0.25, 4.55, 10.55, 10.75, "#f4f2fc")          # GENERATE
panel(10.95, 4.55, 23.75, 10.75, "#edf4fb")         # EVALUATE
panel(0.25, 0.25, 23.75, 4.15, "#fdf7e6")           # DISTILL & ITERATE
ax.text(5.05, 10.38, "G E N E R A T E", fontsize=13, fontweight="bold",
        color=PUR, zorder=4)
ax.text(23.35, 10.30, "E V A L U A T E", fontsize=13, fontweight="bold",
        color="#2196d6", ha="right", zorder=4)
ax.text(0.65, 0.55, "D I S T I L L   &   I T E R A T E", fontsize=13,
        fontweight="bold", color="#d9822b", zorder=4)

# --------------------------------------------------------------- GENERATE
# evaluation set + trace feed the rephraser
box(0.85, 8.55, 4.55, 10.15, "#8a97a6", "#f2f4f6", "evaluation set",
    "original / natural / adversarial", tsize=13, ssize=11, lw=1.8)
chip(0.85, 7.30, 2.95, 8.00, TEA, "#eefaf6", "scene trace", 11)
arrow((3.20, 8.55), (3.20, 6.80), rad=-0.12)
arrow((1.85, 7.30), (1.85, 6.80), rad=0.0, lw=1.8, ms=15)

# rephraser
box(1.15, 5.05, 4.75, 6.80, PUR, PURF, "rephraser", "VLM + current rulebook",
    tsize=16, ssize=11.5)
# rewrites fan
chip(5.55, 8.15, 7.85, 8.85, "#9b8fe0", "#ffffff", "rewrite 1", 12)
chip(5.55, 5.35, 7.85, 6.05, "#9b8fe0", "#ffffff", "rewrite N", 12)
ax.text(6.70, 7.25, "$\\vdots$", fontsize=15, ha="center", color="#9b8fe0",
        zorder=4)
arrow((4.75, 6.35), (5.55, 8.30), rad=-0.18)
arrow((4.75, 5.75), (5.55, 5.85), rad=0.10)
# VLA
box(8.35, 6.15, 10.35, 7.95, BLU, BLUF, "VLA", "executor", tsize=17, ssize=12)
arrow((7.85, 8.40), (8.50, 7.85), rad=-0.15)
arrow((7.85, 5.80), (8.50, 6.30), rad=0.15)

# --------------------------------------------------------------- EVALUATE
chip(11.85, 9.85, 15.05, 10.50, TEA, "#ffffff", "simulatable tasks", 11.5)
box(11.55, 8.15, 15.95, 9.65, TEA, TEAF, "score with simulated", "rollouts",
    tsize=13.5, ssize=12)
box(11.55, 5.45, 15.95, 6.95, ORA, ORAF, "score with trajectory", "proxy",
    tsize=13.5, ssize=12)
chip(11.85, 4.75, 16.25, 5.35, ORA, "#ffffff", "training / real-world tasks", 11)
arrow((10.35, 7.55), (11.55, 8.70), rad=-0.18)
arrow((10.35, 6.55), (11.55, 6.20), rad=0.15)

# evidence bank (stacked look)
ax.add_patch(FancyBboxPatch((18.35, 6.05), 4.6, 2.5,
                            boxstyle="round,pad=0.10,rounding_size=0.28",
                            fc="#eef8ef", ec=GRN, lw=1.4, alpha=0.55, zorder=2))
ax.add_patch(FancyBboxPatch((18.15, 6.25), 4.6, 2.5,
                            boxstyle="round,pad=0.10,rounding_size=0.28",
                            fc="#e8f7ea", ec=GRN, lw=1.7, alpha=0.75, zorder=2))
box(17.95, 6.45, 22.55, 8.95, GRN, GRNF, "evidence bank",
    "scored rewrites, all iterations", tsize=15.5, ssize=11.5)
arrow((15.95, 8.90), (17.95, 8.35), rad=-0.18)
arrow((15.95, 6.20), (17.95, 7.05), rad=0.18)

# ------------------------------------------------------ DISTILL & ITERATE
box(1.15, 1.45, 5.15, 3.35, PUR, PURF, "phrasing rulebook", "updated",
    tsize=15.5, ssize=12)
box(9.85, 1.45, 14.45, 3.35, TEA, "#eafaf5", "LLM judge",
    "did rewrites follow the rules?", tsize=15.5, ssize=11.5)
box(17.35, 1.45, 21.35, 3.35, BLU, BLUF, "LLM", "rules distiller",
    tsize=16, ssize=12)

# rewrites + scores drop into the judge; evidence bank into the distiller
arrow((13.30, 4.55), (13.30, 3.35), rad=0.0)
ax.text(13.60, 4.22, "rewrites + scores", fontsize=10, style="italic",
        color=SLATE, ha="left", zorder=4)
arrow((20.25, 6.35), (19.85, 3.35), rad=-0.10)
arrow((14.45, 2.40), (17.35, 2.40), rad=0.0)
ax.text(15.90, 2.70, "adherence report", fontsize=10, style="italic",
        color=SLATE, ha="center", zorder=4)

# previous rulebook joins the distiller (the plus arc)
arrow((5.15, 3.20), (17.60, 3.40), rad=-0.05, lw=1.9)
circ_x, circ_y = 7.60, 3.55
ax.add_patch(Circle((circ_x, circ_y), 0.26, fc="white", ec=SLATE, lw=1.8,
                    zorder=5))
ax.text(circ_x, circ_y, "$\\oplus$", fontsize=12, ha="center", va="center",
        color=SLATE, zorder=6)
ax.text(7.60, 4.05, "with previous rulebook", fontsize=10.5, style="italic",
        color=SLATE, ha="center", zorder=6)

# distiller emits the updated rulebook, routed under the judge
arrow((19.35, 1.45), (3.60, 1.45), rad=-0.055)

# iterate: rulebook back to the rephraser
arrow((1.45, 3.35), (1.60, 5.05), rad=0.30, color=PUR, lw=2.0, ls="--", ms=17)
ax.text(0.55, 3.95, "iterate", fontsize=11.5, style="italic", color=PUR,
        rotation=70, zorder=4)

out = REPO / "results/charts/training_flow_loop.png"
fig.savefig(out, bbox_inches="tight", facecolor="white")
print("wrote", out)
