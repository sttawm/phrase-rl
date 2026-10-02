#!/usr/bin/env python3
"""results/charts/site_pi05_replication.png — the pi0.5 / LIBERO replication
for the project site, in the main-dashboard idiom (make_rules_dashboard_
compact.py palette): per half of the sealed set, the no-rephraser baseline
as a dashed red line, the no-rules rephraser as a grey bar, and the rulebooks
(mean over the three v2 draws 2-4; draw 1 used the v1 prompt and is not
reported) as a dark green bar. No in-image title; the caption explains.

Numbers: results/analysis/pi05_bank/LIBERO_V2_OVERVIEW.md section 8
(rounds 1-4, n = 200 bases x 50 inits per arm, Gemini applier)."""
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = pathlib.Path(__file__).resolve().parents[1]
C_BASE, C_RULES = "#D9DEE6", "#3F6B52"
EDGE, INK, RED = "#6E7B8B", "#2d3748", "#C0504D"

NO_REPH = {"in": 93.6, "out": 56.7}          # no rephraser
SCAFFOLD = {"in": 93.6, "out": 58.3}         # no-rules rephraser
DRAWS = {"in": [97.6, 97.9, 97.9], "out": [57.0, 56.6, 56.2]}  # draws 2, 3, 4
RULES = {h: sum(v) / len(v) for h, v in DRAWS.items()}        # 97.8 / 56.6

fig, ax = plt.subplots(figsize=(7.2, 4.4))
W = 0.5
for gi, (half, hlabel) in enumerate([("in", "in-finetune tasks (10)"),
                                     ("out", "out-of-finetune tasks (10)")]):
    x0 = gi * 2.0
    xs, xr = x0 - 0.3, x0 + 0.3
    ax.bar(xs, SCAFFOLD[half], W, color=C_BASE, edgecolor=EDGE, lw=0.7, zorder=2)
    ax.text(xs, SCAFFOLD[half] - 1.6, f"{SCAFFOLD[half]:.1f}", ha="center", va="top",
            fontsize=10, fontweight="bold", color=INK, zorder=4,
            bbox=dict(boxstyle="square,pad=0.09", fc=C_BASE, ec="none"))
    ax.bar(xr, RULES[half], W, color=C_RULES, edgecolor="none", zorder=2)
    ax.text(xr, RULES[half] - 1.6, f"{RULES[half]:.1f}", ha="center", va="top",
            fontsize=10, fontweight="bold", color="white", zorder=4,
            bbox=dict(boxstyle="square,pad=0.09", fc=C_RULES, ec="none"))
    ax.plot([x0 - 0.72, x0 + 0.72], [NO_REPH[half]] * 2, color=RED, lw=1.2,
            ls=(0, (4, 3)), zorder=3)
    ax.text(x0 + 0.76, NO_REPH[half], f"{NO_REPH[half]:.1f}", fontsize=8.6, color=RED,
            ha="left", va="center", zorder=4)
    ax.text(x0, -9.0, hlabel, ha="center", va="top", fontsize=11, fontweight="bold",
            clip_on=False)
    ax.text(xs, -1.6, "no rules", ha="center", va="top", fontsize=8.5, color="#4a5568")
    ax.text(xr, -1.6, "rulebooks", ha="center", va="top", fontsize=8.5, color="#4a5568")

ax.set_xticks([])
ax.set_xlim(-1.0, 3.1)
ax.set_ylim(0, 104)
ax.set_ylabel("success %", fontsize=11)
ax.grid(axis="y", alpha=0.16)
ax.spines[["top", "right"]].set_visible(False)
handles = [plt.Line2D([0], [0], color=RED, lw=1.2, ls=(0, (4, 3))),
           plt.Rectangle((0, 0), 1, 1, fc=C_BASE, ec=EDGE),
           plt.Rectangle((0, 0), 1, 1, fc=C_RULES, ec="none")]
ax.legend(handles, ["no rephraser", "no-rules rephraser",
                    "rulebooks (mean over three draws)"],
          fontsize=9, frameon=False, loc="upper right", bbox_to_anchor=(1.0, 1.02))
fig.tight_layout(rect=(0, 0.06, 1, 1))
out = R / "results/charts/site_pi05_replication.png"
fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.16)
print("chart ->", out, RULES)
