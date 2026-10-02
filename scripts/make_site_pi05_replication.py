#!/usr/bin/env python3
"""results/charts/site_pi05_replication.png — the pi0.5 / LIBERO replication
for the project site, in-finetune half only, in the main-dashboard idiom
(make_rules_dashboard_compact.py palette): the no-rephraser baseline as a
dashed red line, the no-rules rephraser as a grey bar, and the rulebooks
(mean over the three v2 draws 2-4; draw 1 used the v1 prompt and is not
reported) as a dark green bar, annotated with the per-draw gain range and
p-value. The out-of-finetune half (no significant rulebook effect) is stated
in the site caption, not charted. No in-image title.

Numbers: results/analysis/pi05_bank/LIBERO_V2_OVERVIEW.md section 8
(rounds 1-4, n = 100 bases x 50 inits per arm in this half, Gemini applier)."""
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = pathlib.Path(__file__).resolve().parents[1]
C_BASE, C_RULES = "#D9DEE6", "#3F6B52"
EDGE, INK, RED = "#6E7B8B", "#2d3748", "#C0504D"

NO_REPH = 93.6                       # no rephraser, in-finetune
SCAFFOLD = 93.6                      # no-rules rephraser, in-finetune (+0.1, p = 0.87)
DRAWS = [97.6, 97.9, 97.9]           # rulebook draws 2, 3, 4 (+4.1 / +4.3 / +4.3, p <= 0.005)
RULES = sum(DRAWS) / len(DRAWS)      # 97.8

fig, ax = plt.subplots(figsize=(5.6, 4.3))
W = 0.5
xs, xr = -0.3, 0.3
ax.bar(xs, SCAFFOLD, W, color=C_BASE, edgecolor=EDGE, lw=0.7, zorder=2)
ax.text(xs, SCAFFOLD - 1.6, f"{SCAFFOLD:.1f}", ha="center", va="top", fontsize=10,
        fontweight="bold", color=INK, zorder=4,
        bbox=dict(boxstyle="square,pad=0.09", fc=C_BASE, ec="none"))
ax.bar(xr, RULES, W, color=C_RULES, edgecolor="none", zorder=2)
ax.text(xr, RULES - 1.6, f"{RULES:.1f}", ha="center", va="top", fontsize=10,
        fontweight="bold", color="white", zorder=4,
        bbox=dict(boxstyle="square,pad=0.09", fc=C_RULES, ec="none"))
ax.plot([-0.72, 0.72], [NO_REPH] * 2, color=RED, lw=1.2, ls=(0, (4, 3)), zorder=3)
ax.text(0.76, NO_REPH, f"{NO_REPH:.1f}", fontsize=8.6, color=RED, ha="left",
        va="center", zorder=4)
ax.text(xr, RULES + 1.2, "+4.1 to +4.3 per rulebook\np ≤ 0.005", ha="center",
        va="bottom", fontsize=8.2, color=C_RULES, zorder=4)
ax.text(xs, -1.6, "no rules", ha="center", va="top", fontsize=8.5, color="#4a5568")
ax.text(xr, -1.6, "rulebooks", ha="center", va="top", fontsize=8.5, color="#4a5568")
ax.text(0, -9.0, "in-finetune tasks (10)", ha="center", va="top", fontsize=11,
        fontweight="bold", clip_on=False)

ax.set_xticks([])
ax.set_xlim(-1.0, 3.0)
ax.set_ylim(0, 110)
ax.set_yticks(range(0, 101, 20))
ax.set_ylabel("success %", fontsize=11)
ax.grid(axis="y", alpha=0.16)
ax.spines[["top", "right"]].set_visible(False)
handles = [plt.Line2D([0], [0], color=RED, lw=1.2, ls=(0, (4, 3))),
           plt.Rectangle((0, 0), 1, 1, fc=C_BASE, ec=EDGE),
           plt.Rectangle((0, 0), 1, 1, fc=C_RULES, ec="none")]
ax.legend(handles, ["no rephraser", "no-rules rephraser",
                    "rulebooks (mean over three draws)"],
          fontsize=8.6, frameon=False, loc="center right", bbox_to_anchor=(1.0, 0.5))
fig.tight_layout(rect=(0, 0.06, 1, 1))
out = R / "results/charts/site_pi05_replication.png"
fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.16)
print("chart ->", out, round(RULES, 1))
