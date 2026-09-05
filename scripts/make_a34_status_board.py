#!/usr/bin/env python3
"""results/charts/a34_status_board.png -- live A34 grid.

The "more-natural" naturals re-run: the A33 image-conditioned natural set with
goal-changing and adversarial-register phrases removed (192 -> 186 -> 181).

Rows: the three v2 books (train-only / rollout-only / combined) + the no-rules
scaffold control. Columns: the three appliers. ONE condition only (Natural) --
unlike the A31 board, which crossed three conditions.

Grey box = in progress. Data: results/analysis/a34_status_cells.json
  {"<row>|<applier>": value, "_baseline": value_or_null}
"""
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

REPO = pathlib.Path(__file__).resolve().parents[1]
cells = json.loads((REPO / "results/analysis/a34_status_cells.json").read_text())

ROWS = [("train_only", "train-only book"), ("sim_only", "rollout-only book"),
        ("both", "combined book"), ("scaffold", "no rules (scaffold)")]
APPS = ["claude", "gemini", "qwen"]
BASELINE = cells.get("_baseline")

fig, ax = plt.subplots(figsize=(7.6, 4.4))
ax.set_xlim(-1.75, len(APPS) + 0.05); ax.set_ylim(-0.75, len(ROWS) + 0.95)
ax.axis("off")


def shade(v):
    t = max(0.0, min(1.0, (v - 20) / 25))
    return (0.55 - 0.35 * t, 0.65 + 0.1 * t, 0.95)


for ai, app in enumerate(APPS):
    ax.text(ai + 0.5, len(ROWS) + 0.12, app, ha="center", fontsize=10,
            color="#4a5568")
ax.text(len(APPS) / 2, len(ROWS) + 0.55, "Natural  (wider-register set, 186 phrases)",
        ha="center", fontsize=12, fontweight="bold")

for ri, (rk, rlabel) in enumerate(ROWS):
    y = len(ROWS) - 1 - ri
    ax.text(-0.15, y + 0.5, rlabel, ha="right", va="center", fontsize=10)
    for ai, app in enumerate(APPS):
        v = cells.get(f"{rk}|{app}")
        if v is None:
            ax.add_patch(Rectangle((ai + 0.05, y + 0.07), 0.9, 0.86,
                                   facecolor="#d7d7d2", edgecolor="#b9b9b2", lw=0.8))
            ax.text(ai + 0.5, y + 0.5, "…", ha="center", va="center",
                    fontsize=15, color="#7a7a72")
        elif v == "X":
            ax.add_patch(Rectangle((ai + 0.05, y + 0.07), 0.9, 0.86,
                                   facecolor="#f1ece2", edgecolor="#d8d2c6", lw=0.8))
            ax.text(ai + 0.5, y + 0.5, "X", ha="center", va="center",
                    fontsize=11, color="#9a9186")
        else:
            ax.add_patch(Rectangle((ai + 0.05, y + 0.07), 0.9, 0.86,
                                   facecolor=shade(v), edgecolor="none"))
            ax.text(ai + 0.5, y + 0.5, f"{v:.1f}", ha="center", va="center",
                    fontsize=13, fontweight="bold", color="white")

ax.add_patch(Rectangle((0, -0.62), len(APPS), 0.42, color="#f1f1ee", zorder=1))
btxt = ("un-rephrased: PENDING"
        if BASELINE is None else f"un-rephrased baseline (no applier): {BASELINE}")
ax.text(len(APPS) / 2, -0.41, btxt, ha="center", va="center",
        fontsize=8.5, color="#718096")

fig.suptitle("A34 — sealed natural condition, 12 tasks (COMPLETE)", fontsize=13, y=0.98)
fig.text(0.5, 0.045,
         "186 phrases × 24 layouts × 1 rep · 13 arms · 156/156 legs · zero failures",
         ha="center", fontsize=7.5, color="#8a8a8a")
fig.text(0.5, 0.015,
         "Cells BASE-WEIGHTED over the 186 bases (not distinct rewrites: appliers collapse duplicates unequally). "
         "claude=opus-5 · gemini thinking_budget 1024",
         ha="center", fontsize=6.6, color="#a0a0a0")

out = REPO / "results/charts/a34_status_board.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.tight_layout(rect=[0, 0.03, 1, 0.94])
fig.savefig(out, dpi=170)
print("wrote", out)
