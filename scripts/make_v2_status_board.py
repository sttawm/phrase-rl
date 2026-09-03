#!/usr/bin/env python3
"""results/charts/v2_status_board.png -- live A31 grid.

Rows: the three v2 books (train-only / rollout-only / combined) + the no-rules
scaffold baseline. Columns: {Adversarial, Natural, Original} x {Claude, Gemini,
Qwen}. Grey box = in progress. Left strip: un-rephrased baseline per condition.
Data: results/analysis/v2_status_cells.json  {"<row>|<cond>|<applier>": value}
"""
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

REPO = pathlib.Path(__file__).resolve().parents[1]
cells = json.loads((REPO / "results/analysis/v2_status_cells.json").read_text())

ROWS = [("train_only", "train-only book"), ("sim_only", "rollout-only book"),
        ("both", "combined book"), ("scaffold", "no rules (scaffold)")]
CONDS = ["Adversarial", "Natural", "Original"]
APPS = ["claude", "gemini", "qwen"]
UNREPHRASED = {"Adversarial": 24.5, "Natural": 28.45, "Original": 36.1}
UNREPH_NOTE = {"Adversarial": "72-attack", "Natural": "v1", "Original": "v1"}

fig, ax = plt.subplots(figsize=(13.2, 4.6))
ax.set_xlim(-2.1, 9); ax.set_ylim(-0.6, len(ROWS) + 1.0)
ax.axis("off")

def shade(v):
    t = max(0.0, min(1.0, (v - 20) / 25))
    return (0.55 - 0.35 * t, 0.65 + 0.1 * t, 0.95 - 0.25 * t * 0)

for ci, cond in enumerate(CONDS):
    for ai, app in enumerate(APPS):
        x = ci * 3 + ai
        ax.text(x + 0.5, len(ROWS) + 0.12, app, ha="center", fontsize=8.5, color="#4a5568")
    ax.text(ci * 3 + 1.5, len(ROWS) + 0.55, cond, ha="center", fontsize=11.5, fontweight="bold")
    ub = UNREPHRASED[cond]
    ax.text(-1.05, len(ROWS) + 0.55, "un-rephrased", ha="center", fontsize=8.5,
            fontweight="bold", color="#4a5568") if ci == 0 else None
    ax.add_patch(Rectangle((-1.6 + 0, len(ROWS) - 1 - 0, ), 0, 0))  # noop keep layout
for ci, cond in enumerate(CONDS):
    ub = UNREPHRASED[cond]
    ax.add_patch(Rectangle((ci * 3, -0.55), 3, 0.4, color="#f1f1ee", zorder=1))
    ax.text(ci * 3 + 1.5, -0.35, f"un-rephrased: {ub}  ({UNREPH_NOTE[cond]})",
            ha="center", va="center", fontsize=8, color="#718096")

for ri, (rk, rlabel) in enumerate(ROWS):
    y = len(ROWS) - 1 - ri
    ax.text(-0.15, y + 0.5, rlabel, ha="right", va="center", fontsize=9.5)
    for ci, cond in enumerate(CONDS):
        for ai, app in enumerate(APPS):
            x = ci * 3 + ai
            v = cells.get(f"{rk}|{cond}|{app}")
            if v is None:
                ax.add_patch(Rectangle((x + 0.05, y + 0.07), 0.9, 0.86,
                                       facecolor="#d7d7d2", edgecolor="#b9b9b2", lw=0.8))
                ax.text(x + 0.5, y + 0.5, "…", ha="center", va="center",
                        fontsize=11, color="#8a8a83")
            elif isinstance(v, str):
                ax.add_patch(Rectangle((x + 0.05, y + 0.07), 0.9, 0.86,
                                       facecolor="#efe9dc", edgecolor="#c9bea6", lw=0.8))
                ax.text(x + 0.5, y + 0.5, v, ha="center", va="center", fontsize=7.4, color="#7a6f57")
            else:
                ax.add_patch(Rectangle((x + 0.05, y + 0.07), 0.9, 0.86,
                                       facecolor=shade(v), edgecolor="#4a5568", lw=0.9))
                ax.text(x + 0.5, y + 0.5, f"{v:.1f}", ha="center", va="center",
                        fontsize=10, fontweight="bold", color="white")
fig.text(0.5, 0.97, "A31 v2 grid -- sealed suite success % (grey = in progress)",
         ha="center", fontsize=12.5)
fig.text(0.99, 0.01, "v1* = PRELIMINARY prior-era promptB numbers (different no-rules prompt + ERT traces); same-protocol re-runs in flight and will replace them. All v2 cells: 24 layouts.",
         ha="right", fontsize=7, color="#718096")
fig.tight_layout(rect=[0, 0.02, 1, 0.93])
out = REPO / "results/charts/v2_status_board.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("wrote", out)
