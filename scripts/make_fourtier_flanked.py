#!/usr/bin/env python3
"""results/charts/fourtier_flanked.png -- pi0.5/LIBERO four-tier chart in the
pi0_conditions idiom: one solid bar per tier pooled over all 28 tasks
(task-weighted), thin flanks = in-finetune (left, 10 tasks) /
out-of-finetune (right, 18 tasks). No in-image title."""
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = pathlib.Path(__file__).resolve().parents[1]
S = json.load(open(R / "results/analysis/fourtier_summary.json"))
IN, OUT = S["macro"]["goal_in_finetune"], S["macro"]["l90_clean"]

C = {"oracle": "#3F6B52", "orig": "#6B5E9B", "natural": "#B08A3E",
     "adversarial": "#A94E4E"}
TIERS = [("oracle_confirm", "oracle\n(searched)", C["oracle"]),
         ("orig", "original\ninstruction", C["orig"]),
         ("natural", "natural\nrephrasings", C["natural"]),
         ("adversarial", "adversarial\nrephrasings", C["adversarial"])]

fig, ax = plt.subplots(figsize=(4.4, 3.3))
ticks, labels = [], []
for i, (key, label, col) in enumerate(TIERS):
    x = i * 1.0
    iv = IN[key]["rate"] * 100
    ov = OUT[key]["rate"] * 100
    pool = (10 * iv + 18 * ov) / 28
    ax.bar(x - 0.20, iv, 0.34, color=col, alpha=0.40, zorder=1, edgecolor="none")
    ax.bar(x + 0.20, ov, 0.34, color=col, alpha=0.40, zorder=1, edgecolor="none")
    ax.text(x - 0.32, iv + 1.6, f"{iv:.0f}", ha="center", fontsize=5.6,
            color="#4a5568", zorder=3)
    ax.text(x + 0.32, ov + 1.6, f"{ov:.0f}", ha="center", fontsize=5.6,
            color="#4a5568", zorder=3)
    ax.bar(x, pool, 0.44, color=col, zorder=2, edgecolor="none")
    ax.text(x, pool - 2.6, f"{pool:.1f}", ha="center", va="top", fontsize=7.8,
            fontweight="bold", color="white", zorder=4,
            bbox=dict(boxstyle="square,pad=0.10", fc=col, ec="none"))
    ticks.append(x)
    labels.append(label)
ax.set_xticks(ticks)
ax.set_xticklabels(labels, fontsize=7.8)
ax.set_ylabel("rollout success (%)", fontsize=8.5)
ax.set_ylim(0, 126)
ax.set_yticks(range(0, 101, 25))
ax.spines[["top", "right"]].set_visible(False)
ax.grid(alpha=0.2, axis="y", zorder=0)
handles = [plt.Rectangle((0, 0), 1, 1, fc="#7A8698"),
           plt.Rectangle((0, 0), 1, 1, fc="#7A8698", alpha=0.40)]
ax.legend(handles, ["pooled (28 tasks)",
                    "flanks: in-finetune (left, 10) / out-of-finetune (right, 18)"],
          fontsize=5.9, loc="upper right", frameon=False)
fig.tight_layout()
out = R / "results/charts/fourtier_flanked.png"
fig.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0.22)
print("chart ->", out)
