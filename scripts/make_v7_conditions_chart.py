#!/usr/bin/env python3
"""v7 checkpoints on the full val-8, both input conditions (polish = rewriting
originals; repair = rewriting adversarial), against the rolled reference lines."""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

pol = sorted((json.loads(l) for l in open("results/analysis/v7_dev8_backfill.jsonl")), key=lambda r: r["step"])
try:
    rep = sorted((json.loads(l) for l in open("results/analysis/v7_dev8adv_backfill.jsonl")), key=lambda r: r["step"])
except FileNotFoundError:
    rep = []

fig, ax = plt.subplots(figsize=(9.5, 5.4))
ax.axhline(54.7, ls="--", color="#822727", lw=1.4)
ax.text(122, 55.0, "Oracle 54.7", color="#822727", fontsize=8.5)
ax.axhline(40.6, ls="--", color="#48bb78", lw=1.4)
ax.text(122, 40.9, "Original phrasing 40.6", color="#2f855a", fontsize=8.5)
ax.axhline(34.5, ls="--", color="#a0aec0", lw=1.4)
ax.text(122, 34.8, "Adversarial phrasing 34.5", color="#718096", fontsize=8.5)
ax.plot([r["step"] for r in pol], [r["pooled"] for r in pol], marker="o", ms=8,
        color="#6b46c1", lw=2, label="v7a rewriting ORIGINALS (polish)")
if rep:
    ax.plot([r["step"] for r in rep], [r["pooled"] for r in rep], marker="s", ms=9,
            color="#dd6b20", lw=2, ls="-", label="v7a rewriting ADVERSARIAL (repair)")
ax.set_xlabel("v7a checkpoint step (run trained to step 340)")
ax.set_ylabel("val-8 rollout success % (8 tasks × 24 layouts, greedy)")
ax.set_ylim(30, 58)
ax.set_title("v7a checkpoints on the full val-8 — both input conditions vs references")
ax.legend(loc="lower right", fontsize=9)
ax.grid(alpha=0.25)
fig.text(0.01, 0.01, "v7a = original v7 run (β=0.15, F=16), trained to step 340. Late checkpoints (280-340) + repair 260 rolling; "
         "v7b (β=0.05) and v7e (grip/C=16) forks not yet val-8-evaluated. n=192/pt (±3.6). References: rolled val8 leg, n=288/task/arm.", fontsize=7, color="#4a5568")
fig.tight_layout(rect=[0, 0.04, 1, 1])
fig.savefig("results/charts/v7_conditions.png", dpi=150, bbox_inches="tight", pad_inches=0.25)
print("chart -> results/charts/v7_conditions.png")
