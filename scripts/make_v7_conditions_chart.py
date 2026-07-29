#!/usr/bin/env python3
"""v7a checkpoints on the full val-8, both input conditions, TAGGED (fixed) evals
solid vs pre-fix untagged evals ghosted. Tagged rows merge from all roll pods'
files (v7_dev8_tagged*.jsonl / v7_dev8adv_tagged*.jsonl); references from the
rolled val8 leg (no policy generation -> unaffected by the tag bug)."""
import glob
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def rows(pattern):
    by_step = {}
    for f in sorted(glob.glob(pattern)):
        for l in open(f):
            r = json.loads(l)
            # prefer the deepest (highest-n) measurement per step
            if r["step"] not in by_step or r.get("n", 0) > by_step[r["step"]].get("n", 0):
                by_step[r["step"]] = r
    return sorted(by_step.values(), key=lambda r: r["step"])


pol_t = rows("results/analysis/v7_dev8_tagged*.jsonl")
rep_t = rows("results/analysis/v7_dev8adv_tagged*.jsonl")
pol_u = rows("results/analysis/v7_dev8_backfill.jsonl")
rep_u = rows("results/analysis/v7_dev8adv_backfill.jsonl")

fig, ax = plt.subplots(figsize=(10.5, 5.6))
tr = ax.get_yaxis_transform()
ax.axhline(54.7, ls="--", color="#822727", lw=1.4)
ax.text(0.01, 55.0, "Oracle 54.7", color="#822727", fontsize=8.5, transform=tr)
ax.axhline(40.6, ls="--", color="#48bb78", lw=1.4)
ax.text(0.01, 40.9, "Original phrasing 40.6", color="#2f855a", fontsize=8.5, transform=tr)
ax.axhline(34.5, ls="--", color="#a0aec0", lw=1.4)
ax.text(0.01, 34.8, "Adversarial phrasing 34.5", color="#718096", fontsize=8.5, transform=tr)

if pol_u:
    ax.plot([r["step"] for r in pol_u], [r["pooled"] for r in pol_u], marker="o", ms=6,
            color="#6b46c1", alpha=0.25, lw=1.2, ls=":", label="polish, UNTAGGED eval (pre-fix)")
if rep_u:
    ax.plot([r["step"] for r in rep_u], [r["pooled"] for r in rep_u], marker="s", ms=6,
            color="#dd6b20", alpha=0.25, lw=1.2, ls=":", label="repair, UNTAGGED eval (pre-fix)")
if pol_t:
    ax.plot([r["step"] for r in pol_t], [r["pooled"] for r in pol_t], marker="D", ms=9,
            color="#6b46c1", lw=2.2, label="v7a rewriting ORIGINALS — TAGGED eval (fixed)")
    dp = [r for r in pol_t if r.get("n", 192) >= 384]
    ax.plot([r["step"] for r in dp], [r["pooled"] for r in dp], marker="D", ms=12, ls="none",
            markerfacecolor="#6b46c1", markeredgecolor="black", markeredgewidth=2, zorder=5)
if rep_t:
    ax.plot([r["step"] for r in rep_t], [r["pooled"] for r in rep_t], marker="s", ms=9,
            color="#e53e3e", lw=2.2, label="v7a rewriting ADVERSARIAL — TAGGED eval (fixed)")
    dr = [r for r in rep_t if r.get("n", 192) >= 384]
    ax.plot([r["step"] for r in dr], [r["pooled"] for r in dr], marker="s", ms=12, ls="none",
            markerfacecolor="#e53e3e", markeredgecolor="black", markeredgewidth=2, zorder=5)
ax.plot([], [], marker="o", ls="none", markerfacecolor="#a0aec0", markeredgecolor="black",
        markeredgewidth=2, ms=10, label="black edge = DEEPENED eval (n=384, ±2.5)")

ax.set_xlabel("v7a checkpoint step (run trained to 340)")
ax.set_ylabel("val-8 rollout success % (8 tasks × 24 layouts, greedy)")
ax.set_ylim(30, 58)
ax.set_xlim(-8, 348)
ax.set_title("v7a on the full val-8 — both input conditions, tagged (fixed) vs pre-fix evals")
ax.legend(loc="lower right", fontsize=8)
ax.grid(alpha=0.25)
fig.text(0.01, 0.01, "TAGGED = inputs carry the training-time tier tag ([input: original wording] / [input: adversarially reworded]) — "
         "the policy never saw untagged inputs in training, so ghosted points are out-of-distribution measurements. "
         "Full tagged matrix (both conditions, 0-340) landing overnight. n=192/pt (±3.6). References: rolled val8 leg, n=288/task/arm.",
         fontsize=7, color="#4a5568")
fig.tight_layout(rect=[0, 0.05, 1, 1])
fig.savefig("results/charts/v7_conditions.png", dpi=150, bbox_inches="tight", pad_inches=0.25)
n = len(pol_t) + len(rep_t)
print(f"chart -> results/charts/v7_conditions.png ({n} tagged points)")
