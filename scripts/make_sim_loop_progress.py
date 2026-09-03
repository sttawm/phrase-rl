#!/usr/bin/env python3
"""Per-iteration rollout success of the r1_sim rules loop, per applier.

Left: training-48 rollout success by iteration (the loop's own signal; the sim
loop has no val split -- stopping was governed by max_iters, gemini extended to
6 by user directive). Right: the holdout-143 verdict for each applier's best
book vs its scaffold and the unrephrased bases.
Output: results/charts/sim_loop_progress.png
"""
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = pathlib.Path(__file__).resolve().parents[1]
RUN = REPO / "results/rules_runs/r1_sim"
BASELINE = 22.9

COLORS = {"claude": "#b45309", "gemini": "#2b6cb0", "qwen": "#0d9488"}
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), width_ratios=[3, 2])

for m, c in COLORS.items():
    xs, ys = [], []
    for it in range(8):
        p = RUN / f"pass_{m}" / f"iter_{it:02d}" / "scores.json"
        if not p.exists():
            continue
        xs.append(it)
        ys.append(100 * json.loads(p.read_text())["train"])
    ax.plot(xs, ys, "-o", color=c, label=f"{m} (best {max(ys):.1f})", lw=2, ms=5)
    best_i = xs[ys.index(max(ys))]
    ax.plot([best_i], [max(ys)], "o", color=c, ms=11, mfc="none", mew=2)
ax.axhline(BASELINE, color="#888", ls="--", lw=1)
ax.text(0.05, BASELINE + 0.4, "unrephrased baseline 22.9", color="#666", fontsize=8)
ax.set_xlabel("iteration (0 = seeded r1 book measured by rollout)")
ax.set_ylabel("training-48 rollout success (%)")
ax.set_title("sim rules loop: train score per iteration\n(circled = best book; no val split, cap = max_iters)")
ax.legend(fontsize=8, loc="upper left")
ax.grid(alpha=0.25)

hold = json.loads((REPO / "results/analysis/holdout_finals.json").read_text())
arms = [("bases", hold["bases"]["mean"], "#888"),
        ("claude\nscaffold", hold["claude_sc"]["mean"], "#e0b48a"),
        ("claude\nbook", hold["claude_book"]["mean"], COLORS["claude"]),
        ("gemini\nscaffold", hold["gemini_sc"]["mean"], "#9bbbdd"),
        ("gemini\n+claude bk", 43.1, COLORS["gemini"]),
        ("qwen\nscaffold", hold["qwen_sc"]["mean"], "#8fd0c8"),
        ("qwen\nbook", hold["qwen_book"]["mean"], COLORS["qwen"])]
xs = range(len(arms))
ax2.bar(xs, [a[1] for a in arms], color=[a[2] for a in arms])
for x, (_, v, _) in zip(xs, arms):
    ax2.text(x, v + 0.5, f"{v:.1f}", ha="center", fontsize=8)
ax2.set_xticks(list(xs))
ax2.set_xticklabels([a[0] for a in arms], fontsize=7)
ax2.set_title("holdout-143 verdict (n=3/phrase)")
ax2.set_ylabel("success (%)")
ax2.grid(alpha=0.25, axis="y")

fig.tight_layout()
out = REPO / "results/charts/sim_loop_progress.png"
fig.savefig(out, dpi=150)
print("wrote", out)
