#!/usr/bin/env python3
"""Ranking accuracy vs ground-truth gap, all five reward designs, raw vs
task-balanced. The two panels disagree about which design wins — that IS the
result: raw is dominated by two phrase-rich tasks, one of which (coke_can_on_
plate) inverts the gripper channel."""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = json.load(open("results/analysis/reward_acc_by_gap.json"))
BANDS = ["0-5 (ties)", "5-10", "10-15", "15-25", "25+"]
BLENDS = [("ens100", "100% ensemble", "#7c3aed"), ("z75g25", "75% ens / 25% grip", "#2b6cb0"),
          ("z50g50", "50 / 50", "#0d9488"), ("c4b", "25% ens / 75% grip  (v11)", "#dd6b20"),
          ("grip", "100% gripper", "#c53030")]

fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.4), sharey=True)
for ax, mode, sub in [(axes[0], "raw", "pooled over all pairs\n(two coke-can tasks = 51% of pairs)"),
                      (axes[1], "balanced", "per-task accuracy, tasks weighted equally")]:
    for key, lbl, col in BLENDS:
        ys = [d[f"{mode}|{b}"][key] for b in BANDS]
        ax.plot(range(len(BANDS)), ys, "o-", color=col, lw=2.2 if key in ("c4b", "grip") else 1.5,
                ms=6, label=lbl, alpha=1.0 if key in ("c4b", "grip", "ens100") else 0.75)
    ax.axhline(50, ls=":", color="#a0aec0", lw=1)
    ax.set_xticks(range(len(BANDS)))
    ax.set_xticklabels(BANDS, fontsize=9)
    ax.set_xlabel("ground-truth success gap between the two phrases (pp)", fontsize=9.5)
    ax.set_title(f"{mode.upper()} — {sub}", fontsize=10)
    ax.grid(alpha=0.25)
axes[0].set_ylabel("correct ranking (%)", fontsize=10)
axes[0].legend(fontsize=8, loc="upper left")
fig.suptitle("Which reward ranks a pair correctly, as a function of how different the phrases really are",
             fontsize=11)
fig.tight_layout()
fig.savefig("results/charts/reward_acc_by_gap.png", dpi=155, bbox_inches="tight", pad_inches=0.2)
print("chart -> results/charts/reward_acc_by_gap.png")
