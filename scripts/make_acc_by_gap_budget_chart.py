#!/usr/bin/env python3
"""Ranking accuracy vs gap, swept over sampling budget F x C (F=4 fixed, C swept).
Rows = weighting (raw / task-balanced), columns = budget."""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = json.load(open("results/analysis/reward_acc_by_gap_budget.json"))
BANDS = ["0-5 (ties)", "5-10", "10-15", "15-25", "25+"]
XT = ["0-5", "5-10", "10-15", "15-25", "25+"]
BUDGETS = ["FxC=4", "FxC=8", "FxC=16", "FxC=32", "FxC=64"]
BLENDS = [("ens100", "100% ensemble", "#7c3aed"), ("z75g25", "75/25", "#2b6cb0"),
          ("z50g50", "50/50", "#0d9488"), ("c4b", "25/75  (v11)", "#dd6b20"),
          ("grip", "100% gripper", "#c53030")]

fig, axes = plt.subplots(2, len(BUDGETS), figsize=(17.5, 6.6), sharey=True, sharex=True)
for ri, mode in enumerate(("raw", "balanced")):
    for ci, bud in enumerate(BUDGETS):
        ax = axes[ri, ci]
        for key, lbl, col in BLENDS:
            ys = [d[bud][mode][b][key] for b in BANDS]
            ax.plot(range(len(BANDS)), ys, "o-", color=col, ms=4.5,
                    lw=2.2 if key in ("c4b", "grip") else 1.4,
                    alpha=1.0 if key in ("c4b", "grip", "ens100") else 0.7, label=lbl)
        ax.axhline(50, ls=":", color="#a0aec0", lw=1)
        ax.grid(alpha=0.22)
        if ri == 0:
            ax.set_title(f"{bud.replace('FxC=', 'budget  F×C = ')}"
                         f"\n(F=4, C={int(bud.split('=')[1]) // 4})", fontsize=9.5)
        if ci == 0:
            ax.set_ylabel(f"{'pooled over pairs' if mode == 'raw' else 'tasks weighted equally'}"
                          "\ncorrect ranking (%)", fontsize=9)
        if ri == 1:
            ax.set_xticks(range(len(BANDS)))
            ax.set_xticklabels(XT, fontsize=8, rotation=30)
            ax.set_xlabel("gap (pp)", fontsize=9)
axes[0, 0].legend(fontsize=7.5, loc="upper left")
fig.suptitle("Reward ranking accuracy vs phrase-pair separation, across sampling budgets — "
             "gripper-family designs overtake the ensemble as budget grows (task-balanced row)",
             fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.955])
fig.savefig("results/charts/reward_acc_by_gap_budget.png", dpi=150, bbox_inches="tight",
            pad_inches=0.2)
print("chart -> results/charts/reward_acc_by_gap_budget.png")
