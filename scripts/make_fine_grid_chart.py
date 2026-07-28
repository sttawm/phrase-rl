#!/usr/bin/env python3
"""Chart the fine-discrimination grid (OOV half so far): per-reward heatmaps of
sign accuracy on the fine bucket (5-10pp gaps, ordering-confidence >= 0.8)."""
import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HALF = sys.argv[1] if len(sys.argv) > 1 else "oov"
g = json.load(open(f"results/analysis/fine_grid_{HALF}.json"))

# read the grid axes from the json itself so chart tracks whatever was measured
cells = list(g["grid"])
F_GRID = sorted({int(k.split("_")[0][1:]) for k in cells})
C_GRID = sorted({int(k.split("_C")[1]) for k in cells})
BLENDS = [("ens100", "100% ensemble"), ("z75g25", "75% ens / 25% grip"), ("z50g50", "50% ens / 50% grip"),
          ("c4b", "C4b (25% ens / 75% grip)"), ("grip", "100% grip")]

fig, axes = plt.subplots(1, 5, figsize=(26, 3.7))
for ax, (bl, name) in zip(axes, BLENDS):
    A = np.array([[g["grid"][f"F{F}_C{C}"][bl]["fine_5-10"] for C in C_GRID] for F in F_GRID])
    im = ax.imshow(A, origin="lower", cmap="viridis", vmin=50, vmax=85, aspect="auto")
    for fi in range(len(F_GRID)):
        for ci in range(len(C_GRID)):
            ax.text(ci, fi, f"{A[fi, ci]:.0f}", ha="center", va="center",
                    color="white", fontsize=11, fontweight="bold")
    ax.set_xticks(range(len(C_GRID))); ax.set_xticklabels(C_GRID)
    ax.set_yticks(range(len(F_GRID))); ax.set_yticklabels(F_GRID)
    ax.set_xlabel("contexts C"); ax.set_ylabel("frames F / episode")
    ax.set_title(f"{name}")
    fig.colorbar(im, ax=ax, label="sign acc %")
fig.suptitle(f"Fine-pair sign accuracy ({HALF} half, sim-grounded): 5-10pp gaps, conf ≥ 0.8 — "
             f"{g['buckets']['fine_5-10']} pairs, B={g['B']}", fontsize=11)
fig.text(0.01, 0.01, "Full-coverage rescore (20,020 rows): every episode carries 6-8 scored frames, so all F rows use the same "
         "63-episode pools (keyboard 10 / wheel 11 / cokeplate 22 / ramekin 20). C=16 still caps at available eps for keyboard/wheel. "
         "Calibration bucket honest benchmark ~55-60%. Native half pending.",
         fontsize=7, color="#4a5568")
fig.tight_layout(rect=[0, 0.05, 1, 0.93])
fig.savefig(f"results/charts/fine_grid_{HALF}.png", dpi=150, bbox_inches="tight", pad_inches=0.25)
print(f"chart -> results/charts/fine_grid_{HALF}.png")
