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

STRATA = [("far_15+", "coarse pairs (>=15pp gaps)"), ("fine_5-10", "fine pairs (5-10pp gaps)")]
fig, axes = plt.subplots(2, 5, figsize=(24, 7.6))
for ri, (bucket, bname) in enumerate(STRATA):
    for ax, (bl, name) in zip(axes[ri], BLENDS):
        A = np.array([[g["grid"][f"F{F}_C{C}"][bl].get(bucket, np.nan) for C in C_GRID] for F in F_GRID])
        im = ax.imshow(A, origin="lower", cmap="viridis", vmin=50, vmax=92, aspect="auto")
        for fi in range(len(F_GRID)):
            for ci in range(len(C_GRID)):
                if not np.isnan(A[fi, ci]):
                    ax.text(ci, fi, f"{A[fi, ci]:.0f}", ha="center", va="center",
                            color="white", fontsize=9.5, fontweight="bold")
        ax.set_xticks(range(len(C_GRID))); ax.set_xticklabels(C_GRID, fontsize=8)
        ax.set_yticks(range(len(F_GRID))); ax.set_yticklabels(F_GRID, fontsize=8)
        if ri == 1: ax.set_xlabel("contexts C", fontsize=9)
        if ax is axes[ri][0]: ax.set_ylabel(f"{bname}\nframes F / episode", fontsize=9)
        if ri == 0: ax.set_title(name, fontsize=10)
fig.suptitle(f"Reward-design pairwise sign accuracy ({HALF} half, sim-grounded), by sampling budget — "
             f"coarse n={g['buckets'].get('far_15+', '?')}, fine n={g['buckets']['fine_5-10']}, B={g['B']}", fontsize=12)
fig.text(0.01, 0.01, "Grip excels on fine discrimination at high budgets; the learned ensemble leads on coarse OOV pairs (grip is blind to gross OOV failures). "
         "C caps at available episodes per task. Ordering-confidence >= 0.8 buckets.",
         fontsize=7, color="#4a5568")
fig.tight_layout(rect=[0, 0.05, 0.945, 0.93])
cb = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.016, pad=0.012)
cb.set_label("agreement with simulator (%)", fontsize=10)
fig.savefig(f"results/charts/fine_grid_{HALF}.png", dpi=150, bbox_inches="tight", pad_inches=0.25)
print(f"chart -> results/charts/fine_grid_{HALF}.png")
