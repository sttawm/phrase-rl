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
F_GRID = [1, 3]
C_GRID = [1, 2, 4, 8, 10]
BLENDS = [("ens100", "100% ensemble"), ("c4b", "C4b (25% ens / 75% grip)"), ("grip", "100% grip")]

fig, axes = plt.subplots(1, 3, figsize=(16.5, 3.6))
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
fig.text(0.01, 0.01, "Calibration bucket (0.5-5pp, honest benchmark ~55-60%): ens ~53-60 (near-optimal), grip ~50-53 (under-resolves). "
         "Max measured cell F3C10; extension to F5/C16 (v7e's cell) pending top-up + native half.",
         fontsize=7.5, color="#4a5568")
fig.tight_layout(rect=[0, 0.05, 1, 0.93])
fig.savefig(f"results/charts/fine_grid_{HALF}.png", dpi=150, bbox_inches="tight", pad_inches=0.25)
print(f"chart -> results/charts/fine_grid_{HALF}.png")
