#!/usr/bin/env python3
"""Paper figure: gripper-channel proxy agreement grids (F x C), coarse and
fine spreads, with operating points marked and dotted connectors."""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

fc = json.load(open("results/analysis/fc_grid.json"))
COARSE = np.array(fc["acc_grip"])          # F=[1,2,4] x C=[1,2,4,8,10,16,20]
CF, CC = [1, 2, 4], [1, 2, 4, 8, 10, 16, 20]

fg = json.load(open("results/analysis/fine_grid_all.json"))["grid"]
FF, FC = [1, 3, 4, 5], [1, 2, 4, 8, 10, 16, 20]
FINE = np.array([[fg[f"F{f}_C{c}"]["grip"]["fine_5-10"] for c in FC] for f in FF])

fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.2))

for ax, A, Fs, Cs, title in [
    (axes[0], COARSE, CF, CC, "coarse spreads"),
    (axes[1], FINE, FF, FC, "fine spreads"),
]:
    im = ax.imshow(A, origin="lower", cmap="viridis", vmin=50, vmax=93, aspect="auto")
    for fi in range(len(Fs)):
        for ci in range(len(Cs)):
            ax.text(ci, fi, f"{A[fi, ci]:.0f}", ha="center", va="center",
                    color="white", fontsize=10.5, fontweight="bold")
    ax.set_xticks(range(len(Cs))); ax.set_xticklabels(Cs, fontsize=9)
    ax.set_yticks(range(len(Fs))); ax.set_yticklabels(Fs, fontsize=9)
    ax.set_xlabel("episodes $C$", fontsize=10)
    ax.set_ylabel("frames per episode $F$", fontsize=10)
    ax.set_title(title, fontsize=10.5)

# level-set contours: the grids read as a scalar field of agreement
for ax, A, Fs, Cs in [(axes[0], COARSE, CF, CC), (axes[1], FINE, FF, FC)]:
    X, Y = np.meshgrid(range(len(Cs)), range(len(Fs)))
    cs = ax.contour(X, Y, A, levels=[65, 75, 85], colors="white",
                    linewidths=1.3, linestyles=":")
    ax.clabel(cs, fmt="%d", fontsize=8.5, colors="white")

cb = fig.colorbar(im, ax=axes, shrink=0.9, pad=0.015)
cb.set_label("agreement with simulator (%)", fontsize=10)
fig.savefig("results/charts/reward_grids_gripper.png", dpi=170, bbox_inches="tight", pad_inches=0.15)
print("chart -> results/charts/reward_grids_gripper.png")
