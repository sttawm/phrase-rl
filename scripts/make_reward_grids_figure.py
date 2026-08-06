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
    (axes[0], COARSE, CF, CC, "coarse spreads (clearly-different pairs, native exam)"),
    (axes[1], FINE, FF, FC, "fine spreads (5–10 pp true gaps, stratified exam)"),
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

# operating points, coarse panel: screen (F1,C4) -> finals (F4,C10) -> triage (F4,C20)
ax = axes[0]
pts = {"screen": (2, 0), "finals": (4, 2), "triage": (6, 2)}
xs, ys = zip(*[pts[k] for k in ("screen", "finals", "triage")])
ax.plot(xs, ys, ls=":", color="white", lw=1.8, zorder=3)
for name, (x, y) in pts.items():
    ax.scatter([x], [y], s=620, facecolor="none", edgecolor="white", lw=2.0, zorder=4)
    ax.annotate(name, (x, y), textcoords="offset points", xytext=(0, -26),
                ha="center", fontsize=9, color="white", fontweight="bold", zorder=4)

# RL point, fine panel: F=4 row, C=5 (between the C=4 and C=8 columns)
ax = axes[1]
x_rl = 2 + (5 - 4) / (8 - 4)   # index-space position of C=5
ax.plot([x_rl, x_rl], [-0.45, 2], ls=":", color="white", lw=1.8, zorder=3)
ax.scatter([x_rl], [2], s=620, facecolor="none", edgecolor="white", lw=2.0, zorder=4)
ax.annotate("RL ($C{=}5$)", (x_rl, 2), textcoords="offset points", xytext=(30, -26),
            ha="center", fontsize=9, color="white", fontweight="bold", zorder=4)

cb = fig.colorbar(im, ax=axes, shrink=0.9, pad=0.015)
cb.set_label("agreement with simulator (%)", fontsize=10)
fig.suptitle("Gripper-error proxy vs. simulated success: pairwise agreement by sampling budget",
             fontsize=11.5)
fig.savefig("results/charts/reward_grids_gripper.png", dpi=170, bbox_inches="tight", pad_inches=0.15)
print("chart -> results/charts/reward_grids_gripper.png")
