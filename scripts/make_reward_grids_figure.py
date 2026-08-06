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

# iso-budget level sets: dotted curves of constant total queries F x C
def axis_pos(vals, x):
    """map a data value to fractional index position on a categorical axis"""
    for i in range(len(vals) - 1):
        if vals[i] <= x <= vals[i + 1]:
            return i + (x - vals[i]) / (vals[i + 1] - vals[i])
    return None

for ax, Fs, Cs in [(axes[0], CF, CC), (axes[1], FF, FC)]:
    for bi, B in enumerate([8, 20, 40, 80]):
        pts = []
        for f in np.linspace(Fs[0], Fs[-1], 200):
            c = B / f
            if Cs[0] <= c <= Cs[-1]:
                xi, yi = axis_pos(Cs, c), axis_pos(Fs, f)
                if xi is not None and yi is not None:
                    pts.append((xi, yi))
        if len(pts) > 5:
            xs, ys = zip(*pts)
            ax.plot(xs, ys, ls=":", color="white", lw=1.5, alpha=0.9, zorder=3)
            ax.annotate(f"$F{{\\times}}C{{=}}{B}$", (xs[-1], ys[-1]),
                        textcoords="offset points", xytext=(-6, 10 + 12 * (bi % 2)), fontsize=8,
                        ha="right", color="white", zorder=4)

cb = fig.colorbar(im, ax=axes, shrink=0.9, pad=0.015)
cb.set_label("agreement with simulator (%)", fontsize=10)
fig.savefig("results/charts/reward_grids_gripper.png", dpi=170, bbox_inches="tight", pad_inches=0.15)
print("chart -> results/charts/reward_grids_gripper.png")
