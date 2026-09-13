#!/usr/bin/env python3
"""Compact task-scene grids for the paper appendix (single-column figures),
replacing the two full-page pairverify PDF scene pages.

  results/charts/scenes_bridge_grid.png  16 SIMPLER Bridge tasks, 4x4
  results/charts/scenes_libero_grid.png  21 LIBERO tasks, 5 cols

Frames: results/charts/pairverify_frames (Bridge, 640x480) and
results/analysis/pairverify_libero/frames (LIBERO, 256x256) — the exact
scenes staged for the minimal-pair docs."""
import glob
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

R = pathlib.Path(__file__).resolve().parents[1]

BRIDGE = [
    "carrot_on_keyboard", "carrot_on_ramekin", "carrot_on_wheel",
    "coke_can_on_keyboard", "coke_can_on_plate", "coke_can_on_ramekin",
    "coke_can_on_wheel", "cube_on_plate", "eggplant_on_keyboard",
    "eggplant_on_sponge", "nut_on_plate", "nut_on_wheel", "pepsi_on_plate",
    "put_eggplant_in_basket", "spoon_on_towel", "stack_cube",
]
SHIPPED = {"put_eggplant_in_basket", "spoon_on_towel", "stack_cube"}


def bridge_frame(t):
    stem = f"widowx_{t}" if t in SHIPPED else f"widowx_{t}_clean"
    return R / f"results/charts/pairverify_frames/{stem}.png"


def grid(items, ncols, thumb_ar, out, label_fs):
    """items: (image path, label). thumb_ar = height/width of the thumbs."""
    n = len(items)
    nrows = -(-n // ncols)
    cw = 4.4 / ncols
    fig_h = nrows * (cw * thumb_ar + 0.145) + 0.02
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.4, fig_h))
    axes = axes.ravel()
    for ax in axes[n:]:
        ax.axis("off")
    for ax, (fp, label) in zip(axes, items):
        ax.imshow(mpimg.imread(fp))
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_edgecolor("#cbd5e0"); s.set_linewidth(0.4)
        ax.set_xlabel(label, fontsize=label_fs, family="monospace",
                      color="#2d3748", labelpad=1.4)
    fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005,
                        wspace=0.06, hspace=0.30)
    fig.savefig(out, dpi=210, bbox_inches="tight", pad_inches=0.02)
    print("chart ->", out)


grid([(bridge_frame(t), t) for t in BRIDGE], 4, 0.75,
     R / "results/charts/scenes_bridge_grid.png", 4.6)

lib = sorted(glob.glob(str(R / "results/analysis/pairverify_libero/frames/*.png")),
             key=lambda f: (pathlib.Path(f).stem.split("__")[0],
                            int(pathlib.Path(f).stem.split("__")[1])))
items = [(f, pathlib.Path(f).stem.replace("__", " / "))
         for f in lib]
grid(items, 5, 1.0, R / "results/charts/scenes_libero_grid.png", 3.9)
