#!/usr/bin/env python3
"""Val-8 reference chart (ROLLED leg, n=288/task/arm): Original / Adversarial /
Oracle on the 8 val tasks, strata split, with the v7 best-checkpoint marker.
Reads results/sealed/val8_reference_x12.parquet + v7_dev8_backfill.jsonl.
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch

IV = ["widowx_carrot_on_plate", "widowx_spoon_on_towel", "widowx_stack_cube",
      "widowx_put_eggplant_in_basket"]

d = pd.read_parquet("results/sealed/val8_reference_x12.parquet")
pt = d.groupby(["arm", "task"]).success.mean().unstack(0) * 100


def strata(arm):
    s = pt[arm]
    iv = s[[t for t in s.index if t in IV]].mean()
    oo = s[[t for t in s.index if t not in IV]].mean()
    return s.mean(), iv, oo


rows = [("Oracle (search-best)", strata("val8_oracle"), "#822727"),
        ("Orig", strata("val8_orig"), "#48bb78"),
        ("Adv", strata("val8_adv"), "#a0aec0")]

try:
    bf = [json.loads(l) for l in open("results/analysis/v7_dev8_backfill.jsonl")]
    v7best = max(r["pooled"] for r in bf)
except Exception:
    v7best = None

fig, ax = plt.subplots(figsize=(9.5, 5.2))
y = [2, 1, 0]
h = 0.24
for yi, (label, (pool, iv, oo), color) in zip(y, rows):
    ax.barh(yi + h, pool, h, color=color)
    ax.barh(yi, iv, h, color=color, alpha=0.55)
    ax.barh(yi - h, oo, h, color=color, alpha=0.3, hatch="//", edgecolor=color, lw=0)
    ax.text(pool + 0.5, yi + h, f"{pool:.1f}", va="center", fontsize=11, fontweight="bold")
    ax.text(iv + 0.5, yi, f"{iv:.1f}", va="center", fontsize=8, color="#4a5568")
    ax.text(oo + 0.5, yi - h, f"{oo:.1f}", va="center", fontsize=8, color="#4a5568")
    print(f"{label:22s} pooled {pool:5.1f}  in-vocab {iv:5.1f}  out-of-vocab {oo:5.1f}")
ax.set_yticks(y)
ax.set_yticklabels([r[0] for r in rows], fontsize=12)
if v7best is not None:
    ax.axvline(v7best, ls="--", color="#9f7aea", lw=1.5)
    ax.text(v7best + 0.4, 2.3, f"v7 best ckpt ({v7best:.0f})", color="#9f7aea",
            fontsize=8.5, rotation=90, va="top")
ax.legend(handles=[Patch(color="#4a5568", label="pooled (8 tasks)"),
                   Patch(color="#4a5568", alpha=0.55, label="in-vocab (4 tasks)"),
                   Patch(facecolor="#4a5568", alpha=0.3, hatch="//", label="out-of-vocab (4 tasks)")],
          loc="lower right", fontsize=8)
ax.set_xlabel("success % — 8 val tasks × 24 layouts × 12 reps per arm")
ax.set_title("Val-8 reference frame: Original / Adversarial / Oracle (+ v7 RL marker)")
ax.grid(axis="x", alpha=0.25)
fig.tight_layout()
fig.savefig("results/charts/val8_reference.png", dpi=150, bbox_inches="tight", pad_inches=0.25)
print("chart -> results/charts/val8_reference.png")
