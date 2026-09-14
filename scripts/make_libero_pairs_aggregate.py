#!/usr/bin/env python3
"""results/charts/pairs_aggregate_libero.png — LIBERO single-edit pairs
(single_edit_pairs_full.csv, 426 pairs / 68 tasks, n=50 per phrase)
aggregated by edit class, split in-finetune vs out-of-finetune.
significant = |delta| >= 18pp and two-proportion p < 0.05 (the file's flag).
No in-image title (paper style). Caveats live in prose: adaptive probing,
goal/7 ladder cross-pairs dominate the in-finetune noun class."""
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

R = pathlib.Path(__file__).resolve().parents[1]
d = pd.read_csv(R / "results/analysis/pi05_bank/single_edit_pairs_full.csv")
order = d.groupby("edit_class").size().sort_values(ascending=False).index.tolist()
C_SIG, C_NULL, INK = "#3F6B52", "#D9DEE6", "#2d3748"

fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.0), sharey=True)
for ax, ft, name in [(axes[0], "in", "in-finetune"), (axes[1], "out", "out-of-finetune")]:
    sub = d[d.finetune == ft]
    agg = sub.groupby("edit_class").agg(sig=("significant", "sum"),
                                        tested=("significant", "size")).reindex(order).fillna(0)
    agg["null"] = agg.tested - agg.sig
    y = range(len(agg))[::-1]
    ax.barh(y, agg.sig, 0.62, color=C_SIG, edgecolor="none")
    ax.barh(y, agg["null"], 0.62, left=agg.sig, color=C_NULL,
            edgecolor="#6E7B8B", lw=0.6)
    for yi, (_, r) in zip(y, agg.iterrows()):
        if r.sig >= 3:
            ax.text(r.sig / 2, yi, str(int(r.sig)), ha="center", va="center",
                    fontsize=7.5, fontweight="bold", color="white")
        if r["null"] >= 5:
            ax.text(r.sig + r["null"] / 2, yi, str(int(r["null"])),
                    ha="center", va="center", fontsize=7.5, color=INK)
        if r.tested:
            ax.text(r.tested + 3, yi, f"{int(r.sig)}/{int(r.tested)}",
                    va="center", fontsize=6.8, color="#4a5568")
    ax.set_yticks(list(y))
    ax.set_yticklabels(agg.index, fontsize=8.5)
    ax.set_xlabel(name, fontsize=9)
    ax.set_xlim(0, 235)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.15)
handles = [plt.Rectangle((0, 0), 1, 1, fc=C_SIG),
           plt.Rectangle((0, 0), 1, 1, fc=C_NULL, ec="#6E7B8B")]
fig.legend(handles, ["significant ($|\\Delta| \\geq 18$ pp, p < 0.05)", "null"],
           fontsize=8, ncol=2, loc="lower center", bbox_to_anchor=(0.5, -0.02),
           frameon=False)
fig.tight_layout(rect=(0, 0.05, 1, 1))
out = R / "results/charts/pairs_aggregate_libero.png"
fig.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0.12)
print("chart ->", out)
