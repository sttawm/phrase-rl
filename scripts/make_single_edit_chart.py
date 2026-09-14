#!/usr/bin/env python3
"""results/charts/single_edit_pairs.png — every single-edit pair as a point.

Left: the 195 canonical-anchored pairs, signed delta (edit minus canonical) by
edit class. Right: all 426 pairs, |delta| by class (sign is arbitrary for
ladder cross-pairs). Filled = significant (|delta| >= 18 pp and p < 0.05),
hollow = null; colour = task finetune status. Dashed lines mark the 18 pp
resolution floor at n=50. Data: results/analysis/pi05_bank/single_edit_pairs_full.csv.
"""
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

R = pathlib.Path(__file__).resolve().parents[1]
P = pd.read_csv(R / "results/analysis/pi05_bank/single_edit_pairs_full.csv")
order = P.edit_class.value_counts().index.tolist()          # most pairs at top
ypos = {c: i for i, c in enumerate(order[::-1])}
COL = {"in": "#1f5fa8", "out": "#d9822b"}
rng = np.random.default_rng(3)

fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True,
                         gridspec_kw={"width_ratios": [1.15, 1]})
for ax, sub, xcol, title in [
        (axes[0], P[P.a_is_canonical], "delta",
         f"canonical-anchored pairs (n={int(P.a_is_canonical.sum())}): edit minus canonical, pp"),
        (axes[1], P, "absdelta", f"all pairs (n={len(P)}): |delta|, pp")]:
    sub = sub.assign(absdelta=sub.delta.abs())
    for fin in ("in", "out"):
        for sig in (False, True):
            d = sub[(sub.finetune == fin) & (sub.significant == sig)]
            y = d.edit_class.map(ypos) + rng.uniform(-0.28, 0.28, len(d))
            ax.scatter(d[xcol], y, s=26, color=COL[fin], alpha=0.85 if sig else 0.55,
                       facecolors=COL[fin] if sig else "none", linewidths=1.1, zorder=3,
                       label=f"{'in' if fin == 'in' else 'out-of'}-finetune, {'significant' if sig else 'null'}")
    for x in ((-18, 18) if xcol == "delta" else (18,)):
        ax.axvline(x, color="0.4", ls="--", lw=0.9, zorder=1)
    if xcol == "delta":
        ax.axvline(0, color="0.75", lw=0.8, zorder=1)
    ax.set_title(title, fontsize=10.5)
    ax.grid(axis="x", color="0.9", zorder=0)
    ax.set_xlabel("percentage points (n=50 per phrase, same 50 inits)")
# per-class counts as tick labels
counts = P.groupby("edit_class").agg(n=("delta", "size"), s=("significant", "sum"))
axes[0].set_yticks(list(ypos.values()))
axes[0].set_yticklabels([f"{c}  ({int(counts.loc[c, 'n'])} pairs, {int(counts.loc[c, 's'])} sig.)" for c in ypos])
axes[0].legend(fontsize=8, loc="lower left", frameon=False)
fig.suptitle("Single-edit phrase pairs, frozen $\\pi_{0.5}$ on LIBERO — "
             f"{int(P.significant.sum())} of {len(P)} pairs move by ≥18 pp; filled = significant, hollow = null",
             fontsize=11)
fig.tight_layout()
out = R / "results/charts/single_edit_pairs.png"
out.parent.mkdir(exist_ok=True)
fig.savefig(out, dpi=170)
print("->", out)
