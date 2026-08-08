#!/usr/bin/env python3
"""v11 RL progress on the nat24 probe: 192 DISTINCT natural rephrases per cell
(one per layout across the val-8 tasks), so every checkpoint is measured on the
same rollout budget but never the same 8 phrases twice.

The question is whether the curve leaves the noise band around its own step-0
baseline. Drawn with +/-2 s.e. of a 192-episode binomial, because a cell that
moves less than that has not moved.

  python3 scripts/make_v11_nat24_chart.py
"""
import glob
import json
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

rows = []
for f in sorted(glob.glob("results/analysis/v11cells/nat24_0*.json")):
    d = json.load(open(f))
    rows.append((int(re.search(r"nat24_(\d+)", f).group(1)), d["pooled"], d["n"]))
rows.sort()
steps = np.array([r[0] for r in rows])
vals = np.array([r[1] for r in rows])
n = rows[0][2]
base = vals[0]
se = np.sqrt((base / 100) * (1 - base / 100) / n) * 100

fig, ax = plt.subplots(figsize=(9.4, 4.9))
ax.axhspan(base - 2 * se, base + 2 * se, color="#e2e8f0", zorder=0,
           label=f"±2 s.e. of a {n}-episode cell (±{2*se:.1f}pp)")
ax.axhline(base, color="#718096", ls="--", lw=1.4)
ax.text(steps.max(), base + 0.25, f"step-0 baseline {base:.2f}", ha="right",
        fontsize=8.5, color="#4a5568")
ax.plot(steps, vals, "-o", color="#2b6cb0", lw=1.9, ms=5.5, zorder=3,
        label="v11 (prompt B+, 25/50/25 mix, c4b reward)")
for s, v in zip(steps, vals):
    ax.annotate(f"{v:.1f}", (s, v), textcoords="offset points", xytext=(0, 7),
                ha="center", fontsize=7.4, color="#2b6cb0")
ax.set_xlabel("RL step", fontsize=10)
ax.set_ylabel("val success on 192 distinct naturals (%)", fontsize=10)
ax.set_ylim(min(vals.min(), base - 3 * se) - 2, max(vals.max(), base + 3 * se) + 3)
ax.grid(alpha=0.25)
ax.legend(fontsize=8.5, loc="lower left")
drift = vals[-5:].mean() - vals[:5].mean()
ax.set_title("v11 after 100 steps: no movement outside the noise band\n"
             f"last-5 mean minus first-5 mean = {drift:+.2f}pp; "
             f"every cell within {np.abs(vals-base).max():.1f}pp of baseline, "
             f"1 s.e. is {se:.1f}pp", fontsize=10.5)
fig.tight_layout()
fig.savefig("results/charts/v11_nat24.png", dpi=155)
print("chart -> results/charts/v11_nat24.png")
print(f"steps {steps.min()}-{steps.max()}, drift {drift:+.2f}pp, 2se {2*se:.2f}pp")
