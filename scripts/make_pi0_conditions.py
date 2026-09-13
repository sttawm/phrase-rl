#!/usr/bin/env python3
"""pi0 (rephrase-augmented INTACT) input-condition chart, CURRENT sealed data:
oracle / canonical / VLM-generated naturals / adversarial on the 12 sealed
tasks x 24 pinned layouts. Pooled bar solid in front (white value, paper
idiom); in-distribution (5 tasks) / out-of-distribution (7) strata behind,
thinner and semi-transparent. No in-image title: explanation lives in the tex
caption. Palette shared with make_paper_fourtier.py (condition -> hue).

Provenance (computed 2026-09-13, base-weighted):
  oracle      48.2 / 50.3 / 46.8   results/sealed/oracle_confirmed_x12.parquet (24x12)
  canonical   36.1 / 48.3 / 27.3   anchors_x12 originals (24x12)
  naturals    26.0 / 38.1 / 17.0   r1_cells.json noreph|nat (186 phrases, 24x1)
  adversarial 24.5 / 25.2 / 23.9   a29pass legs + anchors passthrough (72 attacks)
"""
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = pathlib.Path(__file__).resolve().parents[1]
nr = json.loads((R / "results/analysis/r1_cells.json").read_text())["noreph|nat"]

C_ORACLE, C_CANON = "#3F6B52", "#6B5E9B"
C_NAT, C_ADV = "#B08A3E", "#A94E4E"
INK = "#2d3748"

DATA = [
    ("oracle\n(searched)",        (48.23, 50.28, 46.78), C_ORACLE),
    ("canonical\ninstruction",    (36.08, 48.33, 27.33), C_CANON),
    ("VLM-generated\nnaturals",   (nr["pooled"], nr["iv"], nr["oov"]), C_NAT),
    ("adversarial",               (24.46, 25.17, 23.94), C_ADV),
]

fig, ax = plt.subplots(figsize=(6.8, 4.1))
x, ticks, labels = 0.0, [], []
for name, (pool, iv, oov), col in DATA:
    ax.bar(x - 0.20, iv, 0.34, color=col, alpha=0.40, zorder=1, edgecolor="none")
    ax.bar(x + 0.20, oov, 0.34, color=col, alpha=0.40, zorder=1, edgecolor="none")
    ax.text(x - 0.32, iv + 0.7, f"{iv:.0f}", ha="center", fontsize=7,
            color="#4a5568", zorder=3)
    ax.text(x + 0.32, oov + 0.7, f"{oov:.0f}", ha="center", fontsize=7,
            color="#4a5568", zorder=3)
    ax.bar(x, pool, 0.44, color=col, zorder=2, edgecolor="none")
    ax.text(x, pool - 1.2, f"{pool:.1f}", ha="center", va="top", fontsize=9.5,
            fontweight="bold", color="white", zorder=4,
            bbox=dict(boxstyle="square,pad=0.12", fc=col, ec="none"))
    ticks.append(x); labels.append(name)
    x += 1.05

ax.set_xticks(ticks)
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel("success % (12 sealed tasks, 24 layouts)", fontsize=9.5)
ax.set_ylim(0, 55)
ax.grid(axis="y", alpha=0.18, zorder=0)
ax.tick_params(axis="x", length=0)
ax.spines[["top", "right"]].set_visible(False)

handles = [plt.Rectangle((0, 0), 1, 1, fc="#7A8698"),
           plt.Rectangle((0, 0), 1, 1, fc="#7A8698", alpha=0.40)]
ax.legend(handles, ["pooled (12 tasks)",
                    "flanks: in-distribution (left, 5) / out-of-distribution (right, 7)"],
          fontsize=7.8, loc="upper right", frameon=False)

fig.tight_layout()
out = R / "results/charts/pi0_conditions.png"
fig.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0.15)
print("chart ->", out)
