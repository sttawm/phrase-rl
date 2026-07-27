#!/usr/bin/env python3
"""Survival curve: x = C (episodes/instruction), y = # instructions with >= C
episodes. Two pools: FULL Bridge train (53k eps, empty-label excluded) and OUR
traced 2000-episode sample (parents need traces; extra scoring contexts don't).
Input: /tmp/ctx_counts.json {"full": [counts desc], "ours": [counts desc]}.
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

d = json.load(open("/tmp/ctx_counts.json"))
fig, ax = plt.subplots(figsize=(9, 5.2))
for name, counts, color in [("full Bridge train (needs new traces for parents)", d["full"], "#2b6cb0"),
                            ("our traced 2000-episode pool (no new Gemini)", d["ours"], "#dd6b20")]:
    counts = np.array(counts)
    ks = np.arange(1, counts.max() + 1)
    surv = [(counts >= k).sum() for k in ks]
    ax.plot(ks, surv, lw=1.8, color=color, label=name)
ax.axvline(16, ls=":", color="#822727", lw=1.2)
ax.text(16.5, 3000, "C=16 (v7e-strong)", color="#822727", fontsize=8.5, rotation=90, va="top")
ax.axvline(8, ls=":", color="#a0aec0", lw=1)
ax.text(8.4, 3000, "C=8", color="#718096", fontsize=8, rotation=90, va="top")
full = np.array(d["full"]); ours = np.array(d["ours"])
for C, pool, col, dy in [(16, full, "#2b6cb0", 1.6), (16, ours, "#dd6b20", 0.6)]:
    n = int((pool >= C).sum())
    ax.annotate(f"{n} instr ≥16", (16, max(n, 1)), textcoords="offset points",
                xytext=(18, 0), color=col, fontsize=9, fontweight="bold")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("C — episodes (contexts) available per instruction")
ax.set_ylabel("# instructions with ≥ C episodes")
ax.set_title("Context availability: how many instructions support C-context reward averaging")
ax.legend(fontsize=9)
ax.grid(alpha=0.25, which="both")
fig.text(0.01, 0.01, "Empty-string label (14,532 unlabeled eps) excluded. Scoring contexts need NO traces (pi0 forward only); "
         "only PARENT (generation) contexts need Gemini traces — new traces cost ~$3/1000 episodes (flash).",
         fontsize=7, color="#4a5568")
fig.tight_layout(rect=[0, 0.05, 1, 1])
fig.savefig("results/charts/context_survival.png", dpi=150, bbox_inches="tight", pad_inches=0.25)
print("chart -> results/charts/context_survival.png")
for C in [2, 4, 8, 16, 24]:
    print(f"C={C:2d}: full {int((full>=C).sum()):4d} instr ({int(full[full>=C].sum())} eps) | ours {int((ours>=C).sum()):3d} instr ({int(ours[ours>=C].sum())} eps)")
