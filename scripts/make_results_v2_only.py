#!/usr/bin/env python3
"""results/charts/results_v2_only.png -- ONLY the v2-measured sealed cells.

Everything on this chart was measured in the A29/A30 campaign (2026-09-03):
rollout-derived (sim-loop) books applied by all three appliers, on the widened
adversarial set (6 attacks/task, 72 phrases, 24 layouts x 2) and the natural
set (12x16, 24 layouts, matched traces). Baseline = the 72-attack passthrough.
Nothing carried over from the paper's runs; compare side by side with Fig 9
(results_translating.png). Scaffold (no-rules) arms land separately (A30).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# (pooled, IV, OOV) -- from results/analysis/a29_adv72_cells.json
ADV_BASE = (24.5, 25.2, 23.9)
ADV = {"Gemini-Pro": (31.7, 38.0, 27.1),
       "Claude Fable": (29.3, 35.8, 24.6),
       "Frozen Qwen": (25.4, 30.8, 21.5)}
NAT = {"Gemini-Pro": (32.1, 41.0, 25.7),
       "Claude Fable": (32.1, 41.7, 25.2),
       "Frozen Qwen": (31.0, 44.2, 21.5)}

C_RULES, C_BASEBAR = "#a3bffa", "#8b96a5"
C_IV, C_OOV = "#b2f5ea", "#fed7aa"
YMIN, YMAX = 15, 50

fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2), sharey=True)


def triplet(ax, x, cellv, pooled_color):
    p, iv, oo = cellv
    ax.bar(x - 0.20, iv, 0.34, color=C_IV, alpha=0.55, zorder=2)
    ax.bar(x + 0.20, oo, 0.34, color=C_OOV, alpha=0.55, zorder=2)
    ax.text(x - 0.32, iv + 0.4, f"{iv:.0f}", ha="center", fontsize=7, color="#4a5568", zorder=4)
    ax.text(x + 0.32, oo + 0.4, f"{oo:.0f}", ha="center", fontsize=7, color="#4a5568", zorder=4)
    ax.bar(x, p, 0.42, color=pooled_color, zorder=3, edgecolor="#4a5568", lw=0.9)
    ax.text(x, p + 0.55, f"{p:.1f}", ha="center", fontsize=9, fontweight="bold", zorder=4)


for ax, (cond, data, base) in zip(
        axes, [("Adversarial (6 attacks/task, n=3,456/arm)", ADV, ADV_BASE),
               ("Natural (12x16, matched traces, n=4,608/arm)", NAT, None)]):
    x = 0.0
    ticks, tlabels = [], []
    if base is not None:
        triplet(ax, x, base, C_BASEBAR)
        ticks.append(x); tlabels.append("passthrough\n(no rephraser)")
        x += 1.4
    for model in ("Gemini-Pro", "Claude Fable", "Frozen Qwen"):
        triplet(ax, x, data[model], C_RULES)
        ticks.append(x); tlabels.append(model + "\n+ sim-loop book")
        x += 1.4
    if base is not None:
        ax.axhline(base[0], color="#c53030", ls=":", lw=1.2, zorder=1)
    ax.set_title(cond, fontsize=11, pad=8)
    ax.set_xticks(ticks)
    ax.set_xticklabels(tlabels, fontsize=8)
    ax.set_ylim(YMIN, YMAX)
    ax.grid(alpha=0.22, axis="y")
axes[0].set_ylabel("sealed-suite success %", fontsize=11)
axes[1].axhline(28.45, color="#b7791f", ls=":", lw=1.2, zorder=1)
axes[1].text(2.7, 28.9, "paper natural baseline 28.45", fontsize=7.4, color="#b7791f")
fig.text(0.5, 0.965, "Sealed suite, v2 measurements only (A29 campaign)",
         ha="center", fontsize=13)
fig.text(0.985, 0.012, "narrow bars: in-vocab (teal) / out-of-vocab (orange) strata; "
         "no-rules scaffold arms (A30) land separately",
         ha="right", fontsize=7.2, color="#4a5568")
fig.tight_layout(rect=[0, 0.02, 1, 0.95])
fig.savefig("results/charts/results_v2_only.png", dpi=140, bbox_inches="tight")
print("chart -> results/charts/results_v2_only.png")
