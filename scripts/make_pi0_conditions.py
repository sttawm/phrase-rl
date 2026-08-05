#!/usr/bin/env python3
"""pi0 (rephrase-augmented INTACT) input-condition chart: oracle / original /
adversarial / natural-rephrasing, each as a pooled | in-vocab | OOV triplet.
ALL bars layout-matched to layouts 0-11 (natural is measured there); computed
from raw parquets. Pastel palette from the condition ladders."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# (pooled, IV, OOV) — layouts 0-11, from raw parquets (see compute in git log)
DATA = [
    ("oracle*", (46.88, 48.3, 45.8)),
    ("original\n(canonical)", (34.38, 44.6, 27.1)),
    ("adversarial\n(ERT passthrough)", (26.85, 31.5, 23.5)),
    ("natural human\nrephrases (K=16)", (26.30, 38.6, 17.5)),
]
STRATA = [("pooled", "#a3bffa"), ("in-vocab (5)", "#b2f5ea"), ("OOV (7)", "#fed7aa")]
W = 0.27

fig, ax = plt.subplots(figsize=(9.5, 5.0))
x = 0.0
ticks, labels = [], []
for name, vals in DATA:
    for j, ((sname, col), v) in enumerate(zip(STRATA, vals)):
        ax.bar(x + (j - 1) * W, v, W, color=col, edgecolor="#4a5568", lw=0.7)
        ax.text(x + (j - 1) * W, v + 0.5, f"{v:.1f}", ha="center", fontsize=8)
    ticks.append(x)
    labels.append(name)
    x += 1.35
handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, c in STRATA]
ax.legend(handles, [s for s, _ in STRATA], fontsize=9, loc="upper right")
ax.set_xticks(ticks)
ax.set_xticklabels(labels, fontsize=9.5)
ax.tick_params(axis="x", length=0)
ax.set_ylabel("success % (layouts 0-11)")
ax.set_ylim(0, 54)
ax.grid(axis="y", alpha=0.25)
ax.set_title("pi0 (bridge finetune + rephrase augmentation) by input condition — layout-matched (0-11)\n"
             "*oracle phrases selected on layouts 0-17; natural = 16 rephrases/task x 1 rep, others x 12 reps",
             fontsize=9.5)
fig.tight_layout()
fig.savefig("results/charts/pi0_conditions.png", dpi=140, bbox_inches="tight", pad_inches=0.2)
print("chart -> results/charts/pi0_conditions.png")
