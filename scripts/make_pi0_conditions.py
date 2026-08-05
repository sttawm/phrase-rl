#!/usr/bin/env python3
"""pi0 (rephrase-augmented INTACT) input-condition chart: oracle / original /
adversarial / natural-rephrasing. POOLED is the headline bar (solid, front);
the in-vocab / out-of-vocabulary strata sit BEHIND it, thinner and
semi-transparent, peeking out either side. All bars layout-matched to 0-11."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# (pooled, IV, OOV) — layouts 0-11, from raw parquets
DATA = [
    ("oracle*", (46.88, 48.3, 45.8)),
    ("original\n(canonical)", (34.38, 44.6, 27.1)),
    ("adversarial\n(ERT passthrough)", (26.85, 31.5, 23.5)),
    ("natural human\nrephrases (K=16)", (26.30, 38.6, 17.5)),
]
C_POOL, C_IV, C_OOV = "#a3bffa", "#b2f5ea", "#fed7aa"

fig, ax = plt.subplots(figsize=(9.5, 5.0))
x = 0.0
ticks, labels = [], []
for name, (pool, iv, oov) in DATA:
    # strata behind: thinner, offset, semi-transparent
    ax.bar(x - 0.20, iv, 0.36, color=C_IV, alpha=0.55, zorder=1, edgecolor="none")
    ax.bar(x + 0.20, oov, 0.36, color=C_OOV, alpha=0.55, zorder=1, edgecolor="none")
    ax.text(x - 0.33, iv + 0.5, f"{iv:.0f}", ha="center", fontsize=7, color="#4a5568", zorder=3)
    ax.text(x + 0.33, oov + 0.5, f"{oov:.0f}", ha="center", fontsize=7, color="#4a5568", zorder=3)
    # pooled in front: wide, solid
    ax.bar(x, pool, 0.42, color=C_POOL, zorder=2, edgecolor="#4a5568", lw=0.9)
    ax.text(x, pool + 0.7, f"{pool:.1f}", ha="center", fontsize=10, fontweight="bold", zorder=3)
    ticks.append(x)
    labels.append(name)
    x += 1.35
handles = [plt.Rectangle((0, 0), 1, 1, color=C_POOL),
           plt.Rectangle((0, 0), 1, 1, color=C_IV, alpha=0.55),
           plt.Rectangle((0, 0), 1, 1, color=C_OOV, alpha=0.55)]
ax.legend(handles, ["pooled (5 + 7)", "in-vocab (5)", "out-of-vocabulary (7)"],
          fontsize=9, loc="upper right")
ax.set_xticks(ticks)
ax.set_xticklabels(labels, fontsize=9.5)
ax.tick_params(axis="x", length=0)
ax.set_ylabel("success % (layouts 0-11)")
ax.set_ylim(0, 56)
ax.grid(axis="y", alpha=0.25, zorder=0)
ax.set_title(r"$\pi_0$ (bridge finetune + rephrase augmentation) by input condition — layout-matched (0-11)"
             "\n*oracle phrases selected on layouts 0-17; natural = 16 rephrases/task x 1 rep, others x 12 reps",
             fontsize=9.5)
fig.tight_layout()
fig.savefig("results/charts/pi0_conditions.png", dpi=140, bbox_inches="tight", pad_inches=0.2)
print("chart -> results/charts/pi0_conditions.png")
