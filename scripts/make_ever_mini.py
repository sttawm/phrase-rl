#!/usr/bin/env python3
"""results/charts/ever_mini.png — miniature dual-metric four-arm comparison
(adversarial, full-72 bases x 24 layouts x 1 rep; all arms complete,
base-weighted: applier collisions expanded back to their 72 bases)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ARMS = [("Gemini +\nrollout rules", 28.1, 35.5, "#a3bffa"),
        ("Gemini,\nno rules", 24.1, 32.4, "#8b96a5"),
        ("no rephraser\n(passthrough)", 24.4, 31.8, "#8b96a5"),
        ("CoVer\n(verifier on)", 20.7, 28.5, "#f6ad55")]

fig, ax = plt.subplots(figsize=(6.8, 3.4))
w = 0.38
for i, (name, fin, ever, col) in enumerate(ARMS):
    ax.bar(i - w/2, fin, w, color=col, edgecolor="#4a5568", lw=0.8)
    ax.bar(i + w/2, ever, w, color=col, alpha=0.45, edgecolor="#4a5568", lw=0.8, hatch="//")
    ax.text(i - w/2, fin + 0.6, f"{fin:.1f}", ha="center", fontsize=9, fontweight="bold")
    ax.text(i + w/2, ever + 0.6, f"{ever:.1f}", ha="center", fontsize=9, fontweight="bold",
            color="#4a5568")
ax.set_xticks(range(len(ARMS)))
ax.set_xticklabels([a[0] for a in ARMS], fontsize=8.5)
ax.set_ylim(0, 40)
ax.set_ylabel("adversarial success %", fontsize=10)
ax.grid(axis="y", alpha=0.2)
ax.spines[["top", "right"]].set_visible(False)
solid = plt.Rectangle((0, 0), 1, 1, color="#b0b7c3")
hatched = plt.Rectangle((0, 0), 1, 1, color="#b0b7c3", alpha=0.45, hatch="//")
ax.legend([solid, hatched], ["final state @ 60 steps", "first success (ever)"],
          fontsize=8.5, loc="upper right", framealpha=0.95)
ax.set_title("Adversarial, both metrics — 72 attacks × 24 layouts", fontsize=10.5)
fig.text(0.99, 0.01, "all arms 72/72 bases, base-weighted", ha="right", fontsize=7,
         color="#718096")
fig.tight_layout()
fig.savefig("results/charts/ever_mini.png", dpi=160, bbox_inches="tight", pad_inches=0.15)
print("chart -> results/charts/ever_mini.png")
