#!/usr/bin/env python3
"""Reward discrimination vs frames — what we know today.

Two regimes on one chart:
  - IN THE LIMIT (episode-averaged, 10-20 eps x t x draws ~ up to 640 evals/phrase):
    C4b signs 66/68 = 97.1%, spearman 0.493. Horizontal reference.
  - PER SINGLE CONTEXT (GRPO's operating point): fraction of adjacent-rank pairs
    RESOLVED (100 - coin-flip zone) from the frames-vs-draws decomposition
    (desk analysis on stored per-draw exam features; ledger 2026-07-23):
    F=4: 11, F=8: 38, F=12: 52, F=16: 57, floor (F=64): 75.
    F=32 is an extrapolation of the fitted power law (resolved = 75 - A*F^-p).
Direct measured F-sweep (extraction at 16/32 t) is pending; those points will
replace the model when they land (task #7).
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

F_pts = np.array([4, 8, 12, 16])
resolved = np.array([11.0, 38.0, 52.0, 57.0])
FLOOR = 75.0

# fit resolved(F) = FLOOR - A * F^-p through the measured-feature points
p = np.polyfit(np.log(F_pts), np.log(FLOOR - resolved), 1)
slope, intercept = p[0], p[1]
A = np.exp(intercept)


def model(F):
    return FLOOR - A * F ** slope


Fs = np.linspace(3, 70, 200)
f32 = model(32)

fig, ax = plt.subplots(figsize=(8.6, 5))
ax.axhline(97.1, color="#48bb78", lw=1.6, ls="-")
ax.text(3.2, 97.8, "in-the-limit sign accuracy (C4b 66/68, ≤640 evals/phrase)",
        color="#2f855a", fontsize=8.5)
ax.plot(Fs, model(Fs), color="#a0aec0", lw=1.2, ls="--", label="fitted power law → floor 75%")
ax.axhline(FLOOR, color="#a0aec0", lw=0.9, ls=":")
ax.text(3.2, 75.8, "achievable floor (F→64+)", color="#718096", fontsize=8)
ax.plot(F_pts, resolved, "o", color="#2b6cb0", ms=8, label="decomposition (from stored per-draw features)")
ax.plot([32], [f32], "D", color="#dd6b20", ms=9, label=f"F=32 extrapolated ≈ {f32:.0f}%")
ax.plot([16, 32], [57, f32], color="#dd6b20", lw=1, ls=":")
for F, r in zip(F_pts, resolved):
    ax.annotate(f"{r:.0f}%", (F, r), textcoords="offset points", xytext=(0, -14),
                fontsize=8, ha="center", color="#2b6cb0")
ax.annotate("v7/v7b operate here", (16, 57), textcoords="offset points", xytext=(8, 10),
            fontsize=8.5, color="#2b6cb0")
ax.annotate("v7d proposal", (32, f32), textcoords="offset points", xytext=(8, -4),
            fontsize=8.5, color="#dd6b20")
ax.set_xscale("log", base=2)
ax.set_xticks([4, 8, 16, 32, 64])
ax.set_xticklabels(["4", "8", "16", "32", "64"])
ax.set_xlabel("reward frames F (per single context — GRPO's operating point)")
ax.set_ylabel("% of adjacent-rank phrase pairs RESOLVED (sign stable, |gap| ≥ 1 SE)")
ax.set_ylim(0, 105)
ax.set_title("Reward discrimination vs frames: single-context vs in-the-limit")
ax.legend(loc="lower right", fontsize=8)
ax.grid(alpha=0.25)
fig.text(0.01, 0.01, "Model points derive from stored per-draw exam features (frames-vs-draws decomposition, ledger 2026-07-23). "
         "Direct measured F=16/32 exam pending (task #7) — will replace the extrapolation.",
         fontsize=7, color="#718096")
fig.tight_layout(rect=[0, 0.045, 1, 1])
fig.savefig("results/charts/fsweep_model.png", dpi=150, bbox_inches="tight", pad_inches=0.25)
print(f"chart -> results/charts/fsweep_model.png  (F=32 model: {f32:.1f}% resolved vs 57% at F=16)")
