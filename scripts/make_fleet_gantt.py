#!/usr/bin/env python3
"""Fleet Gantt (2026-08-05 evening push): every machine, its assigned legs,
and projected landing times. Pastel palette; times UTC."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

C = {"sealed": "#a3bffa", "rr": "#fed7aa", "gen": "#b2f5ea",
     "train": "#d6bcfa", "worker": "#c6f6d5", "setup": "#e2e8f0"}
NOW = 51.6  # ~03:35 Aug 7 pod-time

ROWS = [
    ("L40S",    [("v10 RL training -> PAUSED at step ~256 (clean-pause, resumable)", 4.0, 44.5, "train"),
                 ("STOPPED", 44.6, 46.0, "gen")]),
    ("Eval 6",  [("gen 10-90; wedge; 190-250 gens + archive keeper (10-250)", 4.0, 47.0, "worker"),
                 ("STOPPED 03:20 (volume = archive)", 47.1, 48.5, "gen")]),
    ("Eval 7",  [("legs + cells", 19.45, 48.0, "worker"), ("STOPPED 03:20", 48.1, 49.5, "gen")]),
    ("Eval 8",  [("legs + cells", 19.45, 48.0, "worker"), ("STOPPED 03:20", 48.1, 49.5, "gen")]),
    ("Eval 5",  [("cells + gemini_lay12 leg DONE 33.38", 39.0, 47.0, "rr"),
                 ("LEG: promptB_qwen_lay12 (~07:15)", 47.2, 55.2, "rr"),
                 ("stop on landing", 55.3, 56.3, "gen")]),
    ("Eval 3",  [("cells + claude_lay12 leg DONE 32.2", 39.0, 47.5, "rr"),
                 ("SEALED ARM: v10step60_polish (~13:00)", 47.9, 61.0, "sealed"),
                 ("stop", 61.1, 62.1, "gen")]),
    ("Eval 4",  [("cells + qwen rules leg DONE 27.04", 39.0, 47.2, "rr"),
                 ("SEALED ARM: v10step60_repair (~12:30)", 47.5, 60.5, "sealed"),
                 ("stop", 60.6, 61.6, "gen")]),
]

fig, ax = plt.subplots(figsize=(14.5, 5.4))
for i, (pod, bars) in enumerate(ROWS):
    y = len(ROWS) - 1 - i
    for label, s, e, cat in bars:
        ax.barh(y, e - s, left=s, height=0.62, color=C[cat], edgecolor="#4a5568", lw=0.6)
        if e - s > 1.8:
            ax.text((s + e) / 2, y, label, ha="center", va="center", fontsize=7.2, color="#2d3748")
ax.axvline(NOW, color="#e53e3e", lw=1.4, ls="--")
ax.text(NOW + 0.1, len(ROWS) - 0.35, "now 03:35", color="#e53e3e", fontsize=8)
ax.set_yticks(range(len(ROWS)))
ax.set_yticklabels([p for p, _ in reversed(ROWS)], fontsize=9.5)
ticks = list(range(4, 63, 4))
ax.set_xticks(ticks)
ax.set_xticklabels([f"{t % 24:02d}:00" + ("\n(Aug 6)" if 24 <= t < 48 else ("\n(Aug 7)" if t >= 48 else "")) for t in ticks], fontsize=8)
ax.set_xlim(3.5, 62.5)
ax.grid(axis="x", alpha=0.25)
ax.legend(handles=[Patch(color=C[k], label=l) for k, l in
                   [("sealed", "sealed arm (3,456 eps)"), ("rr", "robustness leg (2,304 eps)"),
                    ("gen", "stop"), ("train", "RL training"), ("worker", "gen / cells era")]],
          fontsize=8, loc="lower left")
ax.set_title("Wind-down — 3 arms remain (qwen lay12 ~07:15; step-60 sealed pair ~12:30-13:00); cells program ended at 31; all other machines stopped", fontsize=10)
fig.tight_layout()
fig.savefig("results/charts/fleet_gantt.png", dpi=140, bbox_inches="tight", pad_inches=0.2)
print("chart -> results/charts/fleet_gantt.png")
