#!/usr/bin/env python3
"""Fleet Gantt (2026-08-05 evening push): every machine, its assigned legs,
and projected landing times. Pastel palette; times UTC."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

C = {"sealed": "#a3bffa", "rr": "#fed7aa", "gen": "#b2f5ea",
     "train": "#d6bcfa", "worker": "#c6f6d5", "setup": "#e2e8f0"}
NOW = 52.0  # ~04:00 Aug 7

ROWS = [
    ("L40S",    [("v10 RL (paused step ~256)", 4.0, 44.5, "train"),
                 ("stopped", 44.6, 47.5, "gen"),
                 ("venv rebuild", 51.0, 51.9, "setup"),
                 ("v11 RL TRAINING — prompt B+ / 25-50-25 mix / c4b (runs all night)", 52.0, 74.0, "train")]),
    ("Eval 6",  [("v10 gens + archive keeper", 4.0, 47.0, "worker"),
                 ("stopped", 47.1, 49.0, "gen"),
                 ("v11 naturals (qwen arm) DONE", 50.2, 51.6, "gen"),
                 ("idle -> v11 ckpt-phrase worker (~06:00)", 54.0, 74.0, "worker")]),
    ("Eval 3",  [("v10 legs + cells", 39.0, 47.5, "rr"),
                 ("v10 cells (backfill)", 51.8, 60.0, "worker"),
                 ("-> v11 cell consumer", 60.1, 74.0, "worker")]),
    ("Eval 4",  [("v10 legs + cells", 39.0, 47.2, "rr"),
                 ("v10 cells (backfill)", 51.8, 60.0, "worker"),
                 ("-> v11 cell consumer", 60.1, 74.0, "worker")]),
    ("Eval 5",  [("v10 cells + gemini lay12 leg", 39.0, 47.0, "rr"),
                 ("FINAL LEG: promptB_qwen_lay12 (~07:15)", 47.2, 55.2, "rr"),
                 ("auto-stop on landing", 55.3, 56.6, "gen")]),
    ("Eval 7",  [("legs + cells", 19.45, 48.0, "worker"), ("stopped 03:20", 48.1, 49.6, "gen")]),
    ("Eval 8",  [("legs + cells", 19.45, 48.0, "worker"), ("stopped 03:20", 48.1, 49.6, "gen")]),
    ("Pod 5",   [("qwen gens; degraded host", 22.2, 39.9, "setup"), ("retired 15:55", 40.0, 41.4, "gen")]),
]

fig, ax = plt.subplots(figsize=(15, 5.8))
for i, (pod, bars) in enumerate(ROWS):
    y = len(ROWS) - 1 - i
    for label, s, e, cat in bars:
        ax.barh(y, e - s, left=s, height=0.62, color=C[cat], edgecolor="#4a5568", lw=0.6)
        if e - s > 2.2:
            ax.text((s + e) / 2, y, label, ha="center", va="center", fontsize=7.0, color="#2d3748")
ax.axvline(NOW, color="#e53e3e", lw=1.5, ls="--")
ax.text(NOW + 0.2, len(ROWS) - 0.4, "now 04:00", color="#e53e3e", fontsize=8.5, fontweight="bold")
ax.set_yticks(range(len(ROWS)))
ax.set_yticklabels([p for p, _ in reversed(ROWS)], fontsize=9.5)
ticks = list(range(4, 75, 4))
ax.set_xticks(ticks)
ax.set_xticklabels([f"{t % 24:02d}:00" + ("\n(Aug 6)" if 24 <= t < 48 else ("\n(Aug 7)" if t >= 48 else "")) for t in ticks], fontsize=8)
ax.set_xlim(3.5, 74.5)
ax.grid(axis="x", alpha=0.25)
ax.legend(handles=[Patch(color=C[k], label=l) for k, l in
                   [("train", "RL training"), ("rr", "robustness leg (2,304 eps)"),
                    ("worker", "checkpoint cells / gen"), ("setup", "setup"), ("gen", "stop / short job")]],
          fontsize=8, loc="lower left")
ax.set_title("v10 closed -> v11 launched. Overnight: v11 trains on L40S; e5 finishes the last ladder leg then self-stops; e3/e4 top up v10 cells then become v11 consumers", fontsize=10)
fig.tight_layout()
fig.savefig("results/charts/fleet_gantt.png", dpi=140, bbox_inches="tight", pad_inches=0.2)
print("chart -> results/charts/fleet_gantt.png")
