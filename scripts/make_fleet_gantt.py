#!/usr/bin/env python3
"""Fleet Gantt (2026-08-05 evening push): every machine, its assigned legs,
and projected landing times. Pastel palette; times UTC."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

C = {"sealed": "#a3bffa", "rr": "#fed7aa", "gen": "#b2f5ea",
     "train": "#d6bcfa", "worker": "#c6f6d5", "setup": "#e2e8f0"}
NOW = 77.4  # 05:25 Aug 8 UTC (hours from 00:00 Aug 5)

ROWS = [
    ("L40S",    [("v10 RL", 4.0, 44.5, "train"), ("stopped", 44.6, 47.5, "gen"),
                 ("v11 RL TRAINING — step 95/1000, continues", 52.0, 84.0, "train")]),
    ("Eval 4",  [("v10 legs + cells", 39.0, 47.2, "rr"), ("gen-stack", 51.5, 52.2, "setup"),
                 ("v11 nat24 evaluator — 10 cells, continues", 52.3, 84.0, "worker")]),
    ("Eval 1",  [("(was Pod 5) qwen gens", 22.2, 39.9, "setup"), ("retired", 40.0, 41.4, "gen"),
                 ("repair + disk rescue", 58.3, 58.6, "setup"),
                 ("v3 GEMINI leg — 37.41 (merge recovered)", 58.6, 69.5, "rr"),
                 ("git gc: 65G .git compacting", 70.6, 77.8, "setup"),
                 ("BANK SCORING — 1,648 TRAINING phrases, depth-first, all night", 77.9, 95.0, "worker")]),
    ("Eval 6",  [("v10 gens + archive keeper", 4.0, 51.6, "worker"), ("stopped", 51.9, 53.0, "gen"),
                 ("v3 qwen leg (30.08)", 53.2, 61.0, "rr"),
                 ("v3 CLAUDE leg — layouts 12-23, 72% done", 63.4, 81.7, "rr"),
                 ("sim contexts: val8 + 7 distractor scenes", 81.8, 85.2, "gen"),
                 ("BANK SCORING — 479 SIM phrases", 85.3, 90.5, "worker")]),
    ("Eval 5",  [("v10 cells + legs", 39.0, 55.2, "rr"),
                 ("v3 qwen lay12 (30.82)", 57.3, 65.0, "rr"),
                 ("v3 CLAUDE leg — 36.02", 62.6, 71.0, "rr"),
                 ("STOPPED (auto)", 71.4, 72.6, "gen")]),
    ("Eval 7",  [("legs + cells", 19.45, 48.0, "worker"),
                 ("v3 gemini lay12 — 36.37", 58.4, 65.6, "rr"), ("STOPPED", 65.7, 66.9, "gen")]),
    ("Eval 8",  [("legs + cells", 19.45, 48.0, "worker"),
                 ("nat24 backfill", 62.5, 64.3, "worker"), ("STOPPED", 64.4, 65.6, "gen")]),
    ("Eval 3",  [("v10 legs + cells", 39.0, 47.5, "rr"), ("STOPPED", 51.9, 53.3, "gen")]),
]

fig, ax = plt.subplots(figsize=(16, 5.6))
for i, (pod, bars) in enumerate(ROWS):
    y = len(ROWS) - 1 - i
    for label, s, e, cat in bars:
        ax.barh(y, e - s, left=s, height=0.62, color=C[cat], edgecolor="#4a5568", lw=0.6)
        if e - s > 2.6:
            ax.text((s + e) / 2, y, label, ha="center", va="center", fontsize=6.6, color="#2d3748")
ax.axvline(NOW, color="#e53e3e", lw=1.5, ls="--")
ax.text(NOW + 0.2, len(ROWS) - 0.4, "now 05:25", color="#e53e3e", fontsize=8.5, fontweight="bold")
ax.set_yticks(range(len(ROWS)))
ax.set_yticklabels([p for p, _ in reversed(ROWS)], fontsize=9.5)
ticks = list(range(4, 97, 4))
ax.set_xticks(ticks)
ax.set_xticklabels([f"{t % 24:02d}:00" + ("\n(Aug 6)" if 24 <= t < 48 else ("\n(Aug 7)" if 48 <= t < 72 else ("\n(Aug 8)" if 72 <= t < 96 else ""))) for t in ticks], fontsize=8)
ax.set_xlim(3.5, 95.5)
ax.grid(axis="x", alpha=0.25)
ax.legend(handles=[Patch(color=C[k], label=l) for k, l in
                   [("train", "RL training"), ("rr", "sealed robustness leg"),
                    ("worker", "scoring / evaluation"), ("setup", "setup / repair"), ("gen", "stop")]],
          fontsize=8, loc="lower left")
ax.set_title("Bank init split as asked: e1 takes the 1,648 TRAINING phrases, e6 takes the 479 SIM phrases "
             "(now incl. 7 distractor scenes) once its A28 leg lands.\n"
             "v3 beats v4 on matched layouts for all three appliers; the one hole left is v4-Gemini layouts 0-11, "
             "queued for the next free pod.", fontsize=9.5)
fig.tight_layout()
fig.savefig("results/charts/fleet_gantt.png", dpi=140, bbox_inches="tight", pad_inches=0.2)
print("chart -> results/charts/fleet_gantt.png")
