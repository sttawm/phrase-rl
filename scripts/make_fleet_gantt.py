#!/usr/bin/env python3
"""Fleet Gantt (2026-08-05 evening push): every machine, its assigned legs,
and projected landing times. Pastel palette; times UTC."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

C = {"sealed": "#a3bffa", "rr": "#fed7aa", "gen": "#b2f5ea",
     "train": "#d6bcfa", "worker": "#c6f6d5", "setup": "#e2e8f0"}
NOW = 27.1  # 03:06 UTC Aug 6

ROWS = [
    ("L40S",    [("v10 RL training — step ~180, healthy", 4.0, 44.0, "train")]),
    ("old Pod5",[("gens+0-11 leg STUCK on zombie gate 16h", 22.2, 38.2, "setup"),
                 ("qwen 0-11 + lay12 (restarted)", 38.3, 50.0, "rr")]),
    ("Eval 1",  [("leg done 03:13; idle (MISSED) ", 27.2, 38.1, "setup"), ("STOPPED 14:10", 38.15, 39.3, "gen")]),
    ("Eval 2",  [("promptB_gemini_ert DONE 27.78", 20.95, 31.6, "sealed"), ("STOPPED 07:50", 31.7, 32.9, "gen")]),
    ("Eval 3",  [("promptB_claude_ert DONE 27.98", 19.45, 28.2, "sealed"), ("consumer (fed 14:10)", 28.3, 44.0, "worker")]),
    ("Eval 4",  [("promptB_gem_lay12 rolled 2304 — merging", 26.3, 38.4, "rr"), ("then consumer", 38.5, 44.0, "worker")]),
    ("Eval 5",  [("promptB_cla_lay12 rolled 2304 — merging", 26.5, 38.4, "rr"), ("then consumer", 38.5, 44.0, "worker")]),
    ("Eval 6",  [("v10 gen worker — 36 files freed 14:08", 4.0, 44.0, "worker")]),
    ("Eval 7",  [("rulesv4_gem_lay12 DONE 33.51", 19.45, 28.8, "rr"), ("consumer (fed 14:10)", 28.9, 44.0, "worker")]),
    ("Eval 8",  [("rulesv4_cla_lay12 DONE 31.94", 19.45, 31.2, "rr"), ("qwen rules 0-11 rolled — merging", 31.3, 38.4, "rr"),
                 ("qwen lay12 next", 38.5, 44.5, "rr")]),
    ("Eval 9",  [("STOPPED 03:20 (capability test)", 27.3, 28.5, "gen")]),
]

fig, ax = plt.subplots(figsize=(13.5, 6.2))
for i, (pod, bars) in enumerate(ROWS):
    y = len(ROWS) - 1 - i
    for label, s, e, cat in bars:
        ax.barh(y, e - s, left=s, height=0.62, color=C[cat], edgecolor="#4a5568", lw=0.6)
        if e - s > 1.6:
            ax.text((s + e) / 2, y, label, ha="center", va="center", fontsize=7.4, color="#2d3748")
ax.axvline(NOW, color="#e53e3e", lw=1.4, ls="--")
ax.text(NOW + 0.1, len(ROWS) - 0.35, "now 03:06", color="#e53e3e", fontsize=8)
ax.set_yticks(range(len(ROWS)))
ax.set_yticklabels([p for p, _ in reversed(ROWS)], fontsize=9.5)
ticks = list(range(4, 33, 2))
ax.set_xticks(ticks)
ax.set_xticklabels([f"{t % 24:02d}:00" + ("\n(Aug 6)" if t >= 24 else "") for t in ticks], fontsize=8)
ax.set_xlim(3.5, 32.5)
ax.grid(axis="x", alpha=0.25)
ax.legend(handles=[Patch(color=C[k], label=l) for k, l in
                   [("sealed", "sealed roll (3,456 eps)"), ("rr", "robustness roll (2,304 eps)"),
                    ("gen", "phrase generation"), ("train", "RL training"),
                    ("worker", "checkpoint evals"), ("setup", "setup / seed")]],
          fontsize=8, loc="lower left")
ax.set_title("Fleet schedule — 2026-08-05 parallel push (times UTC; ETAs from measured A6000 throughput)", fontsize=10.5)
fig.tight_layout()
fig.savefig("results/charts/fleet_gantt.png", dpi=140, bbox_inches="tight", pad_inches=0.2)
print("chart -> results/charts/fleet_gantt.png")
