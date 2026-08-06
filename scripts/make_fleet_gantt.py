#!/usr/bin/env python3
"""Fleet Gantt (2026-08-05 evening push): every machine, its assigned legs,
and projected landing times. Pastel palette; times UTC."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

C = {"sealed": "#a3bffa", "rr": "#fed7aa", "gen": "#b2f5ea",
     "train": "#d6bcfa", "worker": "#c6f6d5", "setup": "#e2e8f0"}
NOW = 26.0  # 02:00 UTC Aug 6

ROWS = [
    ("L40S",    [("v10 RL training (prompt B; step ~135)", 4.0, 34.0, "train")]),
    ("old Pod5",[("polish DONE 29.8", 7.65, 20.85, "sealed"), ("gens", 20.9, 22.15, "gen"),
                 ("rr16_promptB_qwen", 22.2, 28.3, "rr")]),
    ("Eval 1",  [("v10step0_repair (sealed)", 19.45, 29.3, "sealed")]),
    ("Eval 2",  [("promptB_gemini_ert (sealed)", 20.95, 30.8, "sealed")]),
    ("Eval 3",  [("promptB_claude_ert (sealed)", 19.45, 29.3, "sealed")]),
    ("Eval 4",  [("rr16_promptB_gemini", 20.95, 27.0, "rr"), ("promptB_gem_lay12 (chained)", 27.05, 33.1, "rr")]),
    ("Eval 5",  [("rr16_promptB_claude", 22.1, 28.0, "rr"), ("promptB_cla_lay12 (chained)", 28.05, 34.1, "rr")]),
    ("Eval 6",  [("v10 checkpoint GEN (rolls -> consumers)", 4.0, 34.0, "worker")]),
    ("Eval 7",  [("rulesv4_claude DONE 30.4", 19.45, 24.55, "rr"), ("consumer", 24.6, 25.7, "worker"),
                 ("rulesv4_gemini_lay12", 25.75, 31.8, "rr")]),
    ("Eval 8",  [("pi0reph_lay12 DONE 30.6", 19.45, 24.85, "rr"), ("rulesv4_claude_lay12", 25.75, 31.8, "rr")]),
    ("Eval 9",  [("pi0base_lay12 DONE 22.6", 19.45, 24.8, "rr"), ("rr16_rulesv4_qwen", 24.85, 30.9, "rr")]),
]

fig, ax = plt.subplots(figsize=(13.5, 6.2))
for i, (pod, bars) in enumerate(ROWS):
    y = len(ROWS) - 1 - i
    for label, s, e, cat in bars:
        ax.barh(y, e - s, left=s, height=0.62, color=C[cat], edgecolor="#4a5568", lw=0.6)
        if e - s > 1.6:
            ax.text((s + e) / 2, y, label, ha="center", va="center", fontsize=7.4, color="#2d3748")
ax.axvline(NOW, color="#e53e3e", lw=1.4, ls="--")
ax.text(NOW + 0.1, len(ROWS) - 0.35, "now 02:00", color="#e53e3e", fontsize=8)
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
