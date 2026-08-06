#!/usr/bin/env python3
"""Fleet Gantt (2026-08-05 evening push): every machine, its assigned legs,
and projected landing times. Pastel palette; times UTC."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

C = {"sealed": "#a3bffa", "rr": "#fed7aa", "gen": "#b2f5ea",
     "train": "#d6bcfa", "worker": "#c6f6d5", "setup": "#e2e8f0"}
NOW = 43.0  # ~19:00 Aug 6 pod-time

ROWS = [
    ("L40S",    [("v10 RL training — step ~240/1000, healthy", 4.0, 52.0, "train"),
                 ("archive drain -> ckpt-l40s branch", 40.5, 52.0, "gen")]),
    ("Eval 3",  [("promptB_claude_ert DONE", 19.45, 28.2, "sealed"),
                 ("wedged", 28.3, 39.0, "setup"),
                 ("cells x5", 39.0, 42.0, "worker"),
                 ("LEG: promptB_claude_lay12 (~01:30)", 42.1, 49.5, "rr")]),
    ("Eval 4",  [("leg2 wait-loop wedged", 26.3, 39.0, "setup"),
                 ("cells x5", 39.0, 42.0, "worker"),
                 ("LEG: rulesv4_qwen 0-11 (~01:30)", 42.1, 49.5, "rr")]),
    ("Eval 5",  [("leg2 wait-loop wedged", 26.5, 39.0, "setup"),
                 ("cells x5", 39.0, 42.0, "worker"),
                 ("LEG: promptB_gemini_lay12 (~01:30)", 42.1, 49.5, "rr")]),
    ("Eval 6",  [("v10 gen 10-90; wedge; freed 14:08", 4.0, 38.1, "worker"),
                 ("gen 190-220 + step-0 + archive keeper (full 10-220)", 42.9, 52.0, "gen")]),
    ("Eval 7",  [("rulesv4_gem_lay12 DONE 33.51", 19.45, 28.8, "rr"),
                 ("hung roll -> bounced", 38.2, 40.0, "setup"),
                 ("cells x6", 40.1, 42.0, "worker"),
                 ("LEG: promptB_qwen 0-11 redo (~01:30)", 42.1, 49.5, "rr")]),
    ("Eval 8",  [("rulesv4_cla_lay12 DONE 31.94", 19.45, 31.2, "rr"),
                 ("qwenchain wedged", 31.3, 39.0, "setup"),
                 ("cells x5", 39.0, 42.0, "worker"),
                 ("LEG: rulesv4_qwen_lay12 (~01:30)", 42.1, 49.5, "rr")]),
    ("queue",   [("promptB_qwen_lay12 -> first pod free (~01:30-08:30)", 49.6, 57.0, "rr")]),
]

fig, ax = plt.subplots(figsize=(14.5, 6.2))
for i, (pod, bars) in enumerate(ROWS):
    y = len(ROWS) - 1 - i
    for label, s, e, cat in bars:
        ax.barh(y, e - s, left=s, height=0.62, color=C[cat], edgecolor="#4a5568", lw=0.6)
        if e - s > 1.6:
            ax.text((s + e) / 2, y, label, ha="center", va="center", fontsize=7.0, color="#2d3748")
ax.axvline(NOW, color="#e53e3e", lw=1.4, ls="--")
ax.text(NOW + 0.1, len(ROWS) - 0.35, "now ~19:00", color="#e53e3e", fontsize=8)
ax.set_yticks(range(len(ROWS)))
ax.set_yticklabels([p for p, _ in reversed(ROWS)], fontsize=9.5)
ticks = list(range(4, 57, 4))
ax.set_xticks(ticks)
ax.set_xticklabels([f"{t % 24:02d}:00" + ("\n(Aug 6)" if 24 <= t < 48 else ("\n(Aug 7)" if t >= 48 else "")) for t in ticks], fontsize=8)
ax.set_xlim(3.5, 57.5)
ax.grid(axis="x", alpha=0.25)
ax.legend(handles=[Patch(color=C[k], label=l) for k, l in
                   [("sealed", "sealed roll"), ("rr", "robustness leg (2,304 eps)"),
                    ("gen", "phrase gen / drain"), ("train", "RL training"),
                    ("worker", "checkpoint cells"), ("setup", "wedge / incident")]],
          fontsize=8, loc="lower left")
ax.set_title("Fleet — curve COMPLETE (28/28 cells incl. step-0); all 5 pods on overnight legs; 6th leg queued (times UTC)", fontsize=10)
fig.tight_layout()
fig.savefig("results/charts/fleet_gantt.png", dpi=140, bbox_inches="tight", pad_inches=0.2)
print("chart -> results/charts/fleet_gantt.png")
