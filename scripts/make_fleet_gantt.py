#!/usr/bin/env python3
"""Fleet Gantt (2026-08-05 evening push): every machine, its assigned legs,
and projected landing times. Pastel palette; times UTC."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

C = {"sealed": "#a3bffa", "rr": "#fed7aa", "gen": "#b2f5ea",
     "train": "#d6bcfa", "worker": "#c6f6d5", "setup": "#e2e8f0"}
NOW = 43.8  # 19:50 Aug 6

ROWS = [
    ("L40S",    [("v10 RL training — step 233/1000, healthy through host-error banner", 4.0, 46.0, "train"),
                 ("archive drain -> ckpt-l40s branch (slow uplink)", 40.5, 46.0, "gen"),
                 ("EVAC 100-180 -> pod6 OK", 43.2, 43.8, "setup")]),
    ("Pod 5",   [("qwen gens + wedged gate", 22.2, 38.2, "setup"),
                 ("promptB_qwen leg: 4.7x degraded host — killed", 38.3, 39.9, "rr"),
                 ("RETIRED 15:55", 39.9, 41.0, "gen"),
                 ("10-min resume: phrase rescue", 42.7, 43.1, "setup")]),
    ("Eval 1",  [("leg done 03:13; idle; STOPPED 14:10 (now terminated)", 27.2, 38.2, "setup")]),
    ("Eval 2",  [("promptB_gemini_ert DONE 27.78; STOPPED (terminated)", 20.95, 31.7, "sealed")]),
    ("Eval 9",  [("STOPPED 03:20 capability test (terminated)", 27.3, 28.5, "gen")]),
    ("Eval 3",  [("promptB_claude_ert DONE 27.98", 19.45, 28.2, "sealed"),
                 ("consumer armed but repo-wedged (0 claims)", 28.3, 39.0, "setup"),
                 ("cells: pol 20/50 + rolling", 39.0, 44.3, "worker"),
                 ("leg queue ->", 44.4, 50.5, "rr")]),
    ("Eval 4",  [("leg2 wait-loop wedged — never rolled", 26.3, 39.0, "setup"),
                 ("cells: adv 40, pol 10 + rolling", 39.0, 44.3, "worker"),
                 ("leg queue ->", 44.4, 50.5, "rr")]),
    ("Eval 5",  [("leg2 wait-loop wedged — never rolled", 26.5, 39.0, "setup"),
                 ("cells: adv 30/90 + rolling", 39.0, 44.3, "worker"),
                 ("leg queue ->", 44.4, 50.5, "rr")]),
    ("Eval 6",  [("v10 gen 10-90 done 03:25; repo wedge (files freed by hand 14:08)", 4.0, 38.1, "worker"),
                 ("gen 190-220 pushed", 42.9, 43.6, "gen"),
                 ("archive keeper: full 10-220 set; awaits step 230+", 43.6, 46.0, "worker")]),
    ("Eval 7",  [("rulesv4_gem_lay12 DONE 33.51", 19.45, 28.8, "rr"),
                 ("adv_10 roll HUNG 160/192 — killed, claim freed", 38.2, 40.0, "setup"),
                 ("cells: adv 50/60 + rolling", 40.1, 44.3, "worker"),
                 ("leg queue ->", 44.4, 50.5, "rr")]),
    ("Eval 8",  [("rulesv4_cla_lay12 DONE 31.94", 19.45, 31.2, "rr"),
                 ("qwenchain wait-loop wedged (phrases never reached origin)", 31.3, 39.0, "setup"),
                 ("cells: pol 60/70 + rolling", 39.0, 44.3, "worker"),
                 ("leg queue ->", 44.4, 50.5, "rr")]),
]

fig, ax = plt.subplots(figsize=(14.5, 6.6))
for i, (pod, bars) in enumerate(ROWS):
    y = len(ROWS) - 1 - i
    for label, s, e, cat in bars:
        ax.barh(y, e - s, left=s, height=0.62, color=C[cat], edgecolor="#4a5568", lw=0.6)
        if e - s > 1.6:
            ax.text((s + e) / 2, y, label, ha="center", va="center", fontsize=6.9, color="#2d3748")
ax.axvline(NOW, color="#e53e3e", lw=1.4, ls="--")
ax.text(NOW + 0.1, len(ROWS) - 0.35, "now 19:50", color="#e53e3e", fontsize=8)
ax.set_yticks(range(len(ROWS)))
ax.set_yticklabels([p for p, _ in reversed(ROWS)], fontsize=9.5)
ticks = list(range(4, 51, 4))
ax.set_xticks(ticks)
ax.set_xticklabels([f"{t % 24:02d}:00" + ("\n(Aug 6)" if 24 <= t < 48 else ("\n(Aug 7)" if t >= 48 else "")) for t in ticks], fontsize=8)
ax.set_xlim(3.5, 50.9)
ax.grid(axis="x", alpha=0.25)
ax.legend(handles=[Patch(color=C[k], label=l) for k, l in
                   [("sealed", "sealed roll (3,456 eps)"), ("rr", "robustness roll (2,304 eps)"),
                    ("gen", "phrase gen / stop"), ("train", "RL training"),
                    ("worker", "checkpoint eval cells"), ("setup", "wedge / incident / setup")]],
          fontsize=8, loc="lower left")
ax.set_title("Fleet — Aug 5-6: wedge era (grey) -> 15:00 repair -> cells flowing; leg queue (2 lay12 + 4 qwen) auto-starts ~20:30 (times UTC)", fontsize=10)
fig.tight_layout()
fig.savefig("results/charts/fleet_gantt.png", dpi=140, bbox_inches="tight", pad_inches=0.2)
print("chart -> results/charts/fleet_gantt.png")
