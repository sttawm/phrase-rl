#!/usr/bin/env python3
"""v7 progress: proxy-reward val curves (left) + REAL rollout-val curve (right).

Left: from the training pod's train_log.jsonl (pass path as argv[1], default
/tmp/v7_train_log.jsonl). Right: results/analysis/v7_rollout_curve.jsonl —
greedy deployment phrases rolled on val-task layouts (n=192/checkpoint).
"""
import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

log = sys.argv[1] if len(sys.argv) > 1 else "/tmp/v7_train_log.jsonl"
vals = []
for line in open(log):
    try:
        d = json.loads(line)
    except json.JSONDecodeError:
        continue
    if d.get("type") == "val":
        vals.append(d)
vals.sort(key=lambda d: d["step"])

curve = sorted((json.loads(x) for x in open("results/analysis/v7_rollout_curve.jsonl")),
               key=lambda d: d["step"])

fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 4.4))
s = [v["step"] for v in vals]
for key, label, kw in [
    ("mean_orig_reward", "orig (no rewrite)", {"color": "#718096", "ls": ":"}),
    ("mean_best_reward", "best-of-16", {"color": "#48bb78"}),
    ("mean_greedy_reward", "greedy (deploy)", {"color": "#2b6cb0"}),
    ("mean_greedy_ert_reward", "greedy on ERT", {"color": "#e53e3e"}),
    ("mean_reward", "sample mean", {"color": "#a0aec0", "ls": "--"}),
]:
    a1.plot(s, [v.get(key) for v in vals], marker="o", ms=3, label=label, **kw)
a1.set_xlabel("step")
a1.set_ylabel("proxy reward (C4b)")
a1.set_title("proxy-reward val (n=40 contexts)")
a1.legend(fontsize=8)
a1.grid(alpha=0.25)

cs = [c["step"] for c in curve]
a2.plot(cs, [c["pooled"] for c in curve], marker="o", color="#2b6cb0", lw=2, label="pooled (4 tasks)")
tasks = sorted(curve[0]["per_task"])
for t, col in zip(tasks, ["#a0aec0", "#48bb78", "#ed8936", "#9f7aea"]):
    a2.plot(cs, [c["per_task"][t] for c in curve], marker=".", ms=4, lw=0.9, alpha=0.7,
            color=col, label=t.replace("widowx_", ""))
a2.set_xlabel("checkpoint step")
a2.set_ylabel("rollout success %")
a2.set_title("REAL rollout val (n=192/pt)")
a2.legend(fontsize=7)
a2.grid(alpha=0.25)

fig.suptitle("v7 RL (C4b reward, F=16, 25/25/50 mix, 0.5 dropout)", fontsize=11)
fig.tight_layout()
fig.savefig("results/charts/v7_progress.png", dpi=140, bbox_inches="tight", pad_inches=0.2)
print("chart -> results/charts/v7_progress.png")
