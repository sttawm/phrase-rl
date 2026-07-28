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

log = sys.argv[1] if len(sys.argv) > 1 else "results/analysis/v7a_train_log.jsonl"
vals = []
for line in open(log):
    try:
        d = json.loads(line)
    except json.JSONDecodeError:
        continue
    if d.get("type") == "val":
        vals.append(d)
vals.sort(key=lambda d: d["step"])

# merge live + backfill curves; prefer backfill per step (it has greedy+sampled)
import os
by_step = {}
for f in ["results/analysis/v7_rollout_curve.jsonl", "results/analysis/v7_curve_backfill.jsonl"]:
    if not os.path.exists(f):
        continue
    for x in open(f):
        d = json.loads(x)
        cur = by_step.get(d["step"])
        if cur is None or ("sampled_pooled" in d and "sampled_pooled" not in cur):
            by_step[d["step"]] = d
curve = sorted(by_step.values(), key=lambda d: d["step"])

trec = [d for d in (json.loads(l) for l in open(log))
        if d.get("type") not in ("val", "probe") and d.get("kl") is not None]
trec.sort(key=lambda d: d["step"])

fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(17.5, 4.4))
s = [v["step"] for v in vals]
for key, label, kw in [
    ("mean_orig_reward", "orig (no rewrite)", {"color": "#718096", "ls": ":"}),
    ("mean_best_reward", "best-of-16", {"color": "#48bb78"}),
    ("mean_greedy_reward", "greedy (deploy)", {"color": "#2b6cb0"}),
    ("mean_greedy_ert_reward", "greedy on ERT", {"color": "#e53e3e"}),
    ("mean_reward", "sample mean", {"color": "#a0aec0", "ls": "--"}),
]:
    a1.plot(s, [v.get(key) for v in vals], marker="o", ms=3, label=label, **kw)
aw = a1.twinx()
aw.plot(s, [100 * v.get("greedy_win_rate", 0) for v in vals], marker="s", ms=3,
        color="#d69e2e", lw=1.1, ls="-.", label="win-rate (greedy > orig)")
aw.set_ylabel("win-rate %", color="#d69e2e", fontsize=8)
aw.set_ylim(0, 50)
aw.tick_params(axis="y", labelcolor="#d69e2e", labelsize=7)
aw.legend(loc="upper right", fontsize=7)
a1.set_xlabel("step")
a1.set_ylabel("proxy reward (C4b)")
a1.set_title("proxy-reward val (n=40 contexts)")
a1.legend(fontsize=8)
a1.grid(alpha=0.25)

cs = [c["step"] for c in curve]
a2.plot(cs, [c["pooled"] for c in curve], marker="o", color="#2b6cb0", lw=1.2, alpha=0.55, label="greedy (4-task probe)")
try:
    d8 = sorted((json.loads(l) for l in open("results/analysis/v7_dev8_backfill.jsonl")), key=lambda r: r["step"])
    a2.plot([r["step"] for r in d8], [r["pooled"] for r in d8], marker="D", ms=7,
            color="#6b46c1", lw=2.2, label="greedy (FULL val-8)")
    a2.axhline(40.6, ls="--", color="#48bb78", lw=1.2)
    a2.text(cs[0], 41.0, "Original 40.6 (val-8)", color="#2f855a", fontsize=7.5)
    a2.axhline(54.7, ls="--", color="#822727", lw=1.2)
    a2.text(cs[0], 55.1, "Oracle 54.7 (val-8)", color="#822727", fontsize=7.5)
except FileNotFoundError:
    pass
try:
    d8a = sorted((json.loads(l) for l in open("results/analysis/v7_dev8adv_backfill.jsonl")), key=lambda r: r["step"])
    a2.plot([r["step"] for r in d8a], [r["pooled"] for r in d8a], marker="s", ms=8,
            color="#e53e3e", lw=2.0, label="greedy on ERT (FULL val-8, repair)")
    a2.axhline(34.5, ls="--", color="#a0aec0", lw=1.2)
    a2.text(cs[0], 34.9, "Adversarial 34.5 (val-8)", color="#718096", fontsize=7.5)
except FileNotFoundError:
    pass
sc = [c for c in curve if c.get("sampled_pooled") is not None]
if sc:
    a2.plot([c["step"] for c in sc], [c["sampled_pooled"] for c in sc],
            marker="D", ms=4, color="#dd6b20", lw=2.0, label="sampled (mean)")
tasks = sorted(curve[0]["per_task"])
for t, col in zip(tasks, ["#a0aec0", "#48bb78", "#90cdf4", "#9f7aea"]):
    a2.plot(cs, [c["per_task"].get(t) for c in curve], marker=".", ms=3, lw=0.7, alpha=0.5,
            color=col, label=t.replace("widowx_", ""))
a2.set_xlabel("checkpoint step")
a2.set_ylabel("rollout success %")
a2.set_title("REAL rollouts: 4-task probe curve + FULL val-8 checkpoints vs references")
a2.legend(fontsize=6.5, ncol=2)
a2.grid(alpha=0.25)

ts = [d["step"] for d in trec]
a3.plot(ts, [d["kl"] for d in trec], color="#805ad5", lw=1.3, label="KL(policy‖ref)")
a3.axhline(0.15, ls=":", color="#a0aec0", lw=1, label="β=0.15 (penalty coef)")
a3.set_ylabel("KL divergence", color="#805ad5", fontsize=9)
a3.tick_params(axis="y", labelcolor="#805ad5")
ag = a3.twinx()
gn = [(d["step"], d["grad_norm"]) for d in trec if d.get("grad_norm") is not None]
if gn:
    ag.plot([s for s, _ in gn], [g for _, g in gn], color="#dd6b20", lw=0.8, alpha=0.6, label="grad norm")
    ag.set_ylabel("grad norm", color="#dd6b20", fontsize=9)
    ag.tick_params(axis="y", labelcolor="#dd6b20", labelsize=7)
a3.set_xlabel("step")
a3.set_title("GRPO dynamics (KL abort = 1.2, far above)")
a3.legend(loc="upper left", fontsize=7)
a3.grid(alpha=0.25)

fig.suptitle("v7 RL — GRPO (C4b reward, F=16, 25/25/50 mix, 0.5 dropout, β=0.15, lr=7e-6)", fontsize=11)
fig.tight_layout()
fig.savefig("results/charts/v7_progress.png", dpi=140, bbox_inches="tight", pad_inches=0.2)
print("chart -> results/charts/v7_progress.png")
