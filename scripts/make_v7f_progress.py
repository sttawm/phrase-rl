#!/usr/bin/env python3
"""v7f progress — same 3-panel style as make_v7_progress.py, for the C=10 grip-pure
fork. Panels: proxy-reward val (grip blend) + win-rate | REAL rollouts vs references
| GRPO dynamics (KL + grad norm).

x-axis = v7f steps == steps past the v7e@20 fork point, so v7e's lone C=16 rollout
probe (its step 25 = fork+5) plots honestly at x=5 as a cross-branch anchor.
Log: results/analysis/v7f_train_log.jsonl (scp'd from L40S before each render).
Rollouts: results/analysis/v7f_rollout_curve.jsonl. v7a-best reference read live
from v7_dev8_backfill.jsonl (auto-rises as late checkpoints land).
"""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

log = sys.argv[1] if len(sys.argv) > 1 else "results/analysis/v7f_train_log.jsonl"
recs = []
for line in open(log):
    try:
        recs.append(json.loads(line))
    except json.JSONDecodeError:
        continue
vals = sorted((d for d in recs if d.get("type") == "val"), key=lambda d: d["step"])
trec = sorted((d for d in recs if d.get("type") not in ("val", "probe") and d.get("kl") is not None),
              key=lambda d: d["step"])

curve = []
if os.path.exists("results/analysis/v7f_rollout_curve.jsonl"):
    curve = sorted((json.loads(l) for l in open("results/analysis/v7f_rollout_curve.jsonl")),
                   key=lambda d: d["step"])

import glob


def _best(pattern):
    vals = [json.loads(l)["pooled"] for f in glob.glob(pattern) for l in open(f)]
    return max(vals) if vals else None


v7a_best = _best("results/analysis/v7_dev8_tagged*.jsonl")  # tagged (fixed) evals
v7a_best_untagged = _best("results/analysis/v7_dev8_backfill.jsonl")

fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(17.5, 4.4))

s = [v["step"] for v in vals]
if vals:
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
    a1.legend(fontsize=8)
else:
    a1.text(0.5, 0.5, "first val at step 20", ha="center", va="center", transform=a1.transAxes,
            color="#718096")
a1.set_xlabel("v7f step (= steps past v7e@20 fork)")
a1.set_ylabel("proxy reward (grip-pure)")
a1.set_title("proxy-reward val (n=40 contexts)")
a1.grid(alpha=0.25)

a2.set_xlim(0, max([c["step"] for c in curve] + [30]) + 5)
tr = a2.get_yaxis_transform()  # x in axes fraction, y in data units
a2.axhline(54.7, ls="--", color="#822727", lw=1.2)
a2.axhline(40.6, ls="--", color="#48bb78", lw=1.2)
a2.axhline(34.5, ls="--", color="#a0aec0", lw=1.2)
a2.text(0.02, 55.1, "Oracle 54.7 (val-8)", color="#822727", fontsize=7.5, transform=tr)
a2.text(0.02, 41.0, "Original 40.6 (val-8)", color="#2f855a", fontsize=7.5, transform=tr)
a2.text(0.02, 34.9, "Adversarial 34.5 (val-8)", color="#718096", fontsize=7.5, transform=tr)
if v7a_best is not None:
    a2.axhline(v7a_best, ls=":", color="#6b46c1", lw=1.6)
    a2.text(0.02, v7a_best + 0.4, f"v7a best (TAGGED evals so far) {v7a_best:.1f}", color="#6b46c1",
            fontsize=7.5, transform=tr)
if v7a_best_untagged is not None:
    a2.axhline(v7a_best_untagged, ls=":", color="#6b46c1", lw=1.0, alpha=0.35)
    a2.text(0.02, v7a_best_untagged + 0.4, f"v7a best (untagged, pre-fix) {v7a_best_untagged:.1f}",
            color="#6b46c1", alpha=0.5, fontsize=7, transform=tr)
if curve:
    cs = [c["step"] for c in curve]
    a2.plot(cs, [c["pooled"] for c in curve], marker="D", ms=7, color="#2b6cb0", lw=2.0,
            label="v7f: GREEDY (rewriting originals, 8-task, n=384)")
    sc = [c for c in curve if c.get("sampled_pooled") is not None]
    a2.plot([c["step"] for c in sc], [c["sampled_pooled"] for c in sc], marker="D", ms=7,
            markerfacecolor="none", markeredgecolor="#2b6cb0", color="#2b6cb0", lw=1.3, ls="--",
            label="v7f: SAMPLED (rewriting originals, n=192)")
if os.path.exists("results/analysis/v7f_adv_curve.jsonl"):
    adv = sorted((json.loads(l) for l in open("results/analysis/v7f_adv_curve.jsonl")),
                 key=lambda d: d["step"])
    a2.plot([c["step"] for c in adv], [c["pooled"] for c in adv], marker="s", ms=8,
            color="#e53e3e", lw=2.0, label="v7f: GREEDY (rewriting ADVERSARIAL, n=192)")
# v7e's lone C=16 probe: its step 25 = fork+5 on this axis
a2.scatter([5], [40.89], marker="X", s=90, color="#a0aec0", zorder=4,
           label="v7e branch (C=16) @fork+5: 40.9 (untagged probe, pre-fix)")
a2.set_xlabel("v7f step (= steps past v7e@20 fork)")
a2.set_ylabel("val-8 rollout success %")
a2.set_ylim(30, 58)
a2.set_title("REAL rollouts: v7f curve vs references")
a2.legend(fontsize=7, loc="lower right")
a2.grid(alpha=0.25)

if trec:
    ts = [d["step"] for d in trec]
    a3.plot(ts, [d["kl"] for d in trec], color="#805ad5", lw=1.3, label="KL(policy‖ref)")
    gn = [(d["step"], d["grad_norm"]) for d in trec if d.get("grad_norm") is not None]
    ag = a3.twinx()
    if gn:
        ag.plot([x for x, _ in gn], [g for _, g in gn], color="#dd6b20", lw=0.8, alpha=0.6,
                label="grad norm")
        ag.set_ylabel("grad norm", color="#dd6b20", fontsize=9)
        ag.tick_params(axis="y", labelcolor="#dd6b20", labelsize=7)
a3.axhline(0.05, ls=":", color="#a0aec0", lw=1, label="β=0.05 (penalty coef)")
a3.set_ylabel("KL divergence", color="#805ad5", fontsize=9)
a3.tick_params(axis="y", labelcolor="#805ad5")
a3.set_xlabel("v7f step")
a3.set_title("GRPO dynamics (KL abort = 1.2)")
a3.legend(loc="upper left", fontsize=7)
a3.grid(alpha=0.25)

fig.suptitle("v7f RL — GRPO (grip-pure reward, F=4, C=10 club contexts, β=0.05, lr=7e-6; "
             "fork lineage v7a@140 → v7e(C=16)@20 → v7f)", fontsize=11)
fig.tight_layout()
fig.savefig("results/charts/v7f_progress.png", dpi=140, bbox_inches="tight", pad_inches=0.2)
print("chart -> results/charts/v7f_progress.png")
