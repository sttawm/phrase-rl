#!/usr/bin/env python3
"""v8 progress — 2-panel: REAL val-8 rollouts (both conditions vs references) |
GRPO dynamics. v8 = tag-free cold start from base Qwen, input-dropout 0.5,
beta=0.15, grip-pure C=10 F=4. Curves from the pod6 all-in-one worker (stride 10,
2-rep greedy, tag-free gen). References: frozen Qwen (true step-0 of this
lineage), v7a-120 val-8 bests, and the sealed rules-v4 bars for context.
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def rows(*files):
    by = {}
    for f in files:
        if not os.path.exists(f):
            continue
        for l in open(f):
            l = l.strip()
            if not l or l.startswith(("<", "=", ">")):
                continue
            try:
                r = json.loads(l)
            except json.JSONDecodeError:
                continue
            if r["step"] not in by or r.get("n", 0) > by[r["step"]].get("n", 0):
                by[r["step"]] = r
    return sorted(by.values(), key=lambda d: d["step"])


adv = rows("results/analysis/v8_adv_curve.jsonl")
pol = rows("results/analysis/v8_pol_curve.jsonl")
tr = [json.loads(l) for l in open("results/analysis/v8_train_log.jsonl")
      if '"kl"' in l]
tr = [d for d in tr if d.get("kl") is not None]

fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.5, 4.6))

xmax = max([r["step"] for r in adv + pol] + [40]) + 8
a1.set_xlim(-2, xmax)
trx = a1.get_yaxis_transform()
for val, lbl, col, ls in [
        (40.58, "Original phrasing 40.6 (no rewrite, val-8)", "#2f855a", "--"),
        (34.51, "Adversarial phrasing 34.5 (no rewrite)", "#718096", "--"),
        (39.06, "FROZEN QWEN rewriting originals 39.1", "#553c9a", ":"),
        (35.94, "FROZEN QWEN rewriting adversarial 35.9", "#c53030", ":"),
        (45.05, "v7a-120 polish best (val-8, 2-rep)", "#6b46c1", "-."),
]:
    a1.axhline(val, ls=ls, color=col, lw=1.2, alpha=0.8)
    a1.text(0.02, val + 0.25, lbl, color=col, fontsize=6.8, transform=trx)
if pol:
    a1.plot([r["step"] for r in pol], [r["pooled"] for r in pol], marker="D", ms=7,
            color="#2b6cb0", lw=2, label="v8 rewriting ORIGINALS (tag-free, n=384)")
if adv:
    a1.plot([r["step"] for r in adv], [r["pooled"] for r in adv], marker="s", ms=7,
            color="#e53e3e", lw=2, label="v8 rewriting ADVERSARIAL (tag-free, n=384)")
a1.set_ylim(30, 48)
a1.set_xlabel("v8 step (cold start from base Qwen)")
a1.set_ylabel("val-8 rollout success %")
a1.set_title("REAL rollouts: v8 vs references")
a1.legend(fontsize=7.5, loc="lower right")
a1.grid(alpha=0.25)

if tr:
    ts = [d["step"] for d in tr]
    a2.plot(ts, [d["kl"] for d in tr], color="#805ad5", lw=1.4, label="KL(policy‖ref)")
    gn = [(d["step"], d["grad_norm"]) for d in tr if d.get("grad_norm") is not None]
    if gn:
        ag = a2.twinx()
        ag.plot([x for x, _ in gn], [g for _, g in gn], color="#dd6b20", lw=0.8,
                alpha=0.6, label="grad norm")
        ag.set_ylabel("grad norm", color="#dd6b20", fontsize=9)
        ag.tick_params(axis="y", labelcolor="#dd6b20", labelsize=7)
a2.set_xlabel("v8 step")
a2.set_ylabel("KL divergence", color="#805ad5")
a2.tick_params(axis="y", labelcolor="#805ad5")
a2.set_title("GRPO dynamics (beta=0.15, KL abort 1.2)")
a2.legend(loc="upper left", fontsize=8)
a2.grid(alpha=0.25)

fig.suptitle("v8 RL — tag-free, cold start, input-dropout 0.5, grip-pure C=10 F=4, beta=0.15   "
             "[sealed bars to beat: rules-v4 nominal 37.18 / ERT 33.30]", fontsize=10)
fig.tight_layout()
fig.savefig("results/charts/v8_progress.png", dpi=140, bbox_inches="tight", pad_inches=0.2)
print("chart -> results/charts/v8_progress.png")
