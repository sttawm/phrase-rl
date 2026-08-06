#!/usr/bin/env python3
"""v10 RL progress — proxy-val trajectory + KL, from the trainer's telemetry.
Rollout val-8 curve joins as the consumer pool merges checkpoint evals."""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOG = "/private/tmp/claude-501/-Users-sttawm-dev-robotics-phrase-rl/a45839fa-48ee-4aea-bdee-72eb4fc9dccf/scratchpad/v10_train_log.jsonl"
rows = [json.loads(l) for l in open(LOG)]
vals = sorted((r for r in rows if r.get("type") == "val"), key=lambda r: r["step"])
tr = sorted((r for r in rows if r.get("kl") is not None), key=lambda r: r["step"])

fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.5, 4.6), gridspec_kw={"width_ratios": [1.3, 1]})

s = [v["step"] for v in vals]
a1.axhline(vals[0]["mean_orig_reward"], ls=":", color="#718096", lw=1.2)
a1.text(2, vals[0]["mean_orig_reward"] + 0.04, "originals (no rewrite) +0.62", fontsize=7.5, color="#718096")
for key, label, kw in [
    ("mean_greedy_reward", "greedy (deploy)", {"color": "#2b6cb0", "lw": 2.2, "marker": "o", "ms": 5}),
    ("mean_best_reward", "best-of-16", {"color": "#48bb78", "lw": 1.2, "marker": "s", "ms": 3.5}),
    ("mean_reward", "sample mean", {"color": "#a0aec0", "ls": "--", "lw": 1.0, "marker": ".", "ms": 3}),
]:
    a1.plot(s, [v.get(key) for v in vals], label=label, **kw)
for x, lbl in [(2, "cold start"), (18, "accum 1"), (40, "quota\nrestart")]:
    a1.axvline(x, ls=":", color="#c05621", lw=0.9, alpha=0.6)
    a1.text(x + 1, a1.get_ylim()[0] if False else -2.85, lbl, fontsize=6.5, color="#c05621")
a1.set_xlabel("v10 step")
a1.set_ylabel("proxy reward (calibrated logit scale)")
a1.set_title("v10 (prompt B, cold start) — greedy proxy val: +0.41 net, plateauing ~-0.7/-0.8\n"
             "vs v9's frozen greedy (Δ+0.06 over its whole run)", fontsize=9.5)
a1.legend(fontsize=8, loc="lower right")
a1.grid(alpha=0.25)

ts = [d["step"] for d in tr]
a2.plot(ts, [d["kl"] for d in tr], color="#805ad5", lw=1.2)
a2.set_xlabel("v10 step")
a2.set_ylabel("KL(policy || ref)", color="#805ad5")
a2.set_title("KL — one 256-cand update/step", fontsize=9.5)
a2.grid(alpha=0.25)

fig.suptitle("v10 RL progress — step %d; sealed anchors: polish 29.8 / repair 29.63 (rollout val-8 curve landing today)"
             % tr[-1]["step"], fontsize=10)
fig.tight_layout()
fig.savefig("results/charts/v10_progress.png", dpi=140, bbox_inches="tight", pad_inches=0.2)
print("chart -> results/charts/v10_progress.png")
