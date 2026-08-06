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

import glob
cells = [json.load(open(f)) for f in glob.glob("results/analysis/v10cells/*.json")]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.5, 4.6), gridspec_kw={"width_ratios": [1.1, 1]})

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

a2.axhspan(35, 41, color="#e2725b", alpha=0.13)
a2.text(2, 35.5, "v9 adversarial band (whole run)", fontsize=7, color="#c05621")
a2.axhline(44.27, ls="--", color="#718096", lw=1.1)
a2.text(2, 44.7, "v9 polish greedy (frozen 44.27)", fontsize=7, color="#718096")
for cond, color, lbl in [("v10_adv", "#c53030", "adversarial repair"), ("v10_pol", "#2b6cb0", "polish (nominal)")]:
    pts = sorted([(c["step"], c["pooled"]) for c in cells if c["probe"] == cond])
    if pts:
        a2.plot([p[0] for p in pts], [p[1] for p in pts], "o-", color=color, ms=7, lw=1.6, label=f"{lbl} ({len(pts)} pts)")
        for x, y in pts:
            a2.annotate(f"{y:.1f}", (x, y), textcoords="offset points", xytext=(0, 8), fontsize=7.5, ha="center", color=color)
a2.set_xlim(0, 230)
a2.set_ylim(20, 50)
a2.set_xlabel("v10 step")
a2.set_ylabel("val-8 rollout success (%)")
n_curve = len([c for c in cells if c["probe"] != "v10_ref"])
a2.set_title(f"GROUND TRUTH: val-8 rollouts (192 eps/cell) — {n_curve}/28 cells (incl. step-0 base)\nadv>pol at every step so far (repair-shaped gains)", fontsize=9)
a2.legend(fontsize=8, loc="lower right")
a2.grid(alpha=0.25)

fig.suptitle("v10 RL progress — step %d; first ground-truth cells: adv 43.8@30, 41.2@40 | pol 37.5@60 (val-8 probe; sealed anchors 29.8/29.63)"
             % tr[-1]["step"], fontsize=10)
fig.tight_layout()
fig.savefig("results/charts/v10_progress.png", dpi=140, bbox_inches="tight", pad_inches=0.2)
print("chart -> results/charts/v10_progress.png")
