#!/usr/bin/env python3
"""results/charts/v12_progress.png -- v12 live progress, two panels.

Left: the policy's reward per step (mean candidate proxy logit, from the
per-candidate channel dump) against the ORIGINAL instruction's reward level on
the same contexts -- their gap is the candidate margin, the one training
metric v11 proved has real dynamic range. Falling gap = policy climbing.
Right: rollout ground truth on the nat24 probe -- checkpoint cells (192
episodes, stride 10) plus reference levels: measured step-0 policy 42.19 and
passthrough (no rephraser) 37.5. The 768-episode judges at steps 0/150 are
drawn as diamonds when they land.

Refresh telemetry from the pod, then:
  .venv/bin/python scripts/make_v12_progress.py
"""
import glob
import json
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

recs = [json.loads(l) for l in open("results/analysis/v12_telemetry/train_log.jsonl")]
steps = [r for r in recs if r.get("step") is not None and r.get("cand_margin_logit") is not None]

cand = pd.read_parquet("results/analysis/v12_telemetry/cand_channels.parquet")
per_step = cand.groupby("step").reward.mean()

cells, judges = [], []
for f in sorted(glob.glob("results/analysis/v12cells/nat24_*.json")):
    d = json.load(open(f))
    m = re.search(r"nat24_(judge_)?(\d+)", f)
    if not m:
        continue
    (judges if m.group(1) else cells).append((int(m.group(2)), d["pooled"]))
cells = sorted([c for c in cells if isinstance(c[0], int)])
judges.sort()

fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.6, 4.6))

xs = [r["step"] for r in steps]
orig_level = [r["cand_margin_logit"] + per_step.get(r["step"], float("nan")) for r in steps]
a1.plot(per_step.index, per_step.values, "o-", color="#0d9488", lw=2.2, ms=6,
        label="candidates (mean proxy logit)")
a1.plot(xs, orig_level, "s--", color="#805ad5", lw=1.6, ms=5,
        label="original instruction (same contexts)")
vals = [r for r in recs if r.get("type") == "val" and r.get("mean_greedy_reward") is not None]
if vals:
    ax = a1.twinx()
    ax.plot([v["step"] for v in vals], [v["mean_greedy_reward"] for v in vals],
            "^-", color="#c05621", lw=2.0, ms=7, label="fixed-val greedy (natural)")
    if vals[0].get("mean_greedy_ert_reward") is not None:
        ax.plot([v["step"] for v in vals], [v["mean_greedy_ert_reward"] for v in vals],
                "v--", color="#9b2c2c", lw=1.6, ms=6, label="fixed-val greedy (adversarial)")
    ax.set_ylabel("fixed-val greedy reward", color="#c05621")
    ax.legend(fontsize=8, loc="lower right")
a1.set_xlabel("training step")
a1.set_ylabel("proxy-logit reward")
a1.set_title("Reward: candidates vs the original (gap = margin)", fontsize=12)
a1.legend(fontsize=9, loc="best")
a1.grid(alpha=0.25)

if cells:
    a2.plot([c[0] for c in cells], [c[1] for c in cells], "o-", color="#0d9488",
            lw=2.2, ms=6, label="checkpoint cells (192 episodes)")
for s, v in judges:
    a2.scatter([s], [v], marker="D", s=90, color="#c53030", zorder=5,
               label="768-episode judge" if (s, v) == judges[0] else None)
refs = {}
for name, f in (("step0", "results/analysis/v12cells/nat24_0000.json"),
                ("pass", "results/analysis/v12cells/nat24_pass.json")):
    try:
        refs[name] = json.load(open(f))["pooled"]
    except FileNotFoundError:
        pass
if "step0" in refs:
    a2.axhline(refs["step0"], ls="--", color="#805ad5", lw=1.4)
    a2.text(0.98, refs["step0"] + 0.4, f"step-0 policy {refs['step0']}",
            color="#805ad5", fontsize=9, ha="right", transform=a2.get_yaxis_transform())
if "pass" in refs:
    a2.axhline(refs["pass"], ls="--", color="#718096", lw=1.4)
    a2.text(0.98, refs["pass"] - 1.1, f"no rephraser {refs['pass']}",
            color="#718096", fontsize=9, ha="right", transform=a2.get_yaxis_transform())
a2.set_ylim(30, 52)
a2.set_xlim(-5, 155)
a2.set_xlabel("training step")
a2.set_ylabel("rollout success on natural probes (%)")
a2.set_title("Ground truth (nat24)", fontsize=12)
if cells or judges:
    a2.legend(fontsize=9, loc="lower right")
a2.grid(alpha=0.25)

fig.tight_layout()
fig.savefig("results/charts/v12_progress.png", dpi=140, bbox_inches="tight", pad_inches=0.2)
print("chart -> results/charts/v12_progress.png")
