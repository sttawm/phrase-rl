#!/usr/bin/env python3
"""The phrasing ladder — five pipelines on IDENTICAL cells:
10 searched tasks x held-out layouts 18-23 x 12 reps (n=720 per bar).

  ERT => pi0                                     (passthrough)
  ERT + gem-trace => Gemini-pro (with reasoning) => pi0        (gemini_pro_bare)
  ERT + gem-trace => Gemini-pro + RULES (with reasoning) => pi0 (rules_v3_gemini_pro)
  nominal => pi0                                 (originals)
  oracle phrase (adaptive search) => pi0         (row-14 confirm winners)

Flash-tier replication of the same ladder differs by <0.4pp per rung.
"""
import glob
import json
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HELD = set(range(18, 24))

conf = sorted(glob.glob("results/search/sealedsearch_*_confirm_results.json"))
tasks = [json.load(open(f))["task"] for f in conf]
oracle = {json.load(open(f))["task"]:
          max(json.load(open(f))["scoreboard"], key=lambda e: e["success_pct"])["success_pct"]
          for f in conf}

legs = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob("results/sealed/*_x12.parquet"))],
                 ignore_index=True)
cells = legs[legs.episode_id.isin(HELD) & legs.task.isin(tasks)]


def task_mean(arm: str) -> tuple[float, float]:
    g = cells[cells.arm == arm].groupby("task").success.agg(["mean", "count"])
    assert len(g) == len(tasks), f"{arm}: {len(g)} tasks"
    var = (g["mean"] * (1 - g["mean"]) / g["count"]).sum() / len(g) ** 2
    return 100 * g["mean"].mean(), 100 * math.sqrt(var)


BARS = [
    ("adversarial phrasing ⇒ π0", "passthrough", "#a0aec0"),
    ("adversarial phrasing + Gemini scene description ⇒ Gemini-pro (with reasoning) ⇒ π0", "gemini_pro_bare", "#63b3ed"),
    ("adversarial phrasing + Gemini scene description ⇒ v6-RL Qwen ⇒ π0", "v6_rl", "#9f7aea"),
    ("adversarial phrasing + Gemini scene description ⇒ frozen Qwen ⇒ π0", "frozen_gemini_trace", "#c3a6e8"),
    ("adversarial phrasing + Gemini scene description ⇒ frozen Qwen + RULES ⇒ π0", "rules_v3_gemini_trace", "#6b46c1"),
    ("adversarial phrasing + Gemini scene description ⇒ Gemini-pro + RULES (with reasoning) ⇒ π0", "rules_v3_gemini_pro", "#2b6cb0"),
    ("original phrasing ⇒ π0", "originals", "#48bb78"),
    ("oracle phrase (adaptive search) ⇒ π0", None, "#822727"),
]

rows = []
for label, arm, color in BARS:
    if arm:
        v, se = task_mean(arm)
    else:
        v = sum(oracle.values()) / len(oracle)
        se = 100 * math.sqrt(sum(p / 100 * (1 - p / 100) / 72 for p in oracle.values())) / len(oracle)
    rows.append((v, se, label, color))
rows.sort(reverse=True)  # best rung on top
BARS = [(label, None, color) for _, _, label, color in rows]
vals = [r[0] for r in rows]
errs = [r[1] for r in rows]
for v, se, label, _ in rows:
    print(f"{v:5.1f} ±{se:3.1f}  {label}")

fig, ax = plt.subplots(figsize=(9.5, 4.6))
y = range(len(BARS))
ax.barh(list(y), vals, 0.62, xerr=errs, color=[c for _, _, c in BARS],
        error_kw={"ecolor": "#4a5568", "capsize": 3, "lw": 1.2})
for i, v in enumerate(vals):
    ax.text(v + errs[i] + 0.7, i, f"{v:.1f}", va="center", fontsize=11, fontweight="bold")
ax.set_yticks(list(y))
ax.set_yticklabels([b[0] for b in BARS], fontsize=10)
ax.invert_yaxis()
ax.set_xlabel(f"success % on {len(tasks)} tasks")
ax.set_title("What phrasing recovers — the intervention ladder (sealed test)")
ax.grid(axis="x", alpha=0.25)
ax.set_xlim(0, max(v + e for v, e in zip(vals, errs)) + 6)
fig.tight_layout()
fig.savefig("results/charts/sealed_phrasing_ladder.png", dpi=150, bbox_inches="tight", pad_inches=0.25)
print("chart -> results/charts/sealed_phrasing_ladder.png")
