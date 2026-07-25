#!/usr/bin/env python3
"""Thin sealed scoreboard — 6 pipelines, one full-grid bar each (12 tasks x 24
layouts x 12 reps). Slots that are still rolling draw as placeholders and fill
in automatically on the next run once their parquet/leg lands:
  - nominal+RULES (row 17): placeholder until rules_pro_nominal_x12.parquet
  - oracle: held-out estimate (asterisk) until oracle_confirmed_x12.parquet
"""
import glob
import json
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

legs = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob("results/sealed/*_x12.parquet"))],
                 ignore_index=True)


def full_grid(arm):
    g = legs[legs.arm == arm].groupby("task").success.agg(["mean", "count"])
    if len(g) < 12:
        return None, None
    se = 100 * math.sqrt((g["mean"] * (1 - g["mean"]) / g["count"]).sum()) / len(g)
    return 100 * g["mean"].mean(), se


def oracle_value():
    v, se = full_grid("oracle_confirmed")
    if v is not None:
        return v, se, "oracle phrase (adaptive search) ⇒ π0"
    conf = sorted(glob.glob("results/search/sealedsearch_*_confirm_results.json"))
    best = [max(e["success_pct"] for e in json.load(open(f))["scoreboard"]) / 100 for f in conf]
    v = 100 * sum(best) / len(best)
    se = 100 * math.sqrt(sum(p * (1 - p) / 72 for p in best)) / len(best)
    return v, se, "oracle phrase (adaptive search) ⇒ π0  *held-out est., full-grid leg rolling"


BARS = [  # (label, arm or callable, color)
    ("oracle", oracle_value, "#822727"),
    ("original phrasing + Gemini scene description ⇒ Gemini-pro + RULES (with reasoning) ⇒ π0", "rules_pro_nominal", "#805ad5"),
    ("original phrasing ⇒ π0", "originals", "#48bb78"),
    ("adversarial phrasing + Gemini scene description ⇒ Gemini-pro + RULES (with reasoning) ⇒ π0", "rules_v3_gemini_pro", "#2b6cb0"),
    ("adversarial phrasing + Gemini scene description ⇒ Gemini-pro (with reasoning) ⇒ π0", "gemini_pro_bare", "#63b3ed"),
    ("adversarial phrasing ⇒ π0", "passthrough", "#a0aec0"),
]

rows = []
for label, src, color in BARS:
    if callable(src):
        v, se, label = src()
    else:
        v, se = full_grid(src)
    rows.append((label, v, se, color))
    print(f"{'PENDING' if v is None else f'{v:5.1f} ±{se:3.1f}':>12s}  {label}")

fig, ax = plt.subplots(figsize=(9.8, 3.9))
y = list(range(len(rows)))[::-1]
for yi, (label, v, se, color) in zip(y, rows):
    if v is None:
        ax.barh(yi, ax.get_xlim()[1] * 0 + 1e-9, 0.6)
        ax.text(0.6, yi, "leg rolling — lands shortly", va="center",
                fontsize=9, style="italic", color="#718096")
    else:
        ax.barh(yi, v, 0.6, xerr=se, color=color,
                error_kw={"ecolor": "#4a5568", "capsize": 3, "lw": 1.1})
        ax.text(v + se + 0.6, yi, f"{v:.1f}", va="center", fontsize=11, fontweight="bold")
ax.set_yticks(y)
ax.set_yticklabels([r[0] for r in rows], fontsize=9.5)
ax.set_xlabel("success % on 12 tasks (24 layouts × 12 reps)")
ax.set_title("Sealed test — thin scoreboard")
ax.grid(axis="x", alpha=0.25)
ax.set_xlim(0, max((r[1] or 0) + (r[2] or 0) for r in rows) + 7)
fig.tight_layout()
fig.savefig("results/charts/sealed_thin.png", dpi=150, bbox_inches="tight", pad_inches=0.25)
print("chart -> results/charts/sealed_thin.png")
