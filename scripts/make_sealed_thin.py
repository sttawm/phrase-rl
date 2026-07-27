#!/usr/bin/env python3
"""Thin sealed scoreboard — 6 pipelines x (pooled / in-vocab / OOV) on the full
grid (12 tasks x 24 layouts x 12 reps). Strata from the ex-ante vocab audit;
legend carries the task counts. Slots still rolling draw as placeholders; the
oracle uses held-out confirm estimates (asterisk) until its full-grid leg lands.
"""
import glob
import json
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch

ENV_OVERRIDE = {"PutGreenCubeOnPlate": "widowx_cube_on_plate_clean",
                "PutPepsiCanOnPlate": "widowx_pepsi_on_plate_clean"}


def env_to_task(env):
    base = re.sub(r"InScene.*$", "", env)
    for k, v in ENV_OVERRIDE.items():
        if base.startswith(k):
            return v
    s = re.sub(r"([a-z])([A-Z])", r"\1_\2", re.sub(r"^Put", "", base)).lower()
    return f"widowx_{s}_clean"


audit = json.load(open("results/analysis/sealed_vocab_audit.json"))
STRAT = {env_to_task(e["env"]): e["stratum"] for e in audit}
legs = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob("results/sealed/*_x12.parquet"))],
                 ignore_index=True)
ALL_TASKS = sorted(STRAT)
IV = [t for t in ALL_TASKS if STRAT[t] != "OOV"]
OO = [t for t in ALL_TASKS if STRAT[t] == "OOV"]


def strata_from_per_task(pt):
    return {"pooled": sum(pt.values()) / len(pt),
            "iv": sum(pt[t] for t in IV) / len(IV),
            "oov": sum(pt[t] for t in OO) / len(OO)}


def arm_per_task(arm):
    g = legs[legs.arm == arm].groupby("task").success.mean() * 100
    return dict(g) if len(g) == 12 else None


def oracle_per_task():
    pt = arm_per_task("oracle_confirmed")
    if pt:
        return pt, ""
    conf = sorted(glob.glob("results/search/sealedsearch_*_confirm_results.json"))
    pt = {json.load(open(f))["task"]:
          max(e["success_pct"] for e in json.load(open(f))["scoreboard"]) for f in conf}
    return pt, "*"


BARS = [
    ("oracle", "ORACLE", "#822727"),
    ("Orig + Scene-Desc ⇒ Gem-Pro + Rules", "rules_pro_nominal", "#805ad5"),
    ("Orig", "originals", "#48bb78"),
    ("Adv + Scene-Desc ⇒ Gem-Pro + Rules", "rules_v3_gemini_pro", "#2b6cb0"),
    ("Adv + Scene-Desc ⇒ Gem-Pro", "gemini_pro_bare", "#63b3ed"),
    ("Adv", "passthrough", "#a0aec0"),
]

rows = []
star = ""
for label, arm, color in BARS:
    if arm == "ORACLE":
        pt, star = oracle_per_task()
        label += star
    else:
        pt = arm_per_task(arm)
    rows.append((label, strata_from_per_task(pt) if pt else None, color))
    s = rows[-1][1]
    print(f"{label:42s} " + ("PENDING" if s is None else
          f"pooled {s['pooled']:4.1f}  in-vocab {s['iv']:4.1f}  OOV {s['oov']:4.1f}"))

rows.sort(key=lambda r: (r[1] is not None, r[1]["pooled"] if r[1] else -1), reverse=True)
fig, ax = plt.subplots(figsize=(10.2, 0.78 * len(rows) + 2.4))
y = list(range(len(rows)))[::-1]
h = 0.24
for yi, (label, s, color) in zip(y, rows):
    if s is None:
        ax.text(0.6, yi, "leg rolling — lands shortly", va="center",
                fontsize=9, style="italic", color="#718096")
        continue
    ax.barh(yi + h, s["pooled"], h, color=color)
    ax.barh(yi, s["iv"], h, color=color, alpha=0.55)
    ax.barh(yi - h, s["oov"], h, color=color, alpha=0.3, hatch="//", edgecolor=color, lw=0)
    ax.text(s["pooled"] + 0.5, yi + h, f"{s['pooled']:.1f}", va="center",
            fontsize=10, fontweight="bold")
    ax.text(s["iv"] + 0.5, yi, f"{s['iv']:.1f}", va="center", fontsize=7.5, color="#4a5568")
    ax.text(s["oov"] + 0.5, yi - h, f"{s['oov']:.1f}", va="center", fontsize=7.5, color="#4a5568")
ax.set_yticks(y)
ax.set_yticklabels([r[0] for r in rows], fontsize=11)
ax.legend(handles=[Patch(color="#4a5568", label="pooled (12 tasks)"),
                   Patch(color="#4a5568", alpha=0.55, label=f"in-vocab ({len(IV)} tasks)"),
                   Patch(facecolor="#4a5568", alpha=0.3, hatch="//", label=f"out-of-vocab ({len(OO)} tasks)")],
          loc="lower right", fontsize=8)
ax.set_xlabel("success % — 12 tasks × 24 layouts × 12 reps per task")
ax.set_title("How instruction phrasing moves π0 — 12 Bridge tasks in SIMPLER")
ax.grid(axis="x", alpha=0.25)
ax.set_xlim(0, 68)
key_lines = ["Key:",
             "  Adv = adversarial phrasing",
             "  Orig = original phrasing",
             "  Scene-Desc = Gemini-written scene description",
             "  Gem-Pro = Gemini-pro rewriter (reasoning on)",
             "  Rules = phrasing rules v3",
             "  every pipeline ends at π0"]
fig.text(0.01, 0.155, "\n".join(key_lines), fontsize=9, color="#4a5568", va="top")
if star:
    fig.text(0.01, 0.012, "* held-out estimate — full-grid leg rolling", fontsize=7, color="#718096")
fig.tight_layout(rect=[0, 0.17, 1, 1])
fig.savefig("results/charts/sealed_thin.png", dpi=150, bbox_inches="tight", pad_inches=0.25)
print("chart -> results/charts/sealed_thin.png")
