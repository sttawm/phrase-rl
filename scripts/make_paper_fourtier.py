#!/usr/bin/env python3
"""results/charts/paper_fourtier.png -- paper figure for the pi0.5/LIBERO
four-tier eval. Panel (a): phrasing tiers pooled by finetune membership
(Wilson 95%). Panel (b): the over-finetuning exhibit -- each finetuned
instruction string shown in its training scene, in a novel scene, and next to
rephrasings evaluated in that same novel scene.

  .venv/bin/python scripts/make_paper_fourtier.py
"""
import glob
import json
import math
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
rows = []
for f in glob.glob(str(ROOT / "results/analysis/fourtier_live/fourtier_*.jsonl")):
    for line in open(f):
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
df = pd.DataFrame(rows).drop_duplicates(subset=["suite", "task_id", "phrase", "init"], keep="last")
S = json.load(open(ROOT / "results/analysis/fourtier_summary.json"))

C = {"original": "#6B5E9B", "natural": "#B08A3E", "adversarial": "#A94E4E",
     "oracle": "#3F6B52", "reph": "#3F6B52"}


def wl(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0, c - h), min(1, c + h)


fig, (a, b) = plt.subplots(1, 2, figsize=(12.6, 4.4), gridspec_kw={"width_ratios": [1.15, 1.0]})
plt.rcParams["font.family"] = "sans-serif"

# ---------- (a) tiers by finetune membership --------------------------------
groups = [("in-finetune tasks\n(libero_goal, 10)", "goal_in_finetune"),
          ("out-of-finetune tasks\n(libero_90, 18)", "l90_clean")]
tiers = ["orig", "natural", "adversarial", "oracle_confirm"]
tier_labels = ["original\ninstruction", "natural\nrephrasings", "adversarial\nrephrasings",
               "oracle\n(searched)"]
tier_colors = [C["original"], C["natural"], C["adversarial"], C["oracle"]]
W, GAP = 0.8, 1.1
x = 0.0
xt, xl = [], []
for glabel, gkey in groups:
    M = S["macro"][gkey]
    gx = []
    for tier, color in zip(tiers, tier_colors):
        v = M.get(tier)
        if not v:
            continue
        p, lo, hi = v["rate"], v["lo"], v["hi"]
        a.bar(x, p * 100, W, color=color, zorder=3)
        a.errorbar(x, p * 100, yerr=[[100 * max(0, p - lo)], [100 * max(0, hi - p)]],
                   color="#1a202c", lw=1.1, capsize=2.5, zorder=4)
        a.text(x, p * 100 + 4.5, "%.0f" % (p * 100), ha="center", fontsize=10,
               fontweight="bold", color=color)
        a.text(x, 2.5, "n=%d" % v["n"], ha="center", fontsize=6.8, color="white",
               rotation=90, va="bottom", zorder=5)
        gx.append(x)
        x += 1.0
    xt.append(sum(gx) / len(gx))
    xl.append(glabel)
    x += GAP
a.set_xticks(xt)
a.set_xticklabels(xl, fontsize=10)
a.set_ylabel("rollout success (%)", fontsize=11)
a.set_ylim(0, 112)
a.set_yticks(range(0, 101, 25))
a.spines["top"].set_visible(False)
a.spines["right"].set_visible(False)
a.grid(alpha=0.2, axis="y", zorder=0)
a.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for c in tier_colors],
         labels=[t.replace("\n", " ") for t in tier_labels], fontsize=8.3,
         loc="lower center", bbox_to_anchor=(0.5, 1.055), frameon=False, ncol=4,
         columnspacing=1.1, handlelength=1.2, handletextpad=0.5)
a.set_title("(a)", loc="left", fontsize=11.5, pad=30)
a.text(2.0, 40.5, "$-$50 pp", fontsize=8.6, color="white",
       fontstyle="italic", ha="center", va="top", zorder=5)

# ---------- (b) the over-finetuning exhibit ---------------------------------
g7 = df[(df.suite == "libero_goal") & (df.task_id == 7) & (df.phrase == "turn on the stove")]
l44 = df[(df.suite == "libero_90") & (df.task_id == 44)]
l44c = l44[l44.phrase == "turn on the stove"]
l44r = l44[(l44.arm.str.startswith("nat")) | (l44.arm.str.startswith("adv"))]
BOOK = "pick up the book and place it in the back compartment of the caddy"
l10 = df[(df.suite == "libero_10") & (df.phrase == BOOK)]
l77 = df[(df.suite == "libero_90") & (df.task_id == 77)]
l77c = l77[l77.phrase == BOOK]
l77r = l77[(l77.arm.str.startswith("nat")) | (l77.arm.str.startswith("adv"))]

pairs = [('finetuned string 1:\n"turn on the stove"', g7, l44c, l44r),
         ('finetuned string 2:\n"pick up the book and place it\nin the back compartment ..."', l10, l77c, l77r)]
x = 0.0
xt, xl = [], []
for label, tr, nv, rp in pairs:
    cells = [(tr, C["original"], "trained\nscene", False),
             (nv, C["original"], "novel\nscene", True),
             (rp, C["reph"], "rephrasings,\nnovel scene", False)]
    gx = []
    for gdat, color, clabel, hatch in cells:
        k, n = int(gdat.success.sum()), len(gdat)
        p, lo, hi = wl(k, n)
        a2 = b.bar(x, p * 100, W, color=color, zorder=3,
                   hatch="///" if hatch else None,
                   edgecolor="white" if hatch else color, lw=0.0)
        b.errorbar(x, p * 100, yerr=[[100 * max(0, p - lo)], [100 * max(0, hi - p)]],
                   color="#1a202c", lw=1.1, capsize=2.5, zorder=4)
        b.text(x, max(p, hi) * 100 + 3.5, "%.0f" % (p * 100), ha="center", fontsize=10,
               fontweight="bold", color=color)
        b.text(x, -7, clabel + "\nn=%d" % n, ha="center", va="top", fontsize=8.2,
               color="#374151")
        gx.append(x)
        x += 1.0
    xt.append(sum(gx) / len(gx))
    xl.append(label)
    x += GAP
for xc, lab in zip(xt, xl):
    b.text(xc, -33, lab, ha="center", va="top", fontsize=9, color="#111827")
b.set_xticks([])
b.set_ylim(0, 112)
b.set_yticks(range(0, 101, 25))
b.set_ylabel("rollout success (%)", fontsize=11)
b.spines["top"].set_visible(False)
b.spines["right"].set_visible(False)
b.grid(alpha=0.2, axis="y", zorder=0)
b.set_title("(b)", loc="left", fontsize=11.5, pad=30)

fig.tight_layout()
out = ROOT / "results/charts/paper_fourtier.png"
fig.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0.22)
print("chart -> %s" % out)
