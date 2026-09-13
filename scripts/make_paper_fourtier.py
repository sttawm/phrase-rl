#!/usr/bin/env python3
"""results/charts/paper_fourtier.png -- paper figure for the pi0.5/LIBERO
four-tier eval: phrasing tiers pooled by finetune membership (Wilson 95%),
bars ordered by height (oracle leftmost). Single panel, no in-image title.

  .venv/bin/python scripts/make_paper_fourtier.py
"""
import json
import math
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[1]
S = json.load(open(ROOT / "results/analysis/fourtier_summary.json"))

C = {"original": "#6B5E9B", "natural": "#B08A3E", "adversarial": "#A94E4E",
     "oracle": "#3F6B52", "reph": "#3F6B52"}


def wl(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0, c - h), min(1, c + h)


fig, a = plt.subplots(figsize=(7.0, 4.4))
plt.rcParams["font.family"] = "sans-serif"

# ---------- (a) tiers by finetune membership --------------------------------
groups = [("in-finetune tasks\n(libero_goal, 10)", "goal_in_finetune"),
          ("out-of-finetune tasks\n(libero_90, 18)", "l90_clean")]
tiers = ["oracle_confirm", "orig", "natural", "adversarial"]
tier_labels = ["oracle\n(searched)", "original\ninstruction", "natural\nrephrasings",
               "adversarial\nrephrasings"]
tier_colors = [C["oracle"], C["original"], C["natural"], C["adversarial"]]
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
a.text(3.0, 40.5, "$-$50 pp", fontsize=8.6, color="white",
       fontstyle="italic", ha="center", va="top", zorder=5)

fig.tight_layout()
out = ROOT / "results/charts/paper_fourtier.png"
fig.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0.22)
print("chart -> %s" % out)
