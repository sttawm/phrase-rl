#!/usr/bin/env python3
"""results/charts/paper_fourtier.png -- paper figure for the pi0.5/LIBERO
four-tier eval: phrasing tiers pooled by finetune membership (Wilson 95%),
bars ordered by height (oracle leftmost). Single panel, single-column
size, no error bars (kept simple to match the pi0 conditions chart), no
in-image title.

  .venv/bin/python scripts/make_paper_fourtier.py
"""
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[1]
S = json.load(open(ROOT / "results/analysis/fourtier_summary.json"))

C = {"original": "#6B5E9B", "natural": "#B08A3E", "adversarial": "#A94E4E",
     "oracle": "#3F6B52", "reph": "#3F6B52"}


fig, a = plt.subplots(figsize=(4.4, 3.3))
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
        p = v["rate"]
        a.bar(x, p * 100, W, color=color, zorder=3)
        a.text(x, p * 100 + 2.0, "%.0f" % (p * 100), ha="center", fontsize=8.5,
               fontweight="bold", color=color)
        gx.append(x)
        x += 1.0
    xt.append(sum(gx) / len(gx))
    xl.append(glabel)
    x += GAP
a.set_xticks(xt)
a.set_xticklabels(xl, fontsize=7.8)
a.set_ylabel("rollout success (%)", fontsize=8.5)
a.set_ylim(0, 108)
a.set_yticks(range(0, 101, 25))
a.spines["top"].set_visible(False)
a.spines["right"].set_visible(False)
a.grid(alpha=0.2, axis="y", zorder=0)
a.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for c in tier_colors],
         labels=[t.replace("\n", " ") for t in tier_labels], fontsize=6.6,
         loc="lower center", bbox_to_anchor=(0.5, 1.02), frameon=False, ncol=2,
         columnspacing=1.0, handlelength=1.1, handletextpad=0.5)
a.text(3.0, 40.5, "$-$50 pp", fontsize=5.8, color="white",
       fontstyle="italic", ha="center", va="top", zorder=5)

fig.tight_layout()
out = ROOT / "results/charts/paper_fourtier.png"
fig.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0.22)
print("chart -> %s" % out)
