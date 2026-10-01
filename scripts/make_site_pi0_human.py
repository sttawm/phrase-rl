#!/usr/bin/env python3
"""Project-site chart: what each ingredient buys on HUMAN-written phrasings
(363 phrases over the 12 sealed Bridge tasks, 24 fixed layouts each):

  non-augmented pi0, pass-through          b41_base_cells.json  base_raw_human
  rephrase-augmented pi0, pass-through     a39_human_cells.json raw_human
  augmented pi0 + rephraser, no rules      a39_human_cells.json sc|<applier>
  augmented pi0 + rephraser, rules         a39_human_cells.json s|,s2|,s3|<applier>
                                           (out-of-finetune rulebooks, averaged
                                           over the three draws, as in the paper)

  APPLIER=claude .venv/bin/python scripts/make_site_pi0_human.py
No in-image title (the site text carries the explanation). Palette = the
paper's pi0 conditions chart (make_pi0_conditions.py): canonical purple,
human-naturals blue (same number as there), naturals gold, rulebook green.
"""
import json
import os
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = pathlib.Path(__file__).resolve().parents[1]
AP = os.environ.get("APPLIER", "claude")
b41 = json.loads((R / "results/analysis/b41_base_cells.json").read_text())
a39 = json.loads((R / "results/analysis/a39_human_cells.json").read_text())
assert b41["aug_raw_human"]["pooled"] == a39["raw_human"]["pooled"]

def tri(rec):
    return (rec["pooled"], rec["iv"], rec["oov"])


base = tri(b41["base_raw_human"])
aug = tri(a39["raw_human"])
scaf = tri(a39[f"sc|{AP}"])
rules = tuple(sum(a39[f"{b}|{AP}"][k] for b in ("s", "s2", "s3")) / 3
              for k in ("pooled", "iv", "oov"))
for n, v in (("base", base), ("aug", aug), ("scaffold", scaf), ("rules avg3", rules)):
    print(f"{AP} {n:10s} pooled {v[0]:.2f}  iv {v[1]:.2f}  oov {v[2]:.2f}")

C_BASE, C_AUG, C_SCAF, C_RULES = "#6B5E9B", "#5E7B9B", "#B08A3E", "#3F6B52"
BARS = [("non-augmented $\\pi_0$\npass-through", base, C_BASE),
        ("rephrase-augmented $\\pi_0$\npass-through", aug, C_AUG),
        ("rephrase-augmented $\\pi_0$\n+ rephraser,\nno rules", scaf, C_SCAF),
        ("rephrase-augmented $\\pi_0$\n+ rephraser\nwith rules $\\bf{(ours)}$", rules, C_RULES)]

fig, ax = plt.subplots(figsize=(6.0, 3.3))
SP = 1.6    # cluster spacing: four long labels need more room than the paper chart


def cluster(x, pool, iv, oov, col):
    # make_pi0_conditions.py idiom: pooled bar solid in front, in-distribution
    # (left) / out-of-distribution (right) flanks behind, thinner, translucent
    ax.bar(x - 0.20, iv, 0.34, color=col, alpha=0.40, zorder=1, edgecolor="none")
    ax.bar(x + 0.20, oov, 0.34, color=col, alpha=0.40, zorder=1, edgecolor="none")
    ax.text(x - 0.32, iv + 0.9, f"{iv:.0f}", ha="center", fontsize=5.6,
            color="#4a5568", zorder=3)
    ax.text(x + 0.32, oov + 0.9, f"{oov:.0f}", ha="center", fontsize=5.6,
            color="#4a5568", zorder=3)
    ax.bar(x, pool, 0.44, color=col, zorder=2, edgecolor="none")
    ax.text(x, pool - 1.4, f"{pool:.1f}", ha="center", va="top", fontsize=7.8,
            fontweight="bold", color="white", zorder=4,
            bbox=dict(boxstyle="square,pad=0.10", fc=col, ec="none"))


for i, (lab, (pool, iv, oov), col) in enumerate(BARS):
    cluster(i * SP, pool, iv, oov, col)
ax.set_xticks([i * SP for i in range(len(BARS))])
ax.set_xticklabels([b[0] for b in BARS], fontsize=6.8)
ax.set_ylabel("rollout success (%)", fontsize=8.0)
ax.set_ylim(0, 45)
ax.set_xlim(-0.62, (len(BARS) - 1) * SP + 0.62)
ax.grid(axis="y", alpha=0.18, zorder=0)
ax.tick_params(axis="x", length=0)
ax.spines[["top", "right"]].set_visible(False)
handles = [plt.Rectangle((0, 0), 1, 1, fc="#7A8698"),
           plt.Rectangle((0, 0), 1, 1, fc="#7A8698", alpha=0.40)]
ax.legend(handles, ["pooled (12 tasks)",
                    "flanks: in-distrib. (left, 5) / out-of-distrib. (right, 7)"],
          fontsize=5.9, loc="upper left", frameon=False)
ax.text(0.99, 0.985, "human-written phrasings\n363 phrases, 24 layouts each",
        transform=ax.transAxes, ha="right", va="top", fontsize=6.4, color="#4a5568")
ax.legend_.set_bbox_to_anchor((0.0, 1.0))   # legend hugs the left; annotation the right
fig.tight_layout()
out = R / "results/charts" / f"site_pi0_human_{AP}.png"
fig.savefig(out, dpi=450, bbox_inches="tight", pad_inches=0.15)
print("chart ->", out)
