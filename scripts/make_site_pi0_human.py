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

base = b41["base_raw_human"]["pooled"]
aug = a39["raw_human"]["pooled"]
scaf = a39[f"sc|{AP}"]["pooled"]
rules = sum(a39[f"{b}|{AP}"]["pooled"] for b in ("s", "s2", "s3")) / 3
print(f"{AP}: base {base:.2f}  aug {aug:.2f}  scaffold {scaf:.2f}  rules(avg 3 draws) {rules:.2f}")

C_BASE, C_AUG, C_SCAF, C_RULES = "#6B5E9B", "#5E7B9B", "#B08A3E", "#3F6B52"
BARS = [("non-augmented $\\pi_0$\npass-through", base, C_BASE),
        ("augmented $\\pi_0$\npass-through", aug, C_AUG),
        ("augmented $\\pi_0$\n+ rephraser,\nno rules", scaf, C_SCAF),
        ("augmented $\\pi_0$\n+ rephraser\nwith rules $\\bf{(ours)}$", rules, C_RULES)]

fig, ax = plt.subplots(figsize=(4.4, 3.3))
for i, (lab, v, col) in enumerate(BARS):
    ax.bar(i, v, 0.62, color=col, edgecolor="none", zorder=2)
    ax.text(i, v - 1.2, f"{v:.1f}", ha="center", va="top", fontsize=8.6,
            fontweight="bold", color="white", zorder=4,
            bbox=dict(boxstyle="square,pad=0.10", fc=col, ec="none"))
ax.set_xticks(range(len(BARS)))
ax.set_xticklabels([b[0] for b in BARS], fontsize=7.0)
ax.set_ylabel("rollout success (%)", fontsize=8.0)
ax.set_ylim(0, 35)
ax.set_xlim(-0.6, len(BARS) - 0.4)
ax.grid(axis="y", alpha=0.18, zorder=0)
ax.tick_params(axis="x", length=0)
ax.spines[["top", "right"]].set_visible(False)
ax.text(0.99, 0.985, "human-written phrasings\n363 phrases, 12 tasks, 24 layouts each",
        transform=ax.transAxes, ha="right", va="top", fontsize=6.4, color="#4a5568")
fig.tight_layout()
out = R / "results/charts" / f"site_pi0_human_{AP}.png"
fig.savefig(out, dpi=450, bbox_inches="tight", pad_inches=0.15)
print("chart ->", out)
