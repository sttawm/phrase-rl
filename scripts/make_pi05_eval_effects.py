#!/usr/bin/env python3
"""results/charts/pi05_eval_effects.png -- pi0.5/LIBERO sealed eval, rulebook effects.

Top: paired effect of each rulebook (rewrite minus its OWN base, pp) per stratum,
with a 4,000-draw bootstrap 95% interval over tasks -- the same interval method
the rulebooks' own rationale sections use. Dot-and-interval on a zero line, not
bars: the question is whether anything departs from zero, not magnitude from an
origin.

Bottom: the share of bases each arm actually rewrote. Without it the top panel is
unreadable -- an effect of zero from a book that edited nothing is a different
statement from an effect of zero from a book that rewrote everything.
"""
import glob
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

REPO = pathlib.Path(__file__).resolve().parents[1]
D = REPO / "results/analysis/pi05_bank"

res = pd.concat([pd.read_parquet(f) for f in
                 sorted(glob.glob(str(REPO / "results/rules_runs/p_eval/jobs/ev*.result.parquet")))],
                ignore_index=True).drop_duplicates(["task", "phrase"])
bases = pd.read_parquet(D / "eval_bases.parquet")
bl = (bases.merge(res[["task", "phrase", "gt_success"]], on=["task", "phrase"], how="left")
      .rename(columns={"gt_success": "base_succ"}))
A = pd.concat([pd.read_parquet(f).merge(
    res[["task", "phrase", "gt_success"]], left_on=["task", "rewrite"],
    right_on=["task", "phrase"], how="left", suffixes=("", "_r"))
    for f in sorted(glob.glob(str(D / "eval_applies/*.parquet")))], ignore_index=True)
A = A.merge(bl[["task", "phrase", "base_succ"]], on=["task", "phrase"], how="left")
A["delta"] = A.gt_success - A.base_succ
A["changed"] = A.rewrite.str.strip().str.lower() != A.phrase.str.strip().str.lower()

# bootstrap over TASKS, 4,000 draws, 2.5/97.5 -- the interval method the
# rulebooks' own rationale sections use. Seeded so the figure is reproducible.
rng = np.random.default_rng(7)
E = pd.DataFrame([
    dict(book=bk, stratum=st, mean=(per := g.groupby("task").delta.mean()).mean(),
         lo=np.percentile(rng.choice(per.values, size=(4000, len(per)), replace=True).mean(1), 2.5),
         hi=np.percentile(rng.choice(per.values, size=(4000, len(per)), replace=True).mean(1), 97.5))
    for (bk, st), g in A.groupby(["book", "stratum"])])

# "no rules" is the control -> neutral ink, so colour is reserved for the three
# books being compared. Slots 1-3 of the validated categorical order.
BOOKS = [("in_only_v1", "in-finetune only", "#2a78d6"),
         ("ood_only_v1", "out-of-finetune only", "#008300"),
         ("in_plus_ood_v2", "both", "#e87ba4"),
         ("none", "no rules (control)", "#8a8a85")]
STRATA = [("A_in_finetune", "A · in-finetune\n8 tasks · base 100%"),
          ("B_ood_range", "B · OOD with range\n5 tasks · base 94%"),
          ("C_ood_marginal", "C · OOD marginal\n5 tasks · base 4%"),
          ("D_ood_floor", "D · OOD floor\n4 tasks · base 0%")]

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#dedcd5"
fig = plt.figure(figsize=(13.6, 7.4))
gs = fig.add_gridspec(2, 4, height_ratios=[2.4, 1.0], hspace=0.30, wspace=0.14)

# ---- top: effect per stratum -------------------------------------------------
lim = 5.2
for si, (skey, slabel) in enumerate(STRATA):
    ax = fig.add_subplot(gs[0, si])
    ax.axvline(0, color=MUTED, lw=1.2, zorder=1)
    for bi, (bkey, blabel, col) in enumerate(BOOKS):
        r = E[(E.book == bkey) & (E.stratum == skey)]
        if r.empty:
            continue
        r = r.iloc[0]
        y = len(BOOKS) - 1 - bi
        ax.plot([r.lo, r.hi], [y, y], color=col, lw=2, solid_capstyle="round", zorder=2)
        ax.plot([r["mean"]], [y], "o", ms=9, color=col, mec="#fcfcfb", mew=2, zorder=3)
        # direct label on every point: the required secondary encoding, and the
        # relief rule for the low-contrast slots
        off = 0.34
        ax.text(r["mean"], y + off, f"{r['mean']:+.2f}", ha="center", va="bottom",
                fontsize=8.5, color=INK)
    ax.set_ylim(-0.75, len(BOOKS) - 0.15)
    ax.set_xlim(-lim, lim)
    ax.set_yticks(range(len(BOOKS)))
    ax.set_yticklabels([b[1] for b in BOOKS][::-1] if si == 0 else [],
                       fontsize=9, color=INK)
    ax.set_title(slabel, fontsize=9.5, color=INK, pad=9)
    ax.tick_params(axis="x", labelsize=8, colors=MUTED, length=0)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_visible(False)
    if si == 0:
        ax.set_xlabel("effect vs own base (pp)", fontsize=8.5, color=MUTED)

# ---- bottom: did the book actually edit anything? ---------------------------
ax2 = fig.add_subplot(gs[1, :])
ch = (A.groupby(["book", "applier"]).changed.mean().mul(100)).unstack()
x = np.arange(len(BOOKS))
w = 0.34
for ai, (app, hatch) in enumerate([("claude", None), ("gemini", "////")]):
    vals = [ch.loc[b[0], app] if b[0] in ch.index else 0 for b in BOOKS]
    ax2.bar(x + (ai - 0.5) * (w + 0.02), vals, w, label=app,
            color=[b[2] for b in BOOKS], edgecolor="#fcfcfb", linewidth=2,
            hatch=hatch, alpha=1.0 if ai == 0 else 0.55, zorder=2)
    for xi, v in zip(x + (ai - 0.5) * (w + 0.02), vals):
        ax2.text(xi, v + 3, f"{v:.0f}%", ha="center", fontsize=8.5, color=INK)
ax2.set_xticks(x)
ax2.set_xticklabels([b[1] for b in BOOKS], fontsize=9, color=INK)
ax2.set_ylim(0, 118)
ax2.set_yticks([0, 50, 100])
ax2.set_ylabel("bases rewritten", fontsize=8.5, color=MUTED)
ax2.tick_params(labelsize=8, colors=MUTED, length=0)
ax2.grid(axis="y", color=GRID, lw=0.7)
ax2.set_axisbelow(True)
for s in ax2.spines.values():
    s.set_visible(False)
ax2.legend(handles=[Patch(facecolor="#6f6e6a", edgecolor="#fcfcfb", label="claude"),
                    Patch(facecolor="#c4c3bd", edgecolor="#fcfcfb", hatch="////",
                          label="gemini")],
           loc="upper left", frameon=False, fontsize=8.5, ncol=2,
           labelcolor=INK, handletextpad=0.5, columnspacing=1.4)

fig.suptitle("pi0.5 / LIBERO sealed eval — no rulebook moved the policy",
             fontsize=13.5, color=INK, x=0.5, y=0.98, fontweight="bold")
fig.text(0.5, 0.925,
         "Paired effect vs each rewrite's own base, 22 tasks x 20 inits (30-49), "
         "8 of 12 arms (qwen pending). Bars = bootstrap 95% CI over tasks.",
         ha="center", fontsize=9, color=MUTED)
fig.text(0.5, 0.055,
         "Every interval spans zero except out-of-finetune-only on stratum B "
         "(-0.81pp, 5 tasks). 0 of 1,120 rewrites collapsed 40pp or more below their base.",
         ha="center", fontsize=8.5, color=MUTED)
fig.text(0.5, 0.021,
         "Strata A/B sit at ceiling and C/D at floor, so this set has no band in which "
         "an effect could show. in-finetune-only is flat because it barely edits.",
         ha="center", fontsize=8.5, color=MUTED)
fig.subplots_adjust(left=0.145, right=0.985, top=0.845, bottom=0.155)
out = "results/charts/pi05_eval_effects.png"
fig.savefig(out, dpi=170, facecolor="#fcfcfb")
print("wrote", out)
