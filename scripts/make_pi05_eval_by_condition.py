#!/usr/bin/env python3
"""Two figures for the pi0.5/LIBERO sealed eval, split by condition.

  results/charts/pi05_eval_effects_by_condition.png
      Paired effect (rewrite minus its OWN base, pp) for each rulebook, one ROW
      per condition x one COLUMN per stratum. Dot-and-interval on a zero line;
      4,000-draw bootstrap over tasks, the interval method the rulebooks' own
      rationale sections use.

  results/charts/pi05_eval_success_levels.png
      Absolute success rates on the same grid, plus the pooled-over-strata row.
      Pooling is shown because it was asked for and is easy to misread: the
      PREREG says a pooled mean is NOT a headline, because strata C and D cannot
      move and dilute anything A and B do.

Regenerate: .venv/bin/python scripts/make_pi05_eval_by_condition.py
"""
import glob
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

REPO = pathlib.Path(__file__).resolve().parents[1]
D = REPO / "results/analysis/pi05_bank"

res = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(
    str(REPO / "results/rules_runs/p_eval/jobs/ev*.result.parquet")))],
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

BOOKS = [("in_only_v1", "in-finetune only", "#2a78d6"),
         ("ood_only_v1", "out-of-finetune only", "#008300"),
         ("in_plus_ood_v2", "both", "#e87ba4"),
         ("none", "no rules (control)", "#8a8a85")]
STRATA = [("A_in_finetune", "A · in-finetune\n8 tasks"),
          ("B_ood_range", "B · OOD with range\n5 tasks"),
          ("C_ood_marginal", "C · OOD marginal\n5 tasks"),
          ("D_ood_floor", "D · OOD floor\n4 tasks")]
CONDS = ["original", "natural", "adversarial"]
INK, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#dedcd5", "#fcfcfb"

rng = np.random.default_rng(7)


def ci(vals):
    d = rng.choice(vals, size=(4000, len(vals)), replace=True).mean(1)
    return np.percentile(d, 2.5), np.percentile(d, 97.5)


def style(ax):
    ax.tick_params(labelsize=8, colors=MUTED, length=0)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_visible(False)


LEG = [Line2D([], [], marker="o", ls="", ms=8, mfc=c, mec=SURF, mew=1.5, label=l)
       for _, l, c in BOOKS]

# ============================ FIGURE 1: paired effect =========================
fig, axes = plt.subplots(3, 4, figsize=(14.2, 9.4), sharex="row")
for ri, cond in enumerate(CONDS):
    for ci_, (skey, slabel) in enumerate(STRATA):
        ax = axes[ri][ci_]
        g0 = A[(A.kind == cond) & (A.stratum == skey)]
        ax.axvline(0, color=MUTED, lw=1.2, zorder=1)
        for bi, (bkey, blabel, col) in enumerate(BOOKS):
            g = g0[g0.book == bkey]
            if g.empty:
                continue
            per = g.groupby("task").delta.mean()
            lo, hi = ci(per.values)
            y = len(BOOKS) - 1 - bi
            ax.plot([lo, hi], [y, y], color=col, lw=2, solid_capstyle="round", zorder=2)
            ax.plot([per.mean()], [y], "o", ms=8.5, color=col, mec=SURF, mew=2, zorder=3)
            ax.text(per.mean(), y + 0.33, f"{per.mean():+.1f}", ha="center",
                    va="bottom", fontsize=8, color=INK)
        ax.set_ylim(-0.7, len(BOOKS) - 0.1)
        ax.set_yticks(range(len(BOOKS)))
        ax.set_yticklabels([b[1] for b in BOOKS][::-1] if ci_ == 0 else [],
                           fontsize=8.5, color=INK)
        if ri == 0:
            ax.set_title(slabel, fontsize=9.5, color=INK, pad=8)
        ax.grid(axis="x", color=GRID, lw=0.7)
        style(ax)
    axes[ri][0].set_ylabel(cond.upper(), fontsize=10.5, color=INK,
                           fontweight="bold", labelpad=14)
    axes[ri][3].set_xlabel("effect vs own base (pp)", fontsize=8, color=MUTED)

fig.suptitle("pi0.5 / LIBERO sealed eval — rulebook effect, split by condition",
             fontsize=13.5, color=INK, y=0.975, fontweight="bold")
fig.text(0.5, 0.938, "Paired: each rewrite minus its own base. Bars = bootstrap "
         "95% CI over tasks. 8 of 12 arms (qwen pending).",
         ha="center", fontsize=9, color=MUTED)
fig.legend(handles=LEG, loc="lower center", ncol=4, frameon=False, fontsize=9,
           labelcolor=INK, bbox_to_anchor=(0.5, 0.018))
fig.subplots_adjust(left=0.135, right=0.99, top=0.895, bottom=0.085, hspace=0.34, wspace=0.12)
o1 = "results/charts/pi05_eval_effects_by_condition.png"
fig.savefig(o1, dpi=165, facecolor=SURF)
print("wrote", o1)

# ============================ FIGURE 2: absolute levels =======================
fig2, axes2 = plt.subplots(3, 5, figsize=(15.4, 9.0))
cols = STRATA + [("POOLED", "pooled over strata\n(not a headline)")]
for ri, cond in enumerate(CONDS):
    for ci_, (skey, slabel) in enumerate(cols):
        ax = axes2[ri][ci_]
        if skey == "POOLED":
            g0, b0 = A[A.kind == cond], bl[bl.kind == cond]
        else:
            g0 = A[(A.kind == cond) & (A.stratum == skey)]
            b0 = bl[(bl.kind == cond) & (bl.stratum == skey)]
        base = b0.base_succ.mean()
        ax.axvline(base, color=MUTED, lw=1.4, ls="--", zorder=1)
        ax.text(base, len(BOOKS) - 0.12, f"base {base:.0f}%", fontsize=7.5,
                color=MUTED, ha="center", va="bottom")
        for bi, (bkey, blabel, col) in enumerate(BOOKS):
            g = g0[g0.book == bkey]
            if g.empty:
                continue
            v = g.gt_success.mean()
            y = len(BOOKS) - 1 - bi
            ax.plot([v], [y], "o", ms=9, color=col, mec=SURF, mew=2, zorder=3)
            ax.text(v, y + 0.30, f"{v:.1f}", ha="center", va="bottom",
                    fontsize=8, color=INK)
        ax.set_xlim(-6, 106)
        ax.set_xticks([0, 50, 100])
        ax.set_ylim(-0.7, len(BOOKS) + 0.35)
        ax.set_yticks(range(len(BOOKS)))
        ax.set_yticklabels([b[1] for b in BOOKS][::-1] if ci_ == 0 else [],
                           fontsize=8.5, color=INK)
        if ri == 0:
            ax.set_title(slabel, fontsize=9.5, color=INK, pad=8)
        ax.grid(axis="x", color=GRID, lw=0.7)
        style(ax)
        if skey == "POOLED":
            ax.set_facecolor("#f4f3ee")
    axes2[ri][0].set_ylabel(cond.upper(), fontsize=10.5, color=INK,
                            fontweight="bold", labelpad=14)
    axes2[ri][4].set_xlabel("success rate (%)", fontsize=8, color=MUTED)

fig2.suptitle("pi0.5 / LIBERO sealed eval — absolute success rates",
              fontsize=13.5, color=INK, y=0.975, fontweight="bold")
fig2.text(0.5, 0.938, "Dashed line = the un-rephrased base for that cell. Strata "
          "A/B sit at ceiling, C/D at floor; the pooled column averages across "
          "both and is shaded as a caution.",
          ha="center", fontsize=9, color=MUTED)
fig2.legend(handles=LEG, loc="lower center", ncol=4, frameon=False, fontsize=9,
            labelcolor=INK, bbox_to_anchor=(0.5, 0.018))
fig2.subplots_adjust(left=0.125, right=0.99, top=0.895, bottom=0.085, hspace=0.34, wspace=0.12)
o2 = "results/charts/pi05_eval_success_levels.png"
fig2.savefig(o2, dpi=165, facecolor=SURF)
print("wrote", o2)
