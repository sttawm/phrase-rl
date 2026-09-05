#!/usr/bin/env python3
"""results/charts/pi05_dashboard.png -- the rules_dashboard idiom, for pi0.5/LIBERO.

Same page shape as make_rules_dashboard.py (bridge): three condition columns,
grey = no-rules rephraser, blue = rulebook cell, red dashed = no rephraser at
all, value labels on every bar, dotted ghosts for cells that have not run.

Row A (slice by APPLIER): applier clusters x rule diets, pooled over strata.
Row B (slice by STRATUM): bridge slices row 2 by rulebook DRAW (r1/r2/r3);
pi0.5 has only one draw per diet, so the dimension that actually varies here is
the canonical-success stratum -- and it is the one that decides whether a cell
could move at all. Same idiom, substituted axis.

qwen never ran (the pod-side apply branch reads bridge traces only), so its
cluster is drawn as ghosts.

Regenerate: .venv/bin/python scripts/make_pi05_dashboard.py
"""
import glob
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

R = pathlib.Path(__file__).resolve().parents[1]
D = R / "results/analysis/pi05_bank"

res = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(
    str(R / "results/rules_runs/p_eval/jobs/ev*.result.parquet")))],
    ignore_index=True).drop_duplicates(["task", "phrase"])
bases = pd.read_parquet(D / "eval_bases.parquet")
bl = (bases.merge(res[["task", "phrase", "gt_success"]], on=["task", "phrase"], how="left")
      .rename(columns={"gt_success": "base_succ"}))
A = pd.concat([pd.read_parquet(f).merge(
    res[["task", "phrase", "gt_success"]], left_on=["task", "rewrite"],
    right_on=["task", "phrase"], how="left", suffixes=("", "_r"))
    for f in sorted(glob.glob(str(D / "eval_applies/*.parquet")))], ignore_index=True)

# diet order mirrors the bridge dashboard: control first, then the books
DIETS = [("none", "no\nrules"), ("ood_only_v1", "OOD\nonly"),
         ("in_plus_ood_v2", "in +\nOOD"), ("in_only_v1", "in-ft\nonly")]
APPS = [("claude", "Claude"), ("gemini", "Gemini"), ("qwen", "Qwen")]
STRATA = [("A_in_finetune", "A · in-ft\n100%"), ("B_ood_range", "B · range\n94%"),
          ("C_ood_marginal", "C · marginal\n4%"), ("D_ood_floor", "D · floor\n0%")]
CONDS = [("natural", "Natural"), ("adversarial", "Adversarial"), ("original", "Original")]

GREY, BLUE, RED = "#8a8a85", "#a8c4ea", "#cc3b34"
INK, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#dedcd5", "#fcfcfb"

fig, axes = plt.subplots(2, 3, figsize=(19.0, 12.4))


def bars(ax, groups, getter, gap=1.6, rot=0, lab_fs=8.2, tick_fs=7.6, tick_rot=0):
    """groups: list of (group_label, [(diet_key, diet_label)]). getter -> value|None."""
    xs, ticks, tlab, gcent = [], [], [], []
    x = 0.0
    for glabel, diets in groups:
        start = x
        for dk, dl in diets:
            v = getter(glabel, dk)
            col = GREY if dk == "none" else BLUE
            if v is None:
                ax.bar(x, 100, 0.82, facecolor="none", edgecolor="#c9c8c2",
                       lw=0.9, ls=":", zorder=2)
                ax.text(x, 0.5, "not\nrun", transform=ax.get_xaxis_transform(),
                        ha="center", va="center", fontsize=7.5,
                        color="#b6b5af", rotation=90)
            else:
                ax.bar(x, v, 0.82, color=col, edgecolor=SURF, lw=1.4, zorder=2)
                ax.text(x, v + 1.6, f"{v:.1f}", ha="center",
                        va="bottom" if rot == 0 else "bottom",
                        rotation=rot, fontsize=lab_fs, color=INK, fontweight="bold")
            ticks.append(x); tlab.append(dl); x += 1.0
        gcent.append((start + x - 1.0) / 2)
        x += gap
    ax.set_xticks(ticks)
    ax.set_xticklabels(tlab, fontsize=tick_fs, color=MUTED, rotation=tick_rot,
                       ha="center" if tick_rot == 0 else "right")
    return gcent


for ci, (ckey, clabel) in enumerate(CONDS):
    base = bl[bl.kind == ckey].base_succ.mean()

    # ---- Row A: by applier (pooled over strata) ----
    ax = axes[0][ci]
    sub = A[A.kind == ckey]

    def g_app(app_label, dk, sub=sub):
        akey = {v: k for k, v in APPS}[app_label]
        g = sub[(sub.applier == akey) & (sub.book == dk)]
        return None if g.empty else g.gt_success.mean()

    cent = bars(ax, [(al, DIETS) for _, al in APPS], g_app)
    ax.axhline(base, color=RED, ls="--", lw=1.6, zorder=3)
    ax.text(ax.get_xlim()[1], base + 1.0, "no rephraser", color=RED, fontsize=8,
            ha="right", va="bottom")
    for c, (_, al) in zip(cent, APPS):
        ax.text(c, -0.135, al, transform=ax.get_xaxis_transform(), ha="center",
                va="top", fontsize=11, color=INK, fontweight="bold")
    ax.set_title(clabel, fontsize=13, color=INK, pad=12)
    if ci == 0:
        ax.set_ylabel("success %   (slice by applier)", fontsize=10, color=MUTED)

    # ---- Row B: by stratum (pooled over appliers) ----
    ax2 = axes[1][ci]

    def g_str(s_label, dk, sub=sub):
        skey = {v: k for k, v in STRATA}[s_label]
        g = sub[(sub.stratum == skey) & (sub.book == dk)]
        return None if g.empty else g.gt_success.mean()

    SHORT = [(k, {"none": "none", "ood_only_v1": "OOD",
                  "in_plus_ood_v2": "both", "in_only_v1": "in-ft"}[k]) for k, _ in DIETS]
    cent2 = bars(ax2, [(sl, SHORT) for _, sl in STRATA], g_str,
                 rot=90, lab_fs=7.6, tick_fs=8.0, tick_rot=90)
    for c, (skey, sl) in zip(cent2, STRATA):
        b = bl[(bl.kind == ckey) & (bl.stratum == skey)].base_succ.mean()
        ax2.plot([c - 2.0, c + 2.0], [b, b], color=RED, ls="--", lw=1.6, zorder=3)
        ax2.text(c, -0.20, sl.split(" · ")[1].split("\n")[0] + f"\nbase {b:.0f}%",
                 transform=ax2.get_xaxis_transform(), ha="center", va="top",
                 fontsize=9.5, color=INK, fontweight="bold")
    if ci == 0:
        ax2.set_ylabel("success %   (slice by stratum)", fontsize=10, color=MUTED)

for ri, row in enumerate(axes):
    for ci2, ax in enumerate(row):
        if ri == 0:   # zoom: these cells differ by ~1pp on a 0-100 axis otherwise
            b = bl[bl.kind == CONDS[ci2][0]].base_succ.mean()
            ax.set_ylim(max(0, b - 14), b + 14)
        else:
            ax.set_ylim(0, 112)
            ax.set_yticks([0, 25, 50, 75, 100])
for ax in axes.ravel():
    ax.tick_params(labelsize=8, colors=MUTED, length=0)
    ax.grid(axis="y", color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_visible(False)

fig.suptitle("pi0.5 / LIBERO rulebook dashboard — sealed 22 tasks, inits 30-49 "
             "(row 1: by applier, pooled over strata; row 2: by stratum, pooled over appliers)",
             fontsize=13.5, color=INK, y=0.975)
fig.legend(handles=[
    Patch(facecolor=GREY, edgecolor=SURF, label="no-rules rephraser"),
    Patch(facecolor=BLUE, edgecolor=SURF, label="rulebook cell"),
    Patch(facecolor="none", edgecolor="#c9c8c2", ls=":", label="not run (qwen)"),
    Line2D([], [], color=RED, ls="--", lw=1.6, label="no rephraser (un-rephrased base)")],
    loc="lower center", ncol=4, frameon=False, fontsize=10, labelcolor=INK,
    bbox_to_anchor=(0.5, 0.004))
fig.subplots_adjust(left=0.055, right=0.995, top=0.918, bottom=0.135, hspace=0.50, wspace=0.10)
out = "results/charts/pi05_dashboard.png"
fig.savefig(out, dpi=160, facecolor=SURF)
print("wrote", out)
