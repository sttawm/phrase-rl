#!/usr/bin/env python3
"""Titleless, column-width (3.39in) vector dumbbell charts of ALL significant
swing sets (swing_sets_{bridge,libero}.csv) for the paper — captions live in
LaTeX, per paper chart style. Axis capped at 100%; the swing column sits
outside the axis under a small header, per the collaborator's layout."""
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

R = pathlib.Path(__file__).resolve().parents[1]
CAT_COLOR = {"source noun": "#4C72B0", "destination noun": "#4C72B0",
             "noun": "#4C72B0", "color/modifier": "#DD8452",
             "verb": "#55A868", "preposition": "#B8962E",
             "case/structure": "#8172B3", "multi-edit": "#8C8C8C"}
LEGEND = [("Noun", "#4C72B0"), ("Color or modifier", "#DD8452"),
          ("Verb", "#55A868"), ("Preposition", "#B8962E"),
          ("Form, case, order", "#8172B3"), ("Multi-edit", "#8C8C8C")]


def chart(df, out):
    sets = (df.groupby("set_id")
              .agg(gap=("set_gap_pp", "first"), task=("task", "first"),
                   category=("category", "first"))
              .sort_values("gap", ascending=False))
    fig, ax = plt.subplots(figsize=(3.39, 0.118 * len(sets) + 0.72))
    for i, (sid, srow) in enumerate(sets.iterrows()):
        y = len(sets) - 1 - i
        g = df[df.set_id == sid]
        c = CAT_COLOR[srow["category"]]
        ax.plot([g.succ_pct.min(), g.succ_pct.max()], [y, y], color=c,
                lw=1.7, alpha=0.55, zorder=1, solid_capstyle="round")
        for _, r in g.iterrows():
            if r.is_best:
                ax.plot(r.succ_pct, y, "o", ms=3.9, color=c, zorder=3)
            else:
                ax.plot(r.succ_pct, y, "o", ms=3.2, mfc="white", mec=c,
                        mew=0.9, zorder=2)
        ax.text(104, y, f"+{srow.gap:.0f}", va="center", fontsize=5.8,
                color="#4a5568", clip_on=False)
    ax.text(104, len(sets) - 0.05, "swing", va="bottom", fontsize=5.8,
            style="italic", color="#4a5568", clip_on=False)
    labels = [f"{r.task.replace('widowx_','').replace('_clean','')} · {r['category']}"
              for _, r in sets.iterrows()]
    ax.set_yticks(range(len(sets))[::-1])
    ax.set_yticklabels(labels, fontsize=5.8, family="monospace")
    ax.set_xlim(-1.5, 101.5)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(["0", "25", "50", "75", "100%"], fontsize=6.5)
    ax.set_ylim(-0.7, len(sets) - 0.3)
    ax.grid(axis="x", alpha=0.25, lw=0.5)
    ax.spines[["bottom", "right", "left"]].set_visible(False)
    ax.spines["top"].set_linewidth(0.6)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", width=0.6, length=2.5, top=True, bottom=False,
                   labeltop=True, labelbottom=False)
    cats = set(sets.category)
    handles, names = zip(*[(plt.Line2D([0], [0], marker="o", ls="", color=c,
                                       ms=3.6), n)
                           for n, c in LEGEND
                           if any(CAT_COLOR[k] == c for k in cats)])
    handles += (plt.Line2D([0], [0], marker="o", ls="", color="#555555",
                           ms=3.6),
                plt.Line2D([0], [0], marker="o", ls="", mfc="white",
                           mec="#555555", mew=0.9, ms=3.2))
    names += ("best phrase", "other phrases")
    fig.legend(handles, names, fontsize=5.4, ncol=3, loc="lower center",
               frameon=False, borderaxespad=0.1, columnspacing=1.1,
               handletextpad=0.25, bbox_to_anchor=(0.55, 0.0))
    fig.tight_layout(rect=(0, 0.055, 1, 1), pad=0.3)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"{out.name}: {len(sets)} sets / {sets.task.nunique()} tasks, "
          f"gaps {sets.gap.min():.0f}-{sets.gap.max():.0f}")


B = pd.read_csv(R / "results/analysis/swing_sets_bridge.csv")
L = pd.read_csv(R / "results/analysis/swing_sets_libero.csv")
C = R / "results/charts"
chart(B, C / "swings_bridge_full.pdf")
chart(L, C / "swings_libero_full.pdf")
