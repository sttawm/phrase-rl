#!/usr/bin/env python3
"""results/charts/swings_lollipop_thinned.png — the full-population dumbbell
chart thinned for the paper: multi-edit sets dropped, and at most one set per
(task, category) — the largest-gap one. LIBERO is untouched by the cap (every
significant set there is on a distinct task)."""
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

R = pathlib.Path(__file__).resolve().parents[1]
CAT_COLOR = {"source noun": "#4C72B0", "destination noun": "#4C72B0",
             "noun": "#4C72B0", "color/modifier": "#DD8452",
             "verb": "#55A868", "preposition": "#B8962E",
             "case/structure": "#8172B3"}


def thin(df):
    sets = (df.groupby("set_id")
              .agg(gap=("set_gap_pp", "first"), task=("task", "first"),
                   category=("category", "first")))
    sets = sets[sets.category != "multi-edit"]
    keep = (sets.sort_values("gap", ascending=False)
                .groupby(["task", "category"]).head(1).index)
    return df[df.set_id.isin(keep)]


def panel(ax, df, title):
    sets = (df.groupby("set_id")
              .agg(gap=("set_gap_pp", "first"), task=("task", "first"),
                   category=("category", "first"))
              .sort_values("gap", ascending=False))
    for i, (sid, srow) in enumerate(sets.iterrows()):
        y = len(sets) - 1 - i
        g = df[df.set_id == sid]
        c = CAT_COLOR[srow["category"]]
        ax.plot([g.succ_pct.min(), g.succ_pct.max()], [y, y], color=c,
                lw=2.2, alpha=0.55, zorder=1, solid_capstyle="round")
        for _, r in g.iterrows():
            if r.is_best:
                ax.plot(r.succ_pct, y, "o", ms=5.2, color=c, zorder=3)
            else:
                ax.plot(r.succ_pct, y, "o", ms=4.3, mfc="white", mec=c,
                        mew=1.1, zorder=2)
        ax.text(104, y, f"+{srow.gap:.0f}", va="center", fontsize=7,
                color="#4a5568")
    labels = [f"{r.task.replace('widowx_','').replace('_clean','')} · {r['category']}"
              for _, r in sets.iterrows()]
    ax.set_yticks(range(len(sets))[::-1])
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlim(-2, 113)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(["0", "25", "50", "75", "100%"], fontsize=8)
    ax.set_ylim(-0.7, len(sets) - 0.3)
    ax.grid(axis="x", alpha=0.25)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_title(title, fontsize=10)


B = thin(pd.read_csv(R / "results/analysis/swing_sets_bridge.csv"))
L = thin(pd.read_csv(R / "results/analysis/swing_sets_libero.csv"))
nb, nl = B.set_id.nunique(), L.set_id.nunique()
fig, (a1, a2) = plt.subplots(2, 1, figsize=(6.6, 0.172 * (nb + nl) + 1.9),
                             gridspec_kw={"height_ratios": [nb, nl]})
panel(a1, B, "$\\pi_0$ on SIMPLER Bridge --- %d significant sets (n = 72 per phrase)" % nb)
panel(a2, L, "$\\pi_{0.5}$ on LIBERO --- %d significant sets (n = 50 per phrase)" % nl)
handles = [plt.Line2D([0], [0], marker="o", ls="", color=c, ms=6)
           for c in ["#4C72B0", "#DD8452", "#55A868", "#B8962E", "#8172B3"]]
fig.legend(handles, ["Noun", "Color or modifier", "Verb",
                     "Preposition or particle", "Form, case, order"],
           fontsize=7.5, ncol=3, loc="lower center", frameon=False,
           bbox_to_anchor=(0.5, 0.0))
fig.tight_layout(rect=(0, 0.04, 1, 1))
out = R / "results/charts/swings_lollipop_thinned.png"
fig.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0.15)
print("chart ->", out, f"({nb} bridge + {nl} libero sets)")
