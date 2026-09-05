#!/usr/bin/env python3
"""results/charts/pi05_dashboard.png -- the pi0.5/LIBERO twin of rules_dashboard.png.

Deliberately IDENTICAL to make_rules_dashboard.py in layout, colour, spacing,
bar geometry and legend, so the two pages can be read without re-learning the
display. Only the data and the diet labels differ.

Row A (slice by APPLIER): per condition, applier clusters x rule diets; bar =
mean over rulebook draws, dots = the individual draws.
Row B (slice by RULEBOOK DRAW): per condition, diet clusters x draws; bar =
mean over the appliers with complete cells for that draw.

pi0.5 has ONE draw per diet (bridge has three: r1 = A31/A34, r2/r3 = A36
replicates), so r2/r3 ghost out exactly the way bridge's unfinished cells do.
The page then says "no replicates exist yet" in the same visual language rather
than by quietly substituting a different axis. qwen never ran -- the pod-side
apply branch reads bridge traces only -- so its cluster ghosts too.

Regenerate: .venv/bin/python scripts/make_pi05_dashboard.py
"""
import glob
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

R = pathlib.Path(__file__).resolve().parents[1]
D = R / "results/analysis/pi05_bank"

res = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(
    str(R / "results/rules_runs/p_eval/jobs/ev*.result.parquet")))],
    ignore_index=True).drop_duplicates(["task", "phrase"])
bases = pd.read_parquet(D / "eval_bases.parquet")
bl = bases.merge(res[["task", "phrase", "gt_success"]], on=["task", "phrase"], how="left")
A = pd.concat([pd.read_parquet(f).merge(
    res[["task", "phrase", "gt_success"]], left_on=["task", "rewrite"],
    right_on=["task", "phrase"], how="left", suffixes=("", "_r"))
    for f in sorted(glob.glob(str(D / "eval_applies/*.parquet")))], ignore_index=True)

APS = ["claude", "gemini", "qwen"]
KIND = {"nat": "natural", "adv": "adversarial", "orig": "original"}
BOOK = {"s": "ood_only_v1", "b": "in_plus_ood_v2", "t": "in_only_v1"}


def cell(cond, book, ap):
    g = A[(A.kind == KIND[cond]) & (A.book == book) & (A.applier == ap)]
    return None if g.empty else g.gt_success.mean()


def draws(cond, diet, ap):
    """{draw_label: value}. pi0.5 has r1 only; r2/r3 are not distilled yet."""
    v = cell(cond, BOOK[diet], ap)
    return {} if v is None else {"r1": v}


SCAFFOLD = {c: {ap: cell(c, "none", ap) for ap in APS} for c in KIND}
NO_REPH = {c: bl[bl.kind == KIND[c]].gt_success.mean() for c in KIND}

C_BASE, C_RULES, INK = "#8b96a5", "#a3bffa", "#2d3748"
DIETS = [("s", "OOD\nonly"), ("b", "in +\nOOD"), ("t", "in-ft\nonly")]
CONDS = [("nat", "Natural"), ("adv", "Adversarial"), ("orig", "Original")]
DMARK = {"r1": "o", "r2": "^", "r3": "s"}
YLIM = (50, 80)
GHOST_H = YLIM[1] - YLIM[0] - 16

fig, axes = plt.subplots(2, 3, figsize=(16.4, 9.2), sharey=True)


def ghost(ax, x, label):
    ax.bar(x, GHOST_H, 0.62, bottom=YLIM[0], color="none", edgecolor="#cbd5e0",
           lw=0.8, ls=":")
    ax.text(x, YLIM[0] + 3.5, label, ha="center", fontsize=5.6, color="#a0aec0",
            rotation=90)


# ---------- Row A: slice by applier ----------
for ax, (cond, cname) in zip(axes[0], CONDS):
    x = 0.0
    ticks, tlabels, gticks = [], [], []
    for ap in APS:
        gxs = []
        sc = (SCAFFOLD[cond] or {}).get(ap)
        if sc is not None:
            ax.bar(x, sc, 0.62, color=C_BASE, edgecolor="#4a5568", lw=0.8)
            ax.text(x, sc + 0.4, f"{sc:.1f}", ha="center", fontsize=7.6,
                    fontweight="bold")
        else:
            ghost(ax, x, "not run")
        ticks.append(x); tlabels.append("no\nrules"); gxs.append(x); x += 0.82
        for diet, dlabel in DIETS:
            dv = draws(cond, diet, ap)
            if not dv:
                ghost(ax, x, "not run")
            else:
                m = sum(dv.values()) / len(dv)
                ax.bar(x, m, 0.62, color=C_RULES, edgecolor="#4a5568", lw=0.8)
                ax.text(x, m + 0.4, f"{m:.1f}", ha="center", fontsize=7.6,
                        fontweight="bold")
                for lb, v in dv.items():
                    ax.plot([x], [v], DMARK[lb], ms=3.6, color=INK, zorder=5,
                            mfc="white", mew=1.0)
            ticks.append(x); tlabels.append(dlabel); gxs.append(x); x += 0.82
        gticks.append(sum(gxs) / len(gxs))
        ax.axvline(x - 0.31, color="#e2e8f0", lw=1.0, zorder=0)
        x += 0.42
    ax.axhline(NO_REPH[cond], color="#c53030", lw=1.2, ls=(0, (4, 3)), zorder=1)
    ax.text(x - 0.75, NO_REPH[cond] + 0.3, "no rephraser", fontsize=6.6,
            color="#c53030", ha="right")
    ax.set_xticks(ticks); ax.set_xticklabels(tlabels, fontsize=6.0, color="#4a5568")
    for gx, ap in zip(gticks, ["Claude", "Gemini", "Qwen"]):
        ax.text(gx, YLIM[0] - 3.6, ap, ha="center", fontsize=10, fontweight="bold",
                clip_on=False)
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.6, x - 0.55)
    ax.grid(axis="y", alpha=0.16)
    ax.set_title(cname, fontsize=13, pad=8)
    ax.spines[["top", "right"]].set_visible(False)
axes[0][0].set_ylabel("success %  (slice by applier)", fontsize=10.5)

# ---------- Row B: slice by rulebook draw ----------
for ax, (cond, cname) in zip(axes[1], CONDS):
    x = 0.0
    ticks, tlabels, gticks = [], [], []
    for diet, dlabel in DIETS:
        gxs = []
        for dr in ["r1", "r2", "r3"]:
            vals = [draws(cond, diet, ap).get(dr) for ap in APS]
            vals = [v for v in vals if v is not None]
            if not vals:
                ghost(ax, x, "not distilled")
            else:
                m = sum(vals) / len(vals)
                ax.bar(x, m, 0.62, color=C_RULES, edgecolor="#4a5568", lw=0.8)
                ax.text(x, m + 0.4, f"{m:.1f}", ha="center", fontsize=7.6,
                        fontweight="bold")
                if len(vals) < 3:
                    ax.text(x, m - 1.6, f"n={len(vals)}", ha="center",
                            fontsize=5.6, color="#744210")
            ticks.append(x); tlabels.append(dr); gxs.append(x); x += 0.82
        gticks.append(sum(gxs) / len(gxs))
        ax.axvline(x - 0.31, color="#e2e8f0", lw=1.0, zorder=0)
        x += 0.42
    scv = [v for v in (SCAFFOLD[cond] or {}).values() if v is not None]
    if scv:
        sc = sum(scv) / len(scv)
        ax.axhline(sc, color="#4a5568", lw=1.2, ls=(0, (2, 2)), zorder=1)
        ax.text(x - 0.75, sc + 0.3, "no rules (mean)", fontsize=6.6,
                color="#4a5568", ha="right")
    ax.axhline(NO_REPH[cond], color="#c53030", lw=1.2, ls=(0, (4, 3)), zorder=1)
    ax.set_xticks(ticks); ax.set_xticklabels(tlabels, fontsize=7.0, color="#4a5568")
    for gx, (diet, dlabel) in zip(gticks, DIETS):
        ax.text(gx, YLIM[0] - 3.6, dlabel.replace("\n", " "), ha="center",
                fontsize=10, fontweight="bold", clip_on=False)
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.6, x - 0.55)
    ax.grid(axis="y", alpha=0.16)
    ax.spines[["top", "right"]].set_visible(False)
axes[1][0].set_ylabel("success %  (slice by rulebook draw)", fontsize=10.5)

for row in axes:
    for ax in row:
        ax.set_ylim(*YLIM)

handles = [plt.Rectangle((0, 0), 1, 1, color=C_BASE),
           plt.Rectangle((0, 0), 1, 1, color=C_RULES),
           plt.Line2D([0], [0], marker="o", color=INK, mfc="white", ls="", ms=5),
           plt.Line2D([0], [0], marker="^", color=INK, mfc="white", ls="", ms=5),
           plt.Line2D([0], [0], marker="s", color=INK, mfc="white", ls="", ms=5),
           plt.Line2D([0], [0], color="#c53030", lw=1.2, ls=(0, (4, 3))),
           plt.Line2D([0], [0], color="#4a5568", lw=1.2, ls=(0, (2, 2)))]
fig.legend(handles, ["no-rules rephraser (scaffold)", "rulebook cell",
                     "draw r1", "draw r2", "draw r3",
                     "no rephraser", "no rules (mean)"],
           fontsize=8, ncol=7, loc="lower center", bbox_to_anchor=(0.5, -0.005),
           framealpha=0.95)
fig.suptitle("pi0.5 / LIBERO rulebook dashboard — sealed 22 tasks "
             "(row 1: by applier, dots = draws; row 2: by draw, mean over appliers)",
             fontsize=12.5, y=0.995)
fig.text(0.99, 0.004,
         "natural 70 bases · adversarial 48 ERT attacks · original 22 canonicals\n"
         "all x20 inits (window 30-49) · r1 only, no replicates distilled yet · qwen never ran",
         ha="right", va="bottom", fontsize=6.6, color="#718096", linespacing=1.5)
fig.tight_layout(rect=(0, 0.035, 1, 0.975))
out = R / "results/charts/pi05_dashboard.png"
fig.savefig(out, dpi=140, bbox_inches="tight", pad_inches=0.22)
print("chart ->", out)
