#!/usr/bin/env python3
"""results/charts/rules_dashboard.png -- one page, four slices.

v2 (2026-09-13, user-directed): condition columns are Natural / Adversarial /
HUMAN NATURALS (the Original column moved to the appendix figure,
make_orig_appendix_dashboard.py); evidence diets labeled in-finetune /
out-of-finetune / both; draws labeled rulebook-1/2/3; pastel palette.

Row 1 (by APPLIER): applier clusters x rule diets, mean over draws.
Row 2 (by RULEBOOK DRAW): diet clusters x draws, mean over appliers.
Row 3 (IN-VOCAB, 5 tasks) and Row 4 (OUT-OF-VOCAB, 7 tasks): diet bars on
the stratum values, pooled over appliers and draws.

All cells BASE-WEIGHTED. Natural/adversarial: r1 cells + strata from
results/analysis/r1_cells.json, replicates from results/analysis/
a36_cells.json (12/12-leg arms). Human naturals: results/analysis/
a39_human_cells.json (363 human phrases, 24x1). Values inside bars in
bar-colored boxes; dashed red = un-rephrased baseline (labeled outside the
right edge); PAPER=1 drops title/footer and writes *_paper.png.
"""
import json
import os
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = pathlib.Path(__file__).resolve().parents[1]
r1 = json.loads((R / "results/analysis/r1_cells.json").read_text())
rep = json.loads((R / "results/analysis/a36_cells.json").read_text())
hum = json.loads((R / "results/analysis/a39_human_cells.json").read_text())

APS = ["claude", "gemini", "qwen"]
DIETS = [("s", "out-of-\nfinetune"), ("b", "both"), ("t", "in-\nfinetune")]
CONDS = [("nat", "Natural"), ("adv", "Adversarial"), ("hum", "Human Naturals")]
DRAW_LBL = ["rulebook-1", "rulebook-2", "rulebook-3"]
HUM_BOOK = {("s", 1): "s", ("s", 2): "s2", ("s", 3): "s3",
            ("b", 1): "b", ("b", 2): "b2", ("b", 3): "b3",
            ("t", 1): "t", ("t", 2): "t2", ("t", 3): "t3"}

# pastel palette (method-diagram family)
C_BASE, C_RULES = "#D9DEE6", "#BFD8F7"
EDGE, INK = "#6E7B8B", "#2d3748"
RED, GREY = "#C0504D", "#7A8698"

NO_REPH = {"nat": r1["noreph|nat"]["pooled"], "adv": 24.5,
           "hum": hum["raw_human"]["pooled"]}
NOREPH_ST = {"nat": {"iv": r1["noreph|nat"]["iv"], "oov": r1["noreph|nat"]["oov"]},
             "adv": {"iv": 25.2, "oov": 23.9},
             "hum": {"iv": hum["raw_human"]["iv"], "oov": hum["raw_human"]["oov"]}}


def rec(cond, diet, ap, dr):
    if cond == "hum":
        return hum.get(f"{HUM_BOOK[(diet, dr)]}|{ap}")
    if dr == 1:
        return r1.get(f"r1{diet}|{ap}|{cond}")
    r = rep.get(f"{diet}{dr}|{ap}|{cond}")
    return r if (r and r.get("legs") == 12) else None


def draws(cond, diet, ap):
    return {dr: rec(cond, diet, ap, dr)["pooled"] for dr in (1, 2, 3)
            if rec(cond, diet, ap, dr)}


def stratum(cond, diet, dr, st):
    vals = [rec(cond, diet, ap, dr)[st] for ap in APS
            if rec(cond, diet, ap, dr) and rec(cond, diet, ap, dr).get(st) is not None]
    return sum(vals) / len(vals) if vals else None


def scaffold(cond, ap=None, st="pooled"):
    if cond == "hum":
        vals = [hum[f"sc|{a}"][st] for a in ([ap] if ap else APS) if f"sc|{a}" in hum]
    else:
        ks = [f"sc|{a}|{cond}" for a in ([ap] if ap else APS)]
        vals = [r1[k][st] for k in ks if k in r1 and r1[k].get(st) is not None]
    return sum(vals) / len(vals) if vals else None


def bar_annot(ax, x, v, fc, dy=0.9):
    ax.text(x, v - dy, f"{v:.1f}", ha="center", va="top", fontsize=7.0,
            fontweight="bold", color=INK, zorder=6,
            bbox=dict(boxstyle="square,pad=0.10", fc=fc, ec="none"))


fig, axes = plt.subplots(4, 3, figsize=(16.4, 17.0), sharey="row")

# ---------- Row 1: slice by applier ----------
for ax, (cond, cname) in zip(axes[0], CONDS):
    x, ticks, tlabels, gticks = 0.0, [], [], []
    for ap in APS:
        gxs = []
        sc = scaffold(cond, ap)
        if sc is not None:
            ax.bar(x, sc, 0.62, color=C_BASE, edgecolor=EDGE, lw=0.8)
            bar_annot(ax, x, sc, C_BASE)
        else:
            ax.text(x, 21.2, "no rules:\nnot run", ha="center", fontsize=5.8,
                    color="#a0aec0")
        ticks.append(x); tlabels.append("no\nrules"); gxs.append(x); x += 0.82
        for diet, dlabel in DIETS:
            dv = draws(cond, diet, ap)
            m = sum(dv.values()) / len(dv)
            ax.bar(x, m, 0.62, color=C_RULES, edgecolor=EDGE, lw=0.8)
            bar_annot(ax, x, m, C_RULES)
            ticks.append(x); tlabels.append(dlabel); gxs.append(x); x += 0.82
        gticks.append(sum(gxs) / len(gxs))
        ax.axvline(x - 0.31, color="#e2e8f0", lw=1.0, zorder=0)
        x += 0.42
    ax.axhline(NO_REPH[cond], color=RED, lw=1.2, ls=(0, (4, 3)), zorder=1)
    ax.text(1.005, NO_REPH[cond], f"{NO_REPH[cond]:.1f}", fontsize=6.6,
            color=RED, ha="left", va="center", transform=ax.get_yaxis_transform())
    ax.set_xticks(ticks); ax.set_xticklabels(tlabels, fontsize=5.8, color="#4a5568")
    for gx, ap in zip(gticks, ["Claude", "Gemini", "Qwen"]):
        ax.text(gx, 16.6, ap, ha="center", fontsize=10, fontweight="bold",
                clip_on=False)
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.6, x - 0.55)
    ax.set_ylim(20, 44)
    ax.grid(axis="y", alpha=0.16)
    ax.set_title(cname, fontsize=13.5, pad=8)
    ax.spines[["top", "right"]].set_visible(False)
axes[0][0].set_ylabel("success %\n(by applier)", fontsize=10.5)

# ---------- Row 2: slice by rulebook draw ----------
for ax, (cond, cname) in zip(axes[1], CONDS):
    x, ticks, tlabels, gticks = 0.0, [], [], []
    for diet, dlabel in DIETS:
        gxs = []
        for dr in (1, 2, 3):
            vals = [rec(cond, diet, ap, dr) for ap in APS]
            vals = [v["pooled"] for v in vals if v]
            if not vals:
                ax.bar(x, 8, 0.62, bottom=20, color="none", edgecolor="#cbd5e0",
                       lw=0.8, ls=":")
            else:
                m = sum(vals) / len(vals)
                ax.bar(x, m, 0.62, color=C_RULES, edgecolor=EDGE, lw=0.8)
                bar_annot(ax, x, m, C_RULES)
            ticks.append(x); tlabels.append(DRAW_LBL[dr - 1]); gxs.append(x)
            x += 0.82
        gticks.append(sum(gxs) / len(gxs))
        ax.axvline(x - 0.31, color="#e2e8f0", lw=1.0, zorder=0)
        x += 0.42
    sc = scaffold(cond)
    if sc is not None:
        ax.axhline(sc, color=GREY, lw=1.2, ls=(0, (2, 2)), zorder=1)
        crowd = abs(sc - NO_REPH[cond]) < 1.2
        ax.text(1.005, sc, f"{sc:.1f}", fontsize=6.6, color=GREY, ha="left",
                va="bottom" if (crowd and sc >= NO_REPH[cond]) else
                   ("top" if crowd else "center"),
                transform=ax.get_yaxis_transform())
    ax.axhline(NO_REPH[cond], color=RED, lw=1.2, ls=(0, (4, 3)), zorder=1)
    crowd = sc is not None and abs(sc - NO_REPH[cond]) < 1.2
    ax.text(1.005, NO_REPH[cond], f"{NO_REPH[cond]:.1f}", fontsize=6.6,
            color=RED, ha="left",
            va="top" if (crowd and sc >= NO_REPH[cond]) else
               ("bottom" if crowd else "center"),
            transform=ax.get_yaxis_transform())
    ax.set_xticks(ticks); ax.set_xticklabels(tlabels, fontsize=5.6, color="#4a5568")
    for gx, (diet, dlabel) in zip(gticks, DIETS):
        ax.text(gx, 16.6, dlabel.replace("\n", ""), ha="center", fontsize=10,
                fontweight="bold", clip_on=False)
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.6, x - 0.55)
    ax.set_ylim(20, 44)
    ax.grid(axis="y", alpha=0.16)
    ax.spines[["top", "right"]].set_visible(False)
axes[1][0].set_ylabel("success %\n(by rulebook draw)", fontsize=10.5)

# ---------- Rows 3-4: slice by vocabulary stratum ----------
for row, st, stname, ylim in [(2, "iv", "IN-VOCAB (5 tasks)", (12, 52)),
                              (3, "oov", "OUT-OF-VOCAB (7 tasks)", (12, 36))]:
    for ax, (cond, cname) in zip(axes[row], CONDS):
        x, ticks, tlabels = 0.0, [], []
        sc = scaffold(cond, st=st)
        if sc is not None:
            ax.bar(x, sc, 0.62, color=C_BASE, edgecolor=EDGE, lw=0.8)
            bar_annot(ax, x, sc, C_BASE, dy=1.2)
        else:
            ax.text(x, ylim[0] + 2.0, "no rules:\nnot run", ha="center",
                    fontsize=5.8, color="#a0aec0")
        ticks.append(x); tlabels.append("no\nrules"); x += 0.9
        for diet, dlabel in DIETS:
            dvals = [stratum(cond, diet, dr, st) for dr in (1, 2, 3)]
            dvals = [v for v in dvals if v is not None]
            m = sum(dvals) / len(dvals)
            ax.bar(x, m, 0.62, color=C_RULES, edgecolor=EDGE, lw=0.8)
            bar_annot(ax, x, m, C_RULES, dy=1.2)
            ticks.append(x); tlabels.append(dlabel); x += 0.9
        nb = NOREPH_ST[cond][st]
        ax.axhline(nb, color=RED, lw=1.2, ls=(0, (4, 3)), zorder=1)
        ax.text(1.005, nb, f"{nb:.1f}", fontsize=6.6, color=RED, ha="left",
                va="center", transform=ax.get_yaxis_transform())
        ax.set_xticks(ticks); ax.set_xticklabels(tlabels, fontsize=5.8,
                                                 color="#4a5568")
        ax.tick_params(axis="x", length=0)
        ax.set_xlim(-0.65, x - 0.6)
        ax.set_ylim(*ylim)
        ax.grid(axis="y", alpha=0.16)
        ax.spines[["top", "right"]].set_visible(False)
    axes[row][0].set_ylabel(f"success %\n{stname}", fontsize=10)

handles = [plt.Rectangle((0, 0), 1, 1, fc=C_BASE, ec=EDGE),
           plt.Rectangle((0, 0), 1, 1, fc=C_RULES, ec=EDGE),
           plt.Line2D([0], [0], color=RED, lw=1.2, ls=(0, (4, 3))),
           plt.Line2D([0], [0], color=GREY, lw=1.2, ls=(0, (2, 2)))]
fig.legend(handles, ["no-rules rephraser (scaffold)", "rulebook cell",
                     "no rephraser", "no rules (mean)"],
           fontsize=8.4, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 0.0))
PAPER = os.environ.get("PAPER") == "1"
if not PAPER:
    fig.suptitle("Rulebook evaluation dashboard — base-weighted, sealed 12 tasks\n"
                 "rows: by applier · by rulebook draw (mean over appliers) · "
                 "in-vocab stratum · out-of-vocab stratum",
                 fontsize=12.5, y=0.995)
    fig.text(0.99, 0.002,
             "natural = 186-phrase image set ×1 · adversarial = 72 attacks ×2 · "
             "human naturals = 363 survey phrases ×1 (A39) · diets: evidence from "
             "outside the finetune (rollouts) / inside it (corpus) / both · "
             "originals column: appendix figure",
             ha="right", fontsize=6.6, color="#718096")
fig.tight_layout(rect=(0, 0.018, 1, 0.972) if not PAPER else (0, 0.018, 1, 1.0))
out = R / ("results/charts/rules_dashboard_paper.png" if PAPER
           else "results/charts/rules_dashboard.png")
fig.savefig(out, dpi=140, bbox_inches="tight", pad_inches=0.22)
print("chart ->", out)
