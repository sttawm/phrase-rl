#!/usr/bin/env python3
"""results/charts/rules_dashboard.png -- one page, four slices.

Row 1 (by APPLIER): applier clusters x rule diets; dots = individual draws.
Row 2 (by RULEBOOK DRAW): diet clusters x draws; bar = mean over appliers.
Row 3 (IN-VOCAB, 5 tasks) and Row 4 (OUT-OF-VOCAB, 7 tasks): diet bars on the
stratum values, dots = per-draw applier-pooled values.

Every panel carries the no-rephraser baseline as a dashed red line (stratum-
matched in rows 3-4). Grey bars = the no-rules scaffold; blue = rulebook cells.

All cells BASE-WEIGHTED. r1 cells + strata from results/analysis/r1_cells.json
(computed from the A31/A34 legs by scripts/r1_strata.py; pooled values cross-
check the published cells). Replicates live from results/analysis/a36_cells.json
(only 12/12-leg arms), so the page improves as A36 legs land.
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

APS = ["claude", "gemini", "qwen"]
DIETS = [("s", "rollout\nonly"), ("b", "rollout\n+ train"), ("t", "train\nonly")]
CONDS = [("nat", "Natural"), ("adv", "Adversarial"), ("orig", "Original")]
DMARK = {"r1": "o", "r2": "^", "r3": "s"}
C_BASE, C_RULES, INK = "#8b96a5", "#a3bffa", "#2d3748"

# pooled r1 fallbacks for adv/orig scaffold & no-rephraser; nat comes measured
NO_REPH = {"nat": r1["noreph|nat"]["pooled"], "adv": 24.5, "orig": 36.1}
NOREPH_ST = {"nat": {"iv": r1["noreph|nat"]["iv"], "oov": r1["noreph|nat"]["oov"]},
             "adv": {"iv": 25.2, "oov": 23.9}, "orig": {"iv": 48.3, "oov": 27.3}}


def rec(cond, diet, ap, dr):
    if dr == "r1":
        return r1.get(f"r1{diet}|{ap}|{cond}")
    r = rep.get(f"{diet}{dr[1]}|{ap}|{cond}")
    return r if (r and r["legs"] == 12) else None


def draws(cond, diet, ap):
    out = {}
    for dr in ["r1", "r2", "r3"]:
        x = rec(cond, diet, ap, dr)
        if x:
            out[dr] = x["pooled"]
    return out


def stratum_draw(cond, diet, dr, st):
    vals = [rec(cond, diet, ap, dr)[st] for ap in APS
            if rec(cond, diet, ap, dr) and rec(cond, diet, ap, dr).get(st) is not None]
    return sum(vals) / len(vals) if vals else None


def scaffold(cond, ap=None, st="pooled"):
    ks = [f"sc|{a}|{cond}" for a in ([ap] if ap else APS)]
    vals = [r1[k][st] for k in ks if k in r1 and r1[k].get(st) is not None]
    return sum(vals) / len(vals) if vals else None


fig, axes = plt.subplots(4, 3, figsize=(16.4, 17.0), sharey="row")

# ---------- Row 1: slice by applier ----------
for ax, (cond, cname) in zip(axes[0], CONDS):
    x = 0.0
    ticks, tlabels, gticks = [], [], []
    for ap in APS:
        gxs = []
        sc = scaffold(cond, ap)
        if sc is not None:
            ax.bar(x, sc, 0.62, color=C_BASE, edgecolor="#4a5568", lw=0.8)
            ax.text(x, sc + 0.4, f"{sc:.1f}", ha="center", fontsize=7.6, fontweight="bold")
        else:
            ax.text(x, 21.2, "no rules:\nnot run", ha="center", fontsize=5.8, color="#a0aec0")
        ticks.append(x); tlabels.append("no\nrules"); gxs.append(x); x += 0.82
        for diet, dlabel in DIETS:
            dv = draws(cond, diet, ap)
            m = sum(dv.values()) / len(dv)
            ax.bar(x, m, 0.62, color=C_RULES, edgecolor="#4a5568", lw=0.8)
            ax.text(x, m + 0.4, f"{m:.1f}", ha="center", fontsize=7.6, fontweight="bold")
            for lb, v in dv.items():
                ax.plot([x], [v], DMARK[lb], ms=3.6, color=INK, mfc="white", mew=1.0, zorder=5)
            ticks.append(x); tlabels.append(dlabel); gxs.append(x); x += 0.82
        gticks.append(sum(gxs) / len(gxs))
        ax.axvline(x - 0.31, color="#e2e8f0", lw=1.0, zorder=0)
        x += 0.42
    ax.axhline(NO_REPH[cond], color="#c53030", lw=1.2, ls=(0, (4, 3)), zorder=1)
    ax.text(x - 0.75, NO_REPH[cond] + 0.3, f"no rephraser {NO_REPH[cond]:.1f}",
            fontsize=6.6, color="#c53030", ha="right")
    ax.set_xticks(ticks); ax.set_xticklabels(tlabels, fontsize=6.0, color="#4a5568")
    for gx, ap in zip(gticks, ["Claude", "Gemini", "Qwen"]):
        ax.text(gx, 16.6, ap, ha="center", fontsize=10, fontweight="bold", clip_on=False)
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.6, x - 0.55)
    ax.set_ylim(20, 44)
    ax.grid(axis="y", alpha=0.16)
    ax.set_title(cname, fontsize=13.5, pad=8)
    ax.spines[["top", "right"]].set_visible(False)
axes[0][0].set_ylabel("success %\n(by applier)", fontsize=10.5)

# ---------- Row 2: slice by rulebook draw ----------
for ax, (cond, cname) in zip(axes[1], CONDS):
    x = 0.0
    ticks, tlabels, gticks = [], [], []
    for diet, dlabel in DIETS:
        gxs = []
        for dr in ["r1", "r2", "r3"]:
            vals = [draws(cond, diet, ap).get(dr) for ap in APS]
            vals = [v for v in vals if v is not None]
            if not vals:
                ax.bar(x, 8, 0.62, bottom=20, color="none", edgecolor="#cbd5e0",
                       lw=0.8, ls=":")
                ax.text(x, 23.5, "rolling", ha="center", fontsize=5.6,
                        color="#a0aec0", rotation=90)
            else:
                m = sum(vals) / len(vals)
                ax.bar(x, m, 0.62, color=C_RULES, edgecolor="#4a5568", lw=0.8)
                ax.text(x, m + 0.4, f"{m:.1f}", ha="center", fontsize=7.6, fontweight="bold")
                if len(vals) < 3:
                    ax.text(x, m - 1.6, f"n={len(vals)}", ha="center", fontsize=5.6,
                            color="#744210")
            ticks.append(x); tlabels.append(dr); gxs.append(x); x += 0.82
        gticks.append(sum(gxs) / len(gxs))
        ax.axvline(x - 0.31, color="#e2e8f0", lw=1.0, zorder=0)
        x += 0.42
    sc = scaffold(cond)
    if sc is not None:
        ax.axhline(sc, color="#4a5568", lw=1.2, ls=(0, (2, 2)), zorder=1)
        ax.text(x - 0.75, sc + 0.3, f"no rules {sc:.1f}", fontsize=6.6,
                color="#4a5568", ha="right")
    ax.axhline(NO_REPH[cond], color="#c53030", lw=1.2, ls=(0, (4, 3)), zorder=1)
    ax.set_xticks(ticks); ax.set_xticklabels(tlabels, fontsize=7.0, color="#4a5568")
    for gx, (diet, dlabel) in zip(gticks, DIETS):
        ax.text(gx, 16.6, dlabel.replace("\n", " "), ha="center", fontsize=10,
                fontweight="bold", clip_on=False)
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.6, x - 0.55)
    ax.set_ylim(20, 44)
    ax.grid(axis="y", alpha=0.16)
    ax.spines[["top", "right"]].set_visible(False)
axes[1][0].set_ylabel("success %\n(by rulebook draw)", fontsize=10.5)

# ---------- Rows 3-4: slice by vocabulary stratum ----------
for row, st, stname, ylim in [(2, "iv", "IN-VOCAB (5 tasks)", (12, 50)),
                              (3, "oov", "OUT-OF-VOCAB (7 tasks)", (12, 36))]:
    for ax, (cond, cname) in zip(axes[row], CONDS):
        x = 0.0
        ticks, tlabels = [], []
        sc = scaffold(cond, st=st)
        if sc is not None:
            ax.bar(x, sc, 0.62, color=C_BASE, edgecolor="#4a5568", lw=0.8)
            ax.text(x, sc + 0.5, f"{sc:.1f}", ha="center", fontsize=7.6, fontweight="bold")
        else:
            ax.text(x, ylim[0] + 2.0, "no rules:\nnot run", ha="center",
                    fontsize=5.8, color="#a0aec0")
        ticks.append(x); tlabels.append("no\nrules"); x += 0.9
        for diet, dlabel in DIETS:
            dvals = {dr: stratum_draw(cond, diet, dr, st) for dr in ["r1", "r2", "r3"]}
            dvals = {k: v for k, v in dvals.items() if v is not None}
            m = sum(dvals.values()) / len(dvals)
            ax.bar(x, m, 0.62, color=C_RULES, edgecolor="#4a5568", lw=0.8)
            ax.text(x, m + 0.5, f"{m:.1f}", ha="center", fontsize=7.6, fontweight="bold")
            for lb, v in dvals.items():
                ax.plot([x], [v], DMARK[lb], ms=3.8, color=INK, mfc="white", mew=1.0, zorder=5)
            ticks.append(x); tlabels.append(dlabel); x += 0.9
        nb = NOREPH_ST[cond][st]
        ax.axhline(nb, color="#c53030", lw=1.2, ls=(0, (4, 3)), zorder=1)
        ax.text(x - 0.65, min(nb + 0.4, ylim[1] - 1.6), f"no rephraser {nb:.1f}",
                fontsize=6.6, color="#c53030", ha="right")
        ax.set_xticks(ticks); ax.set_xticklabels(tlabels, fontsize=6.4, color="#4a5568")
        ax.tick_params(axis="x", length=0)
        ax.set_xlim(-0.65, x - 0.6)
        ax.set_ylim(*ylim)
        ax.grid(axis="y", alpha=0.16)
        ax.spines[["top", "right"]].set_visible(False)
    axes[row][0].set_ylabel(f"success %\n{stname}", fontsize=10)

handles = [plt.Rectangle((0, 0), 1, 1, color=C_BASE),
           plt.Rectangle((0, 0), 1, 1, color=C_RULES),
           plt.Line2D([0], [0], marker="o", color=INK, mfc="white", ls="", ms=5),
           plt.Line2D([0], [0], marker="^", color=INK, mfc="white", ls="", ms=5),
           plt.Line2D([0], [0], marker="s", color=INK, mfc="white", ls="", ms=5),
           plt.Line2D([0], [0], color="#c53030", lw=1.2, ls=(0, (4, 3))),
           plt.Line2D([0], [0], color="#4a5568", lw=1.2, ls=(0, (2, 2)))]
fig.legend(handles, ["no-rules rephraser (scaffold)", "rulebook cell",
                     "draw r1", "draw r2", "draw r3", "no rephraser",
                     "no rules (mean)"],
           fontsize=8.4, ncol=7, loc="lower center", bbox_to_anchor=(0.5, 0.0))
PAPER = os.environ.get("PAPER") == "1"   # paper variant: no title/footer, the
if not PAPER:                            # LaTeX caption carries that text
    fig.suptitle("Rulebook evaluation dashboard — base-weighted, sealed 12 tasks\n"
                 "rows: by applier (dots = draws) · by rulebook draw (mean over appliers) · "
                 "in-vocab stratum · out-of-vocab stratum",
                 fontsize=12.5, y=0.995)
    fig.text(0.99, 0.002,
             "natural = 186-phrase image set ×1 rep · adversarial = 72 attacks ×2 · original = 12 canonicals ×2 · "
             "draws: r1 = A31/A34, r2/r3 = A36 (12/12-leg arms only) · strata from the ex-ante vocabulary audit",
             ha="right", fontsize=6.6, color="#718096")
fig.tight_layout(rect=(0, 0.018, 1, 0.972) if not PAPER else (0, 0.018, 1, 1.0))
out = R / ("results/charts/rules_dashboard_paper.png" if PAPER
           else "results/charts/rules_dashboard.png")
fig.savefig(out, dpi=140, bbox_inches="tight", pad_inches=0.22)
print("chart ->", out)
