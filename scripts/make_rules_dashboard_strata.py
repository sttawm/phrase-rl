#!/usr/bin/env python3
"""results/charts/rules_dashboard_strata.png -- the vocabulary slice.

Rows: in-vocab (5 tasks) / out-of-vocab (7 tasks). Columns: condition.
Clusters: no-rules scaffold (grey) then the three rule diets (blue); bar = mean
over appliers x draws of the STRATUM value; dots = per-draw applier-pooled
values; dashed red line = no rephraser at all (same stratum).

Base-weighted. r1 strata from results/analysis/r1_cells.json (A31/A34 legs);
replicates from results/analysis/a36_cells.json (12/12-leg arms only).
"""
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = pathlib.Path(__file__).resolve().parents[1]
r1 = json.loads((R / "results/analysis/r1_cells.json").read_text())
rep = json.loads((R / "results/analysis/a36_cells.json").read_text())

APS = ["claude", "gemini", "qwen"]
DIETS = [("s", "rollout only"), ("b", "rollout + train"), ("t", "train only")]
CONDS = [("nat", "Natural"), ("adv", "Adversarial"), ("orig", "Original")]
DMARK = {"r1": "o", "r2": "^", "r3": "s"}
# no-rephraser stratum baselines: nat measured (r1_cells), adv/orig = paper Fig-9
NOREPH = {"nat": {"iv": r1["noreph|nat"]["iv"], "oov": r1["noreph|nat"]["oov"]},
          "adv": {"iv": 25.2, "oov": 23.9}, "orig": {"iv": 48.3, "oov": 27.3}}


def stratum_draw(cond, diet, dr, st):
    """applier-pooled stratum value for one draw ('r1' or a36 r2/r3)."""
    vals = []
    for ap in APS:
        if dr == "r1":
            rec = r1.get(f"r1{diet}|{ap}|{cond}")
        else:
            rec = rep.get(f"{diet}{dr[1]}|{ap}|{cond}")
            if rec and rec["legs"] < 12:
                rec = None
        if rec and rec.get(st) is not None:
            vals.append(rec[st])
    return sum(vals) / len(vals) if vals else None


def scaffold_stratum(cond, st):
    vals = [r1[f"sc|{ap}|{cond}"][st] for ap in APS if f"sc|{ap}|{cond}" in r1]
    return sum(vals) / len(vals) if vals else None


C_BASE, C_RULES, INK = "#8b96a5", "#a3bffa", "#2d3748"
YLIM = {"iv": (12, 46), "oov": (12, 36)}

fig, axes = plt.subplots(2, 3, figsize=(15.6, 8.6))
for row, st, stname in [(0, "iv", "IN-VOCAB (5 tasks)"), (1, "oov", "OUT-OF-VOCAB (7 tasks)")]:
    for ax, (cond, cname) in zip(axes[row], CONDS):
        x = 0.0
        ticks, tlabels = [], []
        sc = scaffold_stratum(cond, st)
        if sc is not None:
            ax.bar(x, sc, 0.62, color=C_BASE, edgecolor="#4a5568", lw=0.8)
            ax.text(x, sc + 0.4, f"{sc:.1f}", ha="center", fontsize=7.8, fontweight="bold")
        else:
            ax.text(x, YLIM[st][0] + 2.2, "no rules:\nnot run", ha="center",
                    fontsize=6.0, color="#a0aec0")
        ticks.append(x); tlabels.append("no\nrules"); x += 0.9
        for diet, dlabel in DIETS:
            dvals = {dr: stratum_draw(cond, diet, dr, st) for dr in ["r1", "r2", "r3"]}
            dvals = {k: v for k, v in dvals.items() if v is not None}
            m = sum(dvals.values()) / len(dvals)
            ax.bar(x, m, 0.62, color=C_RULES, edgecolor="#4a5568", lw=0.8)
            ax.text(x, m + 0.4, f"{m:.1f}", ha="center", fontsize=7.8, fontweight="bold")
            for lb, v in dvals.items():
                ax.plot([x], [v], DMARK[lb], ms=3.8, color=INK, mfc="white", mew=1.0, zorder=5)
            ticks.append(x); tlabels.append(dlabel.replace(" ", "\n")); x += 0.9
        nb = NOREPH[cond][st]
        ax.axhline(nb, color="#c53030", lw=1.2, ls=(0, (4, 3)))
        ax.text(x - 0.65, nb + 0.35, f"no rephraser {nb:.1f}", fontsize=6.6,
                color="#c53030", ha="right")
        ax.set_xticks(ticks); ax.set_xticklabels(tlabels, fontsize=6.6, color="#4a5568")
        ax.tick_params(axis="x", length=0)
        ax.set_xlim(-0.65, x - 0.6)
        ax.set_ylim(*YLIM[st])
        ax.grid(axis="y", alpha=0.16)
        ax.spines[["top", "right"]].set_visible(False)
        if row == 0:
            ax.set_title(cname, fontsize=13, pad=8)
    axes[row][0].set_ylabel(f"success %\n{stname}", fontsize=10)

handles = [plt.Rectangle((0, 0), 1, 1, color=C_BASE),
           plt.Rectangle((0, 0), 1, 1, color=C_RULES),
           plt.Line2D([0], [0], marker="o", color=INK, mfc="white", ls="", ms=5),
           plt.Line2D([0], [0], marker="^", color=INK, mfc="white", ls="", ms=5),
           plt.Line2D([0], [0], marker="s", color=INK, mfc="white", ls="", ms=5),
           plt.Line2D([0], [0], color="#c53030", lw=1.2, ls=(0, (4, 3)))]
fig.legend(handles, ["no-rules rephraser (scaffold)", "rulebook (mean over appliers × draws)",
                     "draw r1", "draw r2", "draw r3", "no rephraser"],
           fontsize=8, ncol=6, loc="lower center", bbox_to_anchor=(0.5, -0.005))
fig.suptitle("Vocabulary slice — base-weighted, sealed 12 tasks "
             "(top: the 5 in-vocab tasks; bottom: the 7 out-of-vocab tasks)",
             fontsize=12.5, y=0.985)
fig.text(0.99, 0.005, "strata from the ex-ante vocabulary audit; adv/orig no-rephraser "
                      "strata from the paper Fig-9 measurement; nat measured on the 186-phrase set",
         ha="right", fontsize=6.4, color="#718096")
fig.tight_layout(rect=(0, 0.04, 1, 0.965))
out = R / "results/charts/rules_dashboard_strata.png"
fig.savefig(out, dpi=140, bbox_inches="tight", pad_inches=0.22)
print("chart ->", out)
