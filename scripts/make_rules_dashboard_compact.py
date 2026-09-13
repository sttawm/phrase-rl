#!/usr/bin/env python3
"""results/charts/rules_dashboard_compact.png — single-row dashboard.

One row of three condition panels (Adversarial / Human-Generated Naturals /
LLM-Generated Naturals). Per applier: no-rules scaffold + the three diet
cells (mean over the three rulebook draws), each pooled bar flanked by thin
semi-transparent strata bars — in-distribution (left, 5 tasks) and
out-of-distribution (right, 7 tasks) — the input-condition-chart idiom.
The per-draw breakdown moves to an appendix grid. No in-image title.
"""
import json
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
CONDS = [("adv", "Adversarial"), ("hum", "Human-Generated Naturals"),
         ("nat", "LLM-Generated Naturals")]
HUM_BOOK = {("s", 1): "s", ("s", 2): "s2", ("s", 3): "s3",
            ("b", 1): "b", ("b", 2): "b2", ("b", 3): "b3",
            ("t", 1): "t", ("t", 2): "t2", ("t", 3): "t3"}

C_BASE, C_RULES = "#D9DEE6", "#3F6B52"
EDGE, INK, RED = "#6E7B8B", "#2d3748", "#C0504D"
NO_REPH = {"nat": r1["noreph|nat"]["pooled"], "adv": 24.5,
           "hum": hum["raw_human"]["pooled"]}


def rec(cond, diet, ap, dr):
    if cond == "hum":
        return hum.get(f"{HUM_BOOK[(diet, dr)]}|{ap}")
    if dr == 1:
        return r1.get(f"r1{diet}|{ap}|{cond}")
    r = rep.get(f"{diet}{dr}|{ap}|{cond}")
    return r if (r and r.get("legs") == 12) else None


def cell(cond, diet, ap, st="pooled"):
    vals = [rec(cond, diet, ap, dr)[st] for dr in (1, 2, 3)
            if rec(cond, diet, ap, dr) and rec(cond, diet, ap, dr).get(st) is not None]
    return sum(vals) / len(vals) if vals else None


def scaffold(cond, ap, st="pooled"):
    r = hum.get(f"sc|{ap}") if cond == "hum" else r1.get(f"sc|{ap}|{cond}")
    return r.get(st) if r else None


fig, axes = plt.subplots(1, 3, figsize=(16.2, 4.9), sharey=True)

for ax, (cond, cname) in zip(axes, CONDS):
    x, ticks, tlabels, gticks = 0.0, [], [], []
    for ap in APS:
        gxs = []
        arms = [("sc", "no\nrules", C_BASE)] + \
               [(d, dl, C_RULES) for d, dl in DIETS]
        for key, dlabel, col in arms:
            if key == "sc":
                pool, iv, oov = (scaffold(cond, ap, s) for s in ("pooled", "iv", "oov"))
            else:
                pool, iv, oov = (cell(cond, key, ap, s) for s in ("pooled", "iv", "oov"))
            if pool is not None:
                if iv is not None:
                    ax.bar(x - 0.185, iv, 0.30, color=col, alpha=0.42, zorder=1,
                           edgecolor="none")
                    ax.text(x - 0.29, iv + 0.7, f"{iv:.0f}", ha="center",
                            fontsize=5.0, color="#71808f", zorder=3)
                if oov is not None:
                    ax.bar(x + 0.185, oov, 0.30, color=col, alpha=0.42, zorder=1,
                           edgecolor="none")
                    ax.text(x + 0.29, oov + 0.7, f"{oov:.0f}", ha="center",
                            fontsize=5.0, color="#71808f", zorder=3)
                ax.bar(x, pool, 0.40, color=col, zorder=2,
                       edgecolor=EDGE if col == C_BASE else "none",
                       lw=0.7 if col == C_BASE else 0)
                ax.text(x, pool - 1.1, f"{pool:.1f}", ha="center", va="top",
                        fontsize=5.9, fontweight="bold",
                        color="white" if col == C_RULES else INK, zorder=4,
                        bbox=dict(boxstyle="square,pad=0.09", fc=col, ec="none"))
            ticks.append(x); tlabels.append(dlabel); gxs.append(x); x += 0.86
        gticks.append(sum(gxs) / len(gxs))
        ax.axvline(x - 0.33, color="#e2e8f0", lw=1.0, zorder=0)
        x += 0.46
    ax.axhline(NO_REPH[cond], color=RED, lw=1.1, ls=(0, (4, 3)), zorder=1)
    ax.text(1.005, NO_REPH[cond], f"{NO_REPH[cond]:.1f}", fontsize=6.4,
            color=RED, ha="left", va="center", transform=ax.get_yaxis_transform())
    ax.set_xticks(ticks)
    ax.set_xticklabels(tlabels, fontsize=5.4, color="#4a5568")
    for gx, ap in zip(gticks, ["Claude", "Gemini", "Qwen"]):
        ax.text(gx, 6.0, ap, ha="center", fontsize=10, fontweight="bold",
                clip_on=False)
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.62, x - 0.62)
    ax.set_ylim(12, 47)
    ax.grid(axis="y", alpha=0.16)
    ax.set_title(cname, fontsize=12.5, pad=6)
    ax.spines[["top", "right"]].set_visible(False)

axes[0].set_ylabel("success %", fontsize=10.5)

handles = [plt.Rectangle((0, 0), 1, 1, fc=C_BASE, ec=EDGE),
           plt.Rectangle((0, 0), 1, 1, fc=C_RULES, ec="none"),
           plt.Rectangle((0, 0), 1, 1, fc="#7A8698", alpha=0.42),
           plt.Line2D([0], [0], color=RED, lw=1.1, ls=(0, (4, 3)))]
fig.legend(handles, ["no-rules rephraser (scaffold)",
                     "rulebook cell (mean over three draws)",
                     "flanks: in-distribution (left, 5) / out-of-distribution (right, 7)",
                     "no rephraser"],
           fontsize=8.2, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 0.0),
           frameon=False)
fig.tight_layout(rect=(0, 0.06, 1, 1.0))
out = R / "results/charts/rules_dashboard_compact.png"
fig.savefig(out, dpi=170, bbox_inches="tight", pad_inches=0.18)
print("chart ->", out)
