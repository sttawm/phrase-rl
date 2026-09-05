#!/usr/bin/env python3
"""results/charts/rules_dashboard.png -- one page, three slices.

Row A (slice by APPLIER): per condition, applier clusters x rule diets; bar =
mean over rulebook draws, dark dots = the individual draws (r1/r2/r3).
Row B (slice by RULEBOOK DRAW): per condition, diet clusters x draws; bar =
mean over the appliers with complete cells for that draw.

All cells BASE-WEIGHTED. r1 hardcoded with provenance; replicates read live
from results/analysis/a36_cells.json (only 12/12-leg arms are used), so the
page improves as A36 legs land. Grey = baselines (no-rules scaffold; dashed
line = no rephraser at all). Blue = rulebook cells.
"""
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = pathlib.Path(__file__).resolve().parents[1]
rep = json.loads((R / "results/analysis/a36_cells.json").read_text())

APS = ["claude", "gemini", "qwen"]
# r1 base-weighted: natural = A34 (186 image set); adv/orig = A31 (verified)
R1 = {
    ("nat", "s"): dict(zip(APS, [31.5, 31.4, 28.4])),
    ("nat", "b"): dict(zip(APS, [29.4, 29.0, 29.5])),
    ("nat", "t"): dict(zip(APS, [26.5, 26.8, 28.4])),
    ("adv", "s"): dict(zip(APS, [30.7, 30.4, 28.4])),
    ("adv", "b"): dict(zip(APS, [29.2, 30.0, 26.7])),
    ("adv", "t"): dict(zip(APS, [26.9, 27.6, 27.3])),
    ("orig", "s"): dict(zip(APS, [39.2, 38.9, 28.5])),
    ("orig", "b"): dict(zip(APS, [39.6, 39.9, 37.0])),
    ("orig", "t"): dict(zip(APS, [36.5, 37.5, 38.0])),
}
SCAFFOLD = {"nat": dict(zip(APS, [29.7, 28.7, 28.1])),
            "adv": dict(zip(APS, [24.3, 24.0, 24.7])),
            "orig": None}
NO_REPH = {"nat": 26.0, "adv": 24.5, "orig": 36.1}


def draws(cond, diet, ap):
    """{draw_label: value} for one (condition, diet, applier)."""
    out = {"r1": R1[(cond, diet)][ap]}
    for k, r in rep.items():
        tag, a, c = k.split("|")
        if a == ap and c == cond and tag[0] == diet and r["legs"] == 12:
            out["r" + tag[1]] = r["pooled"]
    return out


C_BASE, C_RULES, INK = "#8b96a5", "#a3bffa", "#2d3748"
DIETS = [("s", "rollout\nonly"), ("b", "rollout\n+ train"), ("t", "train\nonly")]
CONDS = [("nat", "Natural"), ("adv", "Adversarial"), ("orig", "Original")]
DMARK = {"r1": "o", "r2": "^", "r3": "s"}
YLIM = (20, 44)

fig, axes = plt.subplots(2, 3, figsize=(16.4, 9.2), sharey=True)

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
            ax.text(x, YLIM[0] + 1.2, "no rules:\nnot run", ha="center",
                    fontsize=5.8, color="#a0aec0")
        ticks.append(x); tlabels.append("no\nrules"); gxs.append(x); x += 0.82
        for diet, dlabel in DIETS:
            dv = draws(cond, diet, ap)
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
                ax.bar(x, YLIM[1] - YLIM[0] - 16, 0.62, bottom=YLIM[0],
                       color="none", edgecolor="#cbd5e0", lw=0.8, ls=":")
                ax.text(x, YLIM[0] + 3.5, "rolling", ha="center", fontsize=5.6,
                        color="#a0aec0", rotation=90)
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
    if SCAFFOLD[cond]:
        sc = sum(SCAFFOLD[cond].values()) / 3
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
fig.suptitle("Rulebook evaluation dashboard — base-weighted, sealed 12 tasks "
             "(row 1: by applier, dots = draws; row 2: by draw, mean over appliers)",
             fontsize=12.5, y=0.995)
fig.text(0.99, 0.005,
         "natural = 186-phrase image set ×1 rep · adversarial = 72 attacks ×2 · "
         "original = 12 canonicals ×2 · draws: r1 = A31/A34, r2/r3 = A36 replicates "
         "(only 12/12-leg arms included; n<3 = partial applier coverage)",
         ha="right", fontsize=6.6, color="#718096")
fig.tight_layout(rect=(0, 0.035, 1, 0.975))
out = R / "results/charts/rules_dashboard.png"
fig.savefig(out, dpi=140, bbox_inches="tight", pad_inches=0.22)
print("chart ->", out)
