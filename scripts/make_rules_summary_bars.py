#!/usr/bin/env python3
"""results/charts/rules_summary_bars.png -- one-look summary for the paper.

Groups: oracle | Natural | Adversarial | Original. Within each condition:
no rephraser, rephraser with no rules (scaffold), rollout-only rules,
train+rollout rules, train-only rules. Rule-diet bars are the MEAN over all
complete (applier x rulebook-draw) cells; baselines are grey, rules blue.

All cells BASE-WEIGHTED (pooled over phrase bases; the A31 convention).
r1 cells are hardcoded with provenance; replicate draws are read live from
results/analysis/a36_cells.json, so the chart improves as A36 legs land.

Natural condition = the 186-phrase image-conditioned set throughout (A34/A36);
the A31 natural cells (older rephrase16 set) are deliberately NOT mixed in.
"""
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = pathlib.Path(__file__).resolve().parents[1]
rep = json.loads((R / "results/analysis/a36_cells.json").read_text())

# --- r1 cells, base-weighted, (claude, gemini, qwen) ------------------------
R1 = {
    # natural: A34 d34 legs joined to ph_a34 applies (186 image set)
    ("nat", "s"): [31.5, 31.4, 28.4], ("nat", "b"): [29.4, 29.0, 29.5],
    ("nat", "t"): [26.5, 26.8, 28.4],
    # adversarial / original: A31 published cells (verified base-weighted)
    ("adv", "s"): [30.7, 30.4, 28.4], ("adv", "b"): [29.2, 30.0, 26.7],
    ("adv", "t"): [26.9, 27.6, 27.3],
    ("orig", "s"): [39.2, 38.9, 28.5], ("orig", "b"): [39.6, 39.9, 37.0],
    ("orig", "t"): [36.5, 37.5, 38.0],
}
SCAFFOLD = {"nat": [29.7, 28.7, 28.1],      # A34, same 186 set
            "adv": [24.3, 24.0, 24.7],      # A31, same 72 attacks
            "orig": None}                    # never run (X'd out)
NO_REPH = {"nat": 26.0, "adv": 24.5, "orig": 36.1}
ORACLE = 48.2


def diet_mean(cond, diet):
    """Mean over r1 cells + every complete replicate cell in a36_cells.json."""
    vals = list(R1[(cond, diet)])
    for k, r in rep.items():
        tag, ap, c = k.split("|")
        if c == cond and tag[0] == diet and r["legs"] == 12:
            vals.append(r["pooled"])
    return sum(vals) / len(vals), len(vals)


C_BASE, C_RULES = "#8b96a5", "#a3bffa"
ROLES = [("no\nrephraser", None), ("rephraser,\nno rules", None),
         ("rollout\nonly", "s"), ("rollout\n+ train", "b"),
         ("train\nonly", "t")]
CONDS = [("nat", "Natural"), ("adv", "Adversarial"), ("orig", "Original")]

fig, ax = plt.subplots(figsize=(12.8, 4.8))
x = 0.0
ticks, tlabels, gticks, glabels = [], [], [], []
cover = []

ax.bar(x, ORACLE, 0.7, color=C_BASE, edgecolor="#4a5568", lw=0.9)
ax.text(x, ORACLE + 0.5, f"{ORACLE:.1f}", ha="center", fontsize=9, fontweight="bold")
ticks.append(x); tlabels.append("held-out\nbest phrase")
gticks.append(x); glabels.append("Oracle$^{*}$")
x += 1.7

for cond, cname in CONDS:
    gxs = []
    for label, diet in ROLES:
        if diet is None and label.startswith("no\n"):
            v, n = NO_REPH[cond], None
            ax.bar(x, v, 0.7, color=C_BASE, edgecolor="#4a5568", lw=0.9)
        elif diet is None:
            sc = SCAFFOLD[cond]
            if sc is None:
                ax.bar(x, 30, 0.7, color="none", edgecolor="#cbd5e0", lw=0.9, ls=":")
                ax.text(x, 16.2, "not\nmeasured", ha="center", fontsize=6.2,
                        color="#a0aec0")
                ticks.append(x); tlabels.append(label); gxs.append(x); x += 1.0
                continue
            v, n = sum(sc) / len(sc), None
            ax.bar(x, v, 0.7, color=C_BASE, edgecolor="#4a5568", lw=0.9)
        else:
            v, n = diet_mean(cond, diet)
            ax.bar(x, v, 0.7, color=C_RULES, edgecolor="#4a5568", lw=0.9)
            cover.append((cname, label.replace("\n", " "), n))
        ax.text(x, v + 0.5, f"{v:.1f}", ha="center", fontsize=9, fontweight="bold")
        ticks.append(x); tlabels.append(label); gxs.append(x)
        x += 1.12
    gticks.append(sum(gxs) / len(gxs)); glabels.append(cname)
    x += 0.7

ax.set_xticks(ticks)
ax.set_xticklabels(tlabels, fontsize=6.8, color="#4a5568")
for gx, gl in zip(gticks, glabels):
    ax.text(gx, 9.9, gl, ha="center", fontsize=11.5, fontweight="bold",
            transform=ax.transData, clip_on=False)
ax.tick_params(axis="x", length=0)
ax.set_ylim(14, 52)
ax.set_xlim(-0.8, x - 1.0)
ax.set_ylabel("sealed-suite success %", fontsize=11)
ax.grid(axis="y", alpha=0.18)
ax.spines[["top", "right"]].set_visible(False)

handles = [plt.Rectangle((0, 0), 1, 1, color=C_BASE),
           plt.Rectangle((0, 0), 1, 1, color=C_RULES)]
ax.legend(handles, ["baseline", "rulebook (mean over appliers × independent draws)"],
          fontsize=8.2, loc="upper right", framealpha=0.95)

ncells = {f"{c}/{r}": n for c, r, n in cover}
def nc(cond, role):
    return ncells.get(f"{cond}/{role}", "?")
fig.text(0.5, -0.015,
         "Base-weighted over phrase bases; 12 sealed tasks, 24 layouts. "
         "Natural = 186-phrase image-conditioned set (\u00d71 rep); adversarial = 72 attacks (\u00d72); "
         "original = 12 canonicals (\u00d72). Cells averaged per bar (nat/adv/orig) \u2014 "
         f"rollout-only: {nc('Natural','rollout only')}/{nc('Adversarial','rollout only')}/{nc('Original','rollout only')}, "
         f"rollout+train: {nc('Natural','rollout + train')}/{nc('Adversarial','rollout + train')}/{nc('Original','rollout + train')}, "
         f"train-only: 3/3/3 (replicate draws pending). "
         "Oracle$^{*}$: best held-out phrase per task (24\u00d712 grid).",
         ha="center", fontsize=6.8, color="#4a5568")

out = R / "results/charts/rules_summary_bars.png"
fig.savefig(out, dpi=150, bbox_inches="tight", pad_inches=0.28)
print("chart ->", out)
for c, r, n in cover:
    print(f"  {c:<12} {r:<22} mean of {n} cells")
