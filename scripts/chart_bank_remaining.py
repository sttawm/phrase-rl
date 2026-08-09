#!/usr/bin/env python3
"""results/charts/bank_remaining.png -- what is LEFT to score, and who can do it.

The worklist is bank_to_score UNION bank_generated: a phrase that exists only in
bank_generated is invisible to run_bank_scoring, which reads bank_to_score.
"""
import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
FG, MUT = "#1a202c", "#718096"
GREEN, RED, AMBER, BLUE = "#38a169", "#e53e3e", "#d69e2e", "#2b6cb0"

tos = pd.read_parquet(REPO / "results/analysis/bank_to_score.parquet")
gen = pd.read_parquet(REPO / "results/analysis/bank_generated.parquet")
work = pd.concat([tos[["task", "phrase", "source"]],
                  gen[["task", "phrase", "source"]]],
                 ignore_index=True).drop_duplicates(["task", "phrase"])
sc = pd.concat([pd.read_parquet(f) for f in
                sorted(glob.glob(str(REPO / "results/analysis/bank_scores_*.parquet")))],
               ignore_index=True)
sc = sc.dropna(subset=["z"]).drop_duplicates(["task", "phrase"])
m = work.merge(sc[["task", "phrase"]].assign(hit=1), on=["task", "phrase"], how="left")
m["src"] = m.source.str.split("|").str[0]
val8 = set(json.load(open(REPO / "results/rules_runs/live1/splits.json"))["val8"])
m["grp"] = ["val8" if t in val8 else
            ("distractor" if str(t).startswith("widowx") else "native") for t in m.task]

tot, done = len(m), int(m.hit.count())
left = tot - done

fig = plt.figure(figsize=(15.6, 8.8), facecolor="white")
gs = fig.add_gridspec(2, 2, height_ratios=[0.62, 1.0], hspace=0.52, wspace=0.22,
                      left=0.185, right=0.965, top=0.865, bottom=0.10)

fig.text(0.045, 0.945, "Bank scoring — what's left", fontsize=21, fontweight="bold",
         color=FG)
fig.text(0.045, 0.905,
         f"{done:,} of {tot:,} scored ({done/tot*100:.1f}%)   |   "
         f"{left} outstanding, every one of them a sim phrase   |   "
         f"native tasks are complete",
         fontsize=12, color=MUT)

# -------------------------------------------------- progress bar by group
ax = fig.add_subplot(gs[0, :])
g = m.groupby("grp").agg(total=("phrase", "size"), scored=("hit", "count"))
g["left"] = g.total - g.scored
g = g.loc[["native", "val8", "distractor"]]
y = np.arange(len(g))
ax.barh(y, g.scored, color=GREEN, height=0.55, label=f"scored ({int(g.scored.sum()):,})")
ax.barh(y, g["left"], left=g.scored, color=RED, height=0.55,
        label=f"outstanding ({int(g['left'].sum())})")
for i, r in enumerate(g.itertuples()):
    ax.text(r.scored / 2, i, f"{r.scored:,}", ha="center", va="center",
            color="white", fontsize=11, fontweight="bold")
    if r.left:
        ax.text(r.scored + r.left + 60, i, f"{r.left} left", va="center",
                color=RED, fontsize=11, fontweight="bold")
ax.set_yticks(y)
ax.set_yticklabels(["native\n213 bridge tasks", "val8\n8 sim tasks",
                    "distractor\n7 variants"], fontsize=10.5)
ax.set_xlabel("phrases", fontsize=11)
ax.set_xlim(0, g.total.max() * 1.16)
ax.legend(fontsize=10, loc="lower right", frameon=False)
ax.grid(alpha=0.2, axis="x")
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)

# ------------------------------------------------- outstanding, by task
ax = fig.add_subplot(gs[1, 0])
o = m[m.hit.isna()]
per = (o.groupby("task").phrase.size().sort_values(ascending=True))
lab = [t.replace("widowx_", "").replace("_clean", "")
        .replace("_lang_common_distract", " +LCdist").replace("_distract", " +dist")
       for t in per.index]
cols = [BLUE if t in val8 else AMBER for t in per.index]
ax.barh(np.arange(len(per)), per.values, color=cols, height=0.7)
for i, v in enumerate(per.values):
    ax.text(v + 0.4, i, str(v), va="center", fontsize=9.5, color=FG, fontweight="bold")
ax.set_yticks(np.arange(len(per)))
ax.set_yticklabels(lab, fontsize=9)
ax.set_xlabel("phrases outstanding", fontsize=11)
ax.set_title("Every outstanding phrase, by task", fontsize=12.5, fontweight="bold",
             color=FG, loc="left")
ax.text(0.62, 0.06, "■ val8   ■ distractor", transform=ax.transAxes, fontsize=10,
        color=MUT)
ax.set_xlim(0, per.max() * 1.22)
ax.grid(alpha=0.2, axis="x")
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)

# ---------------------------------------------------- who can do the work
ax = fig.add_subplot(gs[1, 1]); ax.axis("off")
v8 = int(m[(m.grp == "val8") & m.hit.isna()].shape[0])
ds = int(m[(m.grp == "distractor") & m.hit.isna()].shape[0])
lines = [
    ("WHERE IT RUNS", "", True),
    ("e6 only", f"contexts_sim_val8.parquet is 170 MB and gitignored — it exists\n"
                f"on e6 and nowhere else. e1 tried and reported NO CONTEXTS on\n"
                f"all 15 tasks, then exited 0.", False),
    ("now archived", "packed to results/analysis/data_archive/ so a lost pod no\n"
                     "longer means re-rolling from scratch.", False),
    ("QUEUED", "", True),
    (f"val8  {v8}", "chained on e6 behind the sim rollouts — runs automatically.", False),
    (f"distractor  {ds}", "blocked on the scenes being context-starved: extraction is\n"
                          "success-only and pi0 rarely succeeds in clutter. The top-up\n"
                          "that would fix it is repaired but not yet re-run.\n"
                          "LOW PRIORITY — distractors are out of the loop.", False),
]
y = 0.97
for k, v, hdr in lines:
    if hdr:
        ax.text(0.0, y, k, fontsize=10.5, fontweight="bold", color=BLUE,
                transform=ax.transAxes); y -= 0.085
        continue
    ax.text(0.0, y, k, fontsize=11, fontweight="bold", color=FG, transform=ax.transAxes)
    ax.text(0.30, y, v, fontsize=10, color="#2d3748", va="top", transform=ax.transAxes,
            linespacing=1.55)
    y -= 0.085 + 0.062 * v.count("\n")

out = REPO / "results/charts/bank_remaining.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=135, facecolor="white")
print("wrote", out)
print(f"  {done}/{tot} scored, {left} outstanding (val8 {v8}, distractor {ds}, native 0)")
