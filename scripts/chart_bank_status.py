#!/usr/bin/env python3
"""results/charts/bank_status.png -- where the phrase bank stands.

Two DIFFERENT measurements live in the bank and the chart keeps them apart:
  PROXY  z/grip from the score server, F*C=64 context-rows. Cheap, every task.
  GT     real rollout success, n=36 episodes. Expensive, sim tasks only.
A phrase can have one, both, or neither.
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
GREEN, RED, BLUE, PURPLE, AMBER = "#38a169", "#e53e3e", "#2b6cb0", "#805ad5", "#d69e2e"

tos = pd.read_parquet(REPO / "results/analysis/bank_to_score.parquet")
sc = pd.concat([pd.read_parquet(f) for f in
                sorted(glob.glob(str(REPO / "results/analysis/bank_scores_*.parquet")))],
               ignore_index=True)
m = tos.merge(sc[["task", "phrase"]].drop_duplicates().assign(hit=1),
              on=["task", "phrase"], how="left")
m["src"] = m.source.str.split("|").str[0]
m["is_sim"] = m.task.astype(str).str.startswith("widowx")

gt = pd.concat([pd.read_parquet(REPO / f)[["task", "phrase", "gt_success", "gt_n"]]
                for f in ("results/analysis/fine_exam_phrases.parquet",
                          "results/analysis/sim_rollouts_natadv_0of2.parquet")],
               ignore_index=True).drop_duplicates(["task", "phrase"])
val8 = json.load(open(REPO / "results/rules_runs/live1/splits.json"))["val8"]

fig = plt.figure(figsize=(16.0, 10.6), facecolor="white")
gs = fig.add_gridspec(2, 2, height_ratios=[0.78, 1.0], width_ratios=[1.0, 1.28],
                      hspace=0.40, wspace=0.24, left=0.115, right=0.955,
                      top=0.855, bottom=0.085)

tot, done = len(m), int(m.hit.count())
fig.text(0.045, 0.955, "Phrase bank status", fontsize=21, fontweight="bold", color=FG)
fig.text(0.045, 0.925,
         f"proxy scoring {done:,} / {tot:,} = {done/tot*100:.1f}%   "
         f"|   ground-truth rollouts {len(gt):,} phrases at n=36   "
         f"|   {tot-done} rows outstanding, all sim",
         fontsize=12, color=MUT)

# ---- panel 1: proxy coverage by source tier -------------------------------
ax = fig.add_subplot(gs[0, 0])
t = (m.groupby("src").agg(total=("phrase", "size"), scored=("hit", "count"))
     .sort_values("total", ascending=True))
t["missing"] = t.total - t.scored
y = np.arange(len(t))
ax.barh(y, t.scored, color=GREEN, label=f"scored ({t.scored.sum():,})")
ax.barh(y, t.missing, left=t.scored, color=RED, label=f"outstanding ({t.missing.sum()})")
for i, (s, ms) in enumerate(zip(t.scored, t.missing)):
    ax.text(s / 2, i, f"{s:,}", ha="center", va="center", color="white",
            fontsize=10, fontweight="bold")
    if ms:
        ax.text(s + ms + 30, i, f"+{ms}", va="center", color=RED, fontsize=10,
                fontweight="bold")
ax.set_yticks(y)
ax.set_yticklabels([s.replace("generated_", "gen ").replace("_", " ") for s in t.index],
                   fontsize=10.5)
ax.set_xlabel("phrases", fontsize=11)
ax.set_title("Proxy coverage by source tier", fontsize=13, fontweight="bold",
             color=FG, loc="left")
ax.legend(fontsize=9.5, loc="lower right", frameon=False)
ax.set_xlim(0, t.total.max() * 1.14)
ax.grid(alpha=0.2, axis="x")
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)

# ---- panel 2: the two measurements ----------------------------------------
ax = fig.add_subplot(gs[0, 1])
sim = m[m.is_sim]
cats = [
    ("native tasks\n(213 bridge instructions)", len(m) - len(sim), len(m[~m.is_sim].dropna(subset=["hit"])), 0),
    ("sim tasks\n(15 widowx)", len(sim), int(sim.hit.count()), len(gt)),
]
x = np.arange(len(cats))
w = 0.27
ax.bar(x - w, [c[1] for c in cats], w, color="#cbd5e0", label="in bank")
ax.bar(x, [c[2] for c in cats], w, color=BLUE, label="proxy scored (F*C=64)")
ax.bar(x + w, [c[3] for c in cats], w, color=PURPLE, label="gt rollouts (n=36)")
for i, c in enumerate(cats):
    for off, v in ((-w, c[1]), (0, c[2]), (w, c[3])):
        if v:
            ax.text(i + off, v + 55, f"{v:,}", ha="center", fontsize=9.5, fontweight="bold",
                    color=FG)
ax.set_xticks(x)
ax.set_xticklabels([c[0] for c in cats], fontsize=10.5)
ax.set_ylabel("phrases", fontsize=11)
ax.set_title("Proxy is cheap and universal; gt is expensive and sim-only", fontsize=13, fontweight="bold",
             color=FG, loc="left")
ax.legend(fontsize=9.5, frameon=False, loc="upper right")
ax.set_ylim(0, 4900)
ax.grid(alpha=0.2, axis="y")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)

# ---- panel 3: per val8 task ------------------------------------------------
ax = fig.add_subplot(gs[1, :])
rows = []
for t_ in val8:
    sub = m[m.task == t_]
    g = gt[gt.task == t_]
    rows.append((t_.replace("widowx_", "").replace("_clean", ""), len(sub),
                 int(sub.hit.count()), len(g),
                 float(g.gt_success.mean()) if len(g) else np.nan))
r = pd.DataFrame(rows, columns=["task", "total", "proxy", "ngt", "succ"]).sort_values("succ")
x = np.arange(len(r))
w = 0.26
ax.bar(x - w, r.total, w, color="#cbd5e0", label="phrases in bank")
ax.bar(x, r.proxy, w, color=BLUE, label="proxy scored")
ax.bar(x + w, r.ngt, w, color=PURPLE, label="gt rollouts (n=36)")
ax.axhline(16, color=FG, ls="--", lw=1.5)
ax.text(len(r) - 0.35, 16.9, "N=16 needed per task", fontsize=10, ha="right",
        fontweight="bold", color=FG)
for i, row in enumerate(r.itertuples()):
    if row.total > row.proxy:
        ax.text(i, row.total + 1.4, f"+{row.total-row.proxy}", ha="center",
                fontsize=9, color=RED, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(r.task, rotation=16, ha="right", fontsize=10)
ax.set_ylabel("phrases", fontsize=11)
ax.set_title("val8 — every task clears N=16 already; the red gaps are the 12 generated phrases per task awaiting proxy scores", fontsize=13, fontweight="bold",
             color=FG, loc="left")
ax.legend(fontsize=10, frameon=False, loc="upper left")
ax.set_ylim(0, r.total.max() * 1.22)
ax.grid(alpha=0.2, axis="y")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)

ax2 = ax.twinx()
ax2.plot(x, r.succ, "o-", color=AMBER, lw=2, ms=7, zorder=5,
         label="rollout success (right axis)")
for i, s in enumerate(r.succ):
    if not np.isnan(s):
        ax2.annotate(f"{s:.0f}%", (i, s), textcoords="offset points", xytext=(0, 9),
                     ha="center", fontsize=9, color="#975a16", fontweight="bold")
ax2.set_ylabel("rollout success (%)", fontsize=11, color="#975a16")
ax2.set_ylim(0, 100)
ax2.tick_params(axis="y", colors="#975a16")
ax2.legend(fontsize=10, frameon=False, loc="upper center")
for s in ("top", "left"):
    ax2.spines[s].set_visible(False)

out = REPO / "results/charts/bank_status.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=135, facecolor="white")
print("wrote", out)
print(f"  proxy {done}/{tot} ({done/tot*100:.1f}%)  outstanding {tot-done}  gt {len(gt)}")
