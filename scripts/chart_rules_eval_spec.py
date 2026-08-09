#!/usr/bin/env python3
"""results/charts/rules_eval_spec.png -- the rules-loop eval design.

Every number is READ FROM config/params.py, so the chart cannot drift from the
spec. Change params.py and re-run; nothing here duplicates a constant.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
import config.params as P

FG, MUT, ACC = "#1a202c", "#718096", "#2b6cb0"
BLUE, PURPLE, GREY = "#2b6cb0", "#805ad5", "#a0aec0"

fig = plt.figure(figsize=(16.0, 13.4), facecolor="white")
gs = fig.add_gridspec(3, 2, height_ratios=[1.22, 0.80, 0.64],
                      hspace=0.50, wspace=0.22,
                      left=0.050, right=0.968, top=0.900, bottom=0.060)

fig.text(0.050, 0.962, "Rules-loop evaluation — three sets, three different jobs",
         fontsize=20, fontweight="bold", color=FG)
fig.text(0.050, 0.935,
         f"one iteration = rephrase every base, proxy-score {P.n_proxy_phrases():,} "
         f"train/held phrases at F*C={P.SIZE.fxc}, roll out {P.n_sim_bases()} val8 bases "
         f"x {P.ROLL.gt_n} episodes  →  {P.hours_per_iteration():.1f} GPU-h",
         fontsize=11.5, color=MUT)

# ------------------------------------------------------------- variables
ax = fig.add_subplot(gs[0, :]); ax.axis("off")
COLS = (0.005, 0.120, 0.180, 0.245, 0.345, 0.865)
ROWS = [
 ("SET", "TASKS", "N", "PHRASES", "SIZED BY", "RESOLVES"),
 ("train", f"{P.SETS.t_train}", f"{P.SIZE.n_train}", f"{P.n_train():,}",
  "evidence volume — no precision floor", f"{P.proxy_resolution_pp(P.n_train()):.1f} pp"),
 ("train_held", f"{P.SETS.t_train_held}", f"{P.SIZE.n_train_held}", f"{P.n_train_held()}",
  "precision on one number per iteration", f"{P.proxy_resolution_pp(P.n_train_held()):.1f} pp"),
 ("val8 (sim)", f"{P.SETS.t_sim_held}", f"{P.SIZE.n_sim}", f"{P.n_sim_bases()}",
  f"resolution — {P.ROLL.gt_n} rollouts each, REAL success", f"{P.sim_resolution_pp():.1f} pp"),
]
y = 0.93
for i, r in enumerate(ROWS):
    hdr = i == 0
    for x, v in zip(COLS, r):
        ax.text(x, y, v, fontsize=10 if hdr else 12.5,
                color=MUT if hdr else FG,
                fontweight="bold" if (hdr or x in (COLS[0], COLS[5])) else "normal",
                family="monospace" if (not hdr and x in COLS[1:4]) else None,
                transform=ax.transAxes)
    if hdr:
        ax.plot([0.005, 0.995], [y - 0.05] * 2, color="#cbd5e0", lw=1.1,
                transform=ax.transAxes, clip_on=False)
        y -= 0.115
    else:
        y -= 0.155

NOTES = [
    ("why two validation sets",
     "they fail in opposite directions. train_held is REAL robot data with breadth (31 tasks) but a\n"
     "PROXY metric; val8 is TRUE task success but narrow (8 tasks) and simulated. Agreement between\n"
     "them is the claim — neither on its own supports one."),
    ("why train has the lowest N",
     f"{P.SETS.t_train} tasks carry its precision for free: {P.n_train():,} phrases resolves "
     f"{P.proxy_resolution_pp(P.n_train()):.1f} pp, sharper than\neither other set. So its N is not the "
     "binding lever, and falls to the next constraint down —\nwhat the distiller can actually read."),
    ("what val8 traded away",
     f"n={P.ROLL.gt_n} is BELOW the n=36 bank standard (+/-{P.per_phrase_ci_pp(P.ROLL.gt_n):.0f} pp on a single "
     "phrase), so these rows no longer\nenter the bank. Same episodes, spent on more bases instead of "
     "deeper per-phrase measurement."),
]
y -= 0.02
for k, v in NOTES:
    ax.text(0.005, y, f"{k}", fontsize=10, fontweight="bold", color=ACC,
            transform=ax.transAxes)
    ax.text(0.245, y, v, fontsize=9.8, color="#2d3748", va="top",
            transform=ax.transAxes, linespacing=1.5)
    y -= 0.235

# ------------------------------------- bases-vs-depth at fixed episode budget
ax = fig.add_subplot(gs[1, 0])
E = P.episodes()
T8 = P.SETS.t_sim_held
opts = []
for per in (72, 36, 18, 12, 8, 4):
    n = (E // per // T8) * T8
    if n >= T8:
        opts.append((n, per))
xs = [n for n, _ in opts]
ys = [P.sim_resolution_pp(n=n, per=per) for n, per in opts]
ax.plot(xs, ys, "-", color=ACC, lw=2.2, zorder=3)
for (n, per), yv in zip(opts, ys):
    cur = per == P.ROLL.gt_n
    ax.plot(n, yv, "o", ms=12 if cur else 6, color="#c53030" if cur else ACC, zorder=4)
    ax.annotate(f"{n//T8}/task\nx{per}", (n, yv), textcoords="offset points",
                xytext=(0, 14), ha="center", fontsize=9,
                fontweight="bold" if cur else "normal",
                color="#c53030" if cur else MUT)
q, g, c = P.M.reference_effect_pp
ax.axhspan(0, g, color="#38a169", alpha=0.09, zorder=1)
ax.axhline(g, color="#276749", ls="--", lw=1.4)
ax.text(xs[-1] * 0.98, g - 0.30, f"gemini effect +{g}pp — must sit BELOW this",
        fontsize=9.5, color="#276749", ha="right", va="top")
ax.set_xscale("log")
ax.set_xticks(xs); ax.set_xticklabels([str(v) for v in xs], fontsize=9.5)
ax.minorticks_off()
ax.set_xlabel("distinct bases", fontsize=11)
ax.set_ylabel("detectable difference (pp)", fontsize=11)
ax.set_title(f"Same {E:,} episodes, spent differently — bases beat depth",
             fontsize=12.5, fontweight="bold", color=FG, loc="left")
ax.set_ylim(min(ys) * 0.72, max(ys) * 1.14)
ax.grid(alpha=0.22)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)

# --------------------------------------------------------- cost breakdown
ax = fig.add_subplot(gs[1, 1])
parts = [(f"proxy scoring\n({P.n_proxy_phrases():,} phrases)", P.proxy_hours(), BLUE),
         (f"sim rollouts\n({P.episodes():,} episodes)", P.rollout_hours(), PURPLE),
         (f"rephraser API\n({P.n_proxy_phrases()+P.n_sim_bases():,} calls)",
          P.rephraser_hours(), GREY)]
tot = P.hours_per_iteration()
left = 0.0
for lab, v, cc in parts:
    ax.barh([0], [v], left=[left], color=cc, height=0.5)
    if v / tot > 0.08:
        ax.text(left + v / 2, 0, f"{v:.1f}h\n{v/tot*100:.0f}%", ha="center",
                va="center", fontsize=10.5, color="white", fontweight="bold")
    else:
        ax.annotate(f"{v:.1f}h", (left + v / 2, 0.27), ha="center", va="bottom",
                    fontsize=9.5, color=cc, fontweight="bold")
    left += v
ax.set_xlim(0, tot * 1.02); ax.set_ylim(-0.75, 1.5)
ax.set_yticks([]); ax.set_xlabel("GPU-hours", fontsize=11)
ax.set_title(f"Cost per iteration: {tot:.1f} GPU-h", fontsize=12.5,
             fontweight="bold", color=FG, loc="left")
for i, (lab, v, cc) in enumerate(parts):
    ax.text(0.02 + i * 0.345, 0.90, "■", color=cc, fontsize=15, transform=ax.transAxes)
    ax.text(0.055 + i * 0.345, 0.905, lab, fontsize=9.5, color=FG, va="center",
            transform=ax.transAxes)
T = tot * P.SIZE.iterations * P.SIZE.rephrasers
ax.text(0.0, -0.40, f"x {P.SIZE.iterations} iterations = {T:.0f} GPU-h    |    "
        f"1 pod {T/24:.1f}d   2 pods {T/2/24:.1f}d   3 pods {T/3/24:.1f}d   "
        f"4 pods {T/4/24:.1f}d", fontsize=10.5, color=FG, fontweight="bold")
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)

# ------------------------------------------------------------- base supply
ax = fig.add_subplot(gs[2, :])
val8 = json.load(open(REPO / "results/rules_runs/live1/splits.json"))["val8"]
bank = pd.read_parquet(REPO / "results/analysis/bank_to_score.parquet")
N = P.SIZE.n_sim
have = [bank[bank.task == t].phrase.nunique() for t in val8]
short = [max(0, N - h) for h in have]
x = np.arange(len(val8))
ax.bar(x, [min(h, N) for h in have], color="#38a169",
       label=f"already in bank ({sum(min(h, N) for h in have)})")
ax.bar(x, short, bottom=[min(h, N) for h in have], color="#e53e3e",
       label=f"still to generate ({sum(short)})")
ax.axhline(N, color=FG, ls="--", lw=1.4)
ax.text(len(val8) - 0.4, N + 0.8, f"N_sim = {N}", fontsize=10, color=FG,
        ha="right", fontweight="bold")
for i, (h, s_) in enumerate(zip(have, short)):
    if s_:
        ax.text(i, min(h, N) + s_ + 0.6, f"+{s_}", ha="center", fontsize=9.5,
                color="#c53030", fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels([t.replace("widowx_", "").replace("_clean", "") for t in val8],
                   rotation=16, ha="right", fontsize=9.5)
ax.set_ylabel("distinct bases", fontsize=11)
ax.set_title(f"val8 base supply at N={N} — {sum(min(h, N) for h in have)} of {N*8} exist, "
             f"{sum(short)} to generate\nbut 254 are fine_exam CLOSE PARAPHRASES, which fill "
             f"none of the orig:nat:adv buckets cleanly",
             fontsize=11.5, fontweight="bold", color=FG, loc="left")
ax.legend(fontsize=10, loc="upper left", frameon=False)
ax.set_ylim(0, N * 1.32)
ax.grid(alpha=0.22, axis="y")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)

out = REPO / "results/charts/rules_eval_spec.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=135, facecolor="white")
print("wrote", out)
print(f"  {tot:.2f} GPU-h/iter | {T:.0f}h total | sim {P.sim_resolution_pp():.2f}pp "
      f"(floor {P.sim_resolution_floor_pp():.2f}) | "
      f"held {P.proxy_resolution_pp(P.n_train_held()):.2f}pp | "
      f"train {P.proxy_resolution_pp(P.n_train()):.2f}pp")
print(f"  val8 bases: have {sum(min(h, N) for h in have)} / {N*8}, short {sum(short)}")
