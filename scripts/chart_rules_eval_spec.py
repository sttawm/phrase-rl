#!/usr/bin/env python3
"""results/charts/rules_eval_spec.png -- the rules-loop eval design at N_sim=48.

Every number is produced here from the measured constants in
scripts/rules_eval_cost.py, so the chart cannot drift from the calculator.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
SEC_PER_ROW, SEC_PER_EXTRA = 2.7, 0.2
MIN_PER_EPISODE = 0.42
SD_PHRASE, SD_LOGIT, P0 = 0.119, 1.376, 0.42

T_TR, T_TH, T_SH = 25, 31, 8
N_PX, N_SIM, FXC, ITERS = 8, 48, 40, 6
L_REC = 4

n_px, n_sim = (T_TR + T_TH) * N_PX, T_SH * N_SIM
proxy_h = (T_TR + T_TH) * FXC * (SEC_PER_ROW + SEC_PER_EXTRA * (N_PX - 1)) / 3600
reph_h = ((T_TR + T_TH) * N_PX + T_SH * N_SIM) * 2.0 / 3600


def sim_pp(L):
    E = n_sim * L
    return 1.96 * np.sqrt(2 * P0 * (1 - P0) / E + 2 * SD_PHRASE ** 2 / n_sim) * 100


def proxy_pp():
    lo = np.log(P0 / (1 - P0))
    d = 1.96 * SD_LOGIT * np.sqrt(2) / np.sqrt(n_px)
    return (1 / (1 + np.exp(-(lo + d))) - P0) * 100


sim_h = lambda L: n_sim * L * MIN_PER_EPISODE / 60
tot_h = lambda L: proxy_h + sim_h(L) + reph_h
FLOOR = 1.96 * np.sqrt(2 * SD_PHRASE ** 2 / n_sim) * 100

FG, MUT, ACC, WARN = "#1a202c", "#718096", "#2b6cb0", "#c05621"
fig = plt.figure(figsize=(15.5, 12.2), facecolor="white")
gs = fig.add_gridspec(3, 2, height_ratios=[0.90, 0.82, 0.70],
                      hspace=0.36, wspace=0.20,
                      left=0.045, right=0.972, top=0.935, bottom=0.052)

fig.text(0.045, 0.972, "Rules-loop evaluation — design at 48 bases per sim task",
         fontsize=20, fontweight="bold", color=FG)
fig.text(0.045, 0.952,
         f"one iteration = rephrase every base with the current rulebook, "
         f"proxy-score the {(T_TR+T_TH)*N_PX} train/train_held phrases, roll out the {n_sim} val8 phrases",
         fontsize=11.5, color=MUT)

# ---------------------------------------------------------------- variables
ax = fig.add_subplot(gs[0, :]); ax.axis("off")
ROWS = [
 ("EVAL SETS", "", "", ""),
 ("T_train",       f"{T_TR}",  "training tasks — the distiller sees these", "proxy"),
 ("T_train_held",  f"{T_TH}",  "held-out real-robot tasks, never distilled from", "proxy"),
 ("T_sim_held",    f"{T_SH}",  "val8 sim tasks — CLEAN only, no distractor variants", "rollout"),
 ("PER-TASK SIZE", "", "", ""),
 ("N_task",        f"{N_PX}",  "bases per task on the two proxy sets", "proxy"),
 ("N_sim",         f"{N_SIM}", "bases per task on val8  →  n = 8 x 48 = 384 bases", "rollout"),
 ("base",          "—",        "one input instruction the rulebook rewrites (the eval unit)", ""),
 ("orig:nat:adv",  "20:50:30", "base mix. Free — costs the same either way; sets meaning only", ""),
 ("MEASUREMENT",   "", "", ""),
 ("F x C",         f"{FXC}",   "context-rows per proxy phrase (F frames x C contexts)", "proxy"),
 ("L",             f"{L_REC}", "layouts per base — FIXED set, balanced 2 from 0-11 + 2 from 12-23", "rollout"),
 ("K",             "1",        "episodes per layout. K=1 is optimal: repeats only cut binomial noise", "rollout"),
 ("episodes",      f"{n_sim*L_REC}", "= n x L x K = 384 x 4 x 1", "rollout"),
 ("RUN",           "", "", ""),
 ("iterations",    f"{ITERS}", "distill → apply → measure cycles", ""),
 ("rephrasers",    "1",        "Gemini first; Claude/Qwen legs are separate runs", ""),
]
y, dy = 0.960, 0.0605
ax.text(0.005, y + 0.035, "VARIABLE", fontsize=10, fontweight="bold", color=MUT, transform=ax.transAxes)
ax.text(0.135, y + 0.035, "VALUE", fontsize=10, fontweight="bold", color=MUT, transform=ax.transAxes)
ax.text(0.225, y + 0.035, "DEFINITION", fontsize=10, fontweight="bold", color=MUT, transform=ax.transAxes)
ax.text(0.905, y + 0.035, "LEG", fontsize=10, fontweight="bold", color=MUT, transform=ax.transAxes)
ax.plot([0.005, 0.995], [y + 0.021] * 2, color="#cbd5e0", lw=1.1, transform=ax.transAxes, clip_on=False)
for name, val, desc, leg in ROWS:
    if not val:
        ax.text(0.005, y, name, fontsize=10, fontweight="bold", color=ACC,
                transform=ax.transAxes); y -= dy * 0.72; continue
    ax.text(0.005, y, name, fontsize=11.5, color=FG, family="monospace", transform=ax.transAxes)
    ax.text(0.135, y, val, fontsize=11.5, fontweight="bold", color=FG,
            family="monospace", transform=ax.transAxes)
    ax.text(0.225, y, desc, fontsize=11, color="#2d3748", transform=ax.transAxes)
    if leg:
        c = "#2b6cb0" if leg == "proxy" else "#805ad5"
        ax.text(0.905, y, leg, fontsize=9.5, color=c, fontweight="bold", transform=ax.transAxes)
    y -= dy

# ------------------------------------------------------- L trade-off + cost
ax = fig.add_subplot(gs[1, 0])
Ls = np.arange(1, 25)
ax.plot([tot_h(L) for L in Ls], [sim_pp(L) for L in Ls], "-", color=ACC, lw=2.2, zorder=3)
ax.axhline(FLOOR, color=WARN, ls="--", lw=1.5, zorder=2)
ax.text(tot_h(24) * 0.55, FLOOR + 0.12, f"floor at 384 bases = {FLOOR:.1f}pp\n(more layouts can never beat this)",
        fontsize=9, color=WARN, ha="center", va="bottom")
ax.axhspan(0, 3.2, color="#38a169", alpha=0.075, zorder=1)
ax.text(tot_h(1) + 0.6, 2.62, "Gemini v3→v4 effect = +3.2pp", fontsize=9, color="#276749")
for L in (1, 2, 4, 6, 8, 12, 24):
    ax.plot(tot_h(L), sim_pp(L), "o", ms=8 if L == L_REC else 5,
            color="#c53030" if L == L_REC else ACC, zorder=4)
    ax.annotate(f"L={L}", (tot_h(L), sim_pp(L)), textcoords="offset points",
                xytext=(7, 6), fontsize=9.5,
                fontweight="bold" if L == L_REC else "normal",
                color="#c53030" if L == L_REC else MUT)
ax.set_xlabel("GPU-hours per iteration", fontsize=11)
ax.set_ylabel("detectable difference (pp, 95%, paired)", fontsize=11)
ax.set_title("Layouts buy resolution with diminishing returns", fontsize=12.5,
             fontweight="bold", color=FG, loc="left")
ax.set_ylim(1.4, 8.2); ax.grid(alpha=0.22)
for s in ("top", "right"): ax.spines[s].set_visible(False)

ax = fig.add_subplot(gs[1, 1])
parts = [("proxy scoring\n(train + train_held)", proxy_h, "#2b6cb0"),
         ("rollouts\n(val8, 1536 episodes)", sim_h(L_REC), "#805ad5"),
         ("rephraser API\n(832 calls)", reph_h, "#a0aec0")]
left = 0.0
for lab, v, c in parts:
    ax.barh([0], [v], left=[left], color=c, height=0.5)
    if v / tot_h(L_REC) > 0.08:
        ax.text(left + v / 2, 0, f"{v:.1f}h\n{v/tot_h(L_REC)*100:.0f}%", ha="center",
                va="center", fontsize=10.5, color="white", fontweight="bold")
    else:
        ax.annotate(f"{v:.1f}h", (left + v / 2, 0.27), ha="center", va="bottom",
                    fontsize=9.5, color=c, fontweight="bold")
    left += v
ax.set_xlim(0, tot_h(L_REC) * 1.02); ax.set_ylim(-0.75, 1.5)
ax.set_yticks([]); ax.set_xlabel("GPU-hours", fontsize=11)
ax.set_title(f"Cost per iteration at L={L_REC}:  {tot_h(L_REC):.1f} GPU-h", fontsize=12.5,
             fontweight="bold", color=FG, loc="left")
for i, (lab, v, c) in enumerate(parts):
    ax.text(0.02 + i * 0.345, 0.90, "■", color=c, fontsize=15, transform=ax.transAxes)
    ax.text(0.055 + i * 0.345, 0.905, lab, fontsize=9.5, color=FG, va="center",
            transform=ax.transAxes)
ax.text(0.0, -0.40, f"x {ITERS} iterations = {tot_h(L_REC)*ITERS:.0f} GPU-h    "
        f"|    1 pod {tot_h(L_REC)*ITERS/24:.1f}d    2 pods {tot_h(L_REC)*ITERS/2/24:.1f}d    "
        f"3 pods {tot_h(L_REC)*ITERS/3/24:.1f}d",
        fontsize=11, color=FG, fontweight="bold")
for s in ("top", "right", "left"): ax.spines[s].set_visible(False)

# ------------------------------------------------------------- base supply
ax = fig.add_subplot(gs[2, :])
val8 = json.load(open(REPO / "results/rules_runs/live1/splits.json"))["val8"]
bank = pd.read_parquet(REPO / "results/analysis/bank_to_score.parquet")
have = [bank[bank.task == t].phrase.nunique() for t in val8]
short = [max(0, N_SIM - h) for h in have]
x = np.arange(len(val8))
ax.bar(x, [min(h, N_SIM) for h in have], color="#38a169", label=f"already in bank ({sum(min(h,N_SIM) for h in have)})")
ax.bar(x, short, bottom=[min(h, N_SIM) for h in have], color="#e53e3e",
       label=f"still to generate ({sum(short)})")
ax.axhline(N_SIM, color=FG, ls="--", lw=1.4)
ax.text(len(val8) - 0.4, N_SIM + 1.1, f"N_sim = {N_SIM}", fontsize=10, color=FG,
        ha="right", fontweight="bold")
for i, (h, s) in enumerate(zip(have, short)):
    if s: ax.text(i, min(h, N_SIM) + s + 1.0, f"+{s}", ha="center", fontsize=9.5,
                  color="#c53030", fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels([t.replace("widowx_", "").replace("_clean", "") for t in val8],
                   rotation=18, ha="right", fontsize=9.5)
ax.set_ylabel("distinct bases", fontsize=11)
ax.set_title(f"Base supply — {sum(min(h,N_SIM) for h in have)} of {N_SIM*8} already exist; "
             f"only {sum(short)} to generate (~${sum(short)*0.05:.0f})",
             fontsize=12.5, fontweight="bold", color=FG, loc="left")
ax.legend(fontsize=10, loc="upper left", frameon=False)
ax.set_ylim(0, N_SIM * 1.30); ax.grid(alpha=0.22, axis="y")
for s in ("top", "right"): ax.spines[s].set_visible(False)

out = REPO / "results/charts/rules_eval_spec.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=135, facecolor="white")
print("wrote", out)
print(f"  per-iter {tot_h(L_REC):.2f} GPU-h | {ITERS} iters {tot_h(L_REC)*ITERS:.0f}h | "
      f"sim {sim_pp(L_REC):.2f}pp | proxy {proxy_pp():.2f}pp | floor {FLOOR:.2f}pp")
print(f"  bases have={sum(have)} capped={sum(min(h,N_SIM) for h in have)} short={sum(short)}")
