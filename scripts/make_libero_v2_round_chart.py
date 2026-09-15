#!/usr/bin/env python3
"""results/charts/libero_v2_<prefix>.png — round chart for the pi0.5/LIBERO sealed-v2 eval.

Left: mean success per arm (no-rephraser / scaffold / rulebook draws) pooled and
split in- vs out-of-finetune, with the paired delta vs the no-rephraser arm and
its sign-flip p above each bar. Right: per-task means per arm, tasks sorted by
baseline, so the reader sees where an arm moves anything. Data: the same
loaders as scripts/analyze_libero_v2.py.

  .venv/bin/python scripts/make_libero_v2_round_chart.py [--prefix v2r1] [--out libero_v2_round1]
"""
import argparse
import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

R = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R / "scripts"))
import analyze_libero_v2 as A  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--run", default="p_v2")
ap.add_argument("--prefix", default="v2r1")
ap.add_argument("--applies-dir", default=str(A.B / "eval_applies_v2"))
ap.add_argument("--bases", default=str(A.B / "naturals_v2.parquet"))
ap.add_argument("--cells", default=str(A.B / "libero_v2_round1_cells.json"))
ap.add_argument("--out", default="libero_v2_round1")
a = ap.parse_args()

files, rates = A.load_results(a.run, a.prefix)
bases = pd.read_parquet(a.bases)[["task", "phrase"]]
bases["task"] = bases.task.map(A.norm_task)
bases["phrase"] = bases.phrase.astype(str).str.strip()
bases = bases.drop_duplicates(["task", "phrase"]).reset_index(drop=True)
none = bases.rename(columns={"phrase": "base"}).assign(phrase=lambda d: d.base)
arms = [("none", none.set_index(["task", "base"])[["phrase"]])] + A.load_arms(pathlib.Path(a.applies_dir), bases)
succ = {lab: m.join(rates, on=["task", "phrase"]).succ for lab, m in arms}
cells = json.load(open(a.cells))["arms"]
NAMES = {"none": "no rephraser"}
def nice(lab):
    return NAMES.get(lab, lab.split("|")[0].replace("v2_", "book ").replace("draw", "draw "))

labels = [lab for lab, _ in arms]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.4), gridspec_kw={"width_ratios": [1, 1.35]})
# ---- left: means per split with paired delta vs none -----------------------------
splits = [("pooled", "pooled"), ("in", "in-finetune (10 tasks)"), ("out", "out-of-finetune (10 tasks)")]
w = 0.8 / len(labels)
colors = plt.cm.tab10(np.arange(len(labels)))
for i, lab in enumerate(labels):
    c = cells.get(lab, {})
    xs = np.arange(len(splits)) + (i - (len(labels) - 1) / 2) * w
    ys = [c.get(f"mean_{s}") or 0 for s, _ in splits]
    ax1.bar(xs, ys, w * 0.92, color=colors[i], label=nice(lab), zorder=3)
    for x, y, (s, _) in zip(xs, ys, splits):
        v = c.get("vs_none", {})
        d, p = v.get(f"delta_{s}"), v.get(f"p_{s}")
        txt = f"{y:.0f}" if lab == "none" or d is None else f"{y:.0f}\nΔ{d:+.1f}\np={p:.2g}"
        ax1.text(x, y + 1, txt, ha="center", va="bottom", fontsize=7.2, zorder=4)
ax1.set_xticks(range(len(splits))); ax1.set_xticklabels([n for _, n in splits], fontsize=9)
ax1.set_ylim(0, 112); ax1.set_ylabel("success % (mean over bases, n=50 inits each)")
ax1.grid(axis="y", color="0.9", zorder=0); ax1.legend(fontsize=8, frameon=False, loc="upper right")
ax1.set_title("arms by half — Δ and p: paired sign-flip vs no rephraser", fontsize=10)
# ---- right: per-task means per arm ------------------------------------------------
# per-task means over the bases rated in EVERY arm (paired), so a half-rolled
# wave cannot show an arm on a different phrase subset
S = pd.DataFrame(succ).dropna()
T = S.groupby(level=0).mean()
T["fin"] = ["out" if t.startswith("libero_90") else "in" for t in T.index]
T = T.sort_values(["fin", "none"], ascending=[True, False])
y = np.arange(len(T))
for i, lab in enumerate(labels):
    ax2.scatter(T[lab], y, s=34, color=colors[i], label=nice(lab), zorder=3,
                marker="o" if lab == "none" else ("s" if "scaffold" in lab else "D"))
for yi, (t, r) in enumerate(T.iterrows()):
    ax2.plot([T.loc[t, labels].min(), T.loc[t, labels].max()], [yi, yi], color="0.8", lw=1, zorder=2)
ax2.set_yticks(y); ax2.set_yticklabels([t.replace(":", "/") for t in T.index], fontsize=8)
ax2.axhline((T.fin == "in").sum() - 0.5, color="0.5", lw=0.8, ls="--")
ax2.text(1, (T.fin == "in").sum() - 0.5, " in-finetune ↑   out-of-finetune ↓", fontsize=8, va="center", color="0.4")
ax2.set_xlim(-2, 102); ax2.set_xlabel("success % (mean of the task's 10 natural phrases)")
ax2.grid(axis="x", color="0.9", zorder=0); ax2.invert_yaxis()
ax2.legend(fontsize=8, frameon=False, loc="lower left")
ax2.set_title("per sealed task", fontsize=10)
n_files, n_missing = len(files), int(len(bases) - len(S))
fig.suptitle(f"Sealed set v2, round {a.prefix[-1]}: 200 natural test phrases × 50 inits, frozen $\\pi_{{0.5}}$ — "
             f"{n_files} legs" + (f", {n_missing} bases not yet rated in every arm" if n_missing else ""), fontsize=11)
fig.tight_layout()
out = R / "results/charts" / f"{a.out}.png"
fig.savefig(out, dpi=170); print("->", out)
