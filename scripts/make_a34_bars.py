#!/usr/bin/env python3
"""results/charts/a34_natural_bars.png -- A34 natural condition in the Fig-9 idiom.

One pooled bar per (book, applier) with the in-vocab / out-of-vocab strata drawn
as narrow flanking bars, exactly as the paper's main-results figure does. Left
block: the un-rephrased and no-rules references. Dotted lines carry the
un-rephrased levels across every panel. Grey bar = legs still rolling.
"""
import collections
import glob
import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

R = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R / "scripts"))
from build_sealed_assets import SEALED

JOBS = R / "results/rules_runs/r1_sim/jobs"
aud = json.loads((R / "results/analysis/sealed_vocab_audit.json").read_text())
by_nom = {a["nominal"]: ("IV" if a["stratum"] != "OOV" else "OOV") for a in aud}
STRAT = {t: by_nom.get(nom) for t, nom in SEALED.items()}

groups = collections.defaultdict(list)
for f in glob.glob(str(JOBS / "d34*.result.parquet")):
    groups[pathlib.Path(f).name.split("_")[0]].append(f)

def cell(stem):
    fs = groups.get(stem, [])
    if len(fs) < 12:
        return None
    d = pd.concat([pd.read_parquet(x) for x in fs]).dropna(subset=["gt_success"]).copy()
    d["s"] = d.task.map(STRAT)
    return (d.gt_success.mean(),
            d[d.s == "IV"].gt_success.mean(),
            d[d.s == "OOV"].gt_success.mean())

BOOKS = [("s", "rollout-only"), ("b", "combined"), ("t", "train-only")]
APPS = [("cl", "Claude"), ("ge", "Gemini"), ("qw", "Qwen")]
C_RULES, C_BASE = "#7fb3a6", "#9aa3a0"
C_IV, C_OOV = "#b2f5ea", "#fed7aa"
YMIN, YMAX = 15, 42

fig, axes = plt.subplots(1, 4, figsize=(17.5, 5.2), sharey=True,
                         gridspec_kw={"width_ratios": [1.5, 3, 3, 3]})

def triplet(ax, x, trip, color, label_n=True):
    p, iv, oo = trip
    ax.bar(x - 0.21, iv, 0.33, color=C_IV, alpha=0.6, zorder=2)
    ax.bar(x + 0.21, oo, 0.33, color=C_OOV, alpha=0.6, zorder=2)
    ax.text(x - 0.34, iv + 0.35, f"{iv:.0f}", ha="center", fontsize=7, color="#4a5568", zorder=4)
    ax.text(x + 0.34, oo + 0.35, f"{oo:.0f}", ha="center", fontsize=7, color="#4a5568", zorder=4)
    ax.bar(x, p, 0.42, color=color, zorder=3, edgecolor="#3d4a46", lw=0.9)
    ax.text(x, p + 0.45, f"{p:.1f}", ha="center", fontsize=9.5, fontweight="bold", zorder=4)

base = cell("d34basen")
axB = axes[0]
refs, ticks, tlabels = [], [], []
if base:
    refs.append(("un-rephrased", base))
sc = [cell(f"d34sc{ak}n") for ak, _ in APPS]
sc_ok = [s for s in sc if s]
if sc_ok:
    refs.append(("no rules\n(mean)", tuple(sum(s[i] for s in sc_ok) / len(sc_ok) for i in range(3))))
for i, (nm, trip) in enumerate(refs):
    triplet(axB, float(i), trip, C_BASE)
    ticks.append(i); tlabels.append(nm)
if not refs:
    axB.text(0.5, 28, "baseline\nrolling", ha="center", color="#8a8a83", fontsize=10)
axB.set_xticks(ticks); axB.set_xticklabels(tlabels, fontsize=9)
axB.set_xlim(-0.7, max(len(refs) - 0.3, 1.0))
axB.set_title("References", fontsize=11.5, pad=10)
axB.set_ylabel("sealed natural success %", fontsize=11)

for ax, (bk, bname) in zip(axes[1:], BOOKS):
    ticks, tlabels = [], []
    for i, (ak, aname) in enumerate(APPS):
        trip = cell(f"d34{bk}{ak}n")
        if trip:
            triplet(ax, float(i), trip, C_RULES)
        else:
            n = len(groups.get(f"d34{bk}{ak}n", []))
            ax.bar(float(i), YMIN + 1.2, 0.42, color="#d9d9d4", edgecolor="#bcbcb5", zorder=3)
            ax.text(i, YMIN + 1.6, f"{n}/12", ha="center", fontsize=8, color="#82827b", zorder=4)
        ticks.append(i); tlabels.append(aname)
    if base:
        ax.axhline(base[0], color="#c53030", ls=":", lw=1.2, zorder=1)
    ax.set_xticks(ticks); ax.set_xticklabels(tlabels, fontsize=9.5)
    ax.set_xlim(-0.7, len(APPS) - 0.3)
    ax.set_title(bname + " book", fontsize=11.5, pad=10)
    ax.grid(alpha=0.22, axis="y")

axes[0].grid(alpha=0.22, axis="y")
axes[0].set_ylim(YMIN, YMAX)
fig.text(0.5, 0.965, "Sealed natural condition — wider-register set (A34)", ha="center", fontsize=13.5)
fig.text(0.5, 0.925,
         "186 phrases, 3 authors, image-conditioned  ·  24 layouts × 1 rep  ·  "
         "narrow bars: in-vocab (teal) / out-of-vocab (orange)  ·  grey = rolling",
         ha="center", fontsize=8.6, color="#6b7370")
fig.tight_layout(rect=[0, 0.01, 1, 0.90])
out = R / "results/charts/a34_natural_bars.png"
fig.savefig(out, dpi=145, bbox_inches="tight")
print("wrote", out)
