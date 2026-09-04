#!/usr/bin/env python3
"""results/charts/sealed_bars_<cond>.png -- the final sealed figures, Fig-9 idiom.

  .venv/bin/python scripts/make_sealed_bars.py adversarial|natural|original

One pooled bar per (book, applier) with in-vocab / out-of-vocab strata flanking.
Left panel: oracle (held-out, sealed_headroom_confirmed.json), no rephraser, and
no rules. Grey = legs still rolling.

Sources per condition:
  adversarial  A31 book legs b31<book><ap>a ; A30 scaffolds a30<ap>adv ;
               no-rephraser = a29pass (60 new attacks) + anchors passthrough (12)
  original     A31 book legs b31<book><ap>o ; no-rephraser = anchors 'originals';
               no-rules was not run (user decision 2026-09-03)
  natural      A34 book legs d34<book><ap>n ; scaffolds d34sc<ap>n ;
               no-rephraser = d34basen
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

COND = sys.argv[1] if len(sys.argv) > 1 else "natural"
JOBS = R / "results/rules_runs/r1_sim/jobs"
aud = json.loads((R / "results/analysis/sealed_vocab_audit.json").read_text())
by_nom = {a["nominal"]: ("IV" if a["stratum"] != "OOV" else "OOV") for a in aud}
STRAT = {t: by_nom.get(nom) for t, nom in SEALED.items()}

groups = collections.defaultdict(list)
for f in glob.glob(str(JOBS / "*.result.parquet")):
    groups[pathlib.Path(f).name.split("_")[0]].append(f)

def strata(d):
    d = d.dropna(subset=["gt_success"]).copy()
    d["s"] = d.task.map(STRAT)
    return (d.gt_success.mean(),
            d[d.s == "IV"].gt_success.mean(),
            d[d.s == "OOV"].gt_success.mean())

def legs(stem, need=12):
    fs = groups.get(stem, [])
    if len(fs) < need:
        return None, len(fs)
    return strata(pd.concat([pd.read_parquet(x) for x in fs])), len(fs)

SPEC = {
    "adversarial": dict(book=lambda b, a: f"b31{b}{a}a", scaf=lambda a: f"a30{a}adv",
                        title="Sealed adversarial condition (72 attacks/task, 24x2)"),
    "original":    dict(book=lambda b, a: f"b31{b}{a}o", scaf=lambda a: None,
                        title="Sealed original condition (12 nominals, 24x2)"),
    "natural":     dict(book=lambda b, a: f"d34{b}{a}n", scaf=lambda a: f"d34sc{a}n",
                        title="Sealed natural condition, wider-register set (186 phrases, 24x1)"),
}[COND]

# --- references ------------------------------------------------------------
hd = json.loads((R / "results/analysis/sealed_headroom_confirmed.json").read_text())
orc = pd.DataFrame(hd["tasks"])
orc["s"] = orc.task.map(STRAT)
ORACLE = (orc.oracle_pct.mean(), orc[orc.s == "IV"].oracle_pct.mean(),
          orc[orc.s == "OOV"].oracle_pct.mean())

def anchors(arm):
    a = pd.read_parquet(R / "results/sealed/anchors_x12.parquet")
    a = a[a.arm == arm].groupby(["task", "phrase"]).success.mean().reset_index()
    a["gt_success"] = 100 * a.success
    return strata(a)

if COND == "natural":
    NOREPH, nr_n = legs("d34basen")
elif COND == "original":
    NOREPH, nr_n = anchors("originals"), 12
else:
    p60, n60 = legs("a29pass")
    pt = pd.read_parquet(R / "results/sealed/anchors_x12.parquet")
    pt = pt[pt.arm == "passthrough"].groupby(["task", "phrase"]).success.mean().reset_index()
    pt["gt_success"] = 100 * pt.success
    if p60:
        fs = groups.get("a29pass", [])
        d60 = pd.concat([pd.read_parquet(x) for x in fs])[["task", "phrase", "gt_success"]]
        NOREPH = strata(pd.concat([d60, pt[["task", "phrase", "gt_success"]]]))
    else:
        NOREPH = None
    nr_n = n60

sc = [legs(SPEC["scaf"](a))[0] for a, _ in [("cl", 0), ("ge", 0), ("qw", 0)]] \
     if SPEC["scaf"]("cl") else []
sc_ok = [s for s in sc if s]
SCAF = tuple(sum(s[i] for s in sc_ok) / len(sc_ok) for i in range(3)) if sc_ok else None

BOOKS = [("s", "rollout-only"), ("b", "combined"), ("t", "train-only")]
APPS = [("cl", "Claude"), ("ge", "Gemini"), ("qw", "Qwen")]
C_RULES, C_IV, C_OOV = "#7fb3a6", "#b2f5ea", "#fed7aa"
YMIN, YMAX = 12, 66

fig, axes = plt.subplots(1, 4, figsize=(18.6, 5.2), sharey=True,
                         gridspec_kw={"width_ratios": [2.6, 3, 3, 3]})

def triplet(ax, x, trip, color):
    p, iv, oo = trip
    ax.bar(x - 0.21, iv, 0.33, color=C_IV, alpha=0.6, zorder=2)
    ax.bar(x + 0.21, oo, 0.33, color=C_OOV, alpha=0.6, zorder=2)
    ax.text(x - 0.34, iv + 0.5, f"{iv:.0f}", ha="center", fontsize=7, color="#4a5568", zorder=4)
    ax.text(x + 0.34, oo + 0.5, f"{oo:.0f}", ha="center", fontsize=7, color="#4a5568", zorder=4)
    ax.bar(x, p, 0.42, color=color, zorder=3, edgecolor="#3d4a46", lw=0.9)
    ax.text(x, p + 0.6, f"{p:.1f}", ha="center", fontsize=9.5, fontweight="bold", zorder=4)

def pending(ax, x, n, need=12):
    ax.bar(x, YMIN + 1.0, 0.42, color="#d9d9d4", edgecolor="#bcbcb5", zorder=3)
    ax.text(x, YMIN + 1.4, f"{n}/{need}", ha="center", fontsize=8, color="#82827b", zorder=4)

REFS = [("oracle\n(held-out)", ORACLE, "#2f855a"),
        ("no rephraser", NOREPH, "#4a5568"),
        ("no rules\n(scaffold)", SCAF, "#9aa3a0")]
axB = axes[0]
ticks, tlabels = [], []
for i, (nm, trip, col) in enumerate(REFS):
    if trip:
        triplet(axB, float(i), trip, col)
    elif nm.startswith("no rules") and SPEC["scaf"]("cl") is None:
        axB.text(i, (YMIN + YMAX) / 2, "not run", ha="center", va="center",
                 fontsize=9, color="#9a9a93", rotation=90)
    else:
        pending(axB, float(i), nr_n)
    ticks.append(i); tlabels.append(nm)
axB.set_xticks(ticks); axB.set_xticklabels(tlabels, fontsize=8.5)
axB.set_xlim(-0.7, len(REFS) - 0.3)
axB.set_title("References", fontsize=11.5, pad=10)
axB.set_ylabel(f"sealed {COND} success %", fontsize=11)
axB.grid(alpha=0.22, axis="y")
axB.set_ylim(YMIN, YMAX)

for ax, (bk, bname) in zip(axes[1:], BOOKS):
    ticks, tlabels = [], []
    for i, (ak, aname) in enumerate(APPS):
        trip, n = legs(SPEC["book"](bk, ak))
        triplet(ax, float(i), trip, C_RULES) if trip else pending(ax, float(i), n)
        ticks.append(i); tlabels.append(aname)
    if NOREPH:
        ax.axhline(NOREPH[0], color="#c53030", ls=":", lw=1.2, zorder=1)
    ax.set_xticks(ticks); ax.set_xticklabels(tlabels, fontsize=9.5)
    ax.set_xlim(-0.7, len(APPS) - 0.3)
    ax.set_title(bname + " book", fontsize=11.5, pad=10)
    ax.grid(alpha=0.22, axis="y")

fig.text(0.5, 0.965, SPEC["title"], ha="center", fontsize=13.5)
fig.text(0.5, 0.925,
         "narrow bars: in-vocab (teal) / out-of-vocab (orange)  ·  dotted line: no-rephraser level  ·  "
         "oracle = best phrase per task, selected on layouts 0-17 and scored on 18-23  ·  grey = rolling",
         ha="center", fontsize=8.4, color="#6b7370")
fig.tight_layout(rect=[0, 0.01, 1, 0.90])
out = R / f"results/charts/sealed_bars_{COND}.png"
fig.savefig(out, dpi=145, bbox_inches="tight")
print("wrote", out)
