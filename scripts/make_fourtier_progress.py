#!/usr/bin/env python3
"""results/charts/fourtier_progress.png -- live pi0.5/LIBERO four-tier progress.

Reads the pulled shard logs in results/analysis/fourtier_live/ (see
scripts/pull_fourtier.sh). Episodes are deduped on (suite, task, phrase, init)
-- the measurement key; arm labels are metadata. Four panels:
  1. pooled tier success by suite (orig / natural / adversarial / oracle
     confirm-winner), Wilson 95% intervals, n annotated;
  2. per-task heatmap (tasks x tiers), text = success %;
  3. board-search status per task (screen best vs canonical, confirm winner);
  4. episode throughput per shard from row timestamps -> finish estimate.

  .venv/bin/python scripts/make_fourtier_progress.py
"""
import glob
import json
import math
import os
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
LIVE = ROOT / "results/analysis/fourtier_live"

rows = []
for f in sorted(glob.glob(str(LIVE / "fourtier_*.jsonl"))):
    shard = pathlib.Path(f).stem.split("_")[1]
    for line in open(f):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
            r["shard"] = shard
            rows.append(r)
        except json.JSONDecodeError:
            pass
if not rows:
    raise SystemExit("no episodes pulled yet")
df = pd.DataFrame(rows)
df = df.drop_duplicates(subset=["suite", "task_id", "phrase", "init"], keep="last")

tasks_meta = {t["canonical"]: t for t in json.load(open(
    os.path.expanduser("~/dev/interactive-pi/pi05_libero/eval/fourtier_tasks.json")))}


def tier_of(arm):
    if arm == "orig":
        return "original"
    if arm.startswith("nat"):
        return "natural"
    if arm.startswith("adv"):
        return "adversarial"
    return arm  # screen / confirm handled separately


df["tier"] = df.arm.map(tier_of)


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


# oracle confirm winner per task: best confirm-arm phrase (tie -> canonical)
def confirm_winner(g):
    conf = g[g.arm == "confirm"]
    if conf.empty:
        return None
    per = conf.groupby("phrase").success.agg(["sum", "count"])
    per = per[per["count"] >= 10]
    if per.empty:
        return None
    canon = g.canonical.iloc[0]
    per = per.assign(is_canon=[p == canon for p in per.index])
    per = per.sort_values(["sum", "is_canon"], ascending=[False, False])
    return per.index[0], int(per["sum"].iloc[0]), int(per["count"].iloc[0])


fig, (a1, a2, a3, a4) = plt.subplots(1, 4, figsize=(23.5, 5.6),
                                     gridspec_kw={"width_ratios": [1.0, 1.35, 1.15, 0.85]})
TIER_COLORS = {"original": "#805ad5", "natural": "#b7791f", "adversarial": "#c53030",
               "oracle": "#0d9488"}

# --- panel 1: pooled tiers by suite -----------------------------------------
xpos, xlab = [], []
x = 0.0
for suite in ("libero_goal", "libero_90"):
    sd = df[df.suite == suite]
    if sd.empty:
        continue
    for tier in ("original", "natural", "adversarial", "oracle"):
        if tier == "oracle":
            wins = [confirm_winner(g) for _, g in sd.groupby("task_id")]
            wins = [w for w in wins if w]
            if not wins:
                continue
            k, n = sum(w[1] for w in wins), sum(w[2] for w in wins)
        else:
            td = sd[sd.tier == tier]
            if td.empty:
                continue
            k, n = int(td.success.sum()), len(td)
        p, lo, hi = wilson(k, n)
        a1.bar(x, p * 100, 0.8, color=TIER_COLORS[tier], alpha=0.9)
        a1.errorbar(x, p * 100, yerr=[[p * 100 - lo * 100], [hi * 100 - p * 100]],
                    color="#2d3748", lw=1.2, capsize=3)
        a1.text(x, 2, "n=%d" % n, ha="center", fontsize=7, color="white", rotation=90)
        xpos.append(x)
        xlab.append(tier[:4])
        x += 1
    x += 0.9
a1.set_xticks(xpos)
a1.set_xticklabels(xlab, fontsize=8)
a1.set_ylabel("rollout success %")
a1.set_ylim(0, 100)
ng = df[df.suite == "libero_goal"].task_id.nunique()
n9 = df[df.suite == "libero_90"].task_id.nunique()
a1.set_title("Pooled tiers -- goal (%d tasks) | libero_90 (%d)" % (ng, n9), fontsize=11)
a1.grid(alpha=0.25, axis="y")

# --- panel 2: per-task heatmap ----------------------------------------------
tier_order = ["original", "natural", "adversarial", "oracle"]
tasks = sorted(df.groupby(["suite", "task_id"]).groups, key=lambda t: (t[0] != "libero_goal", t[1]))
M = np.full((len(tasks), 4), np.nan)
for i, (suite, tid) in enumerate(tasks):
    g = df[(df.suite == suite) & (df.task_id == tid)]
    for j, tier in enumerate(tier_order[:3]):
        td = g[g.tier == tier]
        if len(td):
            M[i, j] = td.success.mean() * 100
    w = confirm_winner(g)
    if w:
        M[i, 3] = 100.0 * w[1] / w[2]
im = a2.imshow(M, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")
a2.set_xticks(range(4))
a2.set_xticklabels(tier_order, fontsize=8)
a2.set_yticks(range(len(tasks)))
a2.set_yticklabels(["%s %d%s" % ("G" if s == "libero_goal" else "L90", t,
                                 "*" if tasks_meta.get(df[(df.suite == s) & (df.task_id == t)].canonical.iloc[0],
                                                       {}).get("string_in_finetune") and s == "libero_90" else "")
                    for s, t in tasks], fontsize=7)
for i in range(len(tasks)):
    for j in range(4):
        if not np.isnan(M[i, j]):
            a2.text(j, i, "%.0f" % M[i, j], ha="center", va="center", fontsize=7,
                    color="#1a202c")
a2.set_title("Per-task success %% (L90* = trained string)", fontsize=11)

# --- panel 3: board-search status -------------------------------------------
labels, canon_scr, best_scr, conf_pct = [], [], [], []
for suite, tid in tasks:
    g = df[(df.suite == suite) & (df.task_id == tid)]
    scr = g[g.init <= 4]  # screen inits; tier episodes double as screens
    canon = g.canonical.iloc[0]
    per = scr.groupby("phrase").success.mean() * 100
    if per.empty:
        continue
    labels.append("%s%d" % ("G" if suite == "libero_goal" else "L", tid))
    canon_scr.append(per.get(canon, np.nan))
    best_scr.append(per.max())
    w = confirm_winner(g)
    conf_pct.append(100.0 * w[1] / w[2] if w else np.nan)
yy = np.arange(len(labels))
a3.barh(yy + 0.2, best_scr, 0.38, color="#0d9488", alpha=0.75, label="best screen")
a3.barh(yy - 0.2, canon_scr, 0.38, color="#805ad5", alpha=0.75, label="canonical screen")
for i, c in enumerate(conf_pct):
    if not np.isnan(c):
        a3.scatter([c], [i], marker="D", s=45, color="#c53030", zorder=5,
                   label="confirm winner (virgin inits)" if i == next(
                       k for k, v in enumerate(conf_pct) if not np.isnan(v)) else None)
a3.set_yticks(yy)
a3.set_yticklabels(labels, fontsize=7)
a3.invert_yaxis()
a3.set_xlim(0, 105)
a3.set_xlabel("success %")
a3.set_title("Oracle: screen best vs canonical", fontsize=11)
a3.legend(fontsize=7, loc="lower right")
a3.grid(alpha=0.25, axis="x")

# --- panel 4: throughput ----------------------------------------------------
df["t"] = pd.to_datetime(df.ts)
for shard, g in df.groupby("shard"):
    g = g.sort_values("t")
    a4.plot(g.t.values, np.arange(1, len(g) + 1), lw=2, label="%s (%d)" % (shard, len(g)))
a4.set_title("Episodes per shard", fontsize=11)
a4.legend(fontsize=8)
a4.grid(alpha=0.25)
a4.tick_params(axis="x", labelrotation=30, labelsize=7)

total = len(df)
done_tasks = sum(1 for s, t in tasks if confirm_winner(df[(df.suite == s) & (df.task_id == t)]))
fig.suptitle("pi0.5 + LIBERO four-tier eval -- LIVE  (%d episodes, %d/20 tasks fully confirmed)"
             % (total, done_tasks), fontsize=13, y=1.005)
fig.tight_layout()
out = ROOT / "results/charts/fourtier_progress.png"
fig.savefig(out, dpi=130, bbox_inches="tight", pad_inches=0.25)
print("chart -> %s  (%d eps, %d tasks confirmed)" % (out, total, done_tasks))
