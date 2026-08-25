#!/usr/bin/env python3
"""Macro + micro analysis of the pi0.5/LIBERO four-tier eval.

Reads results/analysis/fourtier_live/*.jsonl (deduped on (suite, task, phrase,
init)), writes results/analysis/fourtier_summary.json and prints a digest.
Feeds the FOURTIER-LIBERO.md writeup -- every number in that document comes
from this script.

Macro: pooled tier success by stratum (goal in-finetune / libero_90 clean /
libero_90 trained-string) with Wilson 95% CIs; per-task tier table; oracle
ladder per task (screen best -> confirm winner -> confirm2 replication) with
winner's-curse gaps.
Micro: inversion tasks (canonical fails, rephrasings work), no-headroom tasks,
trained-scene tightening (adversarial collapse where canonical is strong),
same-string cross-scene blocks, biggest per-phrase scene gaps.

  .venv/bin/python scripts/analyze_fourtier.py
"""
import glob
import json
import math
import pathlib

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
EVAL = pathlib.Path.home() / "dev/interactive-pi/pi05_libero/eval"

rows = []
for f in glob.glob(str(ROOT / "results/analysis/fourtier_live/fourtier_*.jsonl")):
    for line in open(f):
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
df = pd.DataFrame(rows).drop_duplicates(subset=["suite", "task_id", "phrase", "init"], keep="last")

meta = {}
for fn in ("fourtier_tasks.json", "fourtier_tasks_ext.json"):
    p = EVAL / fn
    if p.exists():
        for t in json.load(open(p)):
            meta[(t["suite"], t["task_id"])] = t


def stratum(suite, tid):
    m = meta.get((suite, tid), {})
    if suite == "libero_goal":
        return "goal_in_finetune"
    return "l90_trained_string" if m.get("string_in_finetune") else "l90_clean"


def wilson(k, n, z=1.96):
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return {"rate": round(p, 4), "lo": round(max(0, c - h), 4), "hi": round(min(1, c + h), 4), "n": int(n)}


CONF_INITS = set(range(20, 30))
CONF2_INITS = set(range(30, 50))
SCREEN_INITS = set(range(5))

summary = {"episodes": int(len(df)), "tasks": {}, "macro": {}, "micro": {}}

per_task = {}
for (suite, tid), g in df.groupby(["suite", "task_id"]):
    orig_rows = g[g.arm == "orig"]
    canon = orig_rows.phrase.iloc[0] if len(orig_rows) else None
    if canon is None:
        continue
    T = {"canonical": canon, "stratum": stratum(suite, tid)}
    canon_all = g[(g.phrase == canon) & (g.init < 30)]  # exclude confirm2 window from headline
    T["orig"] = wilson(canon_all.success.sum(), len(canon_all))
    for kind, pref in (("natural", "nat"), ("adversarial", "adv")):
        tp = set(g[g.arm.str.startswith(pref)].phrase)
        sub = g[g.phrase.isin(tp) & (g.init < 20)]
        T[kind] = wilson(sub.success.sum(), len(sub)) if len(sub) else None
    # oracle ladder
    conf = g[(g.arm == "confirm") & (g.init.isin(CONF_INITS))]
    if len(conf):
        pc = conf.groupby("phrase").success.agg(["sum", "count"])
        pc = pc[pc["count"] >= 10]
        if len(pc):
            pc = pc.assign(canon=[p == canon for p in pc.index]).sort_values(
                ["sum", "canon"], ascending=[False, False])
            w = pc.index[0]
            scr = g[(g.phrase == w) & (g.init.isin(SCREEN_INITS))]
            c2 = g[(g.phrase == w) & (g.arm == "confirm2")]
            T["oracle"] = {
                "winner": w, "winner_is_canonical": w == canon,
                "confirm": wilson(pc["sum"].iloc[0], pc["count"].iloc[0]),
                "screen_of_winner": wilson(scr.success.sum(), len(scr)) if len(scr) else None,
                "confirm2": wilson(c2.success.sum(), len(c2)) if len(c2) else None,
                "canon_confirm": wilson(pc.loc[canon, "sum"], pc.loc[canon, "count"]) if canon in pc.index else None,
            }
    per_task[(suite, tid)] = T
    summary["tasks"]["%s/%d" % (suite, tid)] = T

# ---- macro: pooled tiers by stratum ----------------------------------------
for strat in ("goal_in_finetune", "l90_clean", "l90_trained_string"):
    tasks = [k for k, v in per_task.items() if v["stratum"] == strat]
    if not tasks:
        continue
    M = {"n_tasks": len(tasks)}
    for tier in ("orig", "natural", "adversarial"):
        k = sum(per_task[t][tier]["rate"] * per_task[t][tier]["n"] for t in tasks
                if per_task[t].get(tier))
        n = sum(per_task[t][tier]["n"] for t in tasks if per_task[t].get(tier))
        M[tier] = wilson(round(k), n) if n else None
    ow = [per_task[t]["oracle"]["confirm"] for t in tasks
          if per_task[t].get("oracle") and per_task[t]["oracle"].get("confirm")]
    if ow:
        M["oracle_confirm"] = wilson(sum(o["rate"] * o["n"] for o in ow),
                                     sum(o["n"] for o in ow))
        M["oracle_n_tasks"] = len(ow)
    summary["macro"][strat] = M

# winner's curse: screen-vs-confirm and confirm-vs-confirm2 pooled gaps
gaps = {"screen_minus_confirm": [], "confirm_minus_confirm2": []}
for T in per_task.values():
    o = T.get("oracle") or {}
    if o.get("screen_of_winner") and o.get("confirm"):
        gaps["screen_minus_confirm"].append(o["screen_of_winner"]["rate"] - o["confirm"]["rate"])
    if o.get("confirm2") and o["confirm2"]["n"] >= 10 and o.get("confirm"):
        gaps["confirm_minus_confirm2"].append(o["confirm"]["rate"] - o["confirm2"]["rate"])
summary["macro"]["winners_curse"] = {
    k: {"mean_pp": round(100 * sum(v) / len(v), 2), "n_tasks": len(v)} if v else None
    for k, v in gaps.items()}

# ---- micro detectors --------------------------------------------------------
mic = summary["micro"]
mic["inversions"] = []
mic["no_headroom"] = []
mic["trained_scene_tightening"] = []
for (suite, tid), T in sorted(per_task.items(), key=lambda x: str(x[0])):
    key = "%s/%d" % (suite, tid)
    orig, nat, adv, orc = T.get("orig"), T.get("natural"), T.get("adversarial"), T.get("oracle")
    best_reph = max([x["rate"] for x in (nat, adv) if x] or [0])
    if orig and orig["rate"] <= 0.2 and best_reph >= 0.7:
        mic["inversions"].append({"task": key, "canonical": T["canonical"], "orig": orig,
                                  "best_rephrase_tier_rate": best_reph,
                                  "stratum": T["stratum"]})
    if orc and orc.get("confirm") and orig and orc["winner_is_canonical"] \
       and abs(orc["confirm"]["rate"] - orig["rate"]) <= 0.15:
        mic["no_headroom"].append({"task": key, "orig": orig["rate"], "confirm": orc["confirm"]["rate"]})
    if orig and adv and orig["rate"] >= 0.8 and orig["rate"] - adv["rate"] >= 0.4:
        mic["trained_scene_tightening"].append(
            {"task": key, "orig": orig["rate"], "adversarial": adv["rate"], "stratum": T["stratum"]})

# same-string blocks: tasks sharing a canonical (within our set + cross rows)
by_canon = {}
for (suite, tid), T in per_task.items():
    by_canon.setdefault(T["canonical"], []).append((suite, tid))
mic["same_string_blocks"] = []
for canon, tasks in by_canon.items():
    if len(tasks) < 2:
        continue
    block = {"canonical": canon, "cells": []}
    for suite, tid in sorted(tasks, key=str):
        T = per_task[(suite, tid)]
        block["cells"].append({"task": "%s/%d" % (suite, tid), "stratum": T["stratum"],
                               "orig": T.get("orig"), "natural": T.get("natural"),
                               "adversarial": T.get("adversarial")})
    mic["same_string_blocks"].append(block)

# cross-scene phrase gaps (arm == "cross" rows exist on trained-scene tasks)
cross = df[df.arm == "cross"]
mic["cross_scene"] = []
for (suite, tid), g in cross.groupby(["suite", "task_id"]):
    per = g.groupby("phrase").success.agg(["sum", "count"])
    mic["cross_scene"].append({
        "trained_task": "%s/%d" % (suite, tid),
        "n_phrases": int(len(per)),
        "pooled": wilson(per["sum"].sum(), per["count"].sum()),
        "phrases": [{"phrase": p, "k": int(r["sum"]), "n": int(r["count"])}
                    for p, r in per.sort_values("sum").iterrows()]})

out = ROOT / "results/analysis/fourtier_summary.json"
json.dump(summary, open(out, "w"), indent=1)
print("summary -> %s  (%d episodes, %d tasks)" % (out, summary["episodes"], len(per_task)))
print("\n=== MACRO (pooled, Wilson 95%) ===")
for strat, M in summary["macro"].items():
    if strat == "winners_curse":
        print("winner's curse:", M)
        continue
    parts = []
    for tier in ("orig", "natural", "adversarial", "oracle_confirm"):
        v = M.get(tier)
        if v:
            parts.append("%s %.1f [%.1f-%.1f] n=%d" % (tier, 100 * v["rate"], 100 * v["lo"],
                                                       100 * v["hi"], v["n"]))
    print("%-22s (%d tasks): %s" % (strat, M["n_tasks"], " | ".join(parts)))
print("\n=== MICRO ===")
print("inversions:", [(x["task"], "%.0f vs %.0f" % (100 * x["orig"]["rate"],
                       100 * x["best_rephrase_tier_rate"])) for x in mic["inversions"]])
print("no-headroom:", [(x["task"], round(x["orig"], 2)) for x in mic["no_headroom"]])
print("tightening:", [(x["task"], "%.0f->%.0f" % (100 * x["orig"], 100 * x["adversarial"]))
                      for x in mic["trained_scene_tightening"]])
print("same-string blocks:", len(mic["same_string_blocks"]),
      "| cross-scene tasks:", len(mic["cross_scene"]))
