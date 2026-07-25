#!/usr/bin/env python3
"""Row-14 confirmed headroom: per-task oracle-best (held-out layouts 18-23,
n=72) vs nominal and per-task best arm on the SAME held-out cells extracted
from the sealed x12 parquets (deterministic grid => exact comparability).
Selection happened on layouts 0-17; nothing here was used to pick phrases.
Writes results/analysis/sealed_headroom_confirmed.json.
"""
import glob
import json
import math

import pandas as pd

HELD = set(range(18, 24))

legs = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob("results/sealed/*_x12.parquet"))],
                 ignore_index=True)
held = legs[legs.episode_id.isin(HELD)]
arm_rates = held.groupby(["arm", "task"]).success.mean() * 100

rows = []
for f in sorted(glob.glob("results/search/sealedsearch_*_confirm_results.json")):
    j = json.load(open(f))
    task = j["task"]
    best = max(j["scoreboard"], key=lambda e: e["success_pct"])
    n = best["n"]
    p = best["success_pct"] / 100
    se = 100 * math.sqrt(p * (1 - p) / n)
    nom = arm_rates.get(("originals", task), float("nan"))
    per_arm = {a: arm_rates.get((a, task)) for a in held.arm.unique() if a != "originals"}
    per_arm = {a: v for a, v in per_arm.items() if v == v}
    ba, bv = max(per_arm.items(), key=lambda kv: kv[1])
    rows.append({
        "task": task, "oracle_pct": best["success_pct"], "oracle_se": round(se, 1),
        "oracle_phrase": best["phrase"], "n": n,
        "nominal_heldout": round(float(nom), 1),
        "best_arm": ba, "best_arm_heldout": round(float(bv), 1),
    })

mean = lambda k: sum(r[k] for r in rows) / len(rows)
out = {
    "protocol": "selection layouts 0-17 (n=36); estimation layouts 18-23 (n=72); "
                "comparators = same held-out cells from sealed x12 parquets",
    "tasks": rows,
    "mean_oracle": round(mean("oracle_pct"), 1),
    "mean_nominal_heldout": round(mean("nominal_heldout"), 1),
    "mean_best_arm_heldout": round(mean("best_arm_heldout"), 1),
}
json.dump(out, open("results/analysis/sealed_headroom_confirmed.json", "w"), indent=1)

print(f"{'task':28s} {'oracle':>10s} {'nominal':>8s} {'best-arm':>9s}")
for r in rows:
    t = r["task"].replace("widowx_", "").replace("_clean", "")
    print(f"{t:28s} {r['oracle_pct']:5.1f}±{r['oracle_se']:3.1f} {r['nominal_heldout']:7.1f} "
          f"{r['best_arm_heldout']:8.1f}  ({r['best_arm'][:20]})")
print(f"\n{'MEAN (10 tasks)':28s} {out['mean_oracle']:5.1f}      {out['mean_nominal_heldout']:7.1f} "
      f"{out['mean_best_arm_heldout']:8.1f}")
