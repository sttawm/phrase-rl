#!/usr/bin/env python3
"""Arm D (v7a-120 + rules router) — the sealed composition (Amendments 1/3/5).

Pure arithmetic, zero rolling: joins arm A per-task cells (sealed_v7a120.jsonl,
polish from pod5 + repair from pod7) with the already-rolled rules cells
(polish fallback: rules_pro_nominal_x12; repair fallback: pairA_pro_x12 arm
rules_v3_gemini_pro) through the FROZEN routing table armD_routing_seatA.json.
Prints the full per-task table and appends the composed rows to
results/analysis/armD_composed.jsonl. Degrades gracefully while arm A legs are
still rolling (reports what is missing and exits 0).
"""
import json

import pandas as pd

routing = json.load(open("results/analysis/armD_routing_seatA.json"))["routing"]

armA = {}
try:
    lines = list(open("results/analysis/sealed_v7a120.jsonl"))
except FileNotFoundError:
    lines = []
    print("[waiting] sealed_v7a120.jsonl does not exist yet (no arm A leg merged)")
for l in lines:
    r = json.loads(l)
    a = r.get("arm", "")
    if "polish" in a:
        armA["polish"] = r
    elif "repair" in a and "drift" not in a:
        armA["repair"] = r

rules_cells = {}
rp = pd.read_parquet("results/sealed/rules_pro_nominal_x12.parquet")
rules_cells["polish"] = {t: float(g.success.mean() * 100) for t, g in rp.groupby("task")}
pa = pd.read_parquet("results/sealed/pairA_pro_x12.parquet")
pa = pa[pa.arm == "rules_v3_gemini_pro"]
rules_cells["repair"] = {t: float(g.success.mean() * 100) for t, g in pa.groupby("task")}

out = []
for cond in ("polish", "repair"):
    if cond not in armA:
        print(f"[waiting] arm A {cond} leg not yet merged — composition deferred")
        continue
    at = armA[cond]["per_task"]
    table, cells = [], []
    for task, route in routing[cond].items():
        keep = route["route"] == "KEEP"
        val = at.get(task) if keep else rules_cells[cond].get(task)
        cells.append(val)
        table.append((task, "armA" if keep else "rules", val, at.get(task), rules_cells[cond].get(task)))
    pooled = round(sum(cells) / len(cells), 2)
    print(f"\n=== ARM D ({cond}) — composed pooled: {pooled} ===")
    print(f"{'task':36s} {'src':6s} {'used':>6s} {'armA':>6s} {'rules':>6s}")
    for task, src, val, av, rv in table:
        print(f"{task[:34]:36s} {src:6s} {val:6.1f} {av if av is not None else float('nan'):6.1f} "
              f"{rv if rv is not None else float('nan'):6.1f}")
    comps = {"armA_pooled": armA[cond]["pooled"],
             "rules_pooled": round(sum(rules_cells[cond].values()) / 12, 2)}
    print(f"components: arm A alone {comps['armA_pooled']}  |  rules alone {comps['rules_pooled']}")
    rec = {"arm": f"armD_seatA_{cond}", "pooled": pooled, "n_keep": sum(1 for x in table if x[1] == "armA"),
           "components": comps,
           "per_task": {t: round(v, 1) for t, _, v, _, _ in table}}
    out.append(rec)

if out:
    with open("results/analysis/armD_composed.jsonl", "a") as f:
        for rec in out:
            f.write(json.dumps(rec) + "\n")
    print(f"\nappended {len(out)} rows -> results/analysis/armD_composed.jsonl")
