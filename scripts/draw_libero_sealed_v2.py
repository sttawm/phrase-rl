#!/usr/bin/env python3
"""Seeded uniform draw of the LIBERO sealed set v2 (10 in-finetune + 10 out-of-finetune).

  .venv/bin/python scripts/draw_libero_sealed_v2.py            # walk the permutation with the data on hand
  .venv/bin/python scripts/draw_libero_sealed_v2.py --screen results/analysis/pi05_bank/sealed_v2_screen.jsonl

Procedure (PREREG amendment 2026-09-14): one seeded permutation per pool
(numpy default_rng(SEED).permutation). Walk each permutation in order; a task is
ACCEPTED unless it is FLOOR = 0 successes out of 20 with its canonical string on
inits 0-19 (seed 7, the standard harness), in which case it is skipped. Stop at 10.
Floor status comes from canonical episodes already measured under that exact
protocol (sealed-20 screen, the n=50 rounds restricted to inits 0-19, the
four-tier boards) or from the screening file passed with --screen; a task with no
data halts the walk and is listed under needs_screen -- the order is never
skipped or re-drawn. In-finetune pool = the 40 finetune tasks minus libero_10/5
(855 probe episodes exist on it; splits.json reserve_exclusions).
Writes results/analysis/pi05_bank/sealed_v2_draw.json.
"""
import glob
import json
import pathlib
import sys

import numpy as np
import pandas as pd

R = pathlib.Path(__file__).resolve().parents[1]
B = R / "results/analysis/pi05_bank"
SEED = 20260914
N_PER_HALF = 10
IN_SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
EXCLUDE = {("libero_10", 5)}
# Post-draw skip (user decision 2026-09-14, after the in-finetune walk was seen):
# libero_goal/7 holds 103 of the 426 single-edit pairs (24%) -- the stove ladder --
# so sealing it would remove a quarter of the distillation evidence. It is skipped
# in the walk exactly like a FLOOR task; the pool and permutation are unchanged, so
# the replacement is simply the next task in the recorded order (libero_10/4).
SKIP_POST_DRAW = {"libero_goal/7": "evidence share 103/426 pairs (24%)"}

# ---- canonical episodes on inits 0-19 --------------------------------------
def episodes():
    rows = []
    srcs = [("rounds", f) for d in ["roll50"] + [f"round{i}" for i in range(2, 9)] for f in glob.glob(str(B / d / "*.jsonl"))]
    srcs += [("fourtier", f) for f in glob.glob(str(R / "results/analysis/fourtier_live/*.jsonl"))]
    if "--screen" in sys.argv:
        srcs += [("screen_v2", sys.argv[sys.argv.index("--screen") + 1])]
    for src, f in srcs:
        for line in open(f):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("success") is None or r["phrase"] != r.get("canonical") or int(r["init"]) >= 20:
                continue
            rows.append((src, r["suite"], int(r["task_id"]), int(r["init"]), int(r["success"])))
    ep = pd.DataFrame(rows, columns=["src", "suite", "task_id", "init", "success"])
    return ep.drop_duplicates(["suite", "task_id", "init"])

ep = episodes()
cov = ep.groupby(["suite", "task_id"]).agg(n=("init", "size"), k=("success", "sum"), src=("src", "first"))
# the sealed-20 screen result parquet carries only the aggregate; it is the same protocol (canonical, inits 0-19, n=20)
seal = pd.read_parquet(R / "results/rules_runs/p_seal/jobs/seal_screen_canon_w0.result.parquet")
for r in seal.itertuples():
    s, t = r.task.split(":"); key = (s, int(t))
    if key not in cov.index or cov.loc[key, "n"] < 20:
        cov.loc[key, ["n", "k", "src"]] = [20, round(r.gt_success / 5), "seal_screen"]
for f in ["reserve_screen_canon_w0", "reserve_screen2_canon_w0"]:
    d = pd.read_parquet(R / f"results/rules_runs/p_seal/jobs/{f}.result.parquet")
    for r in d.itertuples():
        s, t = r.task.split(":"); key = (s, int(t))
        if key not in cov.index or cov.loc[key, "n"] < 20:
            cov.loc[key, ["n", "k", "src"]] = [20, round(r.gt_success / 5), "reserve_screen"]

def status(suite, tid):
    key = (suite, tid)
    if key in cov.index and cov.loc[key, "n"] >= 20:
        k = int(cov.loc[key, "k"]); return ("FLOOR" if k == 0 else "ACCEPT"), k, cov.loc[key, "src"]
    return "NEEDS_SCREEN", None, None

tasks = json.load(open(R / "results/analysis/pi05_bank/libero_tasks.json")) if (B / "libero_tasks.json").exists() else None
n_tasks = {"libero_spatial": 10, "libero_object": 10, "libero_goal": 10, "libero_10": 10, "libero_90": 90}
pools = {"in_finetune": [(s, i) for s in IN_SUITES for i in range(n_tasks[s]) if (s, i) not in EXCLUDE],
         "out_of_finetune": [("libero_90", i) for i in range(90)]}
out = {"seed": SEED, "n_per_half": N_PER_HALF, "floor_rule": "0/20 canonical successes on inits 0-19 (seed 7)",
       "in_finetune_exclusions": sorted(f"{s}/{i}" for s, i in EXCLUDE), "post_draw_skips": SKIP_POST_DRAW, "halves": {}}
for name, pool in pools.items():
    rng = np.random.default_rng(SEED)
    perm = [pool[i] for i in rng.permutation(len(pool))]
    walk, accepted, needs = [], [], []
    for s, i in perm:
        st, k, src = status(s, i)
        if f"{s}/{i}" in SKIP_POST_DRAW:
            st, src = "SKIPPED", SKIP_POST_DRAW[f"{s}/{i}"]
        walk.append({"task": f"{s}/{i}", "status": st, "k_of_20": k, "source": src})
        if st == "ACCEPT":
            accepted.append(f"{s}/{i}")
        elif st == "NEEDS_SCREEN":
            needs.append(f"{s}/{i}")
            break
        if len(accepted) == N_PER_HALF:
            break
    out["halves"][name] = {"permutation": [f"{s}/{i}" for s, i in perm], "walk": walk, "accepted": accepted,
                           "needs_screen_next": needs, "complete": len(accepted) == N_PER_HALF}
    print(f"== {name}: accepted {len(accepted)}/{N_PER_HALF}" + ("" if not needs else f"; walk halted at {needs[0]} (no screen)"))
    for w in walk:
        print(f"   {w['task']:<18} {w['status']:<13} {'' if w['k_of_20'] is None else str(w['k_of_20'])+'/20':<6} {w['source'] or ''}")
json.dump(out, open(B / "sealed_v2_draw.json", "w"), indent=1)
print("->", B / "sealed_v2_draw.json")
