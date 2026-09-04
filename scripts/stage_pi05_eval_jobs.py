#!/usr/bin/env python3
"""Shard the sealed grid's rollouts into worker jobs.

Every arm's rewrites plus the un-rephrased bases collapse to a set of unique
(task, phrase) pairs -- identical strings are rolled ONCE and shared across arms
(PREREG: key is (task, phrase, init)). Jobs are balanced by episode count and
written to results/rules_runs/<run>/jobs/ for rules_loop_worker.sh to claim.

  FINAL_EVAL=1 .venv/bin/python scripts/stage_pi05_eval_jobs.py --shards 8
"""
import argparse
import hashlib
import json
import os
import pathlib

import pandas as pd

if os.environ.get("FINAL_EVAL") != "1":
    raise SystemExit("FINAL_EVAL=1 required")

REPO = pathlib.Path(__file__).resolve().parents[1]
D = REPO / "results/analysis/pi05_bank"
INITS = list(range(30, 50))          # PREREG reporting window, disjoint from the screen

ap = argparse.ArgumentParser()
ap.add_argument("--shards", type=int, default=8)
ap.add_argument("--run", default="p_eval")
ap.add_argument("--prefix", default="ev")
a = ap.parse_args()

bases = pd.read_parquet(D / "eval_bases.parquet")
pairs = [bases[["task", "phrase"]].assign(src="base")]
for f in sorted((D / "eval_applies").glob("*.parquet")):
    d = pd.read_parquet(f)
    d = d[d.rewrite.str.strip().str.len() > 0]
    pairs.append(d[["task", "rewrite"]].rename(columns={"rewrite": "phrase"})
                 .assign(src=f.stem))
allp = pd.concat(pairs, ignore_index=True)
uniq = allp[["task", "phrase"]].drop_duplicates().reset_index(drop=True)
print(f"{len(allp)} arm rows -> {len(uniq)} unique (task,phrase); "
      f"{len(uniq)*len(INITS)} episodes")

# dedup ledger: which arms share which string (keeps arm composition auditable)
ledger = (allp.groupby(["task", "phrase"]).src.apply(lambda s: sorted(set(s)))
          .reset_index().rename(columns={"src": "arms"}))
ledger["n_arms"] = ledger.arms.str.len()
ledger.to_parquet(D / "eval_dedup_ledger.parquet", index=False)
print(f"dedup ledger: {int((ledger.n_arms > 1).sum())} strings shared by >1 arm")

# task-affinity sharding (a shard keeps whole tasks together: bank_eval reloads
# the sim scene per task, so splitting a task across shards pays that twice)
sizes = uniq.groupby("task").size().sort_values(ascending=False)
shards = [[] for _ in range(a.shards)]
load = [0] * a.shards
for task, n in sizes.items():
    i = load.index(min(load))
    shards[i].append(task)
    load[i] += n

jd = REPO / f"results/rules_runs/{a.run}/jobs"
jd.mkdir(parents=True, exist_ok=True)
for i, tasks in enumerate(shards):
    if not tasks:
        continue
    pl = uniq[uniq.task.isin(tasks)].reset_index(drop=True)
    jid = f"{a.prefix}{i}_" + hashlib.sha1(
        ("|".join(sorted(f"{t}\x01{p}" for t, p in zip(pl.task, pl.phrase)))
         ).encode()).hexdigest()[:10]
    pl.to_parquet(jd / f"{jid}.payload.parquet", index=False)
    json.dump({"job_id": jid, "kind": "score", "method": "libero_bank_eval",
               "rollout": {"inits": INITS, "port": 8000, "seed": 7}},
              open(jd / f"{jid}.spec.json", "w"), indent=1)
    print(f"  {jid}: {len(tasks)} tasks, {len(pl)} phrases, "
          f"{len(pl)*len(INITS)} episodes")
print(f"staged {sum(1 for s in shards if s)} jobs in {jd}")
