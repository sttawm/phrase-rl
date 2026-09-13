#!/usr/bin/env python3
"""Incrementally queue A39 rollout legs as apply arms land (PREREG A39).

  FINAL_EVAL=1 .venv/bin/python scripts/queue_a39_legs.py

Re-runnable: builds the global unique (task, phrase) roll set from every
apply output present so far (local ph_a39_*.parquet + landed qwen job
results) PLUS the raw human phrases, subtracts phrases already queued in
b39 legs, and queues only the new ones — so rolling starts while slower
appliers are still working. 24 layouts x 1 rep (n=1 per addendum), sharded
SHARD_N phrases per leg for fleet load-balancing. Empty/base-identical
rewrites roll once thanks to global dedup; per-arm attribution happens at
analysis time through the ph_a39_* mapping tables.
"""
import glob
import hashlib
import json
import os
import pathlib

import pandas as pd

if os.environ.get("FINAL_EVAL") != "1":
    raise SystemExit("FINAL_EVAL=1 required")
R = pathlib.Path.home() / "dev/robotics/phrase-rl"
jd = R / "results/rules_runs/r1_sim/jobs"
SHARD_N = 15

SPEC = {"kind": "score", "method": "rollout", "draw": 0, "seed": 7,
        "proxy": {"C": 8.123, "bz": 0.4445, "bg": 11.3193},
        "rollout": {"config": "config/experiment/simpler/pi0_finetune_bridge_ev.yaml",
                    "ckpt": "juexzz/INTACT-pi0-finetune-rephrase-bridge",
                    "seed": 42, "episode_ids": list(range(24)), "repeats": 1}}

frames = []
frames.append(pd.read_parquet(R / "results/human_naturals/a39_human_phrases.parquet")[["task", "phrase"]])
for f in glob.glob(str(R / "results/human_naturals/ph_a39_*.parquet")):
    frames.append(pd.read_parquet(f)[["task", "phrase"]])
for f in glob.glob(str(jd / "a39*qw_*.result.parquet")):
    d = pd.read_parquet(f)
    if "rewrite" in d.columns:
        d = d.rename(columns={"phrase": "base", "rewrite": "phrase"})
        d["phrase"] = d.phrase.fillna("").astype(str)
        d.loc[d.phrase.str.strip() == "", "phrase"] = d.base
    frames.append(d[["task", "phrase"]])
want = pd.concat(frames, ignore_index=True).drop_duplicates().reset_index(drop=True)

queued = set()
for f in glob.glob(str(jd / "b39roll_*.payload.parquet")):
    q = pd.read_parquet(f)
    queued |= set(zip(q.task, q.phrase))
new = want[~want.apply(lambda r: (r.task, r.phrase) in queued, axis=1)]
print(f"roll set so far: {len(want)} unique; already queued {len(queued)}; new {len(new)}")

n = 0
for task, grp in new.groupby("task"):
    rows = grp.reset_index(drop=True)
    for i in range(0, len(rows), SHARD_N):
        shard = rows.iloc[i:i + SHARD_N]
        h = hashlib.sha1(("b39" + task + "\x00".join(sorted(shard.phrase))).encode()).hexdigest()[:10]
        jid = f"b39roll_{h}"
        if (jd / f"{jid}.spec.json").exists():
            continue
        shard[["task", "phrase"]].to_parquet(jd / f"{jid}.payload.parquet", index=False)
        json.dump({"job_id": jid, **SPEC}, open(jd / f"{jid}.spec.json", "w"), indent=1)
        n += 1
print(f"queued {n} new legs ({len(new)} phrases x 24 eps = {len(new) * 24} episodes)")
