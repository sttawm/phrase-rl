#!/usr/bin/env python3
"""Queue B41: the NON-augmented Bridge checkpoint on the 363 human-natural
phrases (A39 raw human set), 24 layouts x 1 rep -- the pass-through baseline
that was never measured for the rephrase-finetuned vs plain-finetuned
comparison on human phrasings (project-site chart).

  FINAL_EVAL=1 .venv/bin/python scripts/queue_b41_base_legs.py

Same shape as queue_a39_legs.py (SHARD_N phrases per leg, SPEC rollout job)
except rollout.ckpt is juexzz/INTACT-pi0-finetune-bridge. Re-runnable: legs
whose spec already exists are skipped.
"""
import hashlib
import json
import os
import pathlib

import pandas as pd

if os.environ.get("FINAL_EVAL") != "1":
    raise SystemExit("FINAL_EVAL=1 required")
R = pathlib.Path(__file__).resolve().parents[1]
jd = R / "results/rules_runs/r1_sim/jobs"
SHARD_N = 15
PREFIX = "b41base"

SPEC = {"kind": "score", "method": "rollout", "draw": 0, "seed": 7,
        "proxy": {"C": 8.123, "bz": 0.4445, "bg": 11.3193},
        "rollout": {"config": "config/experiment/simpler/pi0_finetune_bridge_ev.yaml",
                    "ckpt": "juexzz/INTACT-pi0-finetune-bridge",
                    "seed": 42, "episode_ids": list(range(24)), "repeats": 1}}

want = (pd.read_parquet(R / "results/human_naturals/a39_human_phrases.parquet")
          [["task", "phrase"]].drop_duplicates().reset_index(drop=True))
print(f"roll set: {len(want)} unique (task, phrase) over {want.task.nunique()} tasks")

n = 0
for task, grp in want.groupby("task"):
    rows = grp.reset_index(drop=True)
    for i in range(0, len(rows), SHARD_N):
        shard = rows.iloc[i:i + SHARD_N]
        h = hashlib.sha1(("b41" + task + "\x00".join(sorted(shard.phrase))).encode()).hexdigest()[:10]
        jid = f"{PREFIX}_{h}"
        if (jd / f"{jid}.spec.json").exists():
            continue
        shard[["task", "phrase"]].to_parquet(jd / f"{jid}.payload.parquet", index=False)
        json.dump({"job_id": jid, **SPEC}, open(jd / f"{jid}.spec.json", "w"), indent=1)
        n += 1
print(f"queued {n} new legs ({len(want)} phrases x 24 eps = {len(want) * 24} episodes)")
