#!/usr/bin/env python3
"""Seed the pi0.5/LIBERO phrase bank from the four-tier episode logs.

Aggregates every fourtier_live/*.jsonl episode to (suite, task_id, phrase) rows
with success counts split by init window (screen 0-19 / confirm 20-29 /
confirm2 30-49), keeping only TRAIN-split tasks (train-only banking; the
30 generation-touched tasks are all in train by construction).

  .venv/bin/python scripts/make_pi05_seed_bank.py
  -> results/analysis/pi05_bank/seed_bank.parquet
"""
import glob
import json
import pathlib

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
D = ROOT / "results/analysis/pi05_bank"

splits = json.load(open(D / "splits.json"))
train_keys = {(t["suite"], t["task_id"]) for t in splits["train"]}
val_test_keys = {(t["suite"], t["task_id"])
                 for t in splits["val"] + splits["sealed_test"]}

def arm_kind(a):
    a = str(a)
    if a in ("orig", "original"):
        return "original"
    if a.startswith("nat"):
        return "natural"
    if a.startswith("adv"):
        return "adversarial"
    if a == "cross":
        return "cross_scene"
    # screen / board / confirm / confirm2 / confirmmax / oracle*: phases of the
    # oracle board search -- all are searched candidates
    return "oracle_board"


def window(init):
    return "screen" if init < 20 else ("confirm" if init < 30 else "confirm2")


eps = []
for f in sorted(glob.glob(str(ROOT / "results/analysis/fourtier_live/*.jsonl"))):
    for line in open(f):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        if "phrase" not in d or "success" not in d:
            continue
        eps.append({"suite": d["suite"], "task_id": d["task_id"],
                    "canonical": d.get("canonical", ""), "phrase": d["phrase"],
                    "arm": d.get("arm", "?"), "init": d["init"],
                    "success": int(d["success"]), "window": window(d["init"])})
df = pd.DataFrame(eps)
print(f"episodes: {len(df)} over {df.groupby(['suite','task_id']).ngroups} tasks, "
      f"{df.groupby(['suite','task_id','phrase']).ngroups} (task,phrase) pairs")

leak = df[[tuple(x) in val_test_keys for x in zip(df.suite, df.task_id)]]
assert leak.empty, f"episodes touch val/test tasks!: {leak[['suite','task_id']].drop_duplicates()}"

non_train = df[[tuple(x) not in train_keys for x in zip(df.suite, df.task_id)]]
if len(non_train):
    print("dropped non-train episodes:",
          non_train.groupby(["suite", "task_id"]).size().to_dict())
df = df[[tuple(x) in train_keys for x in zip(df.suite, df.task_id)]]
g = (df.groupby(["suite", "task_id", "canonical", "phrase", "arm", "window"])
       .agg(succ=("success", "sum"), n=("success", "size")).reset_index())
g["kind"] = g.arm.map(arm_kind)
g["gt_success"] = (100.0 * g.succ / g.n).round(1)
g["source"] = "fourtier"

out = D / "seed_bank.parquet"
g.to_parquet(out, index=False)
print(f"bank rows: {len(g)} "
      f"({g.groupby(['suite','task_id','phrase']).ngroups} distinct (task,phrase))")
print(g.kind.value_counts().to_string())
print(g.window.value_counts().to_string())
print(f"-> {out.relative_to(ROOT)}")
