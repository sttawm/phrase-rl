#!/usr/bin/env python3
"""Full pi0.5/LIBERO phrase bank: four-tier seed + the 13,490 new bank episodes.

Train-task rows -> bank.parquet (train-only banking). Val-task canonicals from
the screening pass -> val_canonicals.parquet (baselines for the loop, never
bank evidence). Asserts: no sealed-test episodes anywhere; new-episode count
matches the queues exactly.

  .venv/bin/python scripts/make_pi05_full_bank.py
"""
import glob
import json
import pathlib

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
D = ROOT / "results/analysis/pi05_bank"

splits = json.load(open(D / "splits.json"))
train_keys = {(t["suite"], t["task_id"]) for t in splits["train"]}
val_keys = {(t["suite"], t["task_id"]) for t in splits["val"]}
test_keys = {(t["suite"], t["task_id"]) for t in splits["sealed_test"]}


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
    return "oracle_board"


eps = []
for f in sorted(glob.glob(str(D / "bank_live/*.jsonl"))):
    for line in open(f):
        line = line.strip()
        if line:
            eps.append(json.loads(line))
new = pd.DataFrame(eps)
assert len(new) == 13490, len(new)
leak = new[[tuple(x) in test_keys for x in zip(new.suite, new.task_id)]]
assert leak.empty, "sealed-test episodes present!"

new["window"] = new["init"].map(
    lambda i: "screen" if i < 20 else ("confirm" if i < 30 else "confirm2"))
g = (new.groupby(["suite", "task_id", "canonical", "phrase", "arm", "window"])
        .agg(succ=("success", "sum"), n=("success", "size")).reset_index())
g["kind"] = g.arm.map(arm_kind)
g["gt_success"] = (100.0 * g.succ / g.n).round(1)
g["source"] = "bank_v1"

is_val = g.apply(lambda r: (r.suite, r.task_id) in val_keys, axis=1)
val_rows = g[is_val]
assert set(val_rows.kind) <= {"original"}, "non-canonical val rows!"
val_rows.to_parquet(D / "val_canonicals.parquet", index=False)

train_rows = g[~is_val]
assert all((s, t) in train_keys for s, t in
           zip(train_rows.suite, train_rows.task_id)), "non-train bank rows!"
seed = pd.read_parquet(D / "seed_bank.parquet")
bank = pd.concat([seed, train_rows], ignore_index=True)
bank.to_parquet(D / "bank.parquet", index=False)

print(f"new episodes: {len(new)}; val canonicals: {len(val_rows)} rows "
      f"({val_rows.task_id.nunique()} tasks)")
print(f"bank: {len(bank)} rows, "
      f"{bank.groupby(['suite','task_id','phrase']).ngroups} distinct (task,phrase) "
      f"over {bank.groupby(['suite','task_id']).ngroups} tasks")
print(bank.kind.value_counts().to_string())
