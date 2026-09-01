#!/usr/bin/env python3
"""Interleaved, sharded scoring queues for the pi0.5/LIBERO bank.

Order is the early-stop guarantee (user 2026-09-01): canonicals of the 50
never-rolled train+val tasks first (round-robin), then rounds r=0..9 where
each train task contributes natural[r] then adversarial[r] — so stopping at
any prefix leaves per-task nat/adv counts within 1 of each other and tasks
uniformly covered. Task-affinity sharding into 4 queues balanced by episode
count (env reuse + same-pod comparability within a task).

  .venv/bin/python scripts/make_pi05_bank_queues.py
  -> results/analysis/pi05_bank/queue_lb{1..4}.json
"""
import json
import pathlib

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
D = ROOT / "results/analysis/pi05_bank"
INITS = list(range(10))

splits = json.load(open(D / "splits.json"))
seed = pd.read_parquet(D / "seed_bank.parquet")
have_canon = {(r.suite, r.task_id) for r in
              seed[seed.kind == "original"][["suite", "task_id"]]
              .drop_duplicates().itertuples()}
gen = json.load(open(D / "bank_phrases.json"))   # keyed by canonical string

# --- canonical screening: fresh train + all val tasks ------------------------
fresh = []
for t in splits["train"]:
    if t["suite"] == "libero_90" and not t.get("touched") \
            and (t["suite"], t["task_id"]) not in have_canon:
        fresh.append({"suite": t["suite"], "task_id": t["task_id"],
                      "canonical": t["lang"]})
for t in splits["val"]:
    fresh.append({"suite": t["suite"], "task_id": t["task_id"],
                  "canonical": t["lang"]})
canon_items = [{"suite": f["suite"], "task_id": f["task_id"],
                "canonical": f["canonical"], "phrase": f["canonical"],
                "arm": "orig", "inits": INITS} for f in fresh]

# --- interleaved nat/adv rounds over train tasks -----------------------------
# map every train task to phrases by CANONICAL TEXT: tasks sharing a string
# (goal<->l90 trained-string, l90 same-string pairs) each score the same
# phrases in their own scene -- the cross-scene contrast is wanted evidence
goal_canon = {int(r.task_id): r.canonical
              for r in seed[seed.suite == "libero_goal"][["task_id", "canonical"]]
              .drop_duplicates().itertuples()}
train_tasks = []
for t in splits["train"]:
    canon = goal_canon[t["task_id"]] if t["suite"] == "libero_goal" else t["lang"]
    if canon in gen:
        train_tasks.append((t["suite"], t["task_id"], canon))
print(f"{len(train_tasks)} train tasks mapped to generated phrases")
rounds = []
max_r = max(max(len(v["natural"]), len(v["adversarial"])) for v in gen.values())
for r in range(max_r):
    for suite, tid, canon in train_tasks:
        v = gen[canon]
        for kind, lst in (("nat", v["natural"]), ("adv", v["adversarial"])):
            if r < len(lst):
                rounds.append({"suite": suite, "task_id": tid, "canonical": canon,
                               "phrase": lst[r], "arm": f"{kind}{r+1}",
                               "inits": INITS})

items = canon_items + rounds
print(f"queue: {len(canon_items)} canonicals + {len(rounds)} tier phrases "
      f"= {len(items)} items, {sum(len(i['inits']) for i in items)} episodes")

# --- task-affinity sharding, balanced by episode count -----------------------
eps_per_task = {}
for it in items:
    k = (it["suite"], it["task_id"])
    eps_per_task[k] = eps_per_task.get(k, 0) + len(it["inits"])
shard_of, load = {}, [0, 0, 0, 0]
for k, n in sorted(eps_per_task.items(), key=lambda kv: -kv[1]):
    i = load.index(min(load))
    shard_of[k] = i
    load[i] += n
print("shard episode loads:", load)

queues = [[], [], [], []]
for it in items:                      # preserves global interleave within shard
    queues[shard_of[(it["suite"], it["task_id"])]].append(it)
for i, q in enumerate(queues, 1):
    (D / f"queue_lb{i}.json").write_text(json.dumps(q, indent=1))
    print(f"queue_lb{i}.json: {len(q)} items, "
          f"{sum(len(x['inits']) for x in q)} episodes")
