#!/usr/bin/env python3
"""Assemble the sealed FINAL_EVAL base set for pi0.5/LIBERO.

PREREG (2026-09-04 FINAL_EVAL design) fixes the per-task budget:
  strata A, B : 4 natural + 3 adversarial + 1 original  (8 bases)
  strata C, D : 4 total -- resolved here as 2 natural + 1 adversarial + 1
                original, keeping every tier represented at the floor where a
                tier-blind sample could easily drop one entirely.
Selection among the judged naturals / generated adversarials is seeded
(eval_set.json seed) so the set is reproducible.

  FINAL_EVAL=1 .venv/bin/python scripts/make_pi05_eval_bases.py
  -> results/analysis/pi05_bank/eval_bases.parquet  (task, phrase, kind, k)
"""
import json
import os
import pathlib
import random

import pandas as pd

if os.environ.get("FINAL_EVAL") != "1":
    raise SystemExit("FINAL_EVAL=1 required")

REPO = pathlib.Path(__file__).resolve().parents[1]
D = REPO / "results/analysis/pi05_bank"
ev = json.load(open(D / "eval_set.json"))
TASKS, STRATA, SEED = ev["tasks"], ev["strata"], ev["seed"]
BUDGET = {"A_in_finetune": (4, 3, 1), "B_ood_range": (4, 3, 1),
          "C_ood_marginal": (2, 1, 1), "D_ood_floor": (2, 1, 1)}

nat = pd.read_parquet(REPO / "results/sealed/ph_pi05_natural_v2_img_judged.parquet")
nat = nat[nat.meaning_ok]
adv = pd.read_parquet(REPO / "results/sealed/ph_pi05_adversarial.parquet")

rows, short = [], []
for task, canon in TASKS.items():
    n_nat, n_adv, n_orig = BUDGET[STRATA[task]]
    rng = random.Random(f"{SEED}:{task}")
    pool_n = sorted(nat[nat.task == task].phrase.unique().tolist())
    pool_a = sorted(adv[adv.task == task].phrase.unique().tolist())
    rng.shuffle(pool_n); rng.shuffle(pool_a)
    if len(pool_n) < n_nat or len(pool_a) < n_adv:
        short.append((task, len(pool_n), len(pool_a)))
    for p in pool_n[:n_nat]:
        rows.append({"task": task, "phrase": p, "kind": "natural"})
    for p in pool_a[:n_adv]:
        rows.append({"task": task, "phrase": p, "kind": "adversarial"})
    if n_orig:
        rows.append({"task": task, "phrase": canon, "kind": "original"})

d = pd.DataFrame(rows).drop_duplicates(["task", "phrase"]).reset_index(drop=True)
d["stratum"] = d.task.map(STRATA)
d["k"] = d.groupby(["task", "kind"]).cumcount()
d.to_parquet(D / "eval_bases.parquet", index=False)
if short:
    print("UNDER-BUDGET tasks (task, n_nat, n_adv):", short, flush=True)
print(f"eval bases: {len(d)} over {d.task.nunique()} tasks", flush=True)
print(d.groupby(["stratum", "kind"]).size().to_string(), flush=True)
for r in d.itertuples():
    print(f"PREFLIGHT [{r.stratum}] {r.task} <{r.kind}> {r.phrase}", flush=True)
