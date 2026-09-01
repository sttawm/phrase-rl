#!/usr/bin/env python3
"""Task-level train/val/sealed-test split for the pi0.5/LIBERO phrase bank.

Design (PI05-BANK-HANDOFF.md, agreed 2026-09-01, val widened 10->15):
  TRAIN = 10 libero_goal (in-finetune, generation-touched) + 20 touched l90
          + 35 fresh l90;  VAL = 15 fresh l90;  SEALED TEST = 20 fresh l90.
  Untouched in-finetune suites (spatial/object/10) stay an unassigned reserve.
  Assignment of the 70 untouched l90 tasks: uniform random, SEED below,
  stratified only by the in-vocab label (every token of the task string appears
  in the 40 finetune task strings; computed from text, no rollouts).

Inputs (committed under results/analysis/pi05_bank/ with provenance):
  libero_suite_task_map.py  -- LIBERO github, task names for the 4 finetune suites
  libero90_categories.json  -- interactive-pi categorize_libero90.py output (90 langs)
  fourtier_tasks{,_ext}.json -- the 30 generation-touched tasks

Output: results/analysis/pi05_bank/splits.json
"""
import json
import random
import re
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
D = ROOT / "results/analysis/pi05_bank"
SEED = 20260901

# --- finetune vocabulary: the 40 task strings of the 4 finetune suites --------
ns = {}
exec((D / "libero_suite_task_map.py").read_text(), ns)
tmap = ns["libero_task_map"]
FINETUNE_SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]


def lang_of(name):
    # libero_10/libero_90 names carry a SCENE prefix (e.g. LIVING_ROOM_SCENE2_...)
    name = re.sub(r"^[A-Z_0-9]+_SCENE\d+_", "", name)
    return name.replace("_", " ").lower().strip()


finetune_langs = [lang_of(n) for s in FINETUNE_SUITES for n in tmap[s]]
assert len(finetune_langs) == 40, len(finetune_langs)
VOCAB = set(t for l in finetune_langs for t in re.findall(r"[a-z']+", l))

# --- l90 tasks + in-vocab label ----------------------------------------------
l90 = json.load(open(D / "libero90_categories.json"))
assert len(l90) == 90
touched = sorted(
    x["task_id"] for x in
    json.load(open(D / "fourtier_tasks.json")) + json.load(open(D / "fourtier_tasks_ext.json"))
    if x["suite"] == "libero_90")
assert len(touched) == 20, touched

rows = []
for t in l90:
    toks = set(re.findall(r"[a-z']+", t["lang"].lower()))
    rows.append({"task_id": t["task_id"], "lang": t["lang"], "scene": t["scene"],
                 "in_vocab": toks <= VOCAB, "oov_words": sorted(toks - VOCAB),
                 "touched": t["task_id"] in touched})

n_iv = sum(r["in_vocab"] for r in rows)
print(f"l90 in-vocab: {n_iv}/90; among untouched: "
      f"{sum(r['in_vocab'] for r in rows if not r['touched'])}/70")

# --- stratified uniform assignment of the 70 untouched -----------------------
rng = random.Random(SEED)
assign = {}
for stratum in (True, False):
    pool = sorted(r["task_id"] for r in rows if not r["touched"] and r["in_vocab"] == stratum)
    rng.shuffle(pool)
    n = len(pool)
    # proportional 35/15/20 of 70 -> exact within-stratum counts via rounding
    n_test = round(20 * n / 70)
    n_val = round(15 * n / 70)
    for tid in pool[:n_test]:
        assign[tid] = "test"
    for tid in pool[n_test:n_test + n_val]:
        assign[tid] = "val"
    for tid in pool[n_test + n_val:]:
        assign[tid] = "train"

counts = {}
for v in assign.values():
    counts[v] = counts.get(v, 0) + 1
print("untouched l90 assignment:", counts)

goal_train = [{"suite": "libero_goal", "task_id": i} for i in range(10)]
out = {
    "seed": SEED,
    "design": "PI05-BANK-HANDOFF.md 2026-09-01 (val=15)",
    "vocab_source": "libero_suite_task_map.py (4 finetune suites, 40 strings)",
    "train": goal_train
             + [{"suite": "libero_90", "task_id": r["task_id"], "lang": r["lang"],
                 "in_vocab": r["in_vocab"], "touched": r["touched"]}
                for r in rows if r["touched"] or assign.get(r["task_id"]) == "train"],
    "val": [{"suite": "libero_90", "task_id": r["task_id"], "lang": r["lang"],
             "in_vocab": r["in_vocab"]}
            for r in rows if assign.get(r["task_id"]) == "val"],
    "sealed_test": [{"suite": "libero_90", "task_id": r["task_id"], "lang": r["lang"],
                     "in_vocab": r["in_vocab"]}
                    for r in rows if assign.get(r["task_id"]) == "test"],
    "reserve_in_finetune_suites": ["libero_spatial", "libero_object", "libero_10"],
    "reserve_exclusions": [["libero_10", 5]],  # 855 probe episodes exist -- contaminated
    "in_vocab_labels": rows,
}
(D / "splits.json").write_text(json.dumps(out, indent=1))
tr, va, te = len(out["train"]), len(out["val"]), len(out["sealed_test"])
print(f"splits.json: train={tr} (10 goal + {tr-10} l90), val={va}, sealed_test={te}")
iv = lambda xs: sum(1 for x in xs if x.get("in_vocab"))
print(f"in-vocab by split: train_l90={iv(out['train'])}, val={iv(out['val'])}, "
      f"test={iv(out['sealed_test'])}")
