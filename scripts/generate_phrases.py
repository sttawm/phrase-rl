#!/usr/bin/env python3
"""The generate_phrases step the algorithm calls for, which the loop driver never
implemented.

    generate_phrases(task) = oracle_search + 2 adversarials + 7 naturals

Oracle search has already been run for all 213 training tasks (search_boards.jsonl)
and IS the training half of the bank. What was never produced is the other two
arms: not one natural and not one adversarial phrase exists in the bank, so the
distiller would be inferring repair rules from evidence containing nothing that
needs repairing.

This generates both arms per task with Gemini, reusing phrases we already hold
where they are non-sealed, and appends them to the bank manifest with honest
`kind` labels so the evidence file can separate the regimes.

Sealed tasks are excluded outright -- generating for them would burn sealed
material outside a PREREG amendment.

    GEMINI_API_KEY=... .venv/bin/python scripts/generate_phrases.py [--limit N]
"""
import argparse
import concurrent.futures as cf
import json
import os
import re
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "results/analysis/bank_generated.parquet"
PROMPT = (REPO / "prompts/rules_loop/generate.md").read_text()
SEALED_STEMS = ["orange_juice", "nut_on_plate", "eggplant_on_keyboard",
                "coke_can_on_keyboard", "pepsi", "cube_on_plate",
                "small_plate_on_green_cube", "carrot_on_sponge",
                "eggplant_on_sponge", "carrot_on_ramekin", "coke_can_on_wheel",
                "nut_on_wheel"]

ap = argparse.ArgumentParser()
ap.add_argument("--n-natural", type=int, default=7)
ap.add_argument("--n-adversarial", type=int, default=2)
ap.add_argument("--limit", type=int, default=0, help="0 = all tasks")
ap.add_argument("--workers", type=int, default=8)
ap.add_argument("--model", default="gemini-pro-latest")
args = ap.parse_args()


def is_sealed(task):
    return any(s in str(task) for s in SEALED_STEMS)


bank = pd.read_parquet(REPO / "results/analysis/bank_to_score.parquet")
tasks = [t for t in sorted(bank.task.unique()) if not is_sealed(t)]
if args.limit:
    tasks = tasks[: args.limit]

# The instruction to reword: for training tasks the task name IS the instruction;
# for the sim tasks it is the env's own wording, recorded in the distract split.
sim_instr = {}
split = REPO / "results/analysis/distract_variants_split.json"
if split.exists():
    for grp in json.load(split.open()).values():
        for r in grp:
            sim_instr[f"widowx_{r['task']}"] = r["instruction"]
for t, i in [("widowx_carrot_on_plate", "put carrot on plate"),
             ("widowx_spoon_on_towel", "put the spoon on the towel"),
             ("widowx_stack_cube", "stack the green block on the yellow block"),
             ("widowx_put_eggplant_in_basket", "put eggplant into yellow basket"),
             ("widowx_carrot_on_keyboard_clean", "put carrot on keyboard"),
             ("widowx_carrot_on_wheel_clean", "put the carrot on the wheel"),
             ("widowx_coke_can_on_plate_clean", "put coke can on plate"),
             ("widowx_coke_can_on_ramekin_clean", "put the coke can in the ramekin")]:
    sim_instr.setdefault(t, i)


def instruction_for(task):
    t = str(task)
    return sim_instr.get(t, t) if t.startswith("widowx_") else t


done = {}
if OUT.exists():
    prev = pd.read_parquet(OUT)
    for t, g in prev.groupby("task"):
        done[t] = g
    print(f"resume: {len(done)} tasks already generated", flush=True)

from google import genai                       # noqa: E402
from google.genai import types                 # noqa: E402

client = genai.Client()


def one(task):
    if task in done:
        return done[task]
    prompt = (PROMPT.replace("{{instruction}}", instruction_for(task))
                    .replace("{{n_natural}}", str(args.n_natural))
                    .replace("{{n_adversarial}}", str(args.n_adversarial)))
    try:
        resp = client.models.generate_content(
            model=args.model, contents=prompt,
            config=types.GenerateContentConfig(
                temperature=1.0,
                http_options=types.HttpOptions(timeout=180_000)))
        text = (resp.text or "").strip()
    except Exception as e:
        print(f"  FAILED {task!r}: {type(e).__name__}: {str(e)[:90]}", flush=True)
        return None
    kind, rows = None, []
    for line in text.splitlines():
        s = line.strip().lstrip("-*0123456789. ").strip()
        if not s:
            continue
        up = s.upper()
        if up.startswith("NATURAL"):
            kind = "natural"; continue
        if up.startswith("ADVERSARIAL"):
            kind = "adversarial"; continue
        if kind and len(s) > 3:
            rows.append({"task": task, "phrase": s, "kind": kind,
                         "source": f"generated_{kind}"})
    if not rows:
        print(f"  EMPTY {task!r}", flush=True)
        return None
    df = pd.DataFrame(rows).drop_duplicates(["task", "phrase"])
    return df


results = list(done.values())
todo = [t for t in tasks if t not in done]
print(f"{len(tasks)} tasks ({len(todo)} outstanding); "
      f"{args.n_natural} natural + {args.n_adversarial} adversarial each", flush=True)

with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
    for i, df in enumerate(ex.map(one, todo), 1):
        if df is not None:
            results.append(df)
        if i % 10 == 0 or i == len(todo):
            pd.concat(results, ignore_index=True).to_parquet(OUT, index=False)
            tot = sum(len(d) for d in results)
            print(f"[{i}/{len(todo)}] {len(results)} tasks, {tot} phrases -> {OUT.name}",
                  flush=True)

allp = pd.concat(results, ignore_index=True).drop_duplicates(["task", "phrase"])
allp.to_parquet(OUT, index=False)
print(f"GENERATION-DONE {len(allp)} phrases over {allp.task.nunique()} tasks")
print(allp.kind.value_counts().to_string())
