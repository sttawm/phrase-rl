#!/usr/bin/env python3
"""Natural + adversarial phrases for the pi0.5/LIBERO bank: 10 + 10 per TRAIN
task (splits.json), the exact historical generate.md prompt (same as the
SIMPLER bank and the four-tier eval; text-only, gemini-pro-latest), deduped
against the seed bank's existing phrases.

  GEMINI_API_KEY=... .venv/bin/python scripts/gen_pi05_bank_phrases.py
  -> results/analysis/pi05_bank/bank_phrases.json   {canonical: {natural, adversarial}}
"""
import importlib.util
import json
import os
import pathlib
import time

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
D = ROOT / "results/analysis/pi05_bank"
N_NAT, N_ADV = 10, 10

spec = importlib.util.spec_from_file_location(
    "gen4", os.path.expanduser("~/dev/interactive-pi/pi05_libero/eval/gen_fourtier_phrases.py"))
gen4 = importlib.util.module_from_spec(spec)
import sys
sys.modules["gen4"] = gen4
try:
    spec.loader.exec_module(gen4)          # imports google.genai; main() is guarded
except SystemExit:
    pass
PROMPT, parse, MODEL = gen4.PROMPT, gen4.parse, gen4.MODEL

from google import genai                    # noqa: E402
from google.genai import types              # noqa: E402


def main():
    splits = json.load(open(D / "splits.json"))
    seed = pd.read_parquet(D / "seed_bank.parquet")
    existing = set(seed.phrase.str.strip().str.lower())
    # train canonicals: goal tasks' canonicals come from the seed bank rows
    goal_canon = {int(t): c for _, (s, t, c) in
                  seed[seed.suite == "libero_goal"][["suite", "task_id", "canonical"]]
                  .drop_duplicates().iterrows() for s in [None]} if False else {
        int(r.task_id): r.canonical
        for r in seed[seed.suite == "libero_goal"][["task_id", "canonical"]]
        .drop_duplicates().itertuples()}
    train = []
    for t in splits["train"]:
        if t["suite"] == "libero_goal":
            train.append({"suite": "libero_goal", "task_id": t["task_id"],
                          "canonical": goal_canon[t["task_id"]]})
        else:
            train.append({"suite": "libero_90", "task_id": t["task_id"],
                          "canonical": t["lang"]})
    print(f"{len(train)} train tasks")

    out_path = D / "bank_phrases.json"
    phrases = json.load(open(out_path)) if out_path.exists() else {}
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    for t in train:
        canon = t["canonical"]
        if canon in phrases:
            continue
        for attempt in range(6):
            try:
                out = client.models.generate_content(
                    model=MODEL,
                    contents=PROMPT.format(instruction=canon,
                                           n_natural=N_NAT, n_adversarial=N_ADV),
                    config=types.GenerateContentConfig(
                        temperature=0.9, max_output_tokens=8000),
                ).text or ""
            except Exception as e:
                print(f"   retry {attempt+1}: {type(e).__name__}")
                time.sleep(min(5 * 2 ** attempt, 60))
                continue
            nat, adv = parse(out)
            if len(nat) >= N_NAT and len(adv) >= N_ADV:
                dd = lambda xs: [p for p in xs if p.strip().lower() not in existing]
                phrases[canon] = {"suite": t["suite"], "task_id": t["task_id"],
                                  "natural": dd(nat[:N_NAT]),
                                  "adversarial": dd(adv[:N_ADV])}
                break
            time.sleep(2)
        else:
            raise SystemExit(f"generation failed for {canon!r}")
        json.dump(phrases, open(out_path, "w"), indent=2)
        n = phrases[canon]
        print(f"[{len(phrases)}/{len(train)}] {canon}  "
              f"(+{len(n['natural'])} nat, +{len(n['adversarial'])} adv after dedup)")
    tot = sum(len(v["natural"]) + len(v["adversarial"]) for v in phrases.values())
    print(f"total new phrases: {tot}")


if __name__ == "__main__":
    main()
