#!/usr/bin/env python3
"""Replay probe: actual Gemini thinking-token usage under thinking_budget=16384
for the three pro conditions (rules+adversarial, bare+adversarial, rules+nominal).
Prompts reconstructed exactly as the arm generators built them; outputs are NOT
used for anything — only usage_metadata is recorded.
"""
import json
import os
import time

import pandas as pd
from google import genai
from google.genai import types

RULES = open("results/analysis/b4_phrasing_rules_v3.md").read()
BARE = open("results/analysis/gemini_bare_baseline_prompt.md").read()
ga = pd.read_parquet("results/sealed/sealed_assets_gemini.parquet")
client = genai.Client()

def probe(preamble, instr, trace):
    prompt = (f"{preamble}\n\n---\n\nApply the rules above.\n\nTrace:\n{trace}\n\n"
              f"Incoming instruction: {instr}\n\nReply with ONLY the rewritten instruction.")
    if preamble is BARE:
        prompt = f"{preamble}\n\nTrace:\n{trace}\n\nIncoming instruction: {instr}\n\nReply with ONLY the rewritten instruction."
    for a in range(4):
        try:
            r = client.models.generate_content(
                model="gemini-pro-latest", contents=[prompt],
                config=types.GenerateContentConfig(
                    temperature=0.2, max_output_tokens=20000,
                    thinking_config=types.ThinkingConfig(thinking_budget=16384)))
            u = r.usage_metadata
            return {"thoughts": u.thoughts_token_count or 0,
                    "output": u.candidates_token_count or 0,
                    "prompt": u.prompt_token_count or 0}
        except Exception as e:
            print("retry", a, type(e).__name__, flush=True)
            time.sleep(10 * (a + 1))
    return None

res = {"rules_adversarial": [], "bare_adversarial": [], "rules_nominal": []}
for r in ga.itertuples():
    res["rules_adversarial"].append(probe(RULES, r.ert_instruction, r.trace))
    res["bare_adversarial"].append(probe(BARE, r.ert_instruction, r.trace))
    res["rules_nominal"].append(probe(RULES, r.nominal, r.trace))
    print(f"{r.task}: done", flush=True)

out = {}
for k, v in res.items():
    t = [x["thoughts"] for x in v if x]
    out[k] = {"n": len(t), "mean_thoughts": round(sum(t) / len(t)), "min": min(t), "max": max(t),
              "mean_prompt": round(sum(x["prompt"] for x in v if x) / len(t))}
    print(k, out[k], flush=True)
json.dump({"budget": 16384, "conditions": out,
           "raw": {k: [x["thoughts"] for x in v if x] for k, v in res.items()}},
          open("results/analysis/thinking_token_probe.json", "w"), indent=1)
print("-> results/analysis/thinking_token_probe.json")
