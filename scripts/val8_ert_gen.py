#!/usr/bin/env python3
"""Author one provenance-matched ERT phrase per val-8 task (same prompt + cover
shots as the sealed stage_gemini), then assemble the reference-leg parquet:
per task 3 rows — orig / adv(ERT) / oracle-best. Output:
results/phrase_artifacts/ph_val8_reference.parquet
"""
import glob
import io
import json
import sys

import pandas as pd

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
from google import genai
from google.genai import types
from phrase_rl.gemini_redteam_assets import call_with_retry
from build_sealed_assets import ERT_PROMPT  # noqa: E402
from build_probe8 import VAL4  # noqa: E402

NOM = {"widowx_carrot_on_plate": "put carrot on plate",
       "widowx_spoon_on_towel": "put the spoon on the towel",
       "widowx_stack_cube": "stack the green block on the yellow block",
       "widowx_put_eggplant_in_basket": "put eggplant into yellow basket"}
NOM.update(VAL4)

cover = json.load(open("results/phrase_artifacts/cover_release_ert_rephrases.json"))["instructions"]
shots = "\n".join(f'nominal: "{n}" -> ERT: "{r["ert_rephrases"][0]}"'
                  for n, r in list(cover.items())[:4])
client = genai.Client()

ctx = pd.read_parquet("results/phrase_artifacts/contexts_probe8.parquet")
img_by_task = {r.task: r.image_png for r in ctx.itertuples()}

KEYMAP = {"spoon": "widowx_spoon_on_towel", "stack": "widowx_stack_cube",
          "carrotplate": "widowx_carrot_on_plate", "eggplant": "widowx_put_eggplant_in_basket",
          "keyboard": "widowx_carrot_on_keyboard_clean", "wheel": "widowx_carrot_on_wheel_clean",
          "ramekin": "widowx_coke_can_on_ramekin_clean", "cokeplate": "widowx_coke_can_on_plate_clean"}
best = {}
for f in sorted(glob.glob("results/search/*_results.json")):
    base = f.split("/")[-1]
    if base.startswith("sealedsearch"):
        continue
    key = next((k for k in KEYMAP if base.startswith(k)), None)
    if key is None:
        continue
    t = KEYMAP[key]
    for e in json.load(open(f)).get("scoreboard", []):
        if e.get("n", 0) >= 36 and (t not in best or e["success_pct"] > best[t][1]):
            best[t] = (e["phrase"], e["success_pct"])

rows = []
for task, nominal in NOM.items():
    img = img_by_task[task]
    part = types.Part.from_bytes(data=bytes(img), mime_type="image/png")
    ert = call_with_retry(client, types, "gemini-3.5-flash",
                          [part, ERT_PROMPT.format(shots=shots, nominal=nominal)],
                          temperature=0.8, max_tokens=400).strip().strip('"').split("\n")[0].strip()
    rows += [{"task": task, "arm": "val8_orig", "phrase": nominal, "instruction": nominal},
             {"task": task, "arm": "val8_adv", "phrase": ert, "instruction": ert},
             {"task": task, "arm": "val8_oracle", "phrase": best[task][0], "instruction": best[task][0]}]
    print(f"{task}\n  ERT: {ert!r}")
pd.DataFrame(rows).to_parquet("results/phrase_artifacts/ph_val8_reference.parquet", index=False)
print(f"wrote {len(rows)} rows")
