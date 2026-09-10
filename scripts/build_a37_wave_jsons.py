#!/usr/bin/env python3
"""Assemble A37 wave JSONs in CoVer's rephrase-file schema.

One wave per adversarial attack index k (all 12 tasks, one attack each) plus a
single original-condition wave. Their eval looks entries up by the env's exact
get_language_instruction() string, so keys use the ENV nominal (captured from
the 12/12 settle test on cv2, 2026-09-09) — NOT our audit nominal, which
differs for small_plate ("put the small plate on the green cube" in-env).
The "original" field is what the episode executes (our base); "ert_rephrases"
is the Gemini-generated recovery pool (>=8 unique, eval slices the first 8).

  .venv/bin/python scripts/build_a37_wave_jsons.py
"""
import json
import pathlib
import time

import pandas as pd

R = pathlib.Path(__file__).resolve().parents[1]
OUT = R / "results/sealed/a37_waves"

# env.get_language_instruction() per task — settle-test output, cv2 graft.log
ENV_NOMINAL = {
    "widowx_carrot_on_ramekin_clean": "put carrot on ramekin",
    "widowx_carrot_on_sponge_clean": "put carrot on sponge",
    "widowx_coke_can_on_keyboard_clean": "put coke can on keyboard",
    "widowx_coke_can_on_wheel_clean": "put coke can on wheel",
    "widowx_cube_on_plate_clean": "put green cube on plate",
    "widowx_eggplant_on_keyboard_clean": "put eggplant on keyboard",
    "widowx_eggplant_on_sponge_clean": "put eggplant on sponge",
    "widowx_nut_on_plate_clean": "put nut on plate",
    "widowx_nut_on_wheel_clean": "put nut on wheel",
    "widowx_orange_juice_on_plate_clean": "put orange juice on plate",
    "widowx_pepsi_on_plate_clean": "put pepsi can on plate",
    "widowx_small_plate_on_green_cube_clean": "put the small plate on the green cube",
}

df = pd.read_parquet(R / "results/sealed/a37_cover_rephrases.parquet")
OUT.mkdir(parents=True, exist_ok=True)
stamp = time.strftime("%Y%m%d_%H%M%S")

waves = [("adv", k) for k in sorted(df[df.cond == "adv"].k.unique())]
waves += [("orig", 0)] if (df.cond == "orig").any() else []
waves += [("nat", k) for k in sorted(df[df.cond == "nat"].k.unique())]

for cond, k in waves:
    g = df[(df.cond == cond) & (df.k == k)]
    missing = set(ENV_NOMINAL) - set(g.task)
    # natural waves are ragged (14-16 bases/task); the eval's A37-diag patch
    # skips suite tasks absent from a wave file
    assert not missing or cond == "nat", f"{cond} k{k}: missing tasks {missing}"
    instructions = {}
    for r in g.itertuples():
        reph = list(r.rephrases)
        assert len(reph) >= 8, f"{r.task}/{cond}/k{k}: only {len(reph)} rephrases"
        instructions[ENV_NOMINAL[r.task]] = {
            "original": r.base, "ert_rephrases": reph}
    name = f"wave_{cond}_k{k}.json" if cond == "adv" else "wave_orig.json"
    (OUT / name).write_text(json.dumps(
        {"timestamp": stamp, "a37_wave": f"{cond}_k{k}",
         "model": g.model.iloc[0], "instructions": instructions}, indent=1))
    print(f"{name}: 12 tasks, "
          f"rephrases min={g.rephrases.map(len).min()} max={g.rephrases.map(len).max()}")
print("->", OUT)
