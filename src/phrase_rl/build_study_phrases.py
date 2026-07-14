"""Assemble the proxy-reward diagnostic study's phrase sets.

Per task (4 SIMPLER ID tasks):
  - 16 OWN rephrases (from the validated 0c pools, qwen arm first, gemini fill)
  - 16 CROSS-task phrases (6/5/5 from the other tasks' own sets) — contrastive
  - 1 original + 2 red-team (CoVer ERT + one fresh ERT-style, generated here)

Outputs:
  data/study_phrases_all.parquet      (35/task, for proxy scoring)
  data/study_phrases_rollable.parquet (19/task: own16 + original + 2 RT, for rollouts)

Run on a pod with .venv-gen + GEMINI_API_KEY (CPU-only, ~1 min):
  .venv-gen/bin/python -m phrase_rl.build_study_phrases
"""

import json
import os

import pandas as pd

TASKS = ["widowx_spoon_on_towel", "widowx_carrot_on_plate",
         "widowx_stack_cube", "widowx_put_eggplant_in_basket"]


def main():
    pool = pd.read_parquet("data/phrases_0c_redo.parquet")
    ert1 = {k: v for k, v in json.load(open("results/phrase_artifacts/cover_ert_instructions.json")).items()
            if not k.startswith("_")}

    # fresh ERT-style instruction per task (Gemini, ERT verbatim prompt, image-conditioned)
    ert2_path = "data/ert2_instructions.json"
    if not os.path.exists(ert2_path):
        from google import genai
        from google.genai import types
        ctx = pd.read_parquet("results/phrase_artifacts/contexts_0c_tasks.parquet")
        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        no_think = types.ThinkingConfig(thinking_budget=0)
        SYS = ("You are a quality assurance engineer for a robot. Your goal is to come up with "
               "instructions that describe the given task correctly, is similar to what human users "
               "would possibly give, and yet challenge the robot's capability on accomplishing the task.")
        ert2 = {}
        for r in ctx.itertuples():
            usr = (f"The attached image is an example image of the initial state of a robot that "
                   f"will perform the task: {r.instruction}. Generate a diverse set of exactly 5 instructions.")
            resp = client.models.generate_content(
                model="gemini-3.5-flash",
                contents=[types.Part.from_bytes(data=r.image_png, mime_type="image/png"), usr],
                config=types.GenerateContentConfig(temperature=0.9, max_output_tokens=400,
                                                   thinking_config=no_think))
            import re
            # numbered items ONLY (the response starts with a preamble line — 2026-07-14 bug)
            lines = [re.sub(r"^\s*\d+[.)]\s*", "", l).strip().strip('"')
                     for l in (resp.text or "").split("\n") if re.match(r"^\s*\d+[.)]", l)]
            lines = [l for l in lines if 4 < len(l.split()) < 20]
            assert lines, f"no numbered instructions parsed for {r.task}: {resp.text[:120]!r}"
            # pick the first candidate that differs from CoVer's ERT phrase
            pick = next((l for l in lines if l.lower() != ert1[r.task].lower()), lines[0])
            ert2[r.task] = pick
            print(f"ERT2 {r.task}: {pick!r}")
        json.dump(ert2, open(ert2_path, "w"), indent=1)
    ert2 = json.load(open(ert2_path))

    def own16(task):
        sub = pool[(pool.task == task) & (pool.arm != "original")]
        qwen = [p for p in sub[sub.arm.str.contains("qwen", case=False)].phrase][:16]
        gem = [p for p in sub[sub.arm.str.contains("gemini", case=False)].phrase]
        out = list(dict.fromkeys(qwen + gem))[:16]
        assert len(out) == 16, f"{task}: only {len(out)} own phrases"
        return out

    owns = {t: own16(t) for t in TASKS}
    origs = {r.task: r.phrase for r in pool[pool.arm == "original"].itertuples()}

    rows_all, rows_roll = [], []
    for t in TASKS:
        others = [x for x in TASKS if x != t]
        cross = owns[others[0]][:6] + owns[others[1]][:5] + owns[others[2]][:5]
        for p in owns[t]:
            rows_all.append({"task": t, "arm": "own", "phrase": p})
            rows_roll.append({"task": t, "arm": "own", "phrase": p})
        for p in cross:
            rows_all.append({"task": t, "arm": "cross", "phrase": p})
        for arm, p in [("original", origs[t]), ("redteam1", ert1[t]), ("redteam2", ert2[t])]:
            rows_all.append({"task": t, "arm": arm, "phrase": p})
            rows_roll.append({"task": t, "arm": arm, "phrase": p})
    pd.DataFrame(rows_all).to_parquet("data/study_phrases_all.parquet", index=False)
    pd.DataFrame(rows_roll).to_parquet("data/study_phrases_rollable.parquet", index=False)
    print(f"all: {len(rows_all)} rows | rollable: {len(rows_roll)} rows")


if __name__ == "__main__":
    main()
