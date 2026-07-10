"""Red-team eval assets from Gemini: trace + zeroshot phrase per task (8 calls total).

For each of the 4 SIMPLER tasks: feed Gemini the ERT red-team instruction + the
task frame with (a) the verbatim CoVer template (same call shape as the training
teacher traces, gemini-3.5-flash) -> extract_trace = the test-time trace, and
(b) a single-phrase zeroshot prompt -> the frontier-baseline arm CoVer never ran.

  .venv-gen/bin/python -m phrase_rl.gemini_redteam_assets \
    --out results/phrase_artifacts/redteam_eval_assets.parquet
"""

import argparse
import json
import os
import time

import pandas as pd

ZEROSHOT_PROMPT = (
    "You are helping a tabletop robot. The attached image shows the scene. "
    "A user gave this instruction: \"{instruction}\"\n"
    "Reply with exactly ONE short, simple instruction the robot will understand — "
    "plain words, name the objects plainly. Output only the instruction, nothing else."
)


def call_with_retry(client, types, model, contents, system=None, temperature=0.8,
                    max_tokens=5000, retries=5):
    from phrase_rl.gemini_rephrase import HARD_STOP_MARKERS, server_retry_delay
    no_think = types.ThinkingConfig(thinking_budget=0)  # analysis goes in the BODY; thinking only adds billed tokens
    for attempt in range(retries):
        try:
            cfg = types.GenerateContentConfig(temperature=temperature, max_output_tokens=max_tokens,
                                              thinking_config=no_think)
            if system:
                cfg = types.GenerateContentConfig(system_instruction=system, temperature=temperature,
                                                  max_output_tokens=max_tokens, thinking_config=no_think)
            return client.models.generate_content(model=model, contents=contents, config=cfg).text or ""
        except Exception as e:
            msg = str(e)
            if "429" in msg and any(m in msg.lower() for m in HARD_STOP_MARKERS):
                raise SystemExit(f"HARD STOP (billing): {msg[:200]}")
            if attempt == retries - 1:
                raise
            time.sleep(server_retry_delay(msg) or min(10 * 2 ** attempt, 60))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemini-3.5-flash")  # same model as the training teacher traces
    ap.add_argument("--ert", default="results/phrase_artifacts/cover_ert_instructions.json")
    ap.add_argument("--contexts", default="results/phrase_artifacts/contexts_0c_tasks.parquet")
    ap.add_argument("--n", type=int, default=16, help="list size in the CoVer trace call (match training)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from google import genai
    from google.genai import types

    from phrase_rl.cover_prompt import build_user_prompt, extract_trace, load_system_prompt

    ert = {k: v for k, v in json.load(open(args.ert)).items() if not k.startswith("_")}
    ctx = pd.read_parquet(args.contexts)
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    client.models.generate_content(model=args.model, contents="ok")  # preflight

    rows = []
    for r in ctx.itertuples():
        if r.task not in ert:
            continue
        instruction = ert[r.task]
        img_part = types.Part.from_bytes(data=r.image_png, mime_type="image/png")
        # (a) trace: verbatim CoVer template on the red-team instruction (matches training call shape)
        raw = call_with_retry(client, types, args.model,
                              [img_part, build_user_prompt(instruction, args.n)],
                              system=load_system_prompt())
        trace = extract_trace(raw)
        if not trace:
            raise SystemExit(f"trace extraction failed for {r.task}; raw head: {raw[:200]!r}")
        # (b) zeroshot single phrase (the ablation CoVer never ran)
        zs = call_with_retry(client, types, args.model,
                             [img_part, ZEROSHOT_PROMPT.format(instruction=instruction)],
                             temperature=0.2, max_tokens=100).strip().strip('"').split("\n")[0]
        rows.append({"task": r.task, "episode_index": int(r.episode_index), "t": int(r.t),
                     "ert_instruction": instruction, "trace": trace,
                     "gemini_zeroshot": zs, "raw_response": raw})
        print(f"{r.task}\n  ERT: {instruction!r}\n  zeroshot: {zs!r}\n  trace: {trace[:100]!r}...")

    pd.DataFrame(rows).to_parquet(args.out, index=False)
    print(f"wrote {len(rows)} tasks -> {args.out}")


if __name__ == "__main__":
    main()
