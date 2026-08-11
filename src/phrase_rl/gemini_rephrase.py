"""Generate reasoning traces + N rephrases per context with Gemini.

For each context (image + original instruction), one call produces:
  - a short reasoning trace r (scene/task analysis, later part of the phrase
    model's conditioning c = (o_t, l, r))
  - N rephrases of the instruction (32 for sensitivity phases, 16-lists for SFT)

Interactive async calls with bounded concurrency — fine for the ~250-context
0b set (~$2). Phase 1's 2.5k-context job should move to the batch API (half price).

Requires GEMINI_API_KEY in the environment. Usage:
  python -m phrase_rl.gemini_rephrase \
    --contexts data/contexts_val_0b.parquet \
    --out data/rephrases_val_0b.parquet --n-rephrases 32
"""

import argparse
import asyncio
import json
import os
import re

import pandas as pd
# Lazy: faithfulness_gate (-> phase2_train -> every scoring consumer) imports
# this module only for HARD_STOP_MARKERS/server_retry_delay; score-only pods
# have no google-genai SDK after /root wipes take .venv-gen with them.
try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = types = None
from tqdm import tqdm

# Set when a 429 indicates depleted credits/quota-0 — a state retries can never
# fix. All pending calls abort immediately instead of retry-storming.
HARD_STOP = asyncio.Event()
HARD_STOP_MARKERS = ("depleted", "prepay", "check your plan", "limit: 0")


def server_retry_delay(err_text: str) -> float | None:
    m = re.search(r"retry.{0,20}?(\d+(?:\.\d+)?)\s*s", err_text, re.I)
    return float(m.group(1)) if m else None

PROMPT = """\
You are helping a robot manipulation policy. The robot was given this instruction:

"{instruction}"

The attached image is the robot's current camera view.

First, briefly reason about the scene: which objects are relevant, where they \
are, and what the task requires (2-4 sentences).

Then write exactly {n} rephrasings of the instruction. Rules:
- Preserve the exact task: same object(s), same target location, same action.
- Vary verbs, word order, specificity, and sentence structure as much as possible.
- Keep each rephrase a short imperative sentence, similar in length to the original.
- No numbering artifacts, quotes, or commentary inside the rephrases.

Return JSON: {{"trace": "...", "rephrases": ["...", ...]}} with exactly {n} rephrases.
"""


async def one_call(client, sem, model, row, n, retries=6):
    async with sem:
        for attempt in range(retries):
            if HARD_STOP.is_set():
                return None
            try:
                resp = await client.aio.models.generate_content(
                    model=model,
                    contents=[
                        types.Part.from_bytes(data=row.image_png, mime_type="image/png"),
                        PROMPT.format(instruction=row.instruction, n=n),
                    ],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json", temperature=1.0
                    ),
                )
                out = json.loads(resp.text)
                assert len(out["rephrases"]) == n, f"got {len(out['rephrases'])} rephrases"
                return {
                    "episode_index": row.episode_index,
                    "t": row.t,
                    "instruction": row.instruction,
                    "trace": out["trace"],
                    "rephrases": out["rephrases"],
                }
            except Exception as e:
                msg = str(e)
                if "429" in msg and any(m in msg.lower() for m in HARD_STOP_MARKERS):
                    if not HARD_STOP.is_set():
                        print(f"\nHARD STOP — unretryable quota state: {msg[:180]}\n"
                              "Aborting all pending calls (results so far are saved).")
                        HARD_STOP.set()
                    return None
                if attempt == retries - 1:
                    print(f"FAILED ep={row.episode_index}: {msg[:200]}")
                    return None
                # honor the server's suggested delay when present; else backoff
                delay = server_retry_delay(msg) or min(15 * 2**attempt, 90)
                await asyncio.sleep(delay + 1)


async def run(args):
    if genai is None:
        raise RuntimeError("google-genai is not installed in this venv")
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    # preflight: one tiny call so a dead quota/billing state fails in seconds,
    # not after launching thousands of doomed, retry-storming tasks
    try:
        await client.aio.models.generate_content(model=args.model, contents="ok")
    except Exception as e:
        raise SystemExit(f"PREFLIGHT FAILED for {args.model} — not launching batch:\n{str(e)[:300]}")

    df = pd.read_parquet(args.contexts)

    done = pd.DataFrame()
    if os.path.exists(args.out):  # resume: skip contexts already rephrased
        done = pd.read_parquet(args.out)
        done_keys = set(zip(done["episode_index"], done["t"]))
        df = df[~df.apply(lambda r: (r["episode_index"], r["t"]) in done_keys, axis=1)]
        print(f"resume: {len(done)} done, {len(df)} remaining")

    sem = asyncio.Semaphore(args.concurrency)
    tasks = [
        one_call(client, sem, args.model, row, args.n_rephrases)
        for row in df.itertuples()
    ]
    results, n_ok = [], 0
    for i, coro in enumerate(tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="gemini")):
        r = await coro
        if r is not None:
            results.append(r)
            n_ok += 1
        # write incrementally — a killed/crashed run must never lose paid API results
        if n_ok and (n_ok % 25 == 0 or i == len(tasks) - 1):
            out = pd.concat([done, pd.DataFrame(results)], ignore_index=True) if len(done) else pd.DataFrame(results)
            out.to_parquet(args.out, index=False)
    total = len(done) + n_ok
    print(f"wrote {n_ok} new ({total} total) contexts to {args.out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--contexts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-rephrases", type=int, default=32)
    ap.add_argument("--model", default="gemini-3.1-pro-preview")
    ap.add_argument("--concurrency", type=int, default=6)
    args = ap.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
