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

import pandas as pd
from google import genai
from google.genai import types
from tqdm import tqdm

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
                if attempt == retries - 1:
                    print(f"FAILED ep={row.episode_index}: {e}")
                    return None
                # free-tier 429s ask for ~25s waits; back off generously
                await asyncio.sleep(min(15 * 2**attempt, 90))


async def run(args):
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
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
    results = []
    for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="gemini"):
        results.append(await coro)
    ok = [r for r in results if r is not None]
    out = pd.concat([done, pd.DataFrame(ok)], ignore_index=True) if len(done) else pd.DataFrame(ok)
    if len(out):
        out.to_parquet(args.out, index=False)
    print(f"wrote {len(ok)} new ({len(out)} total) contexts to {args.out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--contexts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-rephrases", type=int, default=32)
    ap.add_argument("--model", default="gemini-3.1-pro-preview")
    ap.add_argument("--concurrency", type=int, default=8)
    args = ap.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
