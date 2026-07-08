"""Generate rephrases with Gemini under CoVer's VERBATIM prompt (frontier baseline).

Same call structure as CoVer's GPT-4o usage: system persona + user-turn template
(inline reasoning -> numbered list), image attached, temperature 0.8. Output rows
keep the full raw response (reasoning included) for analysis.

  python -m phrase_rl.gemini_cover_rephrase --contexts data/contexts_val_0b.parquet \
    --out data/cover_gemini_val.parquet --n 32
"""

import argparse
import asyncio
import os

import pandas as pd
from google import genai
from google.genai import types
from tqdm import tqdm

from phrase_rl.cover_prompt import build_user_prompt, load_system_prompt, parse_reworded
from phrase_rl.gemini_rephrase import HARD_STOP, HARD_STOP_MARKERS, server_retry_delay


async def one_call(client, sem, model, system_prompt, row, n, retries=5):
    async with sem:
        for attempt in range(retries):
            if HARD_STOP.is_set():
                return None
            try:
                resp = await client.aio.models.generate_content(
                    model=model,
                    contents=[
                        types.Part.from_bytes(data=row.image_png, mime_type="image/png"),
                        build_user_prompt(row.instruction, n),
                    ],
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt, temperature=0.8, max_output_tokens=5000
                    ),
                )
                phrases = parse_reworded(resp.text or "")
                if len(phrases) < max(6, n // 2):
                    raise ValueError(f"parsed only {len(phrases)} phrases")
                return {
                    "episode_index": row.episode_index, "t": row.t,
                    "instruction": row.instruction, "rephrases": phrases[:n],
                    "raw_response": resp.text,
                }
            except Exception as e:
                msg = str(e)
                if "429" in msg and any(m in msg.lower() for m in HARD_STOP_MARKERS):
                    if not HARD_STOP.is_set():
                        print(f"\nHARD STOP: {msg[:150]}")
                        HARD_STOP.set()
                    return None
                if attempt == retries - 1:
                    print(f"FAILED ep={row.episode_index}: {msg[:150]}")
                    return None
                await asyncio.sleep(server_retry_delay(msg) or min(10 * 2**attempt, 60))


async def run(args):
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    try:
        await client.aio.models.generate_content(model=args.model, contents="ok")
    except Exception as e:
        raise SystemExit(f"PREFLIGHT FAILED: {str(e)[:200]}")

    df = pd.read_parquet(args.contexts)
    done = pd.DataFrame()
    if os.path.exists(args.out):
        done = pd.read_parquet(args.out)
        keys = set(zip(done["episode_index"], done["t"]))
        df = df[~df.apply(lambda r: (r["episode_index"], r["t"]) in keys, axis=1)]
        print(f"resume: {len(done)} done, {len(df)} remaining")

    system_prompt = load_system_prompt()
    sem = asyncio.Semaphore(args.concurrency)
    tasks = [one_call(client, sem, args.model, system_prompt, row, args.n) for row in df.itertuples()]
    results, n_ok = [], 0
    for i, coro in enumerate(tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="gemini-cover")):
        r = await coro
        if r is not None:
            results.append(r); n_ok += 1
        if n_ok and (n_ok % 25 == 0 or i == len(tasks) - 1):
            out = pd.concat([done, pd.DataFrame(results)], ignore_index=True) if len(done) else pd.DataFrame(results)
            out.to_parquet(args.out, index=False)
    print(f"wrote {n_ok} new ({len(done) + n_ok} total) -> {args.out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--contexts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--model", default="gemini-3.1-pro-preview")
    ap.add_argument("--concurrency", type=int, default=6)
    args = ap.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
