#!/usr/bin/env python3
"""A37: CoVer-pipeline rephrases for the sealed 12 — Gemini VLM (recorded deviation).

Runs CoVer's released rephrase pipeline on our sealed bases: their verbatim
prompt scaffold (phrase_rl.cover_prompt, byte-checked against the reference),
boot-time first-frame image conditioning (results/sealed/sealed_frames.parquet,
the same frames that conditioned the A33 naturals), temperature 0.8,
batch_number=10 per call, their dedup (normalize = lower/strip; drop copies of
the base) and their top-up loop (<=5 replacement attempts) until >=8 unique
rephrases per base. Only the VLM differs: Gemini instead of GPT-4o — the A37
deviation note in results/analysis/PREREG_closing_leg.md.

Bases (adv): the 72 sealed attacks = sealed_assets_gemini.ert_instruction (k=0)
+ a29_new_erts in file order (k=1..5), 6 per task. Bases (orig): the 12
canonical nominals (k=0). Naturals are a later gate — not generated here.

SEALED-SET DISCIPLINE: refuses to run without FINAL_EVAL=1; preflight-prints
every base before the first API call.

  FINAL_EVAL=1 GEMINI_API_KEY=... .venv/bin/python scripts/gen_a37_cover_rephrases.py \
      --conds adv,orig --out results/sealed/a37_cover_rephrases.parquet
"""
import argparse
import asyncio
import os
import pathlib
import sys

import pandas as pd
from google import genai
from google.genai import types
from tqdm import tqdm

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from phrase_rl.cover_prompt import build_user_prompt, load_system_prompt, parse_reworded
from phrase_rl.gemini_rephrase import HARD_STOP, HARD_STOP_MARKERS, server_retry_delay

R = pathlib.Path(__file__).resolve().parents[1]
BATCH = 10                 # their get_rephrase_batch batch_number per call
MIN_UNIQUE = 8             # eval consumes lang_rephrase_num=8
TOPUP_ATTEMPTS = 5         # their MAX_DUPLICATE_REPLACEMENT_ATTEMPTS


def normalize_text(t: str) -> str:
    """CoVer's normalize_text, verbatim (generate_simpler_rephrases_vlm.py)."""
    return t.lower().strip()


def build_bases(conds):
    assets = pd.read_parquet(R / "results/sealed/sealed_assets_gemini.parquet")
    erts = pd.read_parquet(R / "results/sealed/a29_new_erts.parquet")
    rows = []
    for a in assets.itertuples():
        if "adv" in conds:
            rows.append({"task": a.task, "nominal": a.nominal, "cond": "adv",
                         "k": 0, "base": a.ert_instruction})
        if "orig" in conds:
            rows.append({"task": a.task, "nominal": a.nominal, "cond": "orig",
                         "k": 0, "base": a.nominal})
    if "adv" in conds:
        nom = dict(zip(assets.task, assets.nominal))
        for task, g in erts.groupby("task", sort=False):
            for k, p in enumerate(g["phrase"].tolist(), start=1):
                rows.append({"task": task, "nominal": nom[task], "cond": "adv",
                             "k": k, "base": p})
    df = pd.DataFrame(rows)
    n_adv = (df.cond == "adv").sum()
    assert "adv" not in conds or n_adv == 72, f"expected 72 adv bases, got {n_adv}"
    return df


async def gen_one(client, sem, model, system_prompt, row, frame_png, retries=5):
    """One base -> >=MIN_UNIQUE unique rephrases via CoVer's call + top-up loop."""
    async def call(batch_number):
        for attempt in range(retries):
            if HARD_STOP.is_set():
                return None
            try:
                resp = await client.aio.models.generate_content(
                    model=model,
                    contents=[types.Part.from_bytes(data=frame_png, mime_type="image/png"),
                              build_user_prompt(row.base, batch_number)],
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt, temperature=0.8,
                        max_output_tokens=5000),
                )
                return resp.text or ""
            except Exception as e:
                msg = str(e)
                if "429" in msg and any(m in msg.lower() for m in HARD_STOP_MARKERS):
                    if not HARD_STOP.is_set():
                        print(f"\nHARD STOP: {msg[:150]}")
                        HARD_STOP.set()
                    return None
                if attempt == retries - 1:
                    print(f"FAILED {row.task}/{row.cond}/k{row.k}: {msg[:150]}")
                    return None
                await asyncio.sleep(server_retry_delay(msg) or min(10 * 2**attempt, 60))

    async with sem:
        raws = []
        uniq, seen = [], {normalize_text(row.base)}
        for _ in range(1 + TOPUP_ATTEMPTS):
            raw = await call(BATCH)
            if raw is None:
                break
            raws.append(raw)
            for c in parse_reworded(raw):
                n = normalize_text(c)
                if n not in seen:
                    uniq.append(c)
                    seen.add(n)
            if len(uniq) >= MIN_UNIQUE:
                break
        if len(uniq) < MIN_UNIQUE:
            print(f"SHORT {row.task}/{row.cond}/k{row.k}: only {len(uniq)} unique")
            return None
        return {"task": row.task, "nominal": row.nominal, "cond": row.cond,
                "k": int(row.k), "base": row.base, "rephrases": uniq,
                "raw_responses": raws, "model": model, "n_calls": len(raws)}


async def run(args):
    conds = args.conds.split(",")
    bases = build_bases(conds)

    print(f"=== A37 PREFLIGHT: {len(bases)} sealed bases, conds={conds} ===")
    for r in bases.itertuples():
        print(f"  [{r.cond} k{r.k}] {r.task}: {r.base}")
    if os.environ.get("FINAL_EVAL") != "1":
        raise SystemExit("REFUSING: sealed generation requires FINAL_EVAL=1 (A37)")

    frames = {r.task: bytes(r.image_png) for r in
              pd.read_parquet(R / "results/sealed/sealed_frames.parquet").itertuples()}
    assert set(bases.task) <= set(frames), "missing sealed frame(s)"

    done = pd.DataFrame()
    if os.path.exists(args.out):
        done = pd.read_parquet(args.out)
        keys = set(zip(done["task"], done["cond"], done["k"]))
        bases = bases[~bases.apply(lambda r: (r["task"], r["cond"], r["k"]) in keys, axis=1)]
        print(f"resume: {len(done)} done, {len(bases)} remaining")

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    try:
        await client.aio.models.generate_content(model=args.model, contents="ok")
    except Exception as e:
        raise SystemExit(f"MODEL PREFLIGHT FAILED: {str(e)[:200]}")

    system_prompt = load_system_prompt()
    sem = asyncio.Semaphore(args.concurrency)
    jobs = [gen_one(client, sem, args.model, system_prompt, row, frames[row.task])
            for row in bases.itertuples()]
    results, n_ok = [], 0
    for i, coro in enumerate(tqdm(asyncio.as_completed(jobs), total=len(jobs), desc="a37-cover")):
        r = await coro
        if r is not None:
            results.append(r)
            n_ok += 1
        if n_ok and (n_ok % 10 == 0 or i == len(jobs) - 1):
            out = pd.concat([done, pd.DataFrame(results)], ignore_index=True) \
                if len(done) else pd.DataFrame(results)
            out.to_parquet(args.out, index=False)
    print(f"wrote {n_ok} new ({len(done) + n_ok} total) -> {args.out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conds", default="adv,orig")
    ap.add_argument("--out", default=str(R / "results/sealed/a37_cover_rephrases.parquet"))
    ap.add_argument("--model", default="gemini-3.1-pro-preview")
    ap.add_argument("--concurrency", type=int, default=4)
    args = ap.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
