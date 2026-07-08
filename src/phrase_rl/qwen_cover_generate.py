"""Base-Qwen candidate generation under CoVer's verbatim template — step-0 A/B arms.

--arm inline: c = (image, instruction); the model writes its own reasoning inline
              (CoVer template elicits it) then the list. The prompt-parity primary.
--arm trace:  same, but the cached Gemini trace is prepended to the user turn as
              "Scene analysis (from a vision-language model): ..." — tests whether
              frontier reasoning beats the 9B's own (user hypothesis, 2026-07-07).

Runs in .venv-gen. Resumable.
  .venv-gen/bin/python -m phrase_rl.qwen_cover_generate --contexts data/contexts_val_0b.parquet \
    --arm inline --out data/cover_qwen_inline_val.parquet --n 32
  .venv-gen/bin/python -m phrase_rl.qwen_cover_generate --contexts data/contexts_val_0b.parquet \
    --arm trace --traces results/phrase_artifacts/rephrases_val_0b.parquet \
    --out data/cover_qwen_trace_val.parquet --n 32
"""

import argparse
import io
import os

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from phrase_rl.cover_prompt import (build_user_prompt, build_user_prompt_with_trace,
                                    load_system_prompt, parse_reworded)


def build_messages(img, instruction, n, trace=None):
    # two-prompt flow: trace (prompt 1, cached frontier call) -> phrases (prompt 2)
    if trace:
        user_text = build_user_prompt_with_trace(instruction, trace, n)
    else:
        user_text = build_user_prompt(instruction, n)
    return [
        {"role": "system", "content": [{"type": "text", "text": load_system_prompt()}]},
        {"role": "user", "content": [{"type": "image", "image": img},
                                     {"type": "text", "text": user_text}]},
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--contexts", required=True)
    ap.add_argument("--arm", choices=["inline", "trace"], required=True)
    ap.add_argument("--traces", default=None, help="parquet with episode_index,t,trace (required for --arm trace)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0, help="use only first N contexts (0=all)")
    ap.add_argument("--save-every", type=int, default=10)
    args = ap.parse_args()

    traces = {}
    if args.arm == "trace":
        tdf = pd.read_parquet(args.traces)
        traces = {(r.episode_index, r.t): str(r.trace) for r in tdf.itertuples()}

    from transformers import AutoModelForImageTextToText, AutoProcessor
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda")

    df = pd.read_parquet(args.contexts)
    if args.limit:
        df = df.head(args.limit)
    rows = []
    if os.path.exists(args.out):
        prev = pd.read_parquet(args.out)
        rows = prev.to_dict("records")
        keys = set(zip(prev["episode_index"], prev["t"]))
        df = df[~df.apply(lambda r: (r["episode_index"], r["t"]) in keys, axis=1)]
        print(f"resume: {len(rows)} done, {len(df)} remaining")

    for i, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc=f"qwen-cover-{args.arm}")):
        trace = traces.get((row["episode_index"], row["t"])) if args.arm == "trace" else None
        if args.arm == "trace" and trace is None:
            continue  # context without a cached trace can't be in this arm
        img = Image.open(io.BytesIO(row["image_png"]))
        messages = build_messages(img, row["instruction"], args.n, trace)
        inputs = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=True,
            return_tensors="pt", enable_thinking=False,
        ).to(model.device)
        phrases = []
        for attempt in range(3):
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=1400, do_sample=True,
                                     temperature=0.8 + 0.1 * attempt)
            text = processor.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            phrases = parse_reworded(text)
            if len(phrases) >= max(6, args.n // 2):
                break
        rows.append({"episode_index": row["episode_index"], "t": row["t"],
                     "instruction": row["instruction"], "rephrases": phrases[:args.n],
                     "n_generated": len(phrases)})
        if (i + 1) % args.save_every == 0:
            pd.DataFrame(rows).to_parquet(args.out, index=False)

    pd.DataFrame(rows).to_parquet(args.out, index=False)
    short = sum(1 for r in rows if r["n_generated"] < args.n)
    print(f"wrote {len(rows)} contexts ({short} short lists) -> {args.out}")


if __name__ == "__main__":
    main()
