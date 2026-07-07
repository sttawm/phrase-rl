"""Generate the trainable VLM's own rephrase candidates for each context.

Primary 0b arm: the sensitivity gate must be measured on the distribution the
reward will see during RL — the phrase model's own samples (base model, list
prompt). Resumable; writes every --save-every contexts.

Usage (on pod, .venv-gen):
  .venv-gen/bin/python -m phrase_rl.qwen_generate \
    --contexts data/contexts_val_0b.parquet --out data/qwen_rephrases_val_0b.parquet --n 32
"""

import argparse
import io
import os

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from phrase_rl.smoke_qwen import LIST_PROMPT, parse_list


def generate_for_row(model, processor, row, n, max_attempts=3):
    img = Image.open(io.BytesIO(row["image_png"]))
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": LIST_PROMPT.format(instruction=row["instruction"], n=n)},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
    ).to(model.device)
    for attempt in range(max_attempts):
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=800, do_sample=True, temperature=0.9 + 0.1 * attempt
            )
        text = processor.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        items = parse_list(text)
        if len(items) >= n:
            return items[:n]
    return items  # short list after retries; recorded as-is (format-failure signal)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--contexts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--save-every", type=int, default=25)
    args = ap.parse_args()

    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda"
    )

    df = pd.read_parquet(args.contexts)
    done_rows = []
    if os.path.exists(args.out):
        done = pd.read_parquet(args.out)
        done_rows = done.to_dict("records")
        done_keys = set(zip(done["episode_index"], done["t"]))
        df = df[~df.apply(lambda r: (r["episode_index"], r["t"]) in done_keys, axis=1)]
        print(f"resume: {len(done_rows)} done, {len(df)} remaining")

    for i, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc="qwen generate")):
        rephrases = generate_for_row(model, processor, row, args.n)
        done_rows.append(
            {
                "episode_index": row["episode_index"],
                "t": row["t"],
                "instruction": row["instruction"],
                "rephrases": rephrases,
                "n_generated": len(rephrases),
            }
        )
        if (i + 1) % args.save_every == 0:
            pd.DataFrame(done_rows).to_parquet(args.out, index=False)

    pd.DataFrame(done_rows).to_parquet(args.out, index=False)
    short = sum(1 for r in done_rows if r["n_generated"] < args.n)
    print(f"wrote {len(done_rows)} contexts to {args.out} ({short} short lists)")


if __name__ == "__main__":
    main()
