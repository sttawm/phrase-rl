"""Phase 0a smoke test for the trainable phrase VLM (Qwen3.5-9B).

Checks: model + processor load; image input works; the list prompt yields a
parseable numbered list of rephrases. Runs in .venv-gen (transformers >= 5).

Usage (on pod):
  .venv-gen/bin/python -m phrase_rl.smoke_qwen --contexts data/contexts_val_0b.parquet
"""

import argparse
import io
import re

import pandas as pd
import torch
from PIL import Image

LIST_PROMPT = """\
A robot arm was given this instruction: "{instruction}"
The image is the robot's current camera view.

Write exactly {n} different rephrasings of the instruction. Preserve the exact \
task: same object(s), same target location, same action. Vary verbs, word order, \
and specificity. Short imperative sentences only.

Output a numbered list, one rephrase per line, nothing else."""


def parse_list(text: str) -> list[str]:
    # defensive: never parse items out of a thinking block
    if "</think>" in text:
        text = text.split("</think>")[-1]
    items = re.findall(r"^\s*\d+[.)]\s*(.+)$", text, flags=re.M)
    return [s.strip() for s in items]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--contexts", required=True)
    ap.add_argument("--n", type=int, default=8)
    args = ap.parse_args()

    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda"
    )
    print(f"[1] loaded {args.model} ({model.num_parameters() / 1e9:.1f}B params)")

    row = pd.read_parquet(args.contexts).iloc[0]
    img = Image.open(io.BytesIO(row["image_png"]))
    print(f"[2] context: ep={row['episode_index']} instr={row['instruction']!r}")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": LIST_PROMPT.format(instruction=row["instruction"], n=args.n)},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_dict=True,
        return_tensors="pt", enable_thinking=False,
    ).to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=512, do_sample=True, temperature=0.9)
    text = processor.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    rephrases = parse_list(text)
    print(f"[3] generated {len(rephrases)} rephrases (asked {args.n}):")
    for r in rephrases:
        print("   -", r)
    assert len(rephrases) >= args.n - 1, f"list parse failed; raw output:\n{text}"
    print("[4] list prompt OK")


if __name__ == "__main__":
    main()
