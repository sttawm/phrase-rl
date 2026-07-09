"""Deployment-mode phrase generation for the SIMPLER eval: one phrase per task.

For each of the 4 SIMPLER tasks: trace (extracted from the cached CoVer-format
Gemini response) + p_single prompt -> greedy single phrase, from BASE Qwen and
from a tuned LoRA adapter. Output: task, arm(base|tuned|original), phrase.

Runs in .venv-gen on a GPU (pause training first — same card).
  .venv-gen/bin/python -m phrase_rl.phase4_generate_eval \
    --tasks results/phrase_artifacts/cover_gemini_tasks.parquet \
    --adapter results/checkpoints/phase2_v1_degenerated/best_val \
    --out data/phrases_phase4_eval.parquet
"""

import argparse
import io

import pandas as pd
import torch
from PIL import Image

from phrase_rl.cover_prompt import build_single_phrase_prefix, extract_trace, parse_reworded


def gen_one(model, processor, msgs):
    from phrase_rl.phase2_train import apply_template
    inputs = apply_template(processor, msgs, continue_final_message=True).to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, do_sample=False, max_new_tokens=60)
    text = processor.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return text.strip().split("\n")[0].strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--tasks", required=True, help="cover_gemini_tasks.parquet (raw_response has the trace)")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from transformers import AutoModelForImageTextToText, AutoProcessor
    processor = AutoProcessor.from_pretrained(args.model)
    base = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda")

    tasks = pd.read_parquet(args.tasks)
    ctx = pd.read_parquet("data/contexts_0c_tasks.parquet")  # for the frames
    rows = []

    def run_arm(model, arm):
        for r in tasks.itertuples():
            trace = extract_trace(str(r.raw_response))
            frame = ctx[ctx.episode_index == r.episode_index].iloc[0]
            img = Image.open(io.BytesIO(frame["image_png"]))
            msgs = build_single_phrase_prefix(r.instruction, img, trace=trace)
            phrase = gen_one(model, processor, msgs)
            rows.append({"task": frame["task"], "arm": arm, "phrase": phrase})
            print(f"[{arm}] {frame['task']}: {phrase!r}")

    run_arm(base, "base")
    from peft import PeftModel
    tuned = PeftModel.from_pretrained(base, args.adapter, is_trainable=False)
    run_arm(tuned, "tuned")

    for r in tasks.itertuples():
        frame = ctx[ctx.episode_index == r.episode_index].iloc[0]
        rows.append({"task": frame["task"], "arm": "original", "phrase": frame["instruction"]})

    pd.DataFrame(rows).to_parquet(args.out, index=False)
    print(f"wrote {len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
