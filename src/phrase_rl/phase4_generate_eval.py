"""Deployment-mode phrase generation for the SIMPLER eval: one phrase per task.

For each of the 4 SIMPLER tasks: trace + p_single prompt -> greedy single
phrase, from BASE Qwen and from a tuned LoRA adapter. Output: task,
arm(base|tuned|original), phrase.

Trace source: by default the cached CoVer-format Gemini response
(cover_gemini_tasks.parquet, nominal instructions). With --instructions +
--self-trace, the model rephrases ARBITRARY instructions (e.g. CoVer's ERT
red-team set) and generates its OWN trace first — fully self-contained
deployment, no Gemini in the loop.

Runs in .venv-gen on a GPU (pause training first — same card).
  # nominal instructions, cached Gemini traces:
  .venv-gen/bin/python -m phrase_rl.phase4_generate_eval \
    --tasks results/phrase_artifacts/cover_gemini_tasks.parquet \
    --adapter results/checkpoints/phase2_v1_degenerated/best_val \
    --out data/phrases_phase4_eval.parquet
  # CoVer ERT red-team instructions, self-generated traces:
  .venv-gen/bin/python -m phrase_rl.phase4_generate_eval \
    --tasks results/phrase_artifacts/cover_gemini_tasks.parquet \
    --instructions results/phrase_artifacts/cover_ert_instructions.json --self-trace \
    --adapter results/checkpoints/phase2/best_val \
    --out data/phrases_redteam_eval.parquet
"""

import argparse
import io
import json
import re

import pandas as pd
import torch
from PIL import Image

from phrase_rl.cover_prompt import (
    build_qwen_messages,
    build_single_phrase_prefix,
    extract_trace,
    parse_reworded,
)


def gen_one(model, processor, msgs, max_new_tokens=60, continue_final=True):
    from phrase_rl.phase2_train import apply_template
    inputs = apply_template(processor, msgs, continue_final_message=continue_final).to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens)
    return processor.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def first_line(text):
    return text.strip().split("\n")[0].strip()


# v6.3 tier-conditioning tags: the tag belongs in the MODEL'S INPUT slot only.
# If the model echoes it, it must never reach pi0's instruction.
TAG_RE = re.compile(r"^\s*(?:\[input:[^\]]*\]\s*)+")


def strip_tag(text):
    return TAG_RE.sub("", text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--tasks", required=True, help="cover_gemini_tasks.parquet (raw_response has the trace; also maps task->frame)")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--sample-k", type=int, default=0, help="if >0: k sampled phrases per task instead of greedy")
    ap.add_argument("--sample-temp", type=float, default=1.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--input-tag", default=None,
                    help="v6.3 checkpoints: prepend e.g. '[input: adversarially reworded]' to the instruction slot")
    ap.add_argument("--instructions", default=None,
                    help="json {task: instruction} overriding the nominal instructions (e.g. cover_ert_instructions.json)")
    ap.add_argument("--self-trace", action="store_true",
                    help="generate the trace with the model itself (full CoVer prompt -> extract_trace) instead of the cached Gemini trace; required when --instructions changes the instruction")
    ap.add_argument("--ctx", default="data/contexts_0c_tasks.parquet",
                    help="frames parquet (episode_index, task, image_png)")
    ap.add_argument("--assets", default=None,
                    help="parquet(task, ert_instruction, trace) from gemini_redteam_assets: overrides BOTH instruction and trace per task (ALGORITHM.md: Gemini trace at test time)")
    args = ap.parse_args()
    if args.instructions and not (args.self_trace or args.assets):
        raise SystemExit("--instructions changes the instruction; cached Gemini traces would mismatch — pass --self-trace or --assets")

    from transformers import AutoModelForImageTextToText, AutoProcessor
    processor = AutoProcessor.from_pretrained(args.model)
    base = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda")

    tasks = pd.read_parquet(args.tasks)
    ctx = pd.read_parquet(args.ctx)  # for the frames
    overrides = json.load(open(args.instructions)) if args.instructions else {}
    asset_traces = {}
    if args.assets:
        adf = pd.read_parquet(args.assets)
        overrides = {r.task: str(r.ert_instruction) for r in adf.itertuples()}
        asset_traces = {r.task: str(r.trace) for r in adf.itertuples()}
    rows = []

    def run_arm(model, arm):
        for r in tasks.itertuples():
            frame = ctx[ctx.episode_index == r.episode_index].iloc[0]
            task = frame["task"]
            instruction = overrides.get(task, r.instruction)
            if (args.instructions or args.assets) and task not in overrides:
                continue  # override mode: only eval the tasks named in the override set
            img = Image.open(io.BytesIO(frame["image_png"]))
            if task in asset_traces:
                trace = asset_traces[task]
            elif args.self_trace:
                # model writes its own analysis: full CoVer prompt, split off the trace
                raw = gen_one(model, processor,
                              build_qwen_messages(img, instruction, 1),
                              max_new_tokens=700, continue_final=False)
                trace = extract_trace(raw)
            else:
                trace = extract_trace(str(r.raw_response))
            # tuned-only: v6.3 checkpoints were trained with the regime tag in the
            # input slot; the frozen baseline never saw tags and stays untagged
            use_tag = args.input_tag and arm == "tuned"
            gen_in = f"{args.input_tag} {instruction}" if use_tag else instruction
            msgs = build_single_phrase_prefix(gen_in, img, trace=trace)
            if args.sample_k > 0:
                # sampled eval: k draws at the TRAINING temperature — measures the
                # distribution GRPO actually optimizes, and averages the per-task
                # phrase lottery (cluster-floor reduction). No dedupe: true samples.
                from phrase_rl.phase2_train import apply_template
                inputs = apply_template(processor, msgs, continue_final_message=True).to(model.device)
                with torch.no_grad():
                    out = model.generate(**inputs, do_sample=True,
                                         temperature=args.sample_temp,
                                         num_return_sequences=args.sample_k,
                                         max_new_tokens=60)
                plen = inputs["input_ids"].shape[1]
                for si in range(args.sample_k):
                    p = strip_tag(first_line(strip_tag(
                        processor.decode(out[si][plen:], skip_special_tokens=True))))
                    p = p or instruction
                    rows.append({"task": task, "arm": arm, "phrase": p,
                                 "instruction": instruction, "sample_idx": si})
                print(f"[{arm}] {task}: {args.sample_k} samples, e.g. {rows[-1]['phrase']!r}")
            else:
                phrase = strip_tag(first_line(strip_tag(gen_one(model, processor, msgs))))
                if not phrase:
                    phrase = instruction
                rows.append({"task": task, "arm": arm, "phrase": phrase, "instruction": instruction})
                print(f"[{arm}] {task}: {phrase!r}")

    run_arm(base, "base")
    from peft import PeftModel
    tuned = PeftModel.from_pretrained(base, args.adapter, is_trainable=False)
    run_arm(tuned, "tuned")

    for r in tasks.itertuples():
        frame = ctx[ctx.episode_index == r.episode_index].iloc[0]
        task = frame["task"]
        instruction = overrides.get(task, frame["instruction"])
        if (args.instructions or args.assets) and task not in overrides:
            continue
        # "original" arm = the user instruction executed directly (red-team direct in override mode)
        rows.append({"task": task, "arm": "original", "phrase": instruction, "instruction": instruction})

    pd.DataFrame(rows).to_parquet(args.out, index=False)
    print(f"wrote {len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
