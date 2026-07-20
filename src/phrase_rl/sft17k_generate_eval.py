"""Eval-time phrase generation for the SFT-17k reconstructor.

Text-only deployment: hostile instruction -> build_msgs (the exact training
conditioning from sft17k_train: WRAPPER user turn + "Canonical:" prefill,
continue_final_message, enable_thinking=False) -> reconstructed phrase.
No image, no trace, no tier tags. Output schema matches phase4_generate_eval
({task, arm, phrase, instruction[, sample_idx]}) so the standard rollout
pipeline consumes it unchanged.

  .venv-gen/bin/python -m phrase_rl.sft17k_generate_eval \
      --adapter results/checkpoints/sft17k/final \
      --assets data/val_screen_assets.parquet \
      --out data/phrases_sft17k_ert.parquet [--sample-k 8 --sample-temp 1.0]
"""
import argparse
import json

import pandas as pd
import torch

from phrase_rl.phase2_train import apply_template
from phrase_rl.sft17k_train import build_msgs


def clean(text, fallback):
    p = text.strip().split("\n")[0].strip().strip('"').strip()
    if p.lower().startswith("canonical:"):
        p = p[len("canonical:"):].strip()
    return p or fallback


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--assets", default=None,
                    help="parquet(task, ert_instruction): the hostile inputs to repair")
    ap.add_argument("--instructions", default=None, help="json {task: instruction} alternative")
    ap.add_argument("--arm", default="sft")
    ap.add_argument("--include-frozen", action="store_true",
                    help="also run the BASE model under the identical wrapper (arm "
                         "'frozen_bare') — the no-reasoning frozen comparator: same "
                         "conditioning, weights-only difference")
    ap.add_argument("--sample-k", type=int, default=0)
    ap.add_argument("--sample-temp", type=float, default=1.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if not (args.assets or args.instructions):
        raise SystemExit("need --assets or --instructions")

    if args.assets:
        adf = pd.read_parquet(args.assets)
        inputs = {r.task: str(r.ert_instruction) for r in adf.itertuples()}
    else:
        inputs = json.load(open(args.instructions))

    from transformers import AutoModelForImageTextToText, AutoProcessor
    from peft import PeftModel
    processor = AutoProcessor.from_pretrained(args.model)
    base = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.bfloat16,
                                                       device_map="cuda")

    rows = []

    def run_arm(model, arm):
        model.eval()
        for task, instr in inputs.items():
            enc = apply_template(processor, build_msgs(instr),
                                 continue_final_message=True).to(model.device)
            plen = enc["input_ids"].shape[1]
            with torch.no_grad():
                if args.sample_k > 0:
                    out = model.generate(**enc, do_sample=True, temperature=args.sample_temp,
                                         num_return_sequences=args.sample_k, max_new_tokens=60)
                    for si in range(args.sample_k):
                        p = clean(processor.decode(out[si][plen:], skip_special_tokens=True), instr)
                        rows.append({"task": task, "arm": arm, "phrase": p,
                                     "instruction": instr, "sample_idx": si})
                    print(f"[{arm}] {task}: {args.sample_k} samples, e.g. {rows[-1]['phrase']!r}")
                else:
                    out = model.generate(**enc, do_sample=False, max_new_tokens=60)
                    p = clean(processor.decode(out[0][plen:], skip_special_tokens=True), instr)
                    rows.append({"task": task, "arm": arm, "phrase": p, "instruction": instr})
                    print(f"[{arm}] {task}: {instr!r} -> {p!r}")

    if args.include_frozen:
        run_arm(base, "frozen_bare")  # BEFORE adapter injection mutates base
    tuned = PeftModel.from_pretrained(base, args.adapter, is_trainable=False)
    run_arm(tuned, args.arm)

    pd.DataFrame(rows).to_parquet(args.out, index=False)
    print(f"wrote {len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
