#!/usr/bin/env python3
"""How much candidate diversity does temperature buy the FROZEN rephraser?

v11 groups collapsed to a median 6 unique candidates of 16 (mean duplicate
fraction 0.579). Before spending a v12 run on it, measure whether the collapse
is a property of the POLICY (something training did) or of the DECODE
TEMPERATURE (something a flag fixes).

Samples the frozen base model on the same prompts the v11 trainer builds, at a
ladder of temperatures, and reports unique-of-K.

  PYTHONPATH=src .venv-gen/bin/python scripts/temp_sweep_qwen.py \
      --n-contexts 5 --k 16 --temps 0.7 1.0 1.2 1.5

IMPORTANT: loads with AutoModelForImageTextToText. Two earlier attempts used
AutoModelForCausalLM, which silently loads a DIFFERENT module tree -- the
adapter paths are `...language_model...` -- so both runs measured base Qwen
twice and the "sweep" compared a model with itself.
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

ap = argparse.ArgumentParser()
ap.add_argument("--naturals", default="results/phrase_artifacts/naturals_v11_train.parquet")
ap.add_argument("--n-contexts", type=int, default=5)
ap.add_argument("--k", type=int, default=16)
ap.add_argument("--temps", type=float, nargs="+", default=[0.7, 1.0, 1.2, 1.5])
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--adapter", default=None,
                help="optional trained adapter to compare against the frozen base")
ap.add_argument("--out", default="results/analysis/temp_sweep.json")
ap.add_argument("--max-new-tokens", type=int, default=48)
args = ap.parse_args()

PROMPT = """You are helping reword instructions for a tabletop robot task.

Scene: {trace}

Original instruction: {src}

Write one clear rewording of the instruction. Output only the rewording."""

df = pd.read_parquet(REPO / args.naturals)
df = df.sample(n=args.n_contexts, random_state=args.seed).reset_index(drop=True)
print(f"[sweep] {len(df)} contexts x K={args.k} x {len(args.temps)} temps "
      f"= {len(df)*args.k*len(args.temps)} generations", flush=True)

from transformers import AutoModelForImageTextToText, AutoProcessor

proc = AutoProcessor.from_pretrained("Qwen/Qwen3.5-9B")
model = AutoModelForImageTextToText.from_pretrained(
    "Qwen/Qwen3.5-9B", dtype=torch.bfloat16, device_map="cuda").eval()
tag = "frozen"
if args.adapter:
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, args.adapter).eval()
    tag = Path(args.adapter).name
print(f"[sweep] model loaded ({tag}) via AutoModelForImageTextToText", flush=True)

rows = []
for ci, r in enumerate(df.itertuples()):
    msgs = [{"role": "user", "content": [
        {"type": "text", "text": PROMPT.format(trace=r.trace, src=r.instruction)}]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                    enable_thinking=False)
    for T in args.temps:
        torch.manual_seed(args.seed * 1000 + ci)
        inp = proc(text=[text] * args.k, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            g = model.generate(**inp, do_sample=True, temperature=float(T),
                               top_p=0.95, max_new_tokens=args.max_new_tokens)
        outs = [proc.decode(o[inp.input_ids.shape[1]:], skip_special_tokens=True)
                .strip().split("\n")[0].strip() for o in g]
        uniq = len(set(o.lower().rstrip(".") for o in outs if o))
        rows.append({"context": ci, "instruction": r.instruction, "temp": T,
                     "k": args.k, "unique": uniq, "dup_frac": 1 - uniq / args.k,
                     "candidates": outs})
        print(f"  ctx {ci} T={T:<4} unique {uniq:>2}/{args.k}  "
              f"dup_frac {1-uniq/args.k:.3f}  e.g. {outs[0][:60]!r}", flush=True)

out = pd.DataFrame(rows)
summary = (out.groupby("temp")
             .agg(mean_unique=("unique", "mean"), mean_dup=("dup_frac", "mean"))
             .reset_index())
print("\n[sweep] SUMMARY  (v11 training ran at mean dup_frac 0.579, median 6/16 unique)")
print(summary.to_string(index=False))

p = REPO / args.out
p.parent.mkdir(parents=True, exist_ok=True)
json.dump({"model": tag, "k": args.k, "n_contexts": len(df),
           "summary": summary.to_dict("records"), "rows": rows},
          open(p, "w"), indent=1)
print(f"\n[sweep] wrote {p}")
