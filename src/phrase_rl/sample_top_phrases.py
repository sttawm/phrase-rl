"""Sample each adapter's deployment distribution: N draws of the single-phrase
prompt per task (ERT input + Gemini trace, temp 0.8), tallied by frequency.
"Top phrases" = the modes of what the model would actually say.

  .venv-gen/bin/python -m phrase_rl.sample_top_phrases \
    --adapters rollout_s80=... l2_280=... flow_40=... base=NONE \
    --assets results/phrase_artifacts/redteam_eval_assets.parquet \
    --ctx data/contexts_0c_tasks.parquet --n 12 --out data/top_phrases.parquet
"""

import argparse
import io

import pandas as pd
import torch
from PIL import Image

from phrase_rl.cover_prompt import build_single_phrase_prefix


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--adapters", nargs="+", required=True, help="name=path pairs; path NONE = base model")
    ap.add_argument("--assets", required=True)
    ap.add_argument("--ctx", required=True)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from transformers import AutoModelForImageTextToText, AutoProcessor
    from phrase_rl.phase2_train import apply_template
    processor = AutoProcessor.from_pretrained(args.model)
    base = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda")

    assets = pd.read_parquet(args.assets)
    ctx = pd.read_parquet(args.ctx)
    frames = {r.task: r.image_png for r in ctx.itertuples()}

    rows = []
    from peft import PeftModel
    current_adapter = None
    model = base
    for spec in args.adapters:
        name, path = spec.split("=", 1)
        if path != "NONE":
            if current_adapter is not None:
                model = base  # PeftModel wraps base; re-wrap fresh
            model = PeftModel.from_pretrained(base, path, is_trainable=False)
            current_adapter = path
        else:
            model = base
        for a in assets.itertuples():
            if a.task not in frames:
                continue
            img = Image.open(io.BytesIO(frames[a.task]))
            msgs = build_single_phrase_prefix(str(a.ert_instruction), img, trace=str(a.trace))
            inputs = apply_template(processor, msgs, continue_final_message=True).to(model.device)
            torch.manual_seed(7)  # same draw sequence per arm — paired sampling
            for k in range(args.n):
                with torch.no_grad():
                    out = model.generate(**inputs, do_sample=True, temperature=0.8, max_new_tokens=48)
                ph = processor.decode(out[0][inputs["input_ids"].shape[1]:],
                                      skip_special_tokens=True).strip().split("\n")[0].strip()
                rows.append({"arm": name, "task": a.task, "sample": k, "phrase": ph})
            print(f"[{name}] {a.task}: {args.n} samples")
        if path != "NONE":
            model = model.unload() if hasattr(model, "unload") else base  # drop adapter for next
    pd.DataFrame(rows).to_parquet(args.out, index=False)
    print(f"wrote {len(rows)} samples -> {args.out}")


if __name__ == "__main__":
    main()
