"""Qwen self-traces with a MINIMAL prompt (probe lesson: the full CoVer
template collapses 9B into repetition loops / placeholder echoes, while the
reasoning content itself is right 2/2 on OOV receptacles).

Two sections, prefilled, no extraction step — the whole completion is the
trace. Writes an assets parquet (task, ert_instruction, trace) that drops
into phase4_generate_eval --assets, so the self-trace arms reuse the exact
eval plumbing of the Gemini-trace arms.

  .venv-gen/bin/python -m phrase_rl.sft17k_selftrace_gen \
      --assets data/val_screen_assets.parquet \
      --frames data/val_screen_frames.parquet \
      --out data/val_screen_assets_selftrace.parquet
"""
import argparse
import io

import pandas as pd
import torch
from PIL import Image

from phrase_rl.phase2_train import apply_template

PROMPT = ("Look at the image. First describe the scene in one or two sentences. "
          "Then name the concrete objects this instruction refers to, mapping each "
          "described object to its plain name.\nInstruction: {instr}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--assets", default="data/val_screen_assets.parquet")
    ap.add_argument("--frames", default="data/val_screen_frames.parquet")
    ap.add_argument("--out", default="data/val_screen_assets_selftrace.parquet")
    args = ap.parse_args()

    assets = pd.read_parquet(args.assets)
    frames = pd.read_parquet(args.frames)

    from transformers import AutoModelForImageTextToText, AutoProcessor
    proc = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.bfloat16,
                                                        device_map="cuda")
    model.eval()

    rows = []
    for a in assets.itertuples():
        fr = frames[frames.task == a.task].iloc[0]
        img = Image.open(io.BytesIO(fr["image_png"]))
        msgs = [{"role": "user", "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": PROMPT.format(instr=str(a.ert_instruction))}]},
                {"role": "assistant", "content": [{"type": "text", "text": "Scene:"}]}]
        enc = apply_template(proc, msgs, continue_final_message=True).to(model.device)
        with torch.no_grad():
            g = model.generate(**enc, do_sample=False, max_new_tokens=220)
        trace = "Scene: " + proc.decode(g[0][enc["input_ids"].shape[1]:],
                                        skip_special_tokens=True).strip()
        rows.append({"task": a.task, "ert_instruction": str(a.ert_instruction), "trace": trace})
        print(f"[selftrace] {a.task}:\n{trace}\n", flush=True)

    pd.DataFrame(rows).to_parquet(args.out, index=False)
    print(f"wrote {len(rows)} rows -> {args.out}\nSELFTRACE-GEN-DONE", flush=True)


if __name__ == "__main__":
    main()
