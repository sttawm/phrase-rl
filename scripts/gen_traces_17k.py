"""v2 training traces: Qwen reads (initial-scene frame + ONE sampled hostile
variant per GT) and emits a minimal 2-section trace — same prompt family as
sft17k_selftrace_gen, so train and deploy conditioning match.

One trace per GT (not per variant): the referent PLAIN NAMES are identical
across a GT's variants; description left-hand sides may mismatch other
variants, which is accepted (mild regularization) vs 4x generation cost.
Resumable shards. Requires data/frames17k/ (extract_frames_17k) and the
hostile pairs parquet.

  .venv-gen/bin/python scripts/gen_traces_17k.py        # ~5-8h A6000
"""
import glob
import io
import os
import sys

import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, "src")
from phrase_rl.phase2_train import apply_template
from phrase_rl.sft17k_selftrace_gen import PROMPT

OUT_DIR = "data/traces17k"
BS = 16
SHARD = 500


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    frames = pd.concat([pd.read_parquet(f) for f in
                        sorted(glob.glob("data/frames17k/*.parquet"))], ignore_index=True)
    frames = frames.drop_duplicates("gt")
    pairs = pd.read_parquet("results/phrase_artifacts/sft17k_pairs.parquet")
    pick = (pairs.sample(frac=1.0, random_state=0)
                 .drop_duplicates("gt")[["gt", "variant"]])
    todo = frames.merge(pick, on="gt", how="inner").reset_index(drop=True)
    print(f"tracing {len(todo)} gts ({len(frames)} framed, {pick['gt'].nunique()} varianted)",
          flush=True)

    from transformers import AutoModelForImageTextToText, AutoProcessor
    proc = AutoProcessor.from_pretrained("Qwen/Qwen3.5-9B")
    model = AutoModelForImageTextToText.from_pretrained("Qwen/Qwen3.5-9B",
                                                        dtype=torch.bfloat16, device_map="cuda")
    model.eval()

    n_shards = (len(todo) + SHARD - 1) // SHARD
    for si in range(n_shards):
        path = f"{OUT_DIR}/shard_{si:03d}.parquet"
        if os.path.exists(path):
            continue
        chunk = todo.iloc[si * SHARD:(si + 1) * SHARD]
        rows = []
        for bs in range(0, len(chunk), BS):
            batch = chunk.iloc[bs:bs + BS]
            msgs_list = []
            for r in batch.itertuples():
                img = Image.open(io.BytesIO(r.image_png))
                msgs_list.append(
                    [{"role": "user", "content": [
                        {"type": "image", "image": img},
                        {"type": "text", "text": PROMPT.format(instr=r.variant)}]},
                     {"role": "assistant", "content": [{"type": "text", "text": "Scene:"}]}])
            texts = [proc.apply_chat_template(m, tokenize=False, continue_final_message=True,
                                              enable_thinking=False) for m in msgs_list]
            images = [m[0]["content"][0]["image"] for m in msgs_list]
            enc = proc(text=texts, images=images, return_tensors="pt", padding=True,
                       padding_side="left").to(model.device)
            with torch.no_grad():
                out = model.generate(**enc, do_sample=False, max_new_tokens=220,
                                     pad_token_id=proc.tokenizer.eos_token_id)
            plen = enc["input_ids"].shape[1]
            for j, r in enumerate(batch.itertuples()):
                trace = "Scene: " + proc.decode(out[j][plen:], skip_special_tokens=True).strip()
                rows.append({"gt": r.gt, "variant_used": r.variant, "trace": trace})
        pd.DataFrame(rows).to_parquet(path, index=False)
        if si % 4 == 0 or si == n_shards - 1:
            print(f"trace shard {si + 1}/{n_shards} | e.g. {rows[0]['trace'][:110]!r}", flush=True)
    print("TRACES-DONE", flush=True)


if __name__ == "__main__":
    main()
