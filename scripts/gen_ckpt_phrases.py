#!/usr/bin/env python3
"""Generate greedy + sampled deployment phrases from a v7 checkpoint adapter for
the 4 fixed probe contexts. Mirrors phase2_train.run_probes generation exactly
(same build_single_phrase_prefix, greedy = do_sample=False). Writes
data/rval_greedy.parquet + data/rval_sampled.parquet (the rval roller's schema).

  gen_ckpt_phrases.py <adapter_dir> <n_samples>
"""
import io
import os
import re
import sys

import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, "src")
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor

import os as _os

from phrase_rl.cover_prompt import build_single_phrase_prefix, build_single_phrase_prefix_bare

_fam = _os.environ.get("PROMPT_FAMILY", "cover")
if _fam == "bare":
    def build_single_phrase_prefix(src, img, trace=None):  # v10 prompt-B routing
        return build_single_phrase_prefix_bare(src, trace=trace)
elif _fam == "bplus":
    from phrase_rl.cover_prompt import build_single_phrase_prefix_bplus
    def build_single_phrase_prefix(src, img, trace=None):  # v11 prompt-B+ routing
        return build_single_phrase_prefix_bplus(src, trace=trace)
elif _fam == "bplusimg":
    from phrase_rl.cover_prompt import build_single_phrase_prefix_bplusimg
    def build_single_phrase_prefix(src, img, trace=None):  # v12 prompt-B+ with image
        return build_single_phrase_prefix_bplusimg(src, img, trace=trace)
from phrase_rl.phase2_train import apply_template

adapter_dir = sys.argv[1]
n_samples = int(sys.argv[2]) if len(sys.argv) > 2 else 2

proc = AutoProcessor.from_pretrained("Qwen/Qwen3.5-9B")
base = AutoModelForImageTextToText.from_pretrained(
    "Qwen/Qwen3.5-9B", dtype=torch.bfloat16, device_map="cuda")
if adapter_dir == "BASE":  # v8 step-0: frozen base Qwen, no adapter
    model = base.eval()
else:
    model = PeftModel.from_pretrained(base, adapter_dir, is_trainable=False).eval()

ctx = pd.read_parquet(os.environ.get("PROBE_CONTEXTS", "results/phrase_artifacts/contexts_0c_tasks.parquet"))
tr = pd.read_parquet(os.environ.get("PROBE_TRACES", "results/phrase_artifacts/traces_0c_tasks.parquet"))
tmap = {(r.episode_index, r.t): r.trace for r in tr.itertuples()}

# tier-conditioning tag: v6.3+ policies never saw a bare untagged instruction in
# training, so eval-time inputs must carry the condition's tag. Override with
# GEN_TAG="[input: adversarially reworded]" for adversarial-input (repair) evals;
# GEN_TAG="" for pre-tag (v5/v6.0) checkpoints.
TAG = os.environ.get("GEN_TAG", "[input: original wording]")
_TAG_RE = re.compile(r"^\s*(?:\[input:[^\]]*\]\s*)+")


def dec(o, plen):
    t = proc.decode(o[plen:], skip_special_tokens=True).strip().split("\n")[0].strip()
    return _TAG_RE.sub("", t).strip()


greedy, sampled = [], []
for r in ctx.itertuples():
    img = Image.open(io.BytesIO(r.image_png)).convert("RGB")
    src = f"{TAG} {r.instruction}" if TAG else r.instruction
    msgs = build_single_phrase_prefix(src, img, trace=tmap.get((r.episode_index, r.t)))
    inp = apply_template(proc, msgs, continue_final_message=True).to(model.device)
    plen = inp["input_ids"].shape[1]
    with torch.no_grad():
        g = model.generate(**inp, do_sample=False, max_new_tokens=48)
    gp = dec(g[0], plen)
    row = {"task": r.task, "arm": "v7_greedy", "phrase": gp, "instruction": r.instruction}
    if hasattr(r, "episode_id"):
        row["episode_id"] = int(r.episode_id)
    greedy.append(row)
    with torch.no_grad():
        ss = model.generate(**inp, do_sample=True, temperature=1.0,
                            num_return_sequences=max(4, n_samples), max_new_tokens=48)
    seen = []
    for o in ss:
        sp = dec(o, plen)
        if sp and sp not in seen:
            seen.append(sp)
    for sp in seen[:n_samples]:
        srow = {"task": r.task, "arm": "v7_sampled", "phrase": sp, "instruction": r.instruction}
        if hasattr(r, "episode_id"):
            srow["episode_id"] = int(r.episode_id)
        sampled.append(srow)

os.makedirs("data", exist_ok=True)
pd.DataFrame(greedy).to_parquet("data/rval_greedy.parquet", index=False)
pd.DataFrame(sampled).to_parquet("data/rval_sampled.parquet", index=False)
print(f"generated {len(greedy)} greedy, {len(sampled)} sampled from {adapter_dir}")
