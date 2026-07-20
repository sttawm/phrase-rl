"""Local hostile-variant generation for SFT-17k: frozen Qwen3.5-9B produces 3
ERT-style variants per unique train-split Bridge instruction. Text-only (no
scene trace — deviation from the Gemini batches, documented). Eval hostile
sets remain Gemini/CoVer-authored => held-out attack distribution.

v2 fix: enable_thinking=False + prefilled assistant turn (`ERT: "`) — the v1
run used add_generation_prompt with thinking enabled, so all 80 new tokens
went into the reasoning block and 0/48 variants survived the filter.
SMOKE=1 env: 8 instructions, prints every raw extraction for eyeballing.
"""
import json
import os
import re
import unicodedata

import pandas as pd
import torch

def nk(s):
    return " ".join(unicodedata.normalize("NFKC", str(s)).casefold().split())

# ERT must reword, not change the goal: a variant that flips a spatial relation
# (smoke test caught "right of the pot" -> "slightly to the left of the pot")
# is a corrupted training pair, not an adversarial rewording. Reject when the
# variant introduces the antonym of a relation the GT states (and drops the
# original). top/upper and bottom/lower are treated as synonym groups.
SPAT = [({"left"}, {"right"}), ({"top", "upper", "above", "over"}, {"bottom", "lower", "below", "under", "beneath", "underneath"}),
        ({"front"}, {"back", "behind", "rear"}), ({"inside", "into", "in"}, {"outside"})]

def words(s):
    return set(re.findall(r"[a-z]+", nk(s)))

def spatial_flip(gt, v):
    gw, vw = words(gt), words(v)
    for a, b in SPAT:
        for x, y in ((a, b), (b, a)):
            if (gw & x) and not (gw & y) and (vw & y) and not (vw & x):
                return True
    return False

cover = json.load(open("results/phrase_artifacts/cover_release_ert_rephrases.json"))["instructions"]
SHOTS_LONG = "\n".join(f'nominal: "{nom}" -> ERT: "{rec["ert_rephrases"][0]}"'
                       for nom, rec in list(cover.items())[:4])
short_shots = []
for nominal, rec in cover.items():
    for e in [rec["original"]] + rec["ert_rephrases"]:
        if 40 <= len(e) <= 70:
            short_shots.append(f'nominal: "{nominal}" -> ERT: "{e}"')
            break
SHOTS_SHORT = "\n".join(short_shots[:5])
STYLES = {
    0: ("rename objects with plausible synonyms or category words and inject distracting "
        "but scene-plausible attributes", SHOTS_LONG, (30, 220)),
    1: ("add stance/manner constraints and use indirect referring expressions for objects "
        "and locations", SHOTS_LONG, (30, 220)),
    2: ("CONCISE - a single plain sentence of 40 to 70 characters, opening with an ordinary "
        "verb (set/place/balance/arrange/put), renaming the object or receptacle with a "
        "plausible synonym and/or one wrong-but-plausible attribute", SHOTS_SHORT, (20, 90)),
}

from huggingface_hub import hf_hub_download
mp = hf_hub_download("IPEC-COMMUNITY/bridge_orig_lerobot", "meta/episodes.jsonl", repo_type="dataset")
eps = []
for l in open(mp):
    r = json.loads(l)
    task = r.get("tasks", [""])[0] if isinstance(r.get("tasks"), list) else r.get("tasks", "")
    eps.append(str(task).strip())
n = len(eps)
lo = int(0.10 * n)
excl = set(nk(t) for t in eps[:lo])
seen, uniq = set(), []
for t in eps[lo:]:
    k = nk(t)
    if k and k not in excl and k not in seen and 5 <= len(t) <= 120:
        seen.add(k); uniq.append(t)
print(f"unique clean train instructions: {len(uniq)}", flush=True)

SMOKE = os.environ.get("SMOKE") == "1"
if SMOKE:
    uniq = uniq[:8]

from transformers import AutoModelForImageTextToText, AutoProcessor
tok = AutoProcessor.from_pretrained("Qwen/Qwen3.5-9B")
model = AutoModelForImageTextToText.from_pretrained("Qwen/Qwen3.5-9B", dtype=torch.bfloat16, device_map="cuda")
model.eval()

def build_prompt(instr, style_desc, shots):
    txt = (f"You are building an Extreme Rephrasing Test (ERT) for robot instructions: adversarial "
           f"rewordings that keep the task goal but are hostile to a language-conditioned robot policy. "
           f"THIS style: {style_desc}. Examples of the genre:\n{shots}\n\n"
           f"Produce exactly ONE complete ERT rephrasing of this instruction "
           f"(just the instruction, no explanation):\n{instr}")
    msgs = [{"role": "user", "content": [{"type": "text", "text": txt}]},
            {"role": "assistant", "content": [{"type": "text", "text": 'ERT: "'}]}]
    # continue_final_message + enable_thinking=False: same no-reasoning prefill
    # discipline as phase2_train.apply_template (v1 lesson: thinking mode ate
    # the whole token budget and yielded 0 usable variants)
    return tok.apply_chat_template(msgs, tokenize=False, continue_final_message=True,
                                   enable_thinking=False)

BS = 8 if SMOKE else 96
rows = []
for style_id, (desc, shots, (lmin, lmax)) in STYLES.items():
    prompts = [build_prompt(u, desc, shots) for u in uniq]
    kept = 0
    for i in range(0, len(prompts), BS):
        batch = prompts[i:i + BS]
        enc = tok(text=batch, return_tensors="pt", padding=True, padding_side="left").to("cuda")
        with torch.no_grad():
            out = model.generate(**enc, do_sample=True, temperature=1.0, top_p=0.95,
                                 max_new_tokens=80, pad_token_id=tok.tokenizer.eos_token_id)
        for j in range(len(batch)):
            txt = tok.decode(out[j][enc["input_ids"].shape[1]:], skip_special_tokens=True)
            # prefill opened a quote: the phrase is everything up to the closing quote
            v = txt.split('"')[0].strip().split("\n")[0].strip()
            gt = uniq[i + j]
            if SMOKE:
                print(f"  s{style_id} RAW: {txt[:120]!r}\n  s{style_id} GT: {gt!r} -> V: {v!r}", flush=True)
            ok = (lmin <= len(v) <= lmax and nk(v) != nk(gt)
                  and "->" not in v and "nominal" not in v.lower()
                  and not v.lower().startswith(("i ", "sorry", "sure", "ert"))
                  and not spatial_flip(gt, v))
            if ok:
                rows.append({"gt": gt, "variant": v, "style": style_id})
                kept += 1
        if (i // BS) % 20 == 0:
            print(f"style {style_id}: {i + len(batch)}/{len(prompts)} kept={kept}", flush=True)
            if rows and not SMOKE:
                pd.DataFrame(rows).to_parquet("data/sft17k_pairs_partial.parquet", index=False)
    print(f"style {style_id} DONE: kept {kept}/{len(prompts)}", flush=True)

df = pd.DataFrame(rows)
if not SMOKE:
    df.to_parquet("data/sft17k_pairs.parquet", index=False)
ln = df.variant.str.len()
print(f"TOTAL PAIRS: {len(df)} | per style: {df['style'].value_counts().to_dict()} | "
      f"len median {ln.median():.0f} IQR {ln.quantile(.25):.0f}-{ln.quantile(.75):.0f}", flush=True)
print("VARIANTS-DONE", flush=True)
