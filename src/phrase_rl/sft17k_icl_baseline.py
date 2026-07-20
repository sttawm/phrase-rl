"""Pure in-context canonicalization baseline (NO training): frozen Qwen3.5-9B
recovers the canonical Bridge instruction from a hostile input, given

  1. retrieved nearest canonical instructions from the train-clean inventory
     (retrieval also matches THROUGH the OXE paraphrase clusters: a hostile
     input often lands nearer a paraphrase than its GT — score every
     paraphrase, credit its GT),
  2. demonstration mappings reworded->canonical (benign demos from the OXE
     dictionary; hostile-style demos from the SFT pairs parquet when present),
  3. optionally the Gemini scene trace (assets column), per the user sketch.

This is the control for SFT-17k: if frozen weights + inventory access match
the fine-tune, the value was the KNOWLEDGE, not the training. It also cannot
memorize pairs, so it lower-bounds the no-overfit regime.
Inventory/demos are train-split only — no val/test-split instruction keys.

  .venv-gen/bin/python -m phrase_rl.sft17k_icl_baseline \
      --assets data/val_screen_assets.parquet \
      --out data/phrases_icl_ert.parquet [--sample-k 8] [--no-trace]
"""
import argparse
import math
import re
import unicodedata
from collections import Counter

import pandas as pd
import torch

from phrase_rl.phase2_train import apply_template


def nk(s):
    return " ".join(unicodedata.normalize("NFKC", str(s)).casefold().split())


def grams(s, n=3):
    s = f"  {nk(s)} "
    return Counter(s[i:i + n] for i in range(len(s) - n + 1))


def cos(a, b):
    num = sum(v * b[k] for k, v in a.items() if k in b)
    if not num:
        return 0.0
    return num / math.sqrt(sum(v * v for v in a.values()) * sum(v * v for v in b.values()))


def retrieve(query, entry_grams, topn=8):
    """entry_grams: list of (Counter, gt). Returns topn distinct gts by best-member score."""
    q = grams(query)
    best = {}
    for g, gt in entry_grams:
        s = cos(q, g)
        if s > best.get(gt, 0.0):
            best[gt] = s
    return [gt for gt, _ in sorted(best.items(), key=lambda kv: -kv[1])[:topn]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--assets", required=True, help="parquet(task, ert_instruction[, trace])")
    ap.add_argument("--inventory", default="results/phrase_artifacts/bridge_train_uniques.parquet")
    ap.add_argument("--paraphrases", default="results/phrase_artifacts/oxe_paraphrases_bridge.parquet")
    ap.add_argument("--pairs", default="results/phrase_artifacts/sft17k_pairs.parquet",
                    help="hostile demo source; skipped gracefully if absent")
    ap.add_argument("--topn", type=int, default=8)
    ap.add_argument("--arm", default="icl")
    ap.add_argument("--sample-k", type=int, default=0)
    ap.add_argument("--sample-temp", type=float, default=1.0)
    ap.add_argument("--no-trace", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    inv = pd.read_parquet(args.inventory)["gt"].tolist()
    inv_keys = set(nk(g) for g in inv)
    entries = [(g, g) for g in inv]
    try:
        par = pd.read_parquet(args.paraphrases)
        par = par[par["gt"].map(lambda g: nk(g) in inv_keys)]
        entries += list(zip(par["paraphrase"], par["gt"]))
        print(f"retrieval pool: {len(inv)} gts + {len(par)} paraphrases")
    except FileNotFoundError:
        print(f"retrieval pool: {len(inv)} gts (no paraphrase file)")
    entry_grams = [(grams(text), gt) for text, gt in entries]

    # fixed demos (train-split content only), deterministic
    demos = []
    try:
        pares = pd.read_parquet(args.paraphrases)
        for i in (7, 4001):
            r = pares.iloc[i]
            demos.append((str(r["paraphrase"]), str(r["gt"])))
    except FileNotFoundError:
        pass
    try:
        pairs = pd.read_parquet(args.pairs)
        for st in sorted(pairs["style"].unique()):
            r = pairs[pairs["style"] == st].iloc[11]
            demos.append((str(r["variant"]), str(r["gt"])))
    except FileNotFoundError:
        print("(no sft pairs parquet yet — benign demos only)")
    demo_txt = "\n".join(f'reworded: "{v}" -> canonical: "{g}"' for v, g in demos)

    adf = pd.read_parquet(args.assets)
    has_trace = "trace" in adf.columns and not args.no_trace

    from transformers import AutoModelForImageTextToText, AutoProcessor
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.bfloat16,
                                                        device_map="cuda")
    model.eval()

    rows = []
    for r in adf.itertuples():
        instr = str(r.ert_instruction)
        near = retrieve(instr, entry_grams, args.topn)
        near_txt = "\n".join(f"- {g}" for g in near)
        trace_txt = f"\nScene/task analysis:\n{r.trace}\n" if has_trace else "\n"
        prompt = (
            "A robot policy was trained on instructions written in a plain canonical style. "
            "You will see a REWORDED instruction; recover the canonical instruction for the "
            "SAME task (same objects, same placement).\n"
            f"Example mappings:\n{demo_txt}\n\n"
            f"Canonical instructions from the robot's training data most similar to the input:\n"
            f"{near_txt}\n{trace_txt}"
            f'Reworded input: "{instr}"\n'
            "Reply with ONLY the recovered canonical instruction."
        )
        msgs = [{"role": "user", "content": [{"type": "text", "text": prompt}]},
                {"role": "assistant", "content": [{"type": "text", "text": "Canonical:"}]}]
        enc = apply_template(processor, msgs, continue_final_message=True).to(model.device)
        plen = enc["input_ids"].shape[1]

        def clean(t):
            p = t.strip().split("\n")[0].strip().strip('"').strip()
            p = re.sub(r"^canonical:\s*", "", p, flags=re.I).strip().strip('"').strip()
            return p or instr

        with torch.no_grad():
            if args.sample_k > 0:
                out = model.generate(**enc, do_sample=True, temperature=args.sample_temp,
                                     num_return_sequences=args.sample_k, max_new_tokens=60)
                for si in range(args.sample_k):
                    p = clean(processor.decode(out[si][plen:], skip_special_tokens=True))
                    rows.append({"task": r.task, "arm": args.arm, "phrase": p,
                                 "instruction": instr, "sample_idx": si})
                print(f"[{args.arm}] {r.task}: {args.sample_k} samples, e.g. {rows[-1]['phrase']!r}")
            else:
                out = model.generate(**enc, do_sample=False, max_new_tokens=60)
                p = clean(processor.decode(out[0][plen:], skip_special_tokens=True))
                rows.append({"task": r.task, "arm": args.arm, "phrase": p, "instruction": instr})
                print(f"[{args.arm}] {r.task}:\n  in : {instr!r}\n  near: {near[:3]}\n  out: {p!r}")

    pd.DataFrame(rows).to_parquet(args.out, index=False)
    print(f"wrote {len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
