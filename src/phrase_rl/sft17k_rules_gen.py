"""Rules-arm phrase generation: the B4 rule set as a prescriptive prompt for a
frozen executor (Qwen locally; the Gemini variant lives in the sealed harness).

Prompt = the full rules file + the self-trace + the hostile instruction;
no-thinking prefill per codebase discipline. Output schema matches the
standard phrases parquet.

  .venv-gen/bin/python -m phrase_rl.sft17k_rules_gen \
      --assets data/val_screen_assets_selftrace.parquet \
      --out data/phrases_rules_qwen.parquet [--sample-k 8]
"""
import argparse

import pandas as pd
import torch

from phrase_rl.phase2_train import apply_template

RULES_PATH = "results/analysis/b4_phrasing_rules.md"  # override with --rules-path


def build_rules_msgs(rules_text, instr, trace):
    txt = (f"{rules_text}\n\n---\n\nApply the rules above.\n\n"
           f"Trace:\n{trace}\n\n"
           f"Incoming instruction: {instr}\n\n"
           f"Reply with ONLY the rewritten instruction.")
    return [{"role": "user", "content": [{"type": "text", "text": txt}]},
            {"role": "assistant", "content": [{"type": "text", "text": "Rewritten:"}]}]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--assets", required=True, help="parquet(task, ert_instruction, trace)")
    ap.add_argument("--rules-path", default=RULES_PATH)
    ap.add_argument("--arm", default="rules_qwen")
    ap.add_argument("--sample-k", type=int, default=0)
    ap.add_argument("--sample-temp", type=float, default=1.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rules = open(args.rules_path).read()
    adf = pd.read_parquet(args.assets)

    from transformers import AutoModelForImageTextToText, AutoProcessor
    proc = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.bfloat16,
                                                        device_map="cuda")
    model.eval()

    def clean(t, fallback):
        p = t.strip().split("\n")[0].strip().strip('"').strip()
        if p.lower().startswith("rewritten:"):
            p = p[len("rewritten:"):].strip()
        return p or fallback

    rows = []
    for r in adf.itertuples():
        instr = str(r.ert_instruction)
        enc = apply_template(proc, build_rules_msgs(rules, instr, str(r.trace)),
                             continue_final_message=True).to(model.device)
        plen = enc["input_ids"].shape[1]
        with torch.no_grad():
            if args.sample_k > 0:
                out = model.generate(**enc, do_sample=True, temperature=args.sample_temp,
                                     num_return_sequences=args.sample_k, max_new_tokens=40)
                for si in range(args.sample_k):
                    p = clean(proc.decode(out[si][plen:], skip_special_tokens=True), instr)
                    rows.append({"task": r.task, "arm": args.arm, "phrase": p,
                                 "instruction": instr, "sample_idx": si})
                print(f"[{args.arm}] {r.task}: e.g. {rows[-1]['phrase']!r}", flush=True)
            else:
                out = model.generate(**enc, do_sample=False, max_new_tokens=40)
                p = clean(proc.decode(out[0][plen:], skip_special_tokens=True), instr)
                rows.append({"task": r.task, "arm": args.arm, "phrase": p, "instruction": instr})
                print(f"[{args.arm}] {r.task}: {instr[:60]!r} -> {p!r}", flush=True)

    pd.DataFrame(rows).to_parquet(args.out, index=False)
    print(f"wrote {len(rows)} rows -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
