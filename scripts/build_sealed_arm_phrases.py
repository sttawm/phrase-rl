"""Sealed-arm phrase generators for arms not covered by existing tools.

Existing tools cover: #6 sft_v2 (sft17k_generate_eval --trace-assets),
#7/#8 rules->qwen (sft17k_rules_gen --assets <gemini|selftrace> [--rules-path]).
This script covers:
  anchors    -> #1 originals + #2 passthrough (no model)
  frozen     -> #3/#4 frozen Qwen under CoVer prompt (image+trace), one arm per
                trace source:  --trace-source {gemini,selftrace}
  v6         -> #5 v6 RL adapter, gemini traces (its native conditioning)
  rulesgemini-> #9 rules text executed by Gemini
  oracle     -> #10 stage-1: k=16 Qwen rephrases of the NOMINAL (bare, no trace)

All modes read the canonical sealed asset parquets and write standard
(task, arm, phrase, instruction[, sample_idx]) parquets. FINAL_EVAL=1 required.
"""
import argparse
import io
import os
import sys

import pandas as pd

if os.environ.get("FINAL_EVAL") != "1":
    raise SystemExit("FINAL_EVAL=1 required")

sys.path.insert(0, "src")


def load(args):
    ga = pd.read_parquet(args.gemini_assets)
    st = pd.read_parquet(args.selftrace_assets)
    fr = pd.read_parquet(args.frames)
    return ga, st, fr


def mode_anchors(args):
    ga, _, _ = load(args)
    rows = [{"task": r.task, "arm": "originals", "phrase": r.nominal, "instruction": r.nominal}
            for r in ga.itertuples()]
    rows += [{"task": r.task, "arm": "passthrough", "phrase": r.ert_instruction,
              "instruction": r.ert_instruction} for r in ga.itertuples()]
    pd.DataFrame(rows).to_parquet(args.out, index=False)
    print(f"wrote {len(rows)} anchor rows -> {args.out}")


def _cover_generate(assets, frames, adapter, arm, out):
    import torch
    from PIL import Image
    from phrase_rl.cover_prompt import build_single_phrase_prefix
    from phrase_rl.phase2_train import apply_template
    from phrase_rl.phase4_generate_eval import strip_tag, first_line
    from transformers import AutoModelForImageTextToText, AutoProcessor
    proc = AutoProcessor.from_pretrained("Qwen/Qwen3.5-9B")
    model = AutoModelForImageTextToText.from_pretrained("Qwen/Qwen3.5-9B",
                                                        dtype=torch.bfloat16, device_map="cuda")
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter, is_trainable=False)
    model.eval()
    fmap = {r.task: r.image_png for r in frames.itertuples()}
    rows = []
    for r in assets.itertuples():
        img = Image.open(io.BytesIO(fmap[r.task]))
        msgs = build_single_phrase_prefix(str(r.ert_instruction), img, trace=str(r.trace))
        enc = apply_template(proc, msgs, continue_final_message=True).to(model.device)
        with torch.no_grad():
            g = model.generate(**enc, do_sample=False, max_new_tokens=60)
        p = strip_tag(first_line(strip_tag(
            proc.decode(g[0][enc["input_ids"].shape[1]:], skip_special_tokens=True))))
        p = p or str(r.ert_instruction)
        rows.append({"task": r.task, "arm": arm, "phrase": p,
                     "instruction": str(r.ert_instruction)})
        print(f"[{arm}] {r.task}: {p!r}", flush=True)
    pd.DataFrame(rows).to_parquet(out, index=False)
    print(f"wrote {len(rows)} -> {out}")


def mode_frozen(args):
    ga, st, fr = load(args)
    assets = ga if args.trace_source == "gemini" else st.merge(
        ga[["task"]], on="task")  # selftrace parquet already has ert+trace
    arm = "frozen_gemini_trace" if args.trace_source == "gemini" else "frozen_selftrace"
    _cover_generate(assets, fr, None, arm, args.out)


def mode_v6(args):
    ga, _, fr = load(args)
    _cover_generate(ga, fr, args.v6_adapter, "v6_rl", args.out)


def mode_rulesgemini(args):
    ga, _, _ = load(args)
    from phrase_rl.gemini_redteam_assets import call_with_retry
    from google import genai
    from google.genai import types
    rules = open(args.rules_path).read()
    client = genai.Client()
    rows = []
    for r in ga.itertuples():
        prompt = (f"{rules}\n\n---\n\nApply the rules above.\n\nTrace:\n{r.trace}\n\n"
                  f"Incoming instruction: {r.ert_instruction}\n\n"
                  f"Reply with ONLY the rewritten instruction.")
        p = call_with_retry(client, types, "gemini-3.5-flash", [prompt],
                            temperature=0.2, max_tokens=100).strip().strip('"')
        p = p.split("\n")[0].strip() or str(r.ert_instruction)
        rows.append({"task": r.task, "arm": "rules_gemini", "phrase": p,
                     "instruction": str(r.ert_instruction)})
        print(f"[rules_gemini] {r.task}: {p!r}", flush=True)
    pd.DataFrame(rows).to_parquet(args.out, index=False)
    print(f"wrote {len(rows)} -> {args.out}")


def mode_oracle(args):
    import torch
    from phrase_rl.phase2_train import apply_template
    from transformers import AutoModelForImageTextToText, AutoProcessor
    ga, _, _ = load(args)
    proc = AutoProcessor.from_pretrained("Qwen/Qwen3.5-9B")
    model = AutoModelForImageTextToText.from_pretrained("Qwen/Qwen3.5-9B",
                                                        dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    rows = []
    for r in ga.itertuples():
        msgs = [{"role": "user", "content": [{"type": "text", "text":
                 f"Reword this robot instruction. Reply with ONLY the reworded instruction.\n{r.nominal}"}]},
                {"role": "assistant", "content": [{"type": "text", "text": "Reworded:"}]}]
        enc = apply_template(proc, msgs, continue_final_message=True).to(model.device)
        with torch.no_grad():
            g = model.generate(**enc, do_sample=True, temperature=1.0,
                               num_return_sequences=16, max_new_tokens=40)
        plen = enc["input_ids"].shape[1]
        seen = set()
        for si in range(16):
            p = proc.decode(g[si][plen:], skip_special_tokens=True).strip().split("\n")[0].strip().strip('"')
            p = p or str(r.nominal)
            rows.append({"task": r.task, "arm": "oracle_pool", "phrase": p,
                         "instruction": str(r.nominal), "sample_idx": si})
        print(f"[oracle] {r.task}: 16 drawn, e.g. {rows[-1]['phrase']!r}", flush=True)
    pd.DataFrame(rows).to_parquet(args.out, index=False)
    print(f"wrote {len(rows)} -> {args.out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True,
                    choices=["anchors", "frozen", "v6", "rulesgemini", "oracle"])
    ap.add_argument("--trace-source", default="gemini", choices=["gemini", "selftrace"])
    ap.add_argument("--gemini-assets", default="data/sealed_assets_gemini.parquet")
    ap.add_argument("--selftrace-assets", default="data/sealed_assets_selftrace.parquet")
    ap.add_argument("--frames", default="data/sealed_frames.parquet")
    ap.add_argument("--v6-adapter", default="/workspace/adapters/B/v6step_0080")
    ap.add_argument("--rules-path", default="results/analysis/b4_phrasing_rules.md")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    {"anchors": mode_anchors, "frozen": mode_frozen, "v6": mode_v6,
     "rulesgemini": mode_rulesgemini, "oracle": mode_oracle}[args.mode](args)


if __name__ == "__main__":
    main()
