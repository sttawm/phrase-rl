#!/usr/bin/env python3
"""Paper listing of EVERY single-edit pair (nulls included, sealed tasks included).

  .venv/bin/python scripts/make_single_edit_pairs_listing.py

Reads results/analysis/pi05_bank/single_edit_pairs.parquet (built by
scripts/build_libero_single_edit_pairs.py: all pairs of n=50 phrases on one task
differing in one contiguous token span, both sides on the same 50 inits) and
writes results/analysis/pi05_bank/single_edit_pairs_full.{csv,md}: one row per
pair with both phrases, both success rates, delta, z/Fisher/McNemar p, the edit
class, whether the pair was a designed (canonical, edit) probe, the finetune
status of the task, and whether the task is in the sealed set v2. Sorted by task,
then by delta. Nothing is filtered.
"""
import json
import pathlib

import pandas as pd

R = pathlib.Path(__file__).resolve().parents[1]
B = R / "results/analysis/pi05_bank"
P = pd.read_parquet(B / "single_edit_pairs.parquet")
draw = json.load(open(B / "sealed_v2_draw.json"))
sealed = {t for h in draw["halves"].values() for t in h["accepted"]}
P["sealed_v2"] = P.task.isin(sealed)
P["finetune"] = P.suite.map(lambda s: "out" if s == "libero_90" else "in")
P["significant"] = P.significant_18pp
cols = ["task", "finetune", "sealed_v2", "edit_class", "phrase_a", "succ_a", "phrase_b", "succ_b", "delta",
        "p_z", "p_fisher", "p_mcnemar", "a_only", "b_only", "significant", "designed", "a_is_canonical"]
out = P.sort_values(["task", "delta"])[cols]
out.to_csv(B / "single_edit_pairs_full.csv", index=False)

sig = int(out.significant.sum())
md = [f"# Single-edit pairs, complete listing ({len(out)} pairs, {out.task.nunique()} tasks)", "",
      "Every pair of phrases on one LIBERO task that differ in exactly one contiguous token span, both "
      "measured on the same 50 initial states (n=50 each). Success in %, delta = B - A in pp; p = "
      "two-proportion z (Fisher exact and paired McNemar in the CSV). significant = |delta| >= 18 pp and "
      f"p < 0.05 ({sig} of {len(out)}); the rest are nulls and are listed on purpose. sealed = task is in "
      "the sealed set v2 (seed 20260914); those pairs were removed from the distillation evidence but "
      "belong in the paper analysis. Source: results/analysis/pi05_bank/single_edit_pairs_full.csv.", ""]
by = out.groupby("edit_class").agg(pairs=("delta", "size"), significant=("significant", "sum"),
                                   mean_abs_delta=("delta", lambda d: d.abs().mean()))
md += ["| edit class | pairs | significant | mean abs delta |", "|---|---|---|---|"]
md += [f"| {k} | {int(r.pairs)} | {int(r.significant)} | {r.mean_abs_delta:.1f} |" for k, r in by.iterrows()]
md += ["", "| task | sealed | class | A | A % | B | B % | delta | p |", "|---|---|---|---|---|---|---|---|---|"]
for r in out.itertuples():
    p = f"{r.p_z:.4f}" if r.p_z >= 1e-4 else "<0.0001"
    flag = "**" if r.significant else ""
    md.append(f"| {r.task} | {'yes' if r.sealed_v2 else ''} | {r.edit_class} | {r.phrase_a} | {r.succ_a:.0f} | "
              f"{r.phrase_b} | {r.succ_b:.0f} | {flag}{r.delta:+.0f}{flag} | {p} |")
(B / "single_edit_pairs_full.md").write_text("\n".join(md) + "\n")
print(f"{len(out)} pairs ({sig} significant) on {out.task.nunique()} tasks; sealed-task pairs: {int(out.sealed_v2.sum())}")
print(by.round(1).to_string())
print("->", B / "single_edit_pairs_full.csv", "and .md")
