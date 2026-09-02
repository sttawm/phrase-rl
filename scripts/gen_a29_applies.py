#!/usr/bin/env python3
"""A29 sealed applies (gated: FINAL_EVAL=1). Usage: gen_a29_applies.py <applier>

Applies the applier's A29-locked book to both sealed conditions via the r1_sim
loop protocol (prompts/rules_loop/apply.md wrap; traces from the registry,
which now carries the sealed A29 rows; applier configs from r1_sim config).
  claude -> pass_claude/best_rules.md   (local fable, effort high)
  gemini -> pass_claude/best_rules.md   (API; cross-book per holdout 43.1)
  qwen   -> handled pod-side (see a29 qwen queue block in session scratchpad)
Outputs: results/sealed/ph_a29_<applier>_adv.parquet (72 rows: task, base, phrase)
         results/sealed/ph_a29_<applier>_nat.parquet (192 rows: task, k, base, phrase)
"""
import os
import sys

import pandas as pd

if os.environ.get("FINAL_EVAL") != "1":
    raise SystemExit("FINAL_EVAL=1 required")
sys.path.insert(0, "scripts")
import pathlib

import rules_loop_driver as d

applier = sys.argv[1]
BOOKS = {"claude": "results/rules_runs/r1_sim/pass_claude/best_rules.md",
         "gemini": "results/rules_runs/r1_sim/pass_claude/best_rules.md"}
REPO = pathlib.Path.home() / "dev/robotics/phrase-rl"
rules = (REPO / BOOKS[applier]).read_text()

run = d.Run("r1_sim", dry=False)
cfg = d.jread(run.cfg_path)
run.gemini_model = cfg.get("gemini_model", "gemini-pro-latest")
run.gemini_thinking = cfg.get("gemini_thinking_budget", 1024)
run.apply_effort = cfg.get("apply_effort", "high")
run.claude_model = cfg.get("claude_model", "claude-fable-5")
run.env = d.ENVIRONMENTS[cfg["env"]] if isinstance(cfg.get("env"), str) else cfg.get("env")

orig = pd.read_parquet(REPO / "results/sealed/sealed_assets_gemini.parquet").rename(
    columns={"ert_instruction": "phrase"})[["task", "phrase"]]
new = pd.read_parquet(REPO / "results/sealed/a29_new_erts.parquet")[["task", "phrase"]]
adv = pd.concat([orig, new]).drop_duplicates().reset_index(drop=True)
nat = pd.read_parquet(REPO / "results/sealed/ph_sealed_rephrase16.parquet")[
    ["task", "k", "phrase"]].reset_index(drop=True)
print(f"[{applier}] adv bases={len(adv)} nat bases={len(nat)}", flush=True)

for cond, bases in (("adv", adv), ("nat", nat)):
    rw = d.apply_rules(run, cfg, applier, rules, bases[["task", "phrase"]],
                       f"a29{applier[:2]}_{cond}")
    rw = rw.rename(columns={"phrase": "base", "rewrite": "phrase"})
    rw["phrase"] = rw.phrase.fillna("").astype(str)
    rw.loc[rw.phrase.str.strip() == "", "phrase"] = rw.base
    if cond == "nat":
        rw = bases[["task", "k", "phrase"]].rename(columns={"phrase": "base"}).merge(
            rw, on=["task", "base"], how="left").drop_duplicates(["task", "k"])
    out = REPO / f"results/sealed/ph_a29_{applier}_{cond}.parquet"
    rw.to_parquet(out, index=False)
    print(f"[{applier}/{cond}] applied {len(rw)}, changed "
          f"{(rw.phrase != rw.base).mean():.0%} -> {out.name}", flush=True)
    for r in rw.head(3).itertuples():
        print(f"  PREFLIGHT {r.task} | {str(r.base)[:60]} -> {str(r.phrase)[:60]}", flush=True)
print(f"A29-APPLIES-DONE {applier}", flush=True)
