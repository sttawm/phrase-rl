#!/usr/bin/env python3
"""A36 applies: one replicate book, all three sealed conditions, gemini@16384. one book x one applier x all three sealed conditions.
Usage: gen_a31_applies.py <book_path> <applier> <tag>   (FINAL_EVAL=1 gated)
Writes results/sealed/ph_a34_<tag>_<applier>_{adv,nat,orig}.parquet"""
import os
import sys

import pandas as pd

if os.environ.get("FINAL_EVAL") != "1":
    raise SystemExit("FINAL_EVAL=1 required")
sys.path.insert(0, "scripts")
import pathlib

import rules_loop_driver as d

book_path, tag = sys.argv[1], sys.argv[2]
applier = sys.argv[3] if len(sys.argv) > 3 else "gemini"

REPO = pathlib.Path.home() / "dev/robotics/phrase-rl"
rules = (REPO / book_path).read_text()

run = d.Run("r1_sim", dry=False)
cfg = d.jread(run.cfg_path)
run.gemini_model = cfg.get("gemini_model", "gemini-pro-latest")
run.gemini_thinking = cfg.get("gemini_thinking_budget", 1024)
run.apply_effort = cfg.get("apply_effort", "high")
run.claude_model = cfg.get("claude_model", "claude-fable-5")
run.env = d.ENVIRONMENTS[cfg["env"]] if isinstance(cfg.get("env"), str) else cfg.get("env")

orig12 = pd.read_parquet(REPO / "results/sealed/sealed_assets_gemini.parquet")
adv = pd.concat([
    orig12.rename(columns={"ert_instruction": "phrase"})[["task", "phrase"]],
    pd.read_parquet(REPO / "results/sealed/a29_new_erts.parquet")[["task", "phrase"]],
]).drop_duplicates().reset_index(drop=True)
nat = pd.read_parquet(REPO / "results/sealed/ph_sealed_natural_v2_img.parquet")[["task", "k", "phrase"]]
orig = orig12.rename(columns={"nominal": "phrase"})[["task", "phrase"]].drop_duplicates()

import time as _time
for cond, bases in (("adv", adv), ("nat", nat), ("orig", orig)):
    # transient API disconnects abort a whole condition and the cache only
    # writes on completion -- retry the condition instead of dying (2026-09-05:
    # gemini t3 crash-looped for 3.5h without this)
    for _att in range(3):
        try:
            rw = d.apply_rules(run, cfg, applier, rules, bases[["task", "phrase"]],
                               f"a36{tag}{applier[:2]}_{cond}")
            break
        except Exception as e:
            print(f"[{tag}/{applier}/{cond}] attempt {_att+1} failed: {e}", flush=True)
            if _att == 2:
                raise
            _time.sleep(45)
    rw = rw.rename(columns={"phrase": "base", "rewrite": "phrase"})
    rw["phrase"] = rw.phrase.fillna("").astype(str)
    rw.loc[rw.phrase.str.strip() == "", "phrase"] = rw.base
    if cond == "nat":
        rw = bases.rename(columns={"phrase": "base"}).merge(
            rw, on=["task", "base"], how="left").drop_duplicates(["task", "k"])
    out = REPO / f"results/sealed/ph_a36_{tag}_{applier}_{cond}.parquet"
    rw.to_parquet(out, index=False)
    print(f"[{tag}/{applier}/{cond}] applied {len(rw)}, changed "
          f"{(rw.phrase != rw.base).mean():.0%}", flush=True)
print(f"A31-APPLIES-DONE {tag} {applier}", flush=True)
