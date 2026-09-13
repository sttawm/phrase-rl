#!/usr/bin/env python3
"""A39 applies: every draw-book + scaffold over the human naturals.

  FINAL_EVAL=1 .venv/bin/python scripts/gen_a39_applies.py <applier> [book,...]
  FINAL_EVAL=1 .venv/bin/python scripts/gen_a39_applies.py qwen-queue

Books: s b t s2 s3 b2 b3 t2 t3 sc (default: all ten).
Applier settings per PREREG A39: claude effort MAX, gemini thinking 16384
(both above the A31 protocol; deliberate), qwen greedy pod-side via the
r1_sim job queue (jid a39<book>qw_<sha8>). Traces come from the per-base
registry (results/phrase_artifacts/traces_rules_v1.parquet), which
scripts/gen_a39_traces.py output must be appended to first.
Outputs: results/human_naturals/ph_a39_<book>_<applier>.parquet
         (task, base, phrase).
"""
import hashlib
import json
import os
import pathlib
import sys

import pandas as pd

if os.environ.get("FINAL_EVAL") != "1":
    raise SystemExit("FINAL_EVAL=1 required")
sys.path.insert(0, "scripts")
import rules_loop_driver as d  # noqa: E402

R = pathlib.Path.home() / "dev/robotics/phrase-rl"
BOOKS = {"s": "sim_only", "b": "both", "t": "train_only",
         "s2": "sim_only_r2", "s3": "sim_only_r3", "b2": "both_r2",
         "b3": "both_r3", "t2": "train_only_r2", "t3": "train_only_r3"}


def book_text(tag):
    if tag == "sc":
        return (R / "results/rules_runs/v2distill/scaffold.md").read_text()
    return (R / "results/rules_runs/v2distill" / BOOKS[tag] / "rules.md").read_text()


ph = pd.read_parquet(R / "results/human_naturals/a39_human_phrases.parquet")
bases = ph[["task", "phrase"]].drop_duplicates().reset_index(drop=True)
print(f"{len(bases)} unique (task, phrase) bases")

mode = sys.argv[1]
tags = sys.argv[2].split(",") if len(sys.argv) > 2 else list(BOOKS) + ["sc"]

if mode == "qwen-queue":
    jd = R / "results/rules_runs/r1_sim/jobs"
    for tag in tags:
        rules = book_text(tag)
        sha = hashlib.sha1(rules.encode()).hexdigest()
        jid = f"a39{tag}qw_{sha[:8]}"
        if (jd / f"{jid}.spec.json").exists():
            print("exists, skip", jid)
            continue
        bases.to_parquet(jd / f"{jid}.payload.parquet", index=False)
        json.dump({"job_id": jid, "kind": "apply",
                   "rules_text": d.rules_only(rules), "rules_sha": sha[:12]},
                  open(jd / f"{jid}.spec.json", "w"), indent=1)
        print("queued", jid)
    raise SystemExit(0)

applier = mode
run = d.Run("r1_sim", dry=False)
cfg = d.jread(run.cfg_path)
run.env = d.ENVIRONMENTS[cfg["env"]] if isinstance(cfg.get("env"), str) else cfg.get("env")
run.gemini_model = cfg.get("gemini_model", "gemini-pro-latest")
run.claude_model = cfg.get("claude_model", "claude-fable-5")
# A39 protocol: stronger applier settings than A31 (PREREG'd)
run.gemini_thinking = 16384
run.apply_effort = "max" if applier == "claude" else cfg.get("apply_effort", "high")
run.claude_effort = "max"

for tag in tags:
    out = R / f"results/human_naturals/ph_a39_{tag}_{applier}.parquet"
    if out.exists():
        print("exists, skip", out.name)
        continue
    rw = d.apply_rules(run, cfg, applier, book_text(tag), bases, f"a39{tag}{applier[:2]}")
    rw = rw.rename(columns={"phrase": "base", "rewrite": "phrase"})
    rw["phrase"] = rw.phrase.fillna("").astype(str)
    rw.loc[rw.phrase.str.strip() == "", "phrase"] = rw.base
    rw.to_parquet(out, index=False)
    print(f"[{tag}/{applier}] applied {len(rw)}, changed {(rw.phrase != rw.base).mean():.0%}")
print("A39-APPLIES-DONE", applier, tags)
