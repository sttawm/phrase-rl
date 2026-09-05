#!/usr/bin/env python3
"""Compute A36 cells BASE-WEIGHTED, the convention A31 already uses.

  .venv/bin/python scripts/a36_cells.py [tag[,tag...]]

Pools over the phrase BASES (join the apply table to the leg results on
task+phrase, then mean over bases), not over distinct rewrites. The two differ
because appliers collapse different bases onto the same rewrite at very
different rates -- verified 2026-09-04 that all 33 published A31 cells match
base-weighting, and that distinct-weighting is what made the other session's A34
grid disagree.

Also reports n_distinct, because uncertainty must be CLUSTERED on distinct
rewrites: bases sharing a rewrite share one measurement and are not independent.
Treating n=186 as independent understates the SE, worst for the arms that
collapse hardest.
"""
import glob, os, pathlib, re, sys, collections
import pandas as pd

R = pathlib.Path.home() / "dev/robotics/phrase-rl"
JD = R / "results/rules_runs/r1_sim/jobs"
SD = R / "results/sealed"
ONLY = set(sys.argv[1].split(",")) if len(sys.argv) > 1 else None
AP = {"cl": "claude", "ge": "gemini", "qw": "qwen"}
CN = {"a": "adv", "n": "nat", "o": "orig"}
BOOK = {"t2": "train-only r2", "t3": "train-only r3", "s2": "rollout-only r2",
        "s3": "rollout-only r3", "b2": "combined r2", "b3": "combined r3"}

legs = collections.defaultdict(list)
for f in glob.glob(str(JD / "f36*.result.parquet")):
    m = re.match(r"f36([tsb][23])(cl|ge|qw)([ano])_", pathlib.Path(f).name)
    if not m:
        continue
    tag, ap, c = m.groups()
    if ONLY and tag not in ONLY:
        continue
    legs[(tag, ap, c)].append(pd.read_parquet(f))

if not legs:
    raise SystemExit("no f36 leg results on disk -- fetch them first")

print(f'{"book":<16}{"applier":<9}{"cond":<6}{"legs":>5}{"bases":>7}'
      f'{"n_dist":>8}{"BASE-WTD":>10}{"distinct":>10}')
print("-" * 71)
for (tag, ap, c) in sorted(legs):
    L = (pd.concat(legs[(tag, ap, c)], ignore_index=True)
         .groupby(["task", "phrase"], as_index=False)
         .agg(succ=("gt_success", "mean"), neps=("n_ctx", "sum")))
    apf = SD / f"ph_a36_{tag}_{AP[ap]}_{CN[c]}.parquet"
    if not apf.exists():
        print(f"{BOOK[tag]:<16}{AP[ap]:<9}{CN[c]:<6}  apply file missing: {apf.name}")
        continue
    a = pd.read_parquet(apf)
    m = a.merge(L, on=["task", "phrase"], how="left")
    bw = m["succ"].mean()
    dw = (L["succ"] / 100 * L["neps"]).sum() / L["neps"].sum() * 100
    print(f"{BOOK[tag]:<16}{AP[ap]:<9}{CN[c]:<6}{len(legs[(tag,ap,c)]):>5}"
          f"{len(a):>7}{len(L):>8}{bw:>10.1f}{dw:>10.1f}")
print()
print("controls (already measured, do not re-run):")
print("  natural      A34 scaffold  claude 29.7  gemini 28.7  qwen 28.1   "
      "un-rephrased 26.0")
print("  adversarial  A31 scaffold  claude 24.3  gemini 24.0  qwen 24.7")
print("  original     no scaffold exists (X'd out as not planned)")
