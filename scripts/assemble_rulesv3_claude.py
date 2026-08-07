#!/usr/bin/env python3
"""Assemble the Claude rules-v3 arm from the three agent slices and preflight it.

Mirrors the Gemini and Qwen arms exactly: same 192 sealed natural inputs, same
rules-v3 wrap, one rewrite per (task, k). Refuses to write the parquet unless
every check passes -- the arm rolls on sealed tasks, so a malformed input would
burn 14 pod-hours and produce an uninterpretable cell.

  .venv/bin/python scripts/assemble_rulesv3_claude.py
"""
import json
import sys
from pathlib import Path

import pandas as pd

SCRATCH = Path("/private/tmp/claude-501/-Users-sttawm-dev-robotics-phrase-rl/"
               "a45839fa-48ee-4aea-bdee-72eb4fc9dccf/scratchpad/v3_slices")
OUT = Path("results/sealed/ph_rr16_rulesv3_claude.parquet")

rows = []
for s in range(3):
    f = SCRATCH / f"out_{s}.json"
    if not f.exists():
        sys.exit(f"missing {f} -- slice {s} has not landed")
    rows += json.load(f.open())
df = pd.DataFrame(rows)
df["arm"] = "rr16_rulesv3_claude"
df = df.rename(columns={"src": "instruction"})[["task", "k", "arm", "phrase", "instruction"]]

src = pd.read_parquet("results/sealed/ph_sealed_rephrase16.parquet")
qw = pd.read_parquet("results/sealed/ph_rr16_rulesv3_qwen.parquet")
v4 = pd.read_parquet("results/sealed/ph_rr16_rulesv4_claude.parquet") \
    if Path("results/sealed/ph_rr16_rulesv4_claude.parquet").exists() else None

fail = []
if len(df) != 192:
    fail.append(f"row count {len(df)} != 192")
if df.task.nunique() != 12:
    fail.append(f"task count {df.task.nunique()} != 12")
if set(zip(df.task, df.k)) != set(zip(src.task, src.k)):
    fail.append("(task, k) coverage does not match the sealed natural set")
n_empty = int((df.phrase.str.strip() == "").sum())
if n_empty:
    fail.append(f"{n_empty} empty rewrites")
w = df.phrase.str.split().str.len()
if w.max() > 16:
    fail.append(f"a rewrite is {w.max()} words (>16 suggests commentary leaked in)")
if df.phrase.str.contains(r'^["\']|["\']$|^\s*\d+\.', regex=True).any():
    fail.append("quoting or numbering leaked into rewrites")

m = df.set_index(["task", "k"]).phrase
s = src.set_index(["task", "k"]).phrase
echo = int((m.str.lower().str.strip() == s.str.lower().str.strip()).sum())

print(f"rows {len(df)} | tasks {df.task.nunique()} | per task {len(df) // df.task.nunique()}")
print(f"empty {n_empty} | echo-input {echo} | words mean {w.mean():.1f} max {w.max()}")
def overlap(other, label):
    """align on the shared (task, k) index -- v4 arms may cover a different set"""
    o = other.set_index(["task", "k"]).phrase
    j = m.to_frame("a").join(o.rename("b"), how="inner")
    if not len(j):
        print(f"identical to {label}: (no shared rows)")
        return
    same = (j.a.str.strip() == j.b.str.strip()).mean()
    print(f"identical to {label}: {100 * same:.1f}%  -> {100 * (1 - same):.1f}% differ "
          f"(n={len(j)})")

overlap(qw, "the qwen v3 arm")
if v4 is not None:
    overlap(v4, "the claude v4 arm")

if fail:
    print("\nPREFLIGHT FAILED:")
    for f_ in fail:
        print("  -", f_)
    sys.exit(1)

print("\nPREFLIGHT sample (one per task):")
for t, g in df.groupby("task"):
    r = g.iloc[0]
    print(f"  {t.replace('widowx_', '')[:26]:28s} k{int(r.k):02d} -> {r.phrase!r}")

OUT.parent.mkdir(parents=True, exist_ok=True)
df.to_parquet(OUT, index=False)
print(f"\nPREFLIGHT PASS -> {OUT}")
