#!/usr/bin/env python3
"""Summarise one LIBERO minimal-pair round against its canonicals.

  .venv/bin/python scripts/analyze_libero_round.py 8

Reads results/analysis/pi05_bank/round<N>/*.jsonl (per-episode records from
bank_eval.py) plus the round manifest, looks up each canonical's n=50 rate across
every earlier n=50 round (roll50, round2..round<N>), and writes
round<N>_results.parquet with per-phrase pct, canonical pct, delta, and both a
two-proportion z p-value and a Fisher exact p-value. Prints the table sorted by
delta. Episodes are keyed on (suite, task_id, phrase, init); duplicates from a
resumed shard are dropped.
"""
import glob
import json
import math
import pathlib
import sys

import pandas as pd

R = pathlib.Path(__file__).resolve().parents[1]
B = R / "results/analysis/pi05_bank"
N = int(sys.argv[1])


def load(dirs):
    rows = []
    for d in dirs:
        for f in sorted(glob.glob(str(B / d / "*.jsonl"))):
            for line in open(f):
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("success") is None:
                    continue
                rows.append((r["suite"], int(r["task_id"]), r["phrase"], int(r["init"]), int(r["success"])))
    df = pd.DataFrame(rows, columns=["suite", "task_id", "phrase", "init", "success"])
    return df.drop_duplicates(["suite", "task_id", "phrase", "init"])


def two_prop_p(k1, n1, k2, n2):
    p1, p2 = k1 / n1, k2 / n2
    x = (k1 + k2) / (n1 + n2)
    se = math.sqrt(x * (1 - x) * (1 / n1 + 1 / n2)) or 1e-9
    return math.erfc(abs(p1 - p2) / se / math.sqrt(2))


def fisher_p(k1, n1, k2, n2):
    """Two-sided Fisher exact (sum of table probabilities <= observed)."""
    from math import comb
    a, b, c, d = k1, n1 - k1, k2, n2 - k2
    r1, c1, n = a + b, a + c, a + b + c + d
    def pr(x):
        return comb(r1, x) * comb(n - r1, c1 - x) / comb(n, c1)
    p_obs = pr(a)
    lo, hi = max(0, c1 - (n - r1)), min(r1, c1)
    return min(1.0, sum(pr(x) for x in range(lo, hi + 1) if pr(x) <= p_obs * (1 + 1e-9)))


prior = load(["roll50"] + [f"round{i}" for i in range(2, N)])
this = load([f"round{N}"])
man = pd.read_parquet(B / f"round{N}_manifest.parquet")
if "group" not in man:
    man["group"] = ""

allep = pd.concat([prior, this]).drop_duplicates(["suite", "task_id", "phrase", "init"])
agg = (allep.groupby(["suite", "task_id", "phrase"])
             .agg(n=("success", "size"), k=("success", "sum")).reset_index())
agg["pct"] = 100 * agg.k / agg.n

out = man.merge(agg, on=["suite", "task_id", "phrase"], how="left")
can = agg.rename(columns={"phrase": "canonical", "n": "canon_n", "k": "canon_k", "pct": "canon_pct"})
out = out.merge(can, on=["suite", "task_id", "canonical"], how="left")
out["delta"] = out.pct - out.canon_pct
out["p_z"] = [two_prop_p(r.k, r.n, r.canon_k, r.canon_n) if r.n == r.n and r.canon_n == r.canon_n else float("nan")
              for r in out.itertuples()]
out["p_fisher"] = [fisher_p(int(r.k), int(r.n), int(r.canon_k), int(r.canon_n)) if r.n == r.n and r.canon_n == r.canon_n else float("nan")
                   for r in out.itertuples()]
out = out.sort_values("delta")
out.to_parquet(B / f"round{N}_results.parquet", index=False)

pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 60)
print(f"round {N}: {len(this)} episodes, {out.n.notna().sum()}/{len(out)} phrases rolled, "
      f"{(out.n >= 50).sum()} at n>=50")
cols = ["task", "group", "phrase", "n", "pct", "canon_pct", "delta", "p_z", "p_fisher"]
print(out[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}" if abs(v) < 1 else f"{v:.0f}"))
sig = out[(out.p_z < 0.05) & (out.delta.abs() >= 18)]
print(f"\n|delta|>=18 and p_z<0.05: {len(sig)}")
print(sig[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}" if abs(v) < 1 else f"{v:.0f}"))
