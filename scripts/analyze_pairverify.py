#!/usr/bin/env python3
"""Pair-verification analysis: full-grid re-roll vs priors.

Reads every episodes_*.parquet under results/analysis/pairverify/ (per-episode
rows with layout id = episode_id, rep, success), the merged manifest, and the
member table. Emits:
  - per-phrase summary with Wilson 95% intervals
    (results/analysis/pairverify/phrase_summary.csv)
  - per-pair verdicts: prior gap vs full-grid gap, two-proportion z at the new
    n, and whether the pair survives (results/analysis/pairverify/pair_verdicts.csv)
Headline split: single_concept & format_matched pairs first, everything else
reported separately. Within a B-set group (ladders, verb x noun sets), pairs
are formed between every labeled member and the group's best-performing
member is NOT special-cased — we report all within-group adjacent contrasts
sorted by |gap| (the ladder reads naturally that way).
"""
import glob
import math
import pathlib

import pandas as pd

R = pathlib.Path(__file__).resolve().parents[1]
PV = R / "results/analysis/pairverify"


def wilson(p, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z / (1 + z * z / n) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, c - h), min(1.0, c + h))


def two_prop_z(p1, n1, p2, n2):
    x1, x2 = p1 * n1, p2 * n2
    p = (x1 + x2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2)) or 1e-9
    return (p2 - p1) / se


E = pd.concat([pd.read_parquet(f) for f in glob.glob(str(PV / "episodes_*.parquet"))],
              ignore_index=True)
E = E.drop_duplicates(subset=["task", "phrase", "episode_id", "rep"])
print(f"episodes: {len(E)} | phrases: {E.groupby(['task','phrase']).ngroups}")

S = (E.groupby(["task", "phrase"], as_index=False)
     .agg(succ=("success", "mean"), n=("success", "size"),
          n_layouts=("episode_id", "nunique")))
S[["lo95", "hi95"]] = S.apply(lambda r: pd.Series(wilson(r.succ, r.n)), axis=1)
for c in ["succ", "lo95", "hi95"]:
    S[c] = (S[c] * 100).round(1)

M = pd.read_parquet(R / "results/analysis/bridge_pair_manifest_members.parquet")
S2 = S.merge(M, on=["task", "phrase"], how="left")
S2.to_csv(PV / "phrase_summary.csv", index=False)

rows = []
for (task, group), g in M.groupby(["task", "group"]):
    g = g.merge(S, on=["task", "phrase"], how="left")
    g = g[g.n.notna()]
    if len(g) < 2:
        continue
    g = g.sort_values("succ").reset_index(drop=True)
    for i in range(len(g)):
        for j in range(i + 1, len(g)):
            a, b = g.iloc[i], g.iloc[j]
            prior_gap = abs(b.prior_success - a.prior_success)
            new_gap = b.succ - a.succ
            z = two_prop_z(a.succ / 100, a.n, b.succ / 100, b.n)
            rows.append({
                "task": task.replace("widowx_", "").replace("_clean", ""),
                "group": group,
                "single_concept": bool(a.single_concept),
                "format_matched": bool(a.format_matched),
                "phrase_lo": a.phrase, "succ_lo": a.succ, "ci_lo": f"[{a.lo95},{a.hi95}]",
                "phrase_hi": b.phrase, "succ_hi": b.succ, "ci_hi": f"[{b.lo95},{b.hi95}]",
                "n_each": int(min(a.n, b.n)),
                "prior_gap": round(prior_gap, 1), "new_gap": round(new_gap, 1),
                "z": round(z, 2), "survives": bool(abs(z) >= 1.96),
            })
V = pd.DataFrame(rows).sort_values("new_gap", ascending=False)
V.to_csv(PV / "pair_verdicts.csv", index=False)

head = V[V.single_concept & V.format_matched]
print(f"\npair contrasts: {len(V)} total | headline (single-concept, format-matched): {len(head)}")
print(f"survive at z>=1.96: headline {int(head.survives.sum())}/{len(head)}, "
      f"all {int(V.survives.sum())}/{len(V)}")
with pd.option_context("display.width", 240, "display.max_colwidth", 44):
    print("\n=== HEADLINE: single-concept, format-matched, by new gap ===")
    print(head.head(30)[["task", "group", "succ_lo", "succ_hi", "n_each",
                         "prior_gap", "new_gap", "z", "survives"]].to_string(index=False))
