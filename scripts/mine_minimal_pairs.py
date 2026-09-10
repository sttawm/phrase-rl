#!/usr/bin/env python3
"""Mine minimal-edit phrase pairs with large ground-truth success gaps.

Pools every leg result parquet in --legs (task, phrase, gt_success, n_ctx;
gt_success is a percentage, n_ctx an episode count), then, within each task,
finds pairs of phrases that differ by exactly one of:
  - a single-token substitution, classified as preposition / verb / noun-adj
    by wordlist (the paper's "simple change" categories), or
  - a slight restructure: identical content tokens ignoring determiners and
    politeness tokens, but different order or added/dropped determiners.

Candidates ranked by |success gap| with a two-proportion z filter. Layouts and
reps are NOT matched across sources (candidate list only; verification with
matched rollouts comes later) — n and pooled rates are reported so obviously
under-powered pairs can be excluded up front.

  .venv/bin/python scripts/mine_minimal_pairs.py --legs /tmp/legpool \
      --min-n 24 --min-z 2.0 --out results/analysis/minimal_pairs_candidates.csv
"""
import argparse
import glob
import math
import pathlib
import re

import pandas as pd

PREP = {"on", "onto", "upon", "in", "into", "inside", "within", "atop", "to",
        "at", "over", "above", "top"}
VERB = {"put", "place", "set", "move", "drop", "transfer", "deposit", "rest",
        "position", "lay", "stick", "shift", "park", "balance", "release",
        "arrange", "grab", "pick", "take", "relocate", "situate", "plant",
        "perch", "pop", "plop", "slide", "bring", "carry", "lower", "land",
        "lift", "leave", "settle", "install", "load"}
SKIP = {"the", "a", "an", "please"}


def toks(p):
    return re.sub(r"[^a-z0-9 ]", " ", p.lower()).split()


def classify_sub(a, b):
    if a in PREP and b in PREP:
        return "preposition synonym"
    if a in VERB and b in VERB:
        return "verb synonym"
    return "noun/adjective synonym"


def pair_type(t1, t2):
    """None if not a minimal pair; else (category, changed-from, changed-to)."""
    if t1 == t2:
        return ("case/punctuation only", "", "")
    if len(t1) == len(t2):
        diffs = [(a, b) for a, b in zip(t1, t2) if a != b]
        if len(diffs) == 1:
            a, b = diffs[0]
            return (classify_sub(a, b), a, b)
    c1 = [w for w in t1 if w not in SKIP]
    c2 = [w for w in t2 if w not in SKIP]
    if c1 == c2 or (sorted(c1) == sorted(c2) and len(c1) == len(c2)):
        return ("slight restructure", " ".join(t1), " ".join(t2))
    return None


def two_prop_z(p1, n1, p2, n2):
    x1, x2 = p1 * n1, p2 * n2
    p = (x1 + x2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2)) or 1e-9
    return abs(p1 - p2) / se


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--legs", required=True)
    ap.add_argument("--min-n", type=int, default=24)
    ap.add_argument("--min-z", type=float, default=2.0)
    ap.add_argument("--top", type=int, default=60)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    frames = []
    for f in glob.glob(str(pathlib.Path(args.legs) / "*.parquet")):
        try:
            d = pd.read_parquet(f, columns=["task", "phrase", "gt_success", "n_ctx"])
            frames.append(d)
        except Exception:
            continue
    L = pd.concat(frames, ignore_index=True)
    L["wins"] = L.gt_success / 100.0 * L.n_ctx
    P = (L.groupby(["task", "phrase"], as_index=False)
         .agg(wins=("wins", "sum"), n=("n_ctx", "sum")))
    P["succ"] = P.wins / P.n * 100
    P = P[P.n >= args.min_n]
    print(f"pooled: {len(P)} (task, phrase) cells with n>={args.min_n} "
          f"across {P.task.nunique()} tasks, {len(frames)} leg files")

    rows = []
    for task, g in P.groupby("task"):
        g = g.reset_index(drop=True)
        tk = [toks(p) for p in g.phrase]
        for i in range(len(g)):
            for j in range(i + 1, len(g)):
                pt = pair_type(tk[i], tk[j])
                if pt is None:
                    continue
                cat, fr, to = pt
                lo, hi = (i, j) if g.succ[i] <= g.succ[j] else (j, i)
                z = two_prop_z(g.succ[lo] / 100, g.n[lo], g.succ[hi] / 100, g.n[hi])
                p_lo, p_hi = g.phrase[lo], g.phrase[hi]
                stylematch = (p_lo.islower() == p_hi.islower()
                              and p_lo.rstrip().endswith((".", "?", "!"))
                              == p_hi.rstrip().endswith((".", "?", "!")))
                rows.append({
                    "task": task, "category": cat,
                    "style_consistent": bool(stylematch),
                    "phrase_lo": g.phrase[lo], "succ_lo": round(g.succ[lo], 1),
                    "n_lo": int(g.n[lo]),
                    "phrase_hi": g.phrase[hi], "succ_hi": round(g.succ[hi], 1),
                    "n_hi": int(g.n[hi]),
                    "gap_pp": round(g.succ[hi] - g.succ[lo], 1), "z": round(z, 2),
                    "change": f"{fr} -> {to}" if cat != "slight restructure" else "reorder/determiners",
                })
    C = pd.DataFrame(rows)
    C = C[C.z >= args.min_z].sort_values("gap_pp", ascending=False)
    print(f"minimal pairs found: {len(C)} at z>={args.min_z}")
    print(C.groupby("category").size())
    if args.out:
        C.to_csv(args.out, index=False)
        print("->", args.out)
    with pd.option_context("display.width", 250, "display.max_colwidth", 60):
        print(C.head(args.top).to_string(index=False))


if __name__ == "__main__":
    main()
