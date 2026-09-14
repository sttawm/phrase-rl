#!/usr/bin/env python3
"""Enumerate every single-edit phrase pair measured on the full LIBERO init population.

  .venv/bin/python scripts/build_libero_single_edit_pairs.py

Sources (per-episode jsonl from bank_eval.py): results/analysis/pi05_bank/
{roll50, round2..round8, pilot_raw/pilotA, pilot_raw/pilotC, pilot_raw/minpair_partial}.
pilotB re-measured pilotA's 21 phrases on the same inits; it is kept only as a
replicate column, never pooled. A phrase enters the pool when it has exactly one
episode on each of inits 0-49 (n=50). Within a task, two phrases form a pair
when their whitespace tokens differ in exactly ONE contiguous span (difflib
opcodes: one replace/insert/delete). Case and punctuation count as edits.

Outputs results/analysis/pi05_bank/single_edit_pairs.{parquet,csv} (one row per
pair) and n50_phrases.parquet (one row per phrase at n=50). Per pair: success of
each side, delta (b - a), two-proportion z p, Fisher exact p, and -- because both
phrases ran on the same 50 inits with the same seed -- the paired 2x2
(both / a_only / b_only / neither) with an exact McNemar p. phrase_a is the task
canonical when one side is; otherwise the alphabetically first. `designed`
marks pairs that were a (canonical, edit) row in a round manifest, with that
manifest's group/rationale. Split labels: splits.json (seed 20260901) and
eval_set.json (2026-09-04 FINAL_EVAL strata), so test tasks can be filtered.
"""
import difflib
import glob
import json
import math
import pathlib
import re
from math import comb

import pandas as pd

R = pathlib.Path(__file__).resolve().parents[1]
B = R / "results/analysis/pi05_bank"

SOURCES = [("roll50", "roll50/*.jsonl")] + [(f"round{i}", f"round{i}/*.jsonl") for i in range(2, 9)] + [
    ("pilotA", "pilot_raw/pilotA.jsonl"), ("pilotC", "pilot_raw/pilotC.jsonl"),
    ("minpair_partial", "pilot_raw/minpair_partial.jsonl")]


def load(pattern, source):
    rows = []
    for f in sorted(glob.glob(str(B / pattern))):
        for line in open(f):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("success") is None:
                continue
            rows.append((source, r["suite"], int(r["task_id"]), r.get("canonical"), r["phrase"],
                         int(r["init"]), int(r["success"])))
    return pd.DataFrame(rows, columns=["source", "suite", "task_id", "canonical", "phrase", "init", "success"])


ep = pd.concat([load(p, s) for s, p in SOURCES], ignore_index=True)
ep = ep.drop_duplicates(["suite", "task_id", "phrase", "init"], keep="first")   # source priority = list order
ep["task"] = ep.suite + "/" + ep.task_id.astype(str)
rep = load("pilot_raw/pilotB.jsonl", "pilotB")
rep["task"] = rep.suite + "/" + rep.task_id.astype(str)

# canonical string per task (from the episode records; every round carries it)
canon = ep.dropna(subset=["canonical"]).groupby("task").canonical.agg(lambda s: s.mode().iloc[0]).to_dict()

ph = (ep.groupby(["task", "phrase"])
        .agg(n=("success", "size"), k=("success", "sum"), inits=("init", lambda s: len(set(s))),
             source=("source", "first")).reset_index())
ph = ph[(ph.n == 50) & (ph.inits == 50)].copy()
ph["succ"] = 100 * ph.k / ph.n
ph["is_canonical"] = [p == canon.get(t) for t, p in zip(ph.task, ph.phrase)]
repk = rep.groupby(["task", "phrase"]).success.agg(["size", "sum"])
ph["succ_replicate"] = [100 * repk.loc[(t, p), "sum"] / repk.loc[(t, p), "size"] if (t, p) in repk.index else float("nan")
                        for t, p in zip(ph.task, ph.phrase)]

# ---- split labels ----------------------------------------------------------
splits = json.load(open(B / "splits.json"))
split_of = {}
for name in ("train", "val", "sealed_test"):
    for t in splits[name]:
        split_of[f"{t['suite']}/{t['task_id']}"] = name
evalset = json.load(open(B / "eval_set.json"))
stratum_of = {k.replace(":", "/"): v for k, v in evalset["strata"].items()}

# ---- designed pairs from the round manifests ---------------------------------
designed = {}
NOTE_COLS = ("group", "strand", "label", "tests", "hypernym", "rationale", "swap_type", "arm")
for mf in sorted(glob.glob(str(B / "round*_manifest.parquet"))) + [str(B / "roll50_manifest.parquet"), str(B / "minpair_manifest.parquet")]:
    m = pd.read_parquet(mf)
    rnd = pathlib.Path(mf).name.split("_")[0]
    if "task" not in m.columns:
        m["task"] = m.suite + "/" + m.task_id.astype(str)
    for _, r in m.iterrows():
        info = "; ".join(str(r[c]) for c in NOTE_COLS if c in m.columns and pd.notna(r[c]) and str(r[c]))
        if pd.notna(r.get("canonical")) and r.phrase != r.canonical:
            designed[(r.task, r.canonical, r.phrase)] = f"{rnd}: {info}"

# ---- pair enumeration --------------------------------------------------------
COLORS = {"black", "white", "grey", "gray", "red", "yellow", "green", "blue", "orange", "purple", "brown", "wooden"}
PREPS = {"on", "onto", "in", "into", "inside", "at", "to", "of", "from", "atop", "upon", "up", "over", "top", "next", "beside", "near", "by", "the", "a", "an"}
VERBS = {"put", "place", "pick", "lift", "grab", "take", "move", "set", "stack", "turn", "switch", "fire", "start", "open", "close", "push", "pull", "slide", "shut", "flip", "shove", "shift", "get", "bring", "drop", "position", "transfer", "carry", "deposit", "lay", "rest", "insert", "remove", "activate", "ignite", "light", "power", "can", "you", "please", "i", "want", "should", "be", "goes", "go"}


def single_edit(a, b):
    ta, tb = a.split(), b.split()
    ops = [op for op in difflib.SequenceMatcher(a=ta, b=tb, autojunk=False).get_opcodes() if op[0] != "equal"]
    if len(ops) != 1:
        return None
    tag, i1, i2, j1, j2 = ops[0]
    return tag, " ".join(ta[i1:i2]), " ".join(tb[j1:j2]), i1


DETS = {"the", "a", "an", "that", "this", "those", "these", "it", "its"}


def classify(a, b, tag, sa, sb):
    """Coarse label for the changed span (lexicon heuristic; the spans themselves
    are in span_a/span_b for anything finer)."""
    if a.lower() == b.lower():
        return "case"
    if re.sub(r"[.!?,]", "", a) == re.sub(r"[.!?,]", "", b):
        return "punctuation"
    if a.lower() != b.lower() and (a != a.lower() or b != b.lower()):
        se = single_edit(a.lower(), b.lower())      # separate a case change from the word change
        if se:
            tag, sa, sb, _ = se
    words = set((sa + " " + sb).lower().split()) - {"the", "a", "an"}
    if not words or words <= DETS:
        return "determiner"
    if words <= COLORS:
        return "color"
    if words <= PREPS:
        return "preposition/particle"
    if words <= VERBS | PREPS | DETS and words & VERBS:
        return "verb/frame"
    if tag == "insert":
        return "words added"
    if tag == "delete":
        return "words removed"
    return "noun/other"


def z_p(k1, n1, k2, n2):
    p1, p2 = k1 / n1, k2 / n2
    x = (k1 + k2) / (n1 + n2)
    se = math.sqrt(x * (1 - x) * (1 / n1 + 1 / n2)) or 1e-9
    return math.erfc(abs(p1 - p2) / se / math.sqrt(2))


def fisher_p(k1, n1, k2, n2):
    a, b_, c, d = k1, n1 - k1, k2, n2 - k2
    r1, c1, n = a + b_, a + c, a + b_ + c + d
    pr = lambda x: comb(r1, x) * comb(n - r1, c1 - x) / comb(n, c1)
    po = pr(a)
    return min(1.0, sum(pr(x) for x in range(max(0, c1 - (n - r1)), min(r1, c1) + 1) if pr(x) <= po * (1 + 1e-9)))


def mcnemar_p(b_only, a_only):
    n = a_only + b_only
    if n == 0:
        return 1.0
    k = min(a_only, b_only)
    return min(1.0, 2 * sum(comb(n, i) for i in range(0, k + 1)) / 2 ** n)


succ_by_init = {(t, p): dict(zip(g.init, g.success)) for (t, p), g in ep.groupby(["task", "phrase"])}
rows = []
for task, g in ph.groupby("task"):
    phrases = sorted(g.phrase)
    for i in range(len(phrases)):
        for j in range(i + 1, len(phrases)):
            a, b = phrases[i], phrases[j]
            if b == canon.get(task):
                a, b = b, a
            se = single_edit(a, b)
            if se is None:
                continue
            tag, sa, sb, pos = se
            ka, kb = int(g[g.phrase == a].k.iloc[0]), int(g[g.phrase == b].k.iloc[0])
            va, vb = succ_by_init[(task, a)], succ_by_init[(task, b)]
            both = sum(1 for i_ in range(50) if va[i_] and vb[i_])
            a_only = sum(1 for i_ in range(50) if va[i_] and not vb[i_])
            b_only = sum(1 for i_ in range(50) if vb[i_] and not va[i_])
            key = (task, a, b)
            rows.append(dict(
                task=task, suite=task.split("/")[0], task_id=int(task.split("/")[1]),
                canonical=canon.get(task), split_20260901=split_of.get(task, "unassigned"),
                eval_set_20260904=stratum_of.get(task, ""),
                phrase_a=a, phrase_b=b, a_is_canonical=(a == canon.get(task)),
                edit_type=tag, span_a=sa, span_b=sb, span_pos=pos,
                edit_class=classify(a, b, tag, sa, sb),
                succ_a=2 * ka, succ_b=2 * kb, delta=2 * (kb - ka),
                p_z=z_p(ka, 50, kb, 50), p_fisher=fisher_p(ka, 50, kb, 50),
                both=both, a_only=a_only, b_only=b_only, neither=50 - both - a_only - b_only,
                p_mcnemar=mcnemar_p(b_only, a_only),
                designed=key in designed or (task, b, a) in designed,
                design_note=designed.get(key, designed.get((task, b, a), "")),
                source_a=g[g.phrase == a].source.iloc[0], source_b=g[g.phrase == b].source.iloc[0]))
pairs = pd.DataFrame(rows).sort_values(["task", "delta"]).reset_index(drop=True)
pairs["significant_18pp"] = (pairs.delta.abs() >= 18) & (pairs.p_z < 0.05)

ph.to_parquet(B / "n50_phrases.parquet", index=False)
pairs.to_parquet(B / "single_edit_pairs.parquet", index=False)
pairs.to_csv(B / "single_edit_pairs.csv", index=False)
print(f"episodes {len(ep)} | phrases at n=50 {len(ph)} on {ph.task.nunique()} tasks | pairs {len(pairs)} "
      f"(designed {pairs.designed.sum()}, |delta|>=18 & p<0.05: {pairs.significant_18pp.sum()})")
print(pairs.groupby("edit_class").agg(pairs=("delta", "size"), sig=("significant_18pp", "sum"),
                                      mean_abs_delta=("delta", lambda d: d.abs().mean())).round(1).to_string())
print(pairs.groupby("split_20260901").agg(pairs=("delta", "size"), tasks=("task", "nunique"), sig=("significant_18pp", "sum")).to_string())
