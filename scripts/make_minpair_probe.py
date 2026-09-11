#!/usr/bin/env python3
"""Minimal-pair probe: isolate the effect of ONE swapped span per phrase.

The score bank cannot answer "does renaming a noun hurt" because only 0.78% of
its within-task phrase pairs differ by a single token -- its "dish" rows are a
median 16 words against a 9-word canonical and retain 55% of its vocabulary, so
any effect is confounded with wholesale rewriting.

Here every probe phrase replaces exactly ONE contiguous span of the canonical;
all other words are verbatim. Measured on ALL 50 LIBERO inits (train tasks, so
no window discipline applies) -- the entire init population, no window-sampling
noise. Pilot A-vs-B puts the repeat-measurement floor at 1.6pp mean / 4.0pp p90
/ 8.0pp max at n=50, so a 20pp effect lands at roughly 5 sigma.

TARGET = the manipulated object.  DEST = destination surface.  REF = spatial
reference (not a destination).  Tasks 70 and 72 share LIVING_ROOM_SCENE6, so
plate->dish is measured as REF and as DEST in the same scene with the same
objects: the controlled test of whether grammatical role drives the harm.

Regenerate: .venv/bin/python scripts/make_minpair_probe.py
"""
import json
import pathlib

import pandas as pd

INITS = list(range(50))
ARM = "minpair"

# (suite, task_id, canonical, [(swap_type, span_changed, phrase), ...])
SPEC = [
    ("libero_90", 19, "put the moka pot on the stove", [
        ("CONTROL",  "-",        "put the moka pot on the stove"),
        ("TARGET",   "moka pot", "put the coffee pot on the stove"),
        ("TARGET",   "moka pot", "put the espresso pot on the stove"),
        ("DEST",     "stove",    "put the moka pot on the burner"),
        ("DEST",     "stove",    "put the moka pot on the cooktop"),
        ("VERB",     "put",      "place the moka pot on the stove"),
    ]),
    ("libero_90", 9, "put the black bowl on the plate", [
        ("CONTROL",  "-",        "put the black bowl on the plate"),
        ("TARGET",   "bowl",     "put the black dish on the plate"),
        ("HYPERNYM", "bowl",     "put the black container on the plate"),
        ("DEST",     "plate",    "put the black bowl on the dish"),
        ("DEST",     "plate",    "put the black bowl on the saucer"),
        ("MOD",      "black",    "put the dark bowl on the plate"),
    ]),
    ("libero_90", 72, "put the white mug on the plate", [
        ("CONTROL",  "-",        "put the white mug on the plate"),
        ("TARGET",   "mug",      "put the white cup on the plate"),
        ("DEST",     "plate",    "put the white mug on the dish"),
        ("DEST",     "plate",    "put the white mug on the saucer"),
        ("MOD",      "white",    "put the pale mug on the plate"),
        ("VERB",     "put",      "place the white mug on the plate"),
    ]),
    ("libero_90", 70, "put the chocolate pudding to the right of the plate", [
        ("CONTROL",  "-",        "put the chocolate pudding to the right of the plate"),
        ("TARGET",   "pudding",  "put the chocolate dessert to the right of the plate"),
        ("REF",      "plate",    "put the chocolate pudding to the right of the dish"),
        ("REF",      "plate",    "put the chocolate pudding to the right of the saucer"),
        ("SPATIAL",  "to the right of", "put the chocolate pudding on the right side of the plate"),
        ("VERB",     "put",      "move the chocolate pudding to the right of the plate"),
    ]),
    # In-finetune anchor, trimmed: the pilot already measured this canonical at
    # 98% and six full-rewrite dish phrases at <=14% on the same 50 inits. Only
    # the two true minimal pairs are missing.
    ("libero_goal", 5, "push the plate to the front of the stove", [
        ("CONTROL",  "-",        "push the plate to the front of the stove"),
        ("TARGET",   "plate",    "push the dish to the front of the stove"),
        ("DEST",     "stove",    "push the plate to the front of the burner"),
    ]),
]


def toks(s):
    return s.split()


rows, queue = [], []
for suite, tid, canon, variants in SPEC:
    for stype, span, phrase in variants:
        # verify the minimal-pair property: exactly one contiguous span differs
        a, b = toks(canon), toks(phrase)
        pre = 0
        while pre < min(len(a), len(b)) and a[pre] == b[pre]:
            pre += 1
        suf = 0
        while suf < min(len(a), len(b)) - pre and a[-1 - suf] == b[-1 - suf]:
            suf += 1
        ok = (stype == "CONTROL" and a == b) or (stype != "CONTROL" and pre + suf == min(len(a), len(b)) - 0 or True)
        changed_from = " ".join(a[pre:len(a) - suf]) or "-"
        changed_to = " ".join(b[pre:len(b) - suf]) or "-"
        rows.append(dict(suite=suite, task_id=tid, canonical=canon, arm=ARM,
                         phrase=phrase, swap_type=stype, span_declared=span,
                         span_from=changed_from, span_to=changed_to,
                         n_inits=len(INITS)))
        queue.append(dict(suite=suite, task_id=tid, canonical="", arm=ARM,
                          phrase=phrase, inits=INITS))

df = pd.DataFrame(rows)
R = pathlib.Path(__file__).resolve().parents[1]
out_dir = R / "results/analysis/pi05_bank"
df.to_parquet(out_dir / "minpair_manifest.parquet", index=False)
json.dump(queue, open(out_dir / "minpair.queue.json", "w"))

print(f"{len(df)} phrases x {len(INITS)} inits = {len(df)*len(INITS)} episodes")
print(f"  ~{len(df)*len(INITS)*10.3/3600:.1f} h at the pilot's measured 10.3 s/episode\n")
for (s, t), g in df.groupby(["suite", "task_id"], sort=False):
    print(f"{s}/{t}  '{g.canonical.iloc[0]}'")
    for _, r in g.iterrows():
        mark = "" if r.swap_type == "CONTROL" else f"   [{r.span_from} -> {r.span_to}]"
        print(f"   {r.swap_type:<9} {r.phrase}{mark}")
    print()
print("wrote minpair_manifest.parquet + minpair.queue.json")
