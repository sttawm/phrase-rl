#!/usr/bin/env python3
"""A39: parse + clean the human-naturals survey export into a phrase table.

  .venv/bin/python scripts/build_a39_human_naturals.py <formspree_export.json>

Keeps human phrasing VERBATIM (typos included — that is the register being
measured). Removes only invalid entries: placeholder junk ("X", "Na", "Tu"),
and incomplete phrases with no destination (trailing "onto the" / missing
target). Exact duplicate submissions (double form posts) collapse via
drop_duplicates on (task, register, phrase). Writes
results/human_naturals/a39_human_phrases.parquet with columns
(task, register, phrase, age, gender, familiarity, submitted) and prints
everything it removed.
"""
import json
import pathlib
import re
import sys

import pandas as pd

R = pathlib.Path(__file__).resolve().parents[1]
src = pathlib.Path(sys.argv[1])
data = json.load(open(src))

JUNK = {"x", "na", "n/a", "tu", "none", "-", "?"}
# reviewed-by-hand incompletes the pattern rules cannot catch
DROP_EXACT = {"put the object"}   # names no destination
# form-entry duplication glitches, trims confirmed by the user 2026-09-12
TRIM = {"Place the can on its side on the plate place the ca":
            "Place the can on its side on the plate",
        "Put the coke can on the black materialPut the c":
            "Put the coke can on the black material"}
rows, removed = [], []
for sub in data["submissions"]:
    meta = dict(age=sub.get("age", ""), gender=sub.get("gender", ""),
                familiarity=sub.get("ai_robotics_familiarity", ""),
                submitted=sub.get("_date", ""))
    for k, v in sub.items():
        m = re.match(r"(widowx_\w+_clean)_(adult|kid|robot)$", k)
        if not m or not isinstance(v, str):
            continue
        task, register = m.group(1), m.group(2)
        phrase = re.sub(r"\s+", " ", v.replace("\r\n", " ")).strip()
        phrase = TRIM.get(phrase, phrase)
        why = None
        if phrase.lower() in JUNK:
            why = "placeholder junk"
        elif phrase.rstrip(". ").lower() in DROP_EXACT:
            why = "incomplete: no destination (hand-reviewed)"
        elif re.search(r"\b(on|in|into|onto|of|the|a)$", phrase.rstrip(". ").lower().split()[-1] if phrase else ""):
            why = "incomplete: ends mid-preposition/article (no destination)"
        elif len(phrase.split()) < 3:
            why = "too short to name object + destination"
        if why:
            removed.append((task, register, phrase, why))
        else:
            rows.append(dict(task=task, register=register, phrase=phrase, **meta))

df = pd.DataFrame(rows).drop_duplicates(subset=["task", "register", "phrase"])
out = R / "results/human_naturals/a39_human_phrases.parquet"
out.parent.mkdir(parents=True, exist_ok=True)
df.to_parquet(out, index=False)

print(f"kept {len(df)} unique (task, register, phrase) rows "
      f"({len(rows)} raw) from {len(data['submissions'])} submissions")
print(df.groupby('register').size().to_string())
print(f"tasks covered: {df.task.nunique()}/12; per task: "
      f"min {df.groupby('task').size().min()}, max {df.groupby('task').size().max()}")
print("\nREMOVED:")
for t, r, p, w in removed:
    print(f"  [{w}] {t}/{r}: {p!r}")
