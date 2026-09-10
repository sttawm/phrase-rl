#!/usr/bin/env python3
"""Generate results/analysis/pairverify/PAIRVERIFY.md — the full verified-pair
report as a readable document (every contrast, changed tokens bolded).

Categories from group ids; within-group token variation bolded automatically
(case-sensitive, so case-only pairs bold the recased tokens). Non-survivors
reported in their own section — shrinkage is a result, not a failure.
"""
import pathlib
import re

import pandas as pd

R = pathlib.Path(__file__).resolve().parents[1]
PV = R / "results/analysis/pairverify"

S = pd.read_csv(PV / "phrase_summary.csv").drop_duplicates(subset=["task", "phrase"])
M = pd.read_parquet(R / "results/analysis/bridge_pair_manifest_members.parquet")

CAT = [
    (r"A0[12]", "Case and punctuation only"),
    (r"A0[34567]", "Preposition synonyms"),
    (r"A0[89]|A1[012]", "Verb synonyms"),
    (r"A1[34567]", "Noun renames (same referent)"),
    (r"A1[89]|A20", "Restructures (word order, articles, politeness)"),
    (r"Ac\d", "Colour-word insertions"),
    (r"ramekin_ladder", "The destination-noun ladder"),
    (r"stack_cube", "Cube vs block, and verb framing (stack task)"),
    (r"coke_can_on_plate_clean|carrot_on_wheel|carrot_on_keyboard|put_eggplant_in_basket|spoon_on_towel",
     "The B-set families (object/destination nouns, colour, prepositions)"),
]


def cat_of(group):
    for pat, name in CAT:
        if re.match(pat, group) or re.fullmatch(pat, group):
            return name
    return "Other"


def bold_variation(phrases):
    """Bold tokens that vary across the group's phrases (case-sensitive).
    Equal-length groups diff by position; ragged groups bold tokens absent
    from at least one sibling phrase."""
    toksets = [p.split(" ") for p in phrases]
    out = []
    if len({len(t) for t in toksets}) == 1:
        for toks in toksets:
            marked = []
            for i, t in enumerate(toks):
                if len({ts[i] for ts in toksets}) > 1:
                    marked.append(f"**{t}**")
                else:
                    marked.append(t)
            out.append(" ".join(marked))
    else:
        common = set.intersection(*[set(t) for t in toksets])
        for toks in toksets:
            out.append(" ".join(t if t in common else f"**{t}**" for t in toks))
    return out


def ci(r):
    return f"{r.succ:.0f}% [{r.lo95:.0f}–{r.hi95:.0f}]"


lines = []
E_n = sum(len(pd.read_parquet(f)) for f in PV.glob("episodes_*.parquet"))
lines.append("# Verified minimal pairs — full-grid re-roll\n")
lines.append(f"Frozen π₀ (rephrase-bridge) on SIMPLER. Every phrase rolled on its full "
             f"per-task layout grid × 3 reps (n=72; the basket task's grid is 60 → n=180). "
             f"{E_n:,} episodes, layout id recorded on every row. Success = final state at the "
             f"60-step horizon (the grid convention). Wilson 95% intervals. "
             f"Priors were pooled across mixed layouts/reps — the whole point of the re-roll.\n")
lines.append("Survival = two-proportion |z| ≥ 1.96 at full-grid n. "
             "**Bold** marks the tokens that differ within a group. "
             "Full tables: `pair_verdicts.csv`, `phrase_summary.csv`.\n")

groups = M.drop_duplicates(subset=["group"])[["group"]]
bycat = {}
for g in groups.group:
    bycat.setdefault(cat_of(g), []).append(g)

for _, catname in CAT:
    if catname not in bycat:
        continue
    lines.append(f"\n## {catname}\n")
    for g in bycat[catname]:
        mem = M[M.group == g].drop_duplicates(subset=["task", "phrase"])
        rows = mem.merge(S, on=["task", "phrase"], how="left", suffixes=("", "_s"))
        rows = rows[rows.succ.notna()].copy()
        if len(rows) < 2:
            continue
        rows = rows.sort_values("succ", ascending=False).reset_index(drop=True)
        task = rows.task.iloc[0].replace("widowx_", "").replace("_clean", "")
        gap = rows.succ.iloc[0] - rows.succ.iloc[-1]
        marked = bold_variation(list(rows.phrase))
        note = ""
        if g == "A18_order_fronted":
            note = " *(user note: the fronted variant reads adversarial rather than benign — keep out of the 'simple edits' headline)*"
        if g == "A20_polite_colororder":
            note = " *(two concepts: politeness + colour order — flagged, not headline)*"
        if g == "A01_case_period":
            note = " *(two format concepts: case + period; A02 isolates case alone)*"
        lines.append(f"**{task}** — group `{g}`, top-vs-bottom gap **{gap:.0f}pp**{note}\n")
        lines.append("| phrase | success | n |")
        lines.append("|---|---|---|")
        for m_, (_, r) in zip(marked, rows.iterrows()):
            lines.append(f"| {m_} | {ci(r)} | {int(r.n)} |")
        lines.append("")

V = pd.read_csv(PV / "pair_verdicts.csv")
died = V[~V.survives & (V.prior_gap >= 15)].sort_values("prior_gap", ascending=False)
lines.append("\n## Priors that did NOT survive the full grid\n")
lines.append("Shrinkage is the re-roll doing its job — these priors rested on "
             "unbalanced layout compositions or small n:\n")
lines.append("| task | pair | prior gap | full-grid gap | z |")
lines.append("|---|---|---|---|---|")
for _, r in died.head(12).iterrows():
    lines.append(f"| {r.task} | \"{r.phrase_lo}\" vs \"{r.phrase_hi}\" | "
                 f"{r.prior_gap:.0f}pp | {r.new_gap:.0f}pp | {r.z:.1f} |")
lines.append("")
lines.append(f"\nOverall: **{int(V[V.single_concept & V.format_matched].survives.sum())} of "
             f"{len(V[V.single_concept & V.format_matched])}** single-concept, format-matched "
             f"contrasts survive; {int(V.survives.sum())}/{len(V)} of all contrasts.\n")

out = PV / "PAIRVERIFY.md"
out.write_text("\n".join(lines))
print("doc ->", out, f"({len(lines)} lines)")
