#!/usr/bin/env python3
"""Generate results/analysis/pairverify/PAIRVERIFY.md — verified pairs, by task.

Per user spec (2026-09-10): sorted by task with the task's first frame; only
groups with a surviving contrast (two-proportion |z|>=1.96); only phrases at
full grid n (grid x 3 reps); gap size labeled LARGE (>=30pp) / MEDIUM
(15-30) / SMALL (<15); groups whose members differ in leading case or
trailing punctuation beyond the tested edit are flagged as not
format-controlled. Non-significant contrasts live in pair_verdicts.csv only.
"""
import pathlib
import re

import pandas as pd

R = pathlib.Path(__file__).resolve().parents[1]
PV = R / "results/analysis/pairverify"
FRAMES = "../../charts/pairverify_frames"   # relative to PV, committed PNGs

S = pd.read_csv(PV / "phrase_summary.csv").drop_duplicates(subset=["task", "phrase"])
M = pd.read_parquet(R / "results/analysis/bridge_pair_manifest_members.parquet")
U = pd.read_parquet(R / "results/analysis/bridge_pair_manifest.parquet")
V = pd.read_csv(PV / "pair_verdicts.csv")

GRID = dict(zip(U.task, U.grid * 3))
CASE_GROUPS = {"A01_case_period", "A02_brand_case"}
ADVERSARIAL_NOTE = {
    "A18_order_fronted": "fronted word order — reads adversarial, not a benign edit",
    "A20_polite_colororder": "two concepts (politeness + colour order)",
    "A01_case_period": "two format concepts (case + period); A02 isolates case",
}


def gap_label(gap):
    return "LARGE" if gap >= 30 else ("MEDIUM" if gap >= 15 else "SMALL")


def style_sig(p):
    lead = "U" if p[:1].isupper() else "l"
    tail = "." if p.rstrip().endswith((".", "?", "!")) else "-"
    return lead + tail


def bold_variation(phrases):
    toksets = [p.split(" ") for p in phrases]
    if len({len(t) for t in toksets}) == 1:
        return [" ".join(f"**{t}**" if len({ts[i] for ts in toksets}) > 1 else t
                         for i, t in enumerate(toks)) for toks in toksets]
    common = set.intersection(*[set(t) for t in toksets])
    return [" ".join(t if t in common else f"**{t}**" for t in toks) for toks in toksets]


sig_groups = set(V[V.survives].group)

lines = []
E_n = sum(len(pd.read_parquet(f)) for f in PV.glob("episodes_*.parquet"))
lines.append("# Verified minimal pairs — full-grid re-roll (by task)\n")
lines.append(f"Frozen π₀ (rephrase-bridge) on SIMPLER; {E_n:,} episodes, full per-task "
             "layout grid × 3 reps (n=72; basket grid is 60 → n=180), layout id on every "
             "row, success = final state at the 60-step horizon.\n")
lines.append("Brackets are **Wilson 95% score intervals** on the success probability "
             "(≈ ±1.96 SE mid-range, asymmetric near 0/100%) — not ±1 SD. Only groups "
             "with a significant contrast (|z| ≥ 1.96) and phrases at full n are shown. "
             "**Bold** = the tokens that differ. Gap labels: LARGE ≥ 30pp, MEDIUM 15–30pp, "
             "SMALL < 15pp. ⚠ marks groups whose phrases also differ in capitalisation or "
             "trailing punctuation when that is NOT the tested edit (treat the gap as "
             "format-confounded by up to ~10-40pp until style-matched re-roll). "
             "Everything else: `pair_verdicts.csv`, `phrase_summary.csv`.\n")

order = []
for g in M.drop_duplicates(subset=["group"]).group:
    if g in sig_groups:
        t = M[M.group == g].task.iloc[0]
        order.append((t, g))
order.sort()

cur_task = None
for task, g in order:
    mem = M[M.group == g].drop_duplicates(subset=["task", "phrase"])
    rows = mem.merge(S, on=["task", "phrase"], how="left", suffixes=("", "_s"))
    rows = rows[rows.succ.notna()]
    rows = rows[rows.n >= rows.task.map(GRID)]          # full-n only
    if len(rows) < 2:
        continue
    rows = rows.sort_values("succ", ascending=False).reset_index(drop=True)
    short = task.replace("widowx_", "").replace("_clean", "")
    if task != cur_task:
        lines.append(f"\n---\n\n## {short}\n")
        lines.append(f'<img src="{FRAMES}/{task}.png" width="260"/>\n')
        cur_task = task
    gap = rows.succ.iloc[0] - rows.succ.iloc[-1]
    flags = [f"**{gap_label(gap)} gap ({gap:.0f}pp)**"]
    if g not in CASE_GROUPS and len({style_sig(p) for p in rows.phrase}) > 1:
        flags.append("⚠ not format-controlled")
    if g in ADVERSARIAL_NOTE:
        flags.append(f"*{ADVERSARIAL_NOTE[g]}*")
    lines.append(f"`{g}` — " + " · ".join(flags) + "\n")
    lines.append("| phrase | success | n |")
    lines.append("|---|---|---|")
    for m_, (_, r) in zip(bold_variation(list(rows.phrase)), rows.iterrows()):
        lines.append(f"| {m_} | {r.succ:.0f}% [{r.lo95:.0f}–{r.hi95:.0f}] | {int(r.n)} |")
    lines.append("")

hh = V[V.single_concept & V.format_matched]
lines.append(f"\n---\n\nOverall: **{int(hh.survives.sum())} of {len(hh)}** single-concept, "
             f"format-matched contrasts survive at full-grid n "
             f"({int(V.survives.sum())}/{len(V)} of all contrasts).\n")

out = PV / "PAIRVERIFY.md"
out.write_text("\n".join(lines))
print("doc ->", out, f"({len(lines)} lines)")
