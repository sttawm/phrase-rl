#!/usr/bin/env python3
"""Generate results/analysis/pairverify/pairverify_pairs.tex (+ compile PDF).

Format per user spec (2026-09-10, modeled on the LIBERO pair doc): two-column;
per task: first-frame image, then each significant contrast as
  <category>  <delta> pp  ·  p=<two-proportion p>
  GREEN%  phrase with the changed span highlighted green
  RED%    phrase with the changed span highlighted red
No canonical line, no priors. Ladders show all rungs (top green, bottom red,
middle neutral); delta and p are top-vs-bottom. Only surviving groups at full
grid n. Success = final state @ 60 steps; n=72/phrase (basket 180).
"""
import math
import pathlib
import re
import subprocess

import pandas as pd

R = pathlib.Path(__file__).resolve().parents[1]
PV = R / "results/analysis/pairverify"
FRAMES = R / "results/charts/pairverify_frames"

S = pd.read_csv(PV / "phrase_summary.csv").drop_duplicates(subset=["task", "phrase"])
M = pd.read_parquet(R / "results/analysis/bridge_pair_manifest_members.parquet")
U = pd.read_parquet(R / "results/analysis/bridge_pair_manifest.parquet")
V = pd.read_csv(PV / "pair_verdicts.csv")
GRID = dict(zip(U.task, U.grid * 3))

CATLABEL = [
    (r"A0[12]", "case/punct."), (r"A0[34567]", "preposition"),
    (r"A0[89]|A1[012]", "verb"), (r"A1[34567]", "object noun"),
    (r"A18", "word order"), (r"A19", "articles"), (r"A20", "politeness"),
    (r"Ac\d+m?_obj", "object colour"), (r"Ac\d+m?_dest", "destination colour"),
    (r"ramekin_ladder", "noun ladder"),
    (r"Rn\d", "object noun rename"),
    (r"stack_cube", "noun/verb family"),
    (r"carrot_on_wheel|carrot_on_keyboard", "noun family"),
    (r"coke_can_on_plate_clean", "noun family"),
    (r"put_eggplant_in_basket|spoon_on_towel", "colour/prep family"),
]


def cat(group):
    if group.startswith("__disp__"):
        return group[8:]
    for pat, name in CATLABEL:
        if re.match(pat, group):
            return name
    return "edit"


def esc(s):
    return (s.replace("\\", r"\textbackslash{}").replace("&", r"\&")
             .replace("%", r"\%").replace("$", r"\$").replace("#", r"\#")
             .replace("_", r"\_").replace("{", r"\{").replace("}", r"\}"))


def two_prop_p(p1, n1, p2, n2):
    x = (p1 * n1 + p2 * n2) / (n1 + n2)
    se = math.sqrt(x * (1 - x) * (1 / n1 + 1 / n2)) or 1e-9
    z = abs(p1 - p2) / se
    return math.erfc(z / math.sqrt(2))


def _mark_sets(siblings):
    """Per-phrase sets of token indices to highlight.
    Pairs: difflib alignment; an insert/delete highlights the inserted tokens
    AND the next aligned token on BOTH sides, so word additions mark something
    reasonable in each sentence. Larger groups: tokens not common to all."""
    import difflib
    toksets = [p.split(" ") for p in siblings]
    if len(toksets) == 2:
        a, b = toksets
        ha, hb = set(), set()
        sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == "equal":
                continue
            ha.update(range(i1, i2)); hb.update(range(j1, j2))
            if tag in ("insert", "delete"):
                if i2 < len(a): ha.add(i2)
                if j2 < len(b): hb.add(j2)
        return [ha, hb]
    same_len = len({len(t) for t in toksets}) == 1
    common = set.intersection(*[set(t) for t in toksets])
    outs = []
    for toks in toksets:
        if same_len:
            outs.append({i for i, t in enumerate(toks)
                         if len({ts[i] for ts in toksets}) > 1})
        else:
            outs.append({i for i, t in enumerate(toks) if t not in common})
    return outs


def marked_phrase(phrase, siblings, color):
    idx = siblings.index(phrase)
    hi = _mark_sets(siblings)[idx]
    out = []
    for i, t in enumerate(phrase.split(" ")):
        if i in hi:
            out.append(r"\colorbox{" + color + r"}{\strut " + esc(t) + "}")
        else:
            out.append(esc(t))
    return " ".join(out)


sig = set(V[V.survives].group)
blocks = {}
for g in M.drop_duplicates(subset=["group"]).group:
    if g not in sig:
        continue
    mem = M[M.group == g].drop_duplicates(subset=["task", "phrase"])
    rows = mem.merge(S, on=["task", "phrase"], how="left", suffixes=("", "_s"))
    rows = rows[rows.succ.notna()]
    rows = rows[rows.n >= rows.task.map(GRID)]
    if len(rows) < 2:
        continue
    rows = rows.sort_values("succ", ascending=False).reset_index(drop=True)
    blocks.setdefault(rows.task.iloc[0], []).append((g, rows))

# ---- display curation (user edits 2026-09-10; data untouched) --------------
DISPLAY_EDITS = {
    "carrot_on_keyboard_clean": [
        ("object+destination noun", ["put the carrot on the black keyboard",
                                     "put the carrot on the keys"])],
    "carrot_on_wheel_clean": [
        ("destination noun", ["put the carrot on the black wheel",
                              "put the carrot on the wheel",
                              "put the carrot on the tire"])],
    "coke_can_on_plate_clean": [
        ("object noun", ["put the coke on the plate", "put the can on the plate"]),
        ("destination colour", ["put the coke can on the plate",
                                "put the coke can on the yellow plate"]),
        ("destination noun", ["put the coke can on the plate",
                              "put the coke can on the dish"])],
    "A17_noun_dish_plate": [],   # duplicate of the plate-vs-dish split above
    "ramekin_ladder": [
        ("noun ladder", ["place the red cola can inside the white container",
                         "place the red cola can inside the white dish",
                         "place the red cola can inside the white cup",
                         "place the red cola can inside the white ramekin"])],
    "spoon_on_towel": [
        ("colour/prep pair", ["put the spoon atop the towel",
                              "put the spoon on the blue towel"])],
    "stack_cube": [
        ("object noun", ["put the green cube on the yellow cube",
                         "put the green block on the yellow block"]),
        ("verb", ["put the green cube on the yellow cube",
                  "stack the green cube on the yellow cube"]),
        ("object noun", ["green cube on yellow cube",
                         "green block on yellow block"])],
}
for task in list(blocks):
    newlist = []
    for g, rows in blocks[task]:
        if g not in DISPLAY_EDITS:
            newlist.append((g, rows))
            continue
        for k, (label, phrases) in enumerate(DISPLAY_EDITS[g]):
            sub = rows[rows.phrase.isin(phrases)].copy()
            sub = sub.sort_values("succ", ascending=False).reset_index(drop=True)
            assert len(sub) == len(phrases), f"display edit {g}/{label}: missing phrase"
            newlist.append((f"__disp__{label}", sub))
    blocks[task] = newlist

tex = [r"""\documentclass[10pt]{article}
\usepackage[left=0.75cm,right=0.75cm,top=0.6cm,bottom=0.55cm]{geometry}
\usepackage{graphicx,xcolor,multicol,needspace}
\usepackage[T1]{fontenc}
\definecolor{hi}{RGB}{212,241,212}
\definecolor{lo}{RGB}{250,214,214}
\definecolor{hipct}{RGB}{22,122,39}
\definecolor{lopct}{RGB}{178,32,32}
\setlength{\parindent}{0pt}
\setlength{\columnsep}{12pt}
\setlength{\fboxsep}{1.1pt}
\newcommand{\pct}[2]{\makebox[2.1em][r]{\textcolor{#1}{\bfseries #2\%}}}
\begin{document}
{\normalsize\bfseries Verified minimal pairs --- frozen $\pi_0$, SIMPLER Bridge}\enspace
{\scriptsize Full layout grid $\times$ 3 reps (n=72/phrase; basket 180); success = final state @ 60 steps; p: two-proportion z, best vs worst phrase (ladder extremes selected within family --- descriptive). Green = best phrasing, red = worse; highlight = changed span. Scenes on page 2.}
\vspace{2pt}\hrule\vspace{2.5pt}
\begin{multicols}{2}\scriptsize\setlength{\baselineskip}{7.7pt}"""]

first = True
for task in sorted(blocks):
    short = task.replace("widowx_", "").replace("_clean", "").replace("_", r"\_")
    if not first:
        tex.append(r"\vspace{0.5pt}{\color{black!35}\hrule height 0.5pt}\vspace{1.5pt}")
    first = False
    tex.append(r"{\ttfamily\bfseries " + short + r"}\\[0.6pt]")
    for g, rows in blocks[task]:
        top, bot = rows.iloc[0], rows.iloc[-1]
        delta = top.succ - bot.succ
        p = two_prop_p(top.succ / 100, top.n, bot.succ / 100, bot.n)
        pstr = f"p={p:.4f}" if p >= 5e-5 else "p<0.0001"
        tex.append(r"{\leftskip=1.1em")
        tex.append(r"{\bfseries " + esc(cat(g)) + r"}\enspace " +
                   f"{delta:+.0f}\\,pp\\," + r"$\cdot$\," + f" {pstr}" + r"\\[1pt]")
        sibs = list(rows.phrase)
        for j, (_, r) in enumerate(rows.iterrows()):
            color = "hi" if j == 0 else "lo"
            pcol = "hipct" if j == 0 else "lopct"
            tex.append(r"\hangindent=2.4em \pct{" + pcol + r"}{" + f"{r.succ:.0f}" + r"}~\texttt{"
                       + marked_phrase(r.phrase, sibs, color) + r"}\\")
        tex.append(r"[0.9pt]\par}")

tex.append(r"\end{multicols}")
tex.append(r"\newpage")
tex.append(r"{\large\bfseries Task scenes}\\[4pt]")
tex.append(r"\begin{multicols}{3}\footnotesize\centering")
for task in sorted(blocks):
    short = task.replace("widowx_", "").replace("_clean", "").replace("_", r"\_")
    tex.append(r"\includegraphics[width=0.9\linewidth]{frames/" + task + r".png}\\")
    tex.append(r"{\ttfamily " + short + r"}\\[8pt]")
tex.append(r"\end{multicols}")
tex.append(r"\end{document}")

out = PV / "pairverify_pairs.tex"
out.write_text("\n".join(tex))
print("tex ->", out)
import shutil
if not shutil.which("pdflatex"):
    raise SystemExit("no pdflatex locally; compile elsewhere")
r1 = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-output-directory", str(PV),
                     str(out)], capture_output=True, text=True)
r2 = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-output-directory", str(PV),
                     str(out)], capture_output=True, text=True)
pdf = PV / "pairverify_pairs.pdf"
print("pdf ->", pdf, "exists:", pdf.exists())
if not pdf.exists():
    print("\n".join(r1.stdout.splitlines()[-25:]))
