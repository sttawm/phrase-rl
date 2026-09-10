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
    (r"A0[12]", "case / punctuation"), (r"A0[34567]", "preposition"),
    (r"A0[89]|A1[012]", "verb"), (r"A1[34567]", "object noun"),
    (r"A18", "word order"), (r"A19", "articles"), (r"A20", "politeness"),
    (r"Ac\d+m?_obj", "object colour"), (r"Ac\d+m?_dest", "destination colour"),
    (r"ramekin_ladder", "destination noun (ladder)"),
    (r"Rn\d", "object noun rename"),
    (r"stack_cube", "object noun / verb (family)"),
    (r"carrot_on_wheel|carrot_on_keyboard", "destination noun (family)"),
    (r"coke_can_on_plate_clean", "object / destination noun (family)"),
    (r"put_eggplant_in_basket|spoon_on_towel", "colour / preposition (family)"),
]


def cat(group):
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


def marked_phrase(phrase, siblings, color):
    """Wrap tokens that vary across the group in a colorbox."""
    toksets = [p.split(" ") for p in siblings]
    toks = phrase.split(" ")
    same_len = len({len(t) for t in toksets}) == 1
    common = set.intersection(*[set(t) for t in toksets])
    out = []
    for i, t in enumerate(toks):
        varies = (len({ts[i] for ts in toksets}) > 1) if same_len else (t not in common)
        if varies:
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

tex = [r"""\documentclass[10pt]{article}
\usepackage[margin=1.6cm]{geometry}
\usepackage{graphicx,xcolor,multicol,needspace}
\usepackage[T1]{fontenc}
\definecolor{hi}{RGB}{212,241,212}
\definecolor{lo}{RGB}{250,214,214}
\definecolor{hipct}{RGB}{22,122,39}
\definecolor{lopct}{RGB}{178,32,32}
\setlength{\parindent}{0pt}
\setlength{\columnsep}{22pt}
\newcommand{\pct}[2]{\makebox[2.4em][r]{\textcolor{#1}{\bfseries #2\%}}}
\begin{document}
{\Large\bfseries Verified minimal pairs --- frozen $\pi_0$, SIMPLER Bridge}\\[2pt]
{\small Full per-task layout grid $\times$ 3 reps (n=72 per phrase; basket grid 60 $\to$ n=180); success = final state at the 60-step horizon; p = two-proportion, top vs.\ bottom. Highlight marks the changed span.}
\vspace{6pt}\hrule\vspace{8pt}
\begin{multicols}{2}\footnotesize"""]

for task in sorted(blocks):
    short = task.replace("widowx_", "").replace("_clean", "").replace("_", r"\_")
    img = FRAMES / f"{task}.png"
    tex.append(r"\needspace{14\baselineskip}")
    tex.append(r"\begin{center}\includegraphics[width=0.72\linewidth]{frames/" + task + r".png}\end{center}")
    tex.append(r"{\normalsize\bfseries\ttfamily " + short + r"}\\[3pt]")
    for g, rows in blocks[task]:
        top, bot = rows.iloc[0], rows.iloc[-1]
        delta = top.succ - bot.succ
        p = two_prop_p(top.succ / 100, top.n, bot.succ / 100, bot.n)
        pstr = f"p={p:.4f}" if p >= 5e-5 else "p<0.0001"
        tex.append(r"\Needspace{" + str(len(rows) + 2) + r"\baselineskip}")
        tex.append(r"{\bfseries " + esc(cat(g)) + r"}\enspace " +
                   f"{delta:+.0f} pp\\," + r"$\cdot$\," + f" {pstr}" + r"\\[1pt]")
        sibs = list(rows.phrase)
        for j, (_, r) in enumerate(rows.iterrows()):
            color = "hi" if j == 0 else ("lo" if j == len(rows) - 1 else "white")
            pcol = "hipct" if j == 0 else ("lopct" if j == len(rows) - 1 else "black")
            tex.append(r"\hangindent=2.6em \pct{" + pcol + r"}{" + f"{r.succ:.0f}" + r"}~\texttt{"
                       + marked_phrase(r.phrase, sibs, color) + r"}\\")
        tex.append(r"[6pt]")
    tex.append(r"\vspace{4pt}\hrule\vspace{8pt}")

tex.append(r"\end{multicols}\end{document}")

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
