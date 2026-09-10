#!/usr/bin/env python3
"""Generate results/analysis/pairverify_libero/pairverify_libero_pairs.tex (+PDF).

LIBERO twin of scripts/make_pairverify_tex.py -- layout, spacing, colors and
markup are reused verbatim; only the data loading differs. Two-column listing on
page 1, scene frames in a 3-column grid on page 2.

Data: the n=50 rounds in results/analysis/pi05_bank/{roll50,round2}/*.jsonl.
Every phrase is measured on the full LIBERO init population (inits 0-49), so
n=50/phrase and there is no window-composition question. Success is the
environment's own success flag at episode end, never a predicate on text.
Groups are kept only when |z| >= 1.96 between the group's best and worst phrase.
"""
import glob
import json
import math
import pathlib
import re
import shutil
import subprocess

import pandas as pd

R = pathlib.Path(__file__).resolve().parents[1]
PV = R / "results/analysis/pairverify_libero"
BANK = R / "results/analysis/pi05_bank"
PV.mkdir(parents=True, exist_ok=True)
(PV / "frames").mkdir(exist_ok=True)

# ---- data ------------------------------------------------------------------
rows = [json.loads(l)
        for f in glob.glob(str(BANK / "roll50/roll*.jsonl")) + glob.glob(str(BANK / "round2/r2roll*.jsonl"))
        for l in open(f)]
d = pd.DataFrame(rows)
d["task"] = d.suite + "/" + d.task_id.astype(str)
S = (d.groupby(["task", "phrase"]).success
     .agg(succ=lambda s: s.mean() * 100, n="size").reset_index())
FULL_N = 50

# Curated groups: each is one task, one edit dimension, phrases in any order.
GROUPS = [
    ("libero_goal/7", "source noun", [
        "switch on the stove", "switch on the range", "switch on the hob",
        "switch on the cooktop", "switch on the heating element",
        "switch on the griddle", "switch on the hot plate"]),
    ("libero_goal/7", "verb", ["turn on the burners", "start the burners"]),
    ("libero_goal/1", "destination noun", [
        "place the grey bowl on the electric burner",
        "place the grey bowl on the hot plate"]),
    ("libero_90/44", "verb", ["fire up the stove", "turn on the stove"]),
    ("libero_goal/5", "source noun", [
        "slide that plate over to the front of the stove",
        "slide that dish over to the front of the stove"]),
    ("libero_goal/8", "destination noun", [
        "put the bowl on the plate", "put the bowl on the dish"]),
    ("libero_90/10", "preposition", [
        "place the black bowl on top of the cabinet",
        "place the black bowl up onto the cabinet"]),
    ("libero_goal/9", "destination noun", [
        "put the wine bottle on the rack", "put the wine bottle on the stand"]),
    ("libero_90/38", "verb", [
        "pick up the right moka pot and place it on the stove",
        "lift the right moka pot and place it on the stove"]),
]


def esc(s):
    return (s.replace("\\", r"\textbackslash{}").replace("&", r"\&")
             .replace("%", r"\%").replace("$", r"\$").replace("#", r"\#")
             .replace("_", r"\_").replace("{", r"\{").replace("}", r"\}"))


def two_prop_p(p1, n1, p2, n2):
    x = (p1 * n1 + p2 * n2) / (n1 + n2)
    se = math.sqrt(x * (1 - x) * (1 / n1 + 1 / n2)) or 1e-9
    z = abs(p1 - p2) / se
    return math.erfc(z / math.sqrt(2)), z


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


# ---- assemble surviving groups ---------------------------------------------
blocks = {}
for task, label, phrases in GROUPS:
    sub = S[(S.task == task) & (S.phrase.isin(phrases))]
    sub = sub[sub.n >= FULL_N]
    if len(sub) < 2:
        continue
    sub = sub.sort_values("succ", ascending=False).reset_index(drop=True)
    top, bot = sub.iloc[0], sub.iloc[-1]
    p, z = two_prop_p(top.succ / 100, top.n, bot.succ / 100, bot.n)
    if z < 1.96:
        continue
    blocks.setdefault(task, []).append((label, sub, top.succ - bot.succ, p))

TASKORDER = sorted(blocks, key=lambda t: (t.split("/")[0], int(t.split("/")[1])))


def fname(task):
    return task.replace("/", "__")


# stage the scene frames next to the tex. results/ is sparse-excluded, so a pull
# can strip materialized frames; re-fetch from origin when one is missing.
for task in TASKORDER:
    su, ti = task.split("/")
    sub = "eval_frames" if su in ("libero_spatial", "libero_object") else "frames"
    rel = "results/analysis/pi05_bank/%s/%s__%02d.png" % (sub, su, int(ti))
    src = R / rel
    if not src.exists():
        src.parent.mkdir(parents=True, exist_ok=True)
        with open(src, "wb") as fh:
            subprocess.run(["git", "cat-file", "-p", "origin/main:" + rel],
                           cwd=R, stdout=fh, check=True)
    shutil.copyfile(src, PV / "frames" / (fname(task) + ".png"))

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
{\normalsize\bfseries Verified minimal pairs --- frozen $\pi_{0.5}$, LIBERO}\enspace
{\scriptsize Full init population (inits 0--49; n=50/phrase); success = environment success flag at episode end; p: two-proportion z, best vs worst phrase (ladder extremes selected within family --- descriptive). Green = best phrasing, red = worse; highlight = changed span. Scenes on page 2.}
\vspace{2pt}\hrule\vspace{2.5pt}
\begin{multicols}{2}\scriptsize\setlength{\baselineskip}{7.7pt}"""]

first = True
for task in TASKORDER:
    short = task.replace("_", r"\_")
    if not first:
        tex.append(r"\vspace{0.5pt}{\color{black!35}\hrule height 0.5pt}\vspace{1.5pt}")
    first = False
    tex.append(r"{\ttfamily\bfseries " + short + r"}\par\nopagebreak")
    for label, rows_, delta, p in blocks[task]:
        pstr = f"p={p:.4f}" if p >= 5e-5 else "p<0.0001"
        tex.append(r"{\leftskip=1.1em")
        tex.append(r"{\bfseries " + esc(label) + r"}\enspace " +
                   f"{delta:+.0f}\\,pp\\," + r"$\cdot$\," + f" {pstr}" + r"\\[1pt]")
        sibs = list(rows_.phrase)
        for j, (_, r) in enumerate(rows_.iterrows()):
            color = "hi" if j == 0 else "lo"
            pcol = "hipct" if j == 0 else "lopct"
            tex.append(r"\hangindent=2.4em \pct{" + pcol + r"}{" + f"{r.succ:.0f}" + r"}~\texttt{"
                       + marked_phrase(r.phrase, sibs, color) + r"}\\")
        tex.append(r"[0.9pt]\par}")

tex.append(r"\end{multicols}")
tex.append(r"\newpage")
tex.append(r"{\large\bfseries Task scenes}\\[4pt]")
tex.append(r"\begin{multicols}{3}\footnotesize\centering")
for task in TASKORDER:
    tex.append(r"\includegraphics[width=0.9\linewidth]{frames/" + fname(task) + r".png}\\")
    tex.append(r"{\ttfamily " + task.replace("_", r"\_") + r"}\\[8pt]")
tex.append(r"\end{multicols}")
tex.append(r"\end{document}")

out = PV / "pairverify_libero_pairs.tex"
out.write_text("\n".join(tex))
print("tex ->", out)
print("groups:", sum(len(v) for v in blocks.values()), "across", len(blocks), "tasks")
if not shutil.which("pdflatex"):
    raise SystemExit("no pdflatex locally; compile elsewhere")
for _ in range(2):
    subprocess.run(["pdflatex", "-interaction=nonstopmode", "-output-directory", str(PV), str(out)],
                   capture_output=True, text=True)
pdf = PV / "pairverify_libero_pairs.pdf"
print("pdf ->", pdf, "exists:", pdf.exists())
