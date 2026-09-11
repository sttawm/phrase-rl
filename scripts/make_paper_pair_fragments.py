#!/usr/bin/env python3
"""Extract paper-embeddable listing bodies from the two pairverify tex docs.

The paper inlines the minimal-pair listings as native tex (user request
2026-09-11) instead of including compiled PDF pages. This script cuts the
content of the FIRST multicols{2} block (the listing; the scenes grid is the
second) out of each generated standalone doc and writes it as a fragment for
main.tex to \\input. Colors (hi/lo/mid/…) and \\pct are defined in the paper
preamble; font size and fboxsep are set by the wrapper group in main.tex.

Reads (generated; LIBERO one is produced by the other session's generator):
  results/analysis/pairverify/pairverify_pairs.tex
  results/analysis/pairverify_libero/pairverify_libero_pairs.tex
Writes:
  paper/overleaf_staging/figures/pairverify_bridge_body.tex
  paper/overleaf_staging/figures/pairverify_libero_body.tex
"""
import pathlib
import re

R = pathlib.Path(__file__).resolve().parents[1]
OUT = R / "paper/overleaf_staging/figures"

JOBS = [
    (R / "results/analysis/pairverify/pairverify_pairs.tex",
     OUT / "pairverify_bridge_body.tex"),
    (R / "results/analysis/pairverify_libero/pairverify_libero_pairs.tex",
     OUT / "pairverify_libero_body.tex"),
]

PHRASE = re.compile(r"^\\hangindent=2\.4em (.*)\\\\$")
HEADER = re.compile(r"^(\{\\bfseries .*)\\\\\[[\d.]+pt\]$")
ORPHAN = re.compile(r"^\[[\d.]+pt\]\\par\}$")

for src, dst in JOBS:
    text = src.read_text()
    m = re.search(r"\\begin\{multicols\}\{2\}[^\n]*\n(.*?)\\end\{multicols\}",
                  text, re.S)
    assert m, f"no multicols listing block in {src}"
    out = []
    for line in m.group(1).rstrip().splitlines():
        # each phrase becomes its own paragraph with its own hanging indent:
        # group-level \hangindent + \\ mis-tabs wrapped phrases in the
        # paper's narrower columns (hangindent is a paragraph property).
        pm = PHRASE.match(line)
        hm = HEADER.match(line)
        if pm:
            out.append(r"{\hangindent=2.4em\hangafter=1\noindent "
                       + pm.group(1) + r"\par}")
        elif hm:
            out.append(hm.group(1) + r"\par\nopagebreak")
        elif ORPHAN.match(line):
            out.append("}")
        else:
            out.append(line)
    body = "\n".join(out) + "\n"
    assert "\\includegraphics" not in body, f"scene grid leaked into {dst}"
    dst.write_text(body)
    print(f"{dst.relative_to(R)}  ({len(body.splitlines())} lines)")
