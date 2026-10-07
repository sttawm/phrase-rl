#!/usr/bin/env python3
"""Build the arXiv source bundle from the Overleaf clone WITHOUT touching it.

  .venv/bin/python scripts/make_arxiv_bundle.py <overleaf-clone> <out-dir>

Differences from the Overleaf source, all output-identical:
  * the \\ifanon switch is resolved to its named (\\anonfalse) branch;
  * comments are stripped (arXiv publishes the source; the comments carry the
    anonymous site URL and the old title) -- a trailing '%' is kept so line-end
    spacing is unchanged, full-comment lines are dropped;
  * only the referenced figures, the class file and the .bib are copied.
The caller then runs pdflatex/bibtex and adds main.bbl (arXiv runs no BibTeX).
"""
import pathlib
import re
import shutil
import sys

src = pathlib.Path(sys.argv[1]).resolve()
out = pathlib.Path(sys.argv[2]).resolve()
out.mkdir(parents=True, exist_ok=True)
s = (src / "main.tex").read_text()

# --- 1. resolve the anonymity switch -------------------------------------
a = s.index("% ---- Anonymity switch")
b = s.index("\\fi\n", s.index("\\newcommand{\\artifactsrepo}{https://github.com")) + len("\\fi\n")
block = s[a:b]
m = re.search(r"\\else\n(.*?)\\fi\n", block, re.S)
named_defs = "\n".join(l.strip() for l in m.group(1).strip().splitlines()) + "\n"
assert "sttawm" in named_defs and "4open" not in named_defs
s = s[:a] + named_defs + s[b:]

m = re.search(r"\\author\{%\n\\ifanon\n.*?\\else\n(.*?)\\fi\n\}", s, re.S)
assert m and "Watts" in m.group(1)
s = s[:m.start()] + "\\author{%\n" + m.group(1) + "}" + s[m.end():]

old = "\\ifanon\\else We thank Amber Kleiner for designing Fig.~\\ref{fig:method}. \\fi\n"
assert s.count(old) == 1
s = s.replace(old, "We thank Amber Kleiner for designing Fig.~\\ref{fig:method}.\n")
assert "ifanon" not in s and "anontrue" not in s and "anonfalse" not in s

# --- 2. strip comments ----------------------------------------------------
def strip(line):
    i, n = 0, len(line)
    while i < n:
        if line[i] == "\\":
            i += 2            # skip the escaped character (handles \% and \\)
            continue
        if line[i] == "%":
            head = line[:i]
            return None if head.strip() == "" else head + "%"
        i += 1
    return line

lines = [strip(l) for l in s.splitlines()]
s = "\n".join(l for l in lines if l is not None) + "\n"
for bad in ["4open", "rephrase-before-you-act.github.io", "Anonymous Authors", "anonym", "ICRA", "\\todo{"]:
    assert bad not in s, bad
(out / "main.tex").write_text(s)

# --- 3. copy what the source references ----------------------------------
for f in ["ieeeconf.cls", "references.bib", "IEEEtran.bst"]:
    shutil.copy(src / f, out / f)
(out / "figures").mkdir(exist_ok=True)
refs = re.findall(r"\\includegraphics\[[^\]]*\]\{([^}]*)\}|\\input\{([^}]*)\}", s)
n = 0
for g, i in refs:
    rel = g or (i + ".tex")
    shutil.copy(src / rel, out / rel); n += 1
print(f"bundle -> {out}: main.tex ({len(lines)} lines in, {sum(l is not None for l in lines)} kept), {n} referenced files, class+bib+bst")
