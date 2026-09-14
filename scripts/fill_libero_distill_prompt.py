#!/usr/bin/env python3
"""Fill prompts/distill_minimal.md's {{evidence_file}} slot with the LIBERO
single-edit evidence (distill_evidence_v2.csv rendered as a markdown table) and
write the exact prompt a distiller call would receive, for review before any run.

  .venv/bin/python scripts/fill_libero_distill_prompt.py
  -> results/analysis/pi05_bank/distill_prompt_v2_filled.md
"""
import argparse
import pathlib

R = pathlib.Path(__file__).resolve().parents[1]
B = R / "results/analysis/pi05_bank"
ap = argparse.ArgumentParser()
ap.add_argument("--template", default="prompts/distill_minimal.md")
ap.add_argument("--out", default="distill_prompt_v2_filled.md")
a = ap.parse_args()
tmpl = (R / a.template).read_text().split("---", 1)[1].strip()   # body below the header note
evidence = (B / "distill_evidence_v2.md").read_text().strip()
assert "{{evidence_file}}" in tmpl
filled = tmpl.replace("{{evidence_file}}", evidence)
(B / a.out).write_text(filled + "\n")
words = len(filled.split())
print(f"-> {B / a.out}: {words} words (~{int(words * 1.4)} tokens), "
      f"{evidence.count(chr(10)) - 1} evidence rows")
