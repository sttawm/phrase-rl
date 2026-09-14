#!/usr/bin/env python3
"""Fill prompts/distill_minimal.md's {{evidence_file}} slot with the LIBERO
single-edit evidence (distill_evidence_v2.csv rendered as a markdown table) and
write the exact prompt a distiller call would receive, for review before any run.

  .venv/bin/python scripts/fill_libero_distill_prompt.py
  -> results/analysis/pi05_bank/distill_prompt_v2_filled.md
"""
import pathlib

R = pathlib.Path(__file__).resolve().parents[1]
B = R / "results/analysis/pi05_bank"
tmpl = (R / "prompts/distill_minimal.md").read_text().split("---", 1)[1].strip()   # body below the header note
evidence = (B / "distill_evidence_v2.md").read_text().strip()
assert "{{evidence_file}}" in tmpl
filled = tmpl.replace("{{evidence_file}}", evidence)
(B / "distill_prompt_v2_filled.md").write_text(filled + "\n")
words = len(filled.split())
print(f"-> {B / 'distill_prompt_v2_filled.md'}: {words} words (~{int(words * 1.4)} tokens), "
      f"{evidence.count(chr(10)) - 1} evidence rows")
