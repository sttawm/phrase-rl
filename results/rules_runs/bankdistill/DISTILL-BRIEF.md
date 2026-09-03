# One-shot rulebook distillation — full bank + corpus statistics

You are distilling a phrasing rulebook for a frozen pi0 robot policy (Bridge
corpus). Your book will be applied MECHANICALLY, one instruction at a time, by
a rule-following language model (Gemini) that sees only your rules, the
incoming instruction, and a scene trace. It follows rules literally; it cannot
ask questions or see this evidence.

This is NOT the iterative loop: you get ONE shot, and your evidence is the
FULL measured record, not a 48-phrase training sample. Prior loop books
overfit: they turned broad principles into noun tables mined from 8 tasks
("any cola -> the coke can"), which misfires on unseen tasks (a Pepsi is not a
coke can). Your advantage over those books is breadth — use it.

Files in this directory:

  evidence.csv (+ .README.md, evidence_summary.csv)
      5,236 measured phrases across 228 tasks. Rows with gt_success are REAL
      rollout success rates (0-100) — 516 of them, weighted by n_ctx; the rest
      are proxy estimates (ranking signal only). The README explains columns.

  corpus_stats.md
      Statistics of the policy's TRAINING CORPUS text: what vocabulary,
      lengths, and forms the policy actually saw. Rules grounded here
      generalize to any task drawn from this corpus.

  08_phrase_search.md
      The oracle phrase-search field notes: measured single-concept edits with
      effect sizes and n, winner's-curse warnings, per-scene mechanism reads.

  06_mechanism_cases.md, 09_rules_v1_failures.md, 03_contrast_pairs_139.json
      Measured mechanism case studies, documented failures of past rulebooks,
      and 139 measured contrast pairs.

Write the best rulebook you can. Requirements:

- State rules at the level of CORPUS PRINCIPLES (vocabulary membership,
  register, form) wherever the evidence supports them; use instance-level
  substitutions ONLY where repeatedly measured AND state the principle they
  instantiate, so the applier can extrapolate to unseen objects.
- Decide-then-edit: the strongest measured finding across every evidence base
  is that most plain inputs are best left untouched. Give the applier a crisp
  triage test before any editing rule.
- The applier must never leak reasoning: rule 1 should pin the output format
  (one line, the rewritten instruction, nothing else).
- Numbers in the book are welcome where they help the applier weigh collisions
  (e.g. "noun edits dominate structure edits"), but every rule must be
  executable without them.

Output EXACTLY this structure to a file named rules_bank.md in this directory:

===RULES===
1. ...
2. ...
===RATIONALE===
(short: what evidence drove each major rule)
