# One-shot rulebook distillation — the reusable brief

Generalised from `results/rules_runs/bankdistill/DISTILL-BRIEF.md`, which stays
frozen as that run's record. Substitute the bracketed values per run.

You are distilling a phrasing rulebook for a frozen [POLICY] robot policy
([CORPUS] corpus). Your book will be applied MECHANICALLY, one instruction at a
time, by a rule-following language model that sees only your rules, the incoming
instruction, and a scene trace. It follows rules literally; it cannot ask
questions and it cannot see this evidence.

You get ONE shot, and your evidence is the FULL measured record. Prior books
overfit by turning broad principles into noun tables mined from a handful of
tasks ("any cola -> the coke can"), which misfires on unseen tasks (a Pepsi is
not a coke can). Your advantage is breadth — use it.

Files in this directory: [EVIDENCE MANIFEST]

Write the best rulebook you can. Requirements:

- State rules at the level of CORPUS PRINCIPLES (vocabulary membership,
  register, form) wherever the evidence supports them; use instance-level
  substitutions ONLY where repeatedly measured AND state the principle they
  instantiate, so the applier can extrapolate to unseen objects.

- Decide-then-edit: the strongest measured finding across every evidence base is
  that most plain inputs are best left untouched. Give the applier a crisp
  triage test before any editing rule.

- **A rule that PERMITS an edit needs different evidence from one that FORBIDS
  one.** Forbidding needs demonstrated harm. Permitting needs demonstrated
  BENEFIT. "No measured penalty on average" never licenses an edit: a mean near
  zero is equally consistent with an edit that is usually harmless and
  occasionally catastrophic, and the applier will make that edit on every
  instruction it sees. Before allowing anything, count the tails instead of the
  average — on how many measured phrases does this edit lose >=20pp, and on how
  many does it gain >=10pp? If the losses outnumber the gains, forbid it however
  small the mean. State both counts in the rationale.

  This is not hypothetical. A previous book licensed single-word noun synonyms
  ("bowl" -> "dish", "stove" -> "burner") as carrying "no measured penalty". On
  its own evidence that edit gained >=10pp on 12% of phrases and lost >=20pp on
  30%; on tasks outside its evidence diet the same swap cost 56pp measured at
  n=50. The mean was near zero and the rule was still badly wrong.

- **Your evidence covers less of the world than the applier will meet.** A rule
  qualified by a condition the applier cannot check from the instruction in
  front of it — "out of finetune", "on tasks like these", "for easy scenes" —
  will be applied everywhere regardless, because nothing at apply time tells it
  which case it is in. Either write the rule so it is safe when the condition
  does NOT hold, or do not write it.

- Distinguish measurement error from effect. Repeat measurements of the SAME
  phrase in this program have differed by 15pp or more; a contrast smaller than
  the repeat-measurement spread is not evidence. Where the evidence records how
  many episodes and which initial states a number rests on, weight by it, and
  treat a contrast built from phrases measured on different initial-state blocks
  as unreliable regardless of its size.

- The applier must never leak reasoning: rule 1 should pin the output format
  (one line, the rewritten instruction, nothing else).

- Numbers in the book are welcome where they help the applier weigh collisions
  (e.g. "noun edits dominate structure edits"), but every rule must be
  executable without them.

Output EXACTLY this structure to a file named [OUTPUT].md in this directory:

===RULES===
1. ...
2. ...
===RATIONALE===
(short: what evidence drove each major rule; for every rule that PERMITS an
edit, the loss/gain tail counts that justify it)
