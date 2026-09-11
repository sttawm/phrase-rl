# One-shot rulebook distillation — r4 draw (single-prompt procedure)

Instantiated from `prompts/distill_oneshot.md` (2026-09-11). NOTE: draws
r1–r3 were produced by multi-agent panels (3 lens-distillers → adversarial
critic → synthesizer); a book distilled with THIS single prompt is a
different procedure and must be labeled r4-oneshot in any pooled analysis,
not treated as a fourth exchangeable draw.

---

You are distilling an instruction-phrasing rulebook for a frozen pi0 robot
policy (Bridge corpus). Your book will be applied MECHANICALLY, one
instruction at a time, by a rule-following language model that sees only your
rules, the incoming instruction, and a scene trace. It follows rules
literally; it cannot ask questions and it cannot see this evidence.

You get ONE shot. Your evidence is the directory
`results/rules_runs/v2distill/sim_only/` (rollout-only diet; substitute
`train_only/` or `both/` to distill the other diets):

- `evidence.csv` + `evidence.csv.README.md` — the measured phrases. Rows
  with `gt_success` are REAL rollout success rates (0–100), weighted by
  `n_ctx`; rows with only `z`/`grip`/`logit` are proxy estimates (ranking
  signal within a task, never a success rate; z higher is better, grip
  lower is better). When they disagree, trust `gt_success`.
- The `train_only/` diet carries NO `gt_success` column (proxy + corpus
  evidence only) and adds `corpus_stats.md`, `08_phrase_search.md`,
  `06_mechanism_cases.md`, `09_rules_v1_failures.md`,
  `03_contrast_pairs_139.json`. The `sim_only/` diet has no corpus
  artifacts — treat the policy as a black box measured only through its
  rollouts. `both/` has everything.

Write the best rulebook you can. Requirements:

- Rule 1 pins the output format: one line containing only the rewritten
  instruction — no reasoning, no labels, no second line.
- Decide-then-edit: the strongest measured finding is that most plain
  inputs are best left untouched. Give the applier a crisp triage test
  before any editing rule.
- State rules at the level of CORPUS PRINCIPLES (vocabulary membership,
  register, form); use instance-level substitutions ONLY where repeatedly
  measured AND state the principle they instantiate.
- A rule that PERMITS an edit needs demonstrated benefit; one that FORBIDS
  needs demonstrated harm. Never license an edit on "no measured penalty on
  average": count the tails (on how many measured phrases does the edit
  lose >=20pp, on how many does it gain >=10pp) and put both counts in the
  rationale. If losses outnumber gains, forbid it.
- No rule may hinge on a condition the applier cannot check from the
  instruction and trace in front of it.
- Repeat measurements of the same phrase differ by 15pp or more in this
  program; a contrast smaller than that spread is not evidence. Weight by
  episode counts where recorded.

Output EXACTLY this structure:

===RULES===
1. ...
2. ...
===RATIONALE===
(short: what evidence drove each major rule; for every rule that PERMITS an
edit, the loss/gain tail counts that justify it)
