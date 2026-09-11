# Recommended rules-distiller prompt (2026-09-11 synthesis)

What the distillation prompt should/could have been, knowing everything the
program has since measured. One self-contained prompt for a single strong
distiller; it internalizes the three-lens panel (register / mechanism /
conservative) plus the critic as explicit passes. Substitute the bracketed
values. Provenance: the recovered panel scripts
(reference/distill_panels/), the bankdistill brief, prompts/distill_oneshot.md,
and the pair-verification findings.

---

You are distilling an instruction-phrasing rulebook for a frozen [pi0] robot
manipulation policy ([Bridge] corpus). Phrasing choices move this policy's
task success by tens of points; your book is how that knowledge gets applied.

THE APPLIER. Your book will be applied MECHANICALLY, one instruction at a
time, by rule-following language models — the weakest a 9B model — that see
only your rules, the incoming instruction, and a scene-description trace.
They follow rules literally, cannot ask questions, and cannot see your
evidence. So: crisp triggers, no unstated judgment, and no rule may hinge on
a condition the applier cannot check from the instruction and trace in front
of it ("out of vocabulary", "for easy scenes" — unusable). Rule 1 must pin
the output format: one line containing only the rewritten instruction, all
lowercase edits per your own normalization rules, no reasoning text ever.

THE EVIDENCE. Everything you may rely on is in [EVIDENCE DIR]:
[evidence.csv + README — measured phrases; rows with gt_success are real
rollout success rates weighted by n_ctx episodes; z/grip/logit rows are the
gripper-reward proxy, a within-task RANKING signal, never a success rate;
when they conflict, rollouts win. Additional artifacts per the README.]
Do not use knowledge of tasks or objects the evidence does not cover.

MAKE THREE PASSES, then synthesize:
1. REGISTER pass — triage before edits. From the evidence, classify which
   input registers (plain imperative, ornate/wrapped, telegraphic,
   frozen-form) were unbeatable as-is vs reliably repairable. The strongest
   finding across every evidence base in this program is that most plain
   inputs are best left untouched: your book's first substantive rule is a
   decide-then-edit triage test, and the default is COPY.
2. MECHANISM pass — build each editing rule from the largest reliable
   measured contrasts (noun choice, added/dropped detail, verb frame,
   preposition, format). For every contrast, check it is FORMAT-CONTROLLED:
   capitalization and trailing punctuation are themselves worth tens of
   points on this policy, so a lexical contrast whose members also differ in
   case or punctuation is confounded — split the effects or discard it.
3. CONSERVATIVE pass — for every edit class, count the tails on your own
   evidence: on how many measured phrases does this edit lose >=20pp, and on
   how many does it gain >=10pp? A rule that PERMITS an edit needs measured
   benefit; "no penalty on average" licenses nothing (a near-zero mean is
   consistent with usually-harmless-occasionally-catastrophic, and the
   applier will make the edit every time). If losses outnumber gains, write
   the prohibition instead.

SELF-CRITIQUE before finalizing ([pick the diet's check]):
- train-only diet: LEAKAGE — flag any rule or example naming objects or
  tasks the evidence cannot support.
- rollout-only diet: OVERFIT — flag every instance-level mapping stated
  without the generalizing principle it instantiates; a mapping seen on one
  task is a hypothesis, not a rule.
- combined diet: CONSISTENCY — flag any rule where proxy and rollout
  evidence disagree and you took the proxy side.
Also strike any rule the 9B applier could not execute exactly as written.

DISCIPLINE ON NOISE. Repeat measurements of the same phrase in this program
differ by 15pp or more; a contrast smaller than that spread, or built from
phrases measured on different initial-state blocks, is not evidence. Weight
everything by episode count where recorded.

Write 10–18 numbered rules. Instance-level substitutions are allowed only
where repeatedly measured AND accompanied by the principle they instantiate,
so the applier can extrapolate to unseen objects. Numbers in the book are
welcome where they help the applier weigh collisions, but every rule must be
executable without them.

This is an INDEPENDENT draw: do not attempt to reproduce any previous or
canonical book. Read the evidence yourself and reach your own conclusions.

Output EXACTLY:
===RULES===
1. ...
===RATIONALE===
(short: the measurements behind each major rule; for every rule that PERMITS
an edit, the loss/gain tail counts that justify it)
