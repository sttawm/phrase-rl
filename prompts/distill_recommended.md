# Recommended rules-distiller prompt (2026-09-11 synthesis, rev 2)

What the distillation prompt should/could have been, knowing everything the
program has since measured. Three stages: a train-only draw distilled from
the gripper score alone, a rollout-only draw distilled from rollout success
alone, and a combiner that merges the two BOOKS under the joint evidence.
Substitute {{evidence_file}} (and bracketed values) per run. Provenance: the
recovered panel scripts (reference/distill_panels/), the bankdistill brief,
prompts/distill_oneshot.md, and the pair-verification findings.

---

## Common context (prepend to every stage)

You are distilling an instruction-phrasing rulebook for a frozen robot
manipulation policy. Phrasing choices move this policy's task success by
tens of points; your book is how that knowledge gets applied.

THE APPLIER. Your book will be applied MECHANICALLY, one instruction at a
time, by a strong rule-following language model that sees only your rules,
the incoming instruction, and a scene-description trace. It follows rules
literally, cannot ask questions, and cannot see your evidence. So: crisp
triggers, no unstated judgment, and no rule may hinge on a condition the
applier cannot check from the instruction and trace in front of it ("out of
vocabulary", "for easy scenes" — unusable). Rule 1 must pin the output
format: one line containing only the rewritten instruction, no reasoning
text ever.

DISCIPLINE ON NOISE. Repeat measurements of the same phrase in this program
differ by 15pp or more; a contrast smaller than that spread, or built from
phrases measured on different initial-state blocks, is not evidence. Weight
everything by episode count where recorded. Check every contrast is
FORMAT-CONTROLLED: capitalization and trailing punctuation are themselves
worth tens of points on this policy, so a lexical contrast whose members
also differ in case or punctuation is confounded — split the effects or
discard it.

MAKE THREE PASSES, then synthesize:
1. REGISTER pass — triage before edits. Classify which input registers
   (plain imperative, ornate/wrapped, telegraphic, frozen-form) were
   unbeatable as-is vs reliably repairable. The strongest finding across
   every evidence base in this program is that most plain inputs are best
   left untouched: your book's first substantive rule is a decide-then-edit
   triage test, and the default is COPY.
2. MECHANISM pass — build each editing rule from the largest reliable
   measured contrasts (noun choice, added/dropped detail, verb frame,
   preposition, format).
3. CONSERVATIVE pass — for every edit class, count the tails on your own
   evidence: on how many measured phrases does this edit lose >=20pp, and
   on how many does it gain >=10pp? A rule that PERMITS an edit needs
   measured benefit; "no penalty on average" licenses nothing (a near-zero
   mean is consistent with usually-harmless-occasionally-catastrophic, and
   the applier will make the edit every time). If losses outnumber gains,
   write the prohibition instead.

Write 10–18 numbered rules. Instance-level substitutions are allowed only
where repeatedly measured AND accompanied by the principle they instantiate,
so the applier can extrapolate to unseen objects. Strike any rule that
cannot be executed exactly as written from the instruction and trace alone.

This is an INDEPENDENT draw: do not attempt to reproduce any previous or
canonical book. Read the evidence yourself and reach your own conclusions.

Output EXACTLY:
===RULES===
1. ...
===RATIONALE===
(short: the measurements behind each major rule; for every rule that PERMITS
an edit, the loss/gain tail counts that justify it)

## Stage T — train-only draw

EVIDENCE: {{evidence_file}} (+ its README) — phrases from the policy's
training corpus scored ONLY by the gripper score (lower is better): a
within-task RANKING signal, never a success rate. Use ONLY the gripper
score; ignore any other score column present. Plus the corpus-statistics
artifacts listed in the README. The simulation and sealed tasks are
deliberately absent from this evidence: do not speculate about them, and
write rules grounded in corpus properties (vocabulary membership, register,
form) that generalize to any task drawn from this corpus.

SELF-CRITIQUE before finalizing — LEAKAGE: flag and remove any rule or
example naming objects or tasks this evidence cannot support.

## Stage S — rollout-only draw

EVIDENCE: {{evidence_file}} (+ its README) — phrases scored ONLY by real
rollout success rates (0–100, weighted by episode count). Use ONLY the
rollout score; ignore any proxy columns present. No corpus statistics, no
training-data artifacts: treat the policy as a BLACK-BOX VLA you know only
through these measured rollouts. Be specific where the data is deep (high-n
contrasts) but do not overfit: a noun mapping seen on one task is a
hypothesis, not a rule — state the principle it instantiates so the applier
can extrapolate, and prefer rules supported by contrasts on several tasks.

SELF-CRITIQUE before finalizing — OVERFIT: flag and fix every
instance-level mapping stated without its generalizing principle, and every
rule a novel-object task would break.

## Stage B — combined book (input: the two finished books)

You are given the Stage T rulebook, the Stage S rulebook, and the JOINT
evidence at {{evidence_file}} (both score channels present; where they
conflict, rollout success wins). Do not distill from scratch: COMBINE the
two books. Take the stronger structure as the spine; keep a rule from
either book only where the joint evidence still supports it; where the two
books disagree, resolve with measurements, not taste; drop any rule the
joint evidence contradicts. The result must obey every requirement of the
common context (triage-first, tail counts for permitting rules, executable
by the applier from instruction + trace alone).

SELF-CRITIQUE before finalizing — CONSISTENCY: flag any kept rule where
the two channels disagree and the book took the gripper side.
