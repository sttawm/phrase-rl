You are writing a rulebook for rewriting task instructions given to a robot
policy. The rulebook will be handed to another language model, which must apply
it to instructions it has never seen. Your goal is a rulebook that, when applied,
raises the policy's measured success rate.

You have working files. Read them before you answer — do not rely on this
message alone:

  {{corpus_file}}
      What the policy's training instructions look like: vocabulary,
      grammatical shape, register, notable absences.

The evidence table is large — aggregate it, do not skim it. You have a shell.
`evidence_summary.csv` sits beside it with one row per task (phrase count,
best/worst/median, spread, how much rests on real rollouts versus thin sampling,
sorted by spread) as a starting point, but compute whatever else you need.

  {{evidence_file}}
      Every phrase measured so far, grouped by task. This is the full spread,
      not a summary — read enough of it to see within-task variation, not just
      the extremes. Where a phrase has a real-rollout number (`gt_success`) it
      is marked; those are more reliable than estimates.
      Rank by the `score` column, which is normalised WITHIN each task: 1.0 is
      that task's best measured phrase, 0.0 its worst; do not compare it across
      tasks. `logit` is the same estimator un-normalised, on one common scale
      across tasks — differences are meaningful, levels are not probabilities
      and not success rates. `z` and `grip` are the two raw measured channels
      behind it (z: higher is better; grip: LOWER is better) — a phrase strong
      on one channel and weak on the other is a real, reportable pattern.

  {{probe_file}}
      Answers to YOUR OWN experiments: the probe phrases you asked for last
      iteration, now measured. These rows are also in the evidence file -- they
      are split out because the answers to your questions would otherwise be a
      handful of rows among thousands. Read this before the full table.

  {{rewrites_file}}
      What your last rulebook actually did, phrase by phrase: each base
      instruction PAIRED with the rewrite the rulebook produced for it, both
      measured, with the delta (rewrite_logit - base_logit; positive = the
      rewrite scored better than leaving the instruction alone) and base_kind
      (which input regime was being repaired). The whole-book mean says whether
      the book gained; this file says WHERE it gained and lost. Sorted worst
      delta first within each regime.

  {{best_eval_file}}
      The auditor's read on the BEST rulebook: whether each rule was actually
      followed, and notes on the rewrites. It carries the BOOK's measured delta,
      not a number per rule. A .json twin sits beside it. Absent on the first
      iteration.

  {{regressed_eval_file}}
      How the REGRESSED rulebook performed, in the same format -- present only
      when the last attempt scored worse than the best. Absent otherwise.

BEST RULEBOOK SO FAR (verbatim; empty on the first iteration). This is what you
are revising -- build from it:
{{best_rules}}

{{regressed_block}}

Guidance:
- Learn what works from the EVIDENCE file — it is the accumulated record of every
  phrase ever measured, and it grows every iteration. The eval files are the
  auditor's read on whether your rules were followed, not a second source of
  performance numbers.
- When you add or reword a rule, you are making a claim that has not been
  measured yet. Say what measurement would test it, and name it in your plan —
  those measurements land in the evidence and are there for every later
  iteration.
- **A rule that PERMITS an edit needs different evidence from one that FORBIDS
  one.** Forbidding needs only demonstrated harm. Permitting needs demonstrated
  BENEFIT. "No significant penalty on average" is never grounds to license an
  edit: a mean near zero is equally consistent with an edit that is usually
  harmless and occasionally catastrophic, and the applier will make that edit on
  every instruction it sees. Before writing any rule that allows something,
  count the tails rather than the average — how many phrases does this edit lose
  20pp or more on, and how many does it gain 10pp or more on? If the losses
  outnumber the gains, forbid it however small the mean. State both counts in
  the rationale.
  This is not hypothetical. A previous rulebook licensed single-word noun
  synonyms ("bowl" -> "dish", "stove" -> "burner") as carrying "no measured
  penalty". On its own evidence that edit gained >=10pp on 12% of phrases and
  lost >=20pp on 30% of them; measured at n=50 on a task outside its evidence,
  the same swap cost 56pp.
- **Your evidence covers less of the world than the applier will meet.** A rule
  qualified by a condition the applier cannot check from the instruction in
  front of it — "out of finetune", "on tasks like these", "for easy scenes" —
  will be applied everywhere regardless, because nothing at apply time tells it
  which case it is in. Either write the rule so it is safe when the condition
  does NOT hold, or do not write it.
- When a regressed rulebook is shown, diff it against the best one and identify
  which specific change cost the ground. Their two eval files let you compare the
  same rules measured under both -- a rule present in both whose delta moved is
  more informative than one that only appears in the loser. Do not simply revert;
  say what you learned and why your new change is not the same mistake.
- If a rule is being poorly adhered to, the fix may be to state it more simply
  rather than to drop it. Poor adherence and poor performance are different
  failures and have different remedies.
- Where new evidence contradicts a rule you previously wrote, decide which wins
  and say so explicitly in the rationale.
- Prefer stability: a rulebook that changes wholesale every iteration cannot
  converge. Change what the evidence says to change.
- **Reproduce every rule you are keeping word for word.** A rule's identity is
  its wording: keep it identical and its existing measurements carry over;
  reword it at all -- even cosmetically -- and it becomes a rule with no
  evidence behind it, which must be measured again from scratch. Reword only
  when you mean to change what the rule instructs.
- Rules must be executable by a model reading them cold, with no access to this
  evidence and no examples beyond what you write into the rule itself.

Reply in EXACTLY this format, with these two delimiter lines present verbatim
and nothing before the first one:

===RULES===
1. <one rule, imperative, self-contained>
2. <one rule>
(as many as the evidence supports)
===RATIONALE===
<free prose: what changed, what you kept and why, which conflicts you resolved
and on what evidence. Numbered lines are fine here.>
