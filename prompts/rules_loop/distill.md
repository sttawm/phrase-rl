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
      Every phrase measured so far, with its estimated success rate, grouped by
      task. This is the full spread, not a summary — read enough of it to see
      within-task variation, not just the extremes. Where a phrase has a
      real-rollout number it is marked; those are more reliable than estimates.
      Rank by the `score` column, which is normalised WITHIN each task: 1.0 is
      that task's best measured phrase, 0.0 its worst. Do not compare scores
      across tasks, and do not read the `proxy` column as a success rate unless
      that row's `calibrated_ok` is True — on the training instructions it is
      normally False, where `proxy` sits pinned near 1.0 and carries no
      information. A column of 0.999s is not evidence that the phrasing is
      already perfect; the discrimination lives in `score`.

  {{new_file}}
      Only what has been measured SINCE your last rulebook: the probe phrases you
      asked for, and the rewrites your own rulebook produced. These rows are also
      in the evidence file -- they are split out here because the answers to your
      own experiments would otherwise be a handful of rows among thousands. If you
      proposed an experiment last time, this is what it returned. Read it before
      the full table.

  {{best_eval_file}}
      How the BEST rulebook performed: per-rule adherence, the measured
      single-edit effect of each individual rule, and an auditor's notes. A .json
      twin sits beside it if you want to compute over it. Absent on the first
      iteration.

  {{regressed_eval_file}}
      How the REGRESSED rulebook performed, in the same format -- present only
      when the last attempt scored worse than the best. Absent otherwise.

BEST RULEBOOK SO FAR (verbatim; empty on the first iteration). This is what you
are revising -- build from it:
{{best_rules}}

{{regressed_block}}

Guidance:
- A rule earns its place by evidence, but note what the evidence can and cannot
  resolve. The BOOK's delta is measured. A single rule's contribution is NOT --
  it is well under 1pp and would need ~8,700 measurements to resolve, so no
  per-rule number is given and you should not infer one. Judge rules on whether
  the applier obeyed them and on what the rewrites look like; judge the book on
  its delta. If you want a rule isolated, ask for that experiment in your plan.
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
