# evidence.csv

3561 measured phrases across 156 tasks, sorted by
task then estimated success.

Columns:
  task       the instruction/task the phrase was measured on
  phrase     the exact wording measured
  score      THE COLUMN TO RANK BY WITHIN A TASK. 0-1 within its task:
             1.0 is the best phrase measured for that task, 0.0 the
             worst. NOT comparable across tasks, and not a success
             rate. Blank when a task has only one measured phrase.
  logit      the raw proxy value that `score` normalises: a fixed
             linear mixture of the two measured channels below, on one
             common scale across tasks. Higher is better. It is NOT a
             probability and NOT a success rate; treat differences,
             not levels, as meaningful.
  z          raw verifier-ensemble channel (higher = the trajectory
             matches the instruction better, as judged by a frozen
             learned verifier).
  grip       raw gripper-error channel (LOWER is better: mean distance
             between the policy's gripper action and the demonstrated
             one). The two channels are independent measurements; a
             phrase strong on one and weak on the other is a real
             pattern worth noting, not an error.
  kind       what the phrase IS. Five kinds:
               original     the task's own canonical instruction -- the
                            wording the policy was trained on
               natural      a fluent rewording, the kind of thing a
                            person would actually say
               adversarial  a deliberately awkward, ornate or indirect
                            rewording -- the hard case rules exist for
               rephrased    the output of applying a rulebook
               search       surfaced by automated phrasing search, or
                            proposed as a probe; exploratory wordings
             plus `unknown` for measurements that predate labelling.
  proxy_imputed  True when one channel was missing for this row and
             was filled with the column mean. Those rows are driven by
             the surviving channel alone -- weaker evidence.
  base_kind  for `rephrased` rows only: the kind of the instruction it
             was rewritten FROM. Blank otherwise. This is the column that
             says what a rewrite was REPAIRING.

             COMPARE WITHIN A REGIME. Rules exist to repair inputs, and
             the regimes behave differently: a rule that rescues
             adversarial wordings may do nothing at all for natural ones.
             A single number averaged across them hides both effects. For
             a rephrased row the regime is `base_kind`, not `kind` --
             every rewrite is `rephrased`, so that column alone tells you
             nothing about which problem the rule was solving. When the
             evidence supports it, say which regime a rule is for.
  n_ctx      how many scored contexts back that estimate. LOW n_ctx =
             a weak claim. If two phrases differ but both have small
             n_ctx, the difference may be noise -- you can propose
             re-measuring them (see the experiment format).
  gt_success real measured success rate, 0-100, blank if never rolled.
             Where present, trust this over every estimated column.
  source     where the measurement came from


evidence_summary.csv sits beside it: one row per task with phrase count,
best/worst/median logit, SPREAD (best-worst, in logit units), how many
phrases carry a real rollout number, and how many rest on thin sampling.
Sorted by spread, descending -- the tasks where wording matters most are
at the top. Read it first, then go into evidence.csv for the tasks it
points you to. The within-task spread is the signal, not the extremes.
