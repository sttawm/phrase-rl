# evidence.csv

1430 measured phrases across 178 tasks, sorted by
task then estimated success.

Columns:
  task       the instruction/task the phrase was measured on
  phrase     the exact wording measured
  proxy      estimated success rate, 0-1 (a calibrated estimate)
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
             Where present, trust this over `proxy`.
  source     where the measurement came from

You have a shell. Group, sort and aggregate this rather than reading it
row by row -- the within-task SPREAD is the signal, not the extremes.
