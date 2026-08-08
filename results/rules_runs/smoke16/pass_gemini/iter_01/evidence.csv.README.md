# evidence.csv

1406 measured phrases across 178 tasks, sorted by
task then estimated success.

Columns:
  task       the instruction/task the phrase was measured on
  phrase     the exact wording measured
  proxy      estimated success rate, 0-1 (a calibrated estimate)
  kind       what sort of input the phrase is. Four kinds:
               original     the task's own canonical instruction -- the
                            wording the policy was trained on
               natural      a fluent rewording, the kind of thing a
                            person would actually say
               adversarial  a deliberately awkward, ornate or indirect
                            rewording -- the hard case rules exist for
               search       surfaced by automated phrasing search, or
                            proposed as a probe; exploratory wordings
             plus `unknown` for measurements that predate labelling.
             Rewrites inherit the kind of the instruction they were
             rewritten from.

             COMPARE WITHIN A KIND. Rules exist to repair inputs, and the
             regimes behave differently: a rule that rescues adversarial
             wordings may do nothing at all for natural ones. A single
             number averaged across kinds hides both effects. When the
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
