# evidence.csv

1406 measured phrases across 178 tasks, sorted by
task then estimated success.

Columns:
  task       the instruction/task the phrase was measured on
  phrase     the exact wording measured
  proxy      estimated success rate, 0-1 (a calibrated estimate)
  kind       what sort of input the phrase is:
               original    the task's own canonical instruction
               natural     a fluent rewording, the kind a person would say
               adversarial a deliberately awkward or ornate rewording
               search      surfaced by an automated phrasing search
               rule_output produced by applying a rulebook
               probe       proposed to settle a specific question
               unknown     predates labelling -- do not guess
             Rules are for REPAIRING inputs, so compare within a kind:
             a rule that helps adversarial inputs may do nothing for
             natural ones, and averaging the two hides both effects.
  n_ctx      how many scored contexts back that estimate. LOW n_ctx =
             a weak claim. If two phrases differ but both have small
             n_ctx, the difference may be noise -- you can propose
             re-measuring them (see the experiment format).
  gt_success real measured success rate, 0-100, blank if never rolled.
             Where present, trust this over `proxy`.
  source     where the measurement came from

You have a shell. Group, sort and aggregate this rather than reading it
row by row -- the within-task SPREAD is the signal, not the extremes.
