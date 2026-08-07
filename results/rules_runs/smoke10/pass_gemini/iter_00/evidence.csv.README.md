# evidence.csv

1382 measured phrases across 178 tasks, sorted by
task then estimated success.

Columns:
  task       the instruction/task the phrase was measured on
  phrase     the exact wording measured
  proxy      estimated success rate, 0-1 (a calibrated estimate)
  gt_success real measured success rate, 0-100, blank if never rolled.
             Where present, trust this over `proxy`.
  source     where the measurement came from

You have a shell. Group, sort and aggregate this rather than reading it
row by row -- the within-task SPREAD is the signal, not the extremes.
