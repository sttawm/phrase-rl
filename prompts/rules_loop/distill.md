You are distilling instruction-phrasing rules for a frozen pi0 robot policy
(Bridge corpus: short plain kitchen-tabletop imperatives). The rules will be
executed by {{rephraser}} — tailor their number and complexity to what that
model can reliably follow.

PREVIOUS RULES
{{prev_rules}}

CORPUS SUMMARY
{{corpus_summary}}

EVIDENCE — proxy-scored phrases, worst 3 and best 3 per task (proxy = calibrated
estimated success rate; gt = real rollout success where measured):
{{evidence}}

LAST ITERATION'S PER-RULE EVALUATION (single-edit deltas + judge notes; empty on
the first iteration):
{{eval_summary}}

Where the evidence conflicts with a previous rule, decide which wins and say why
in one line. Prefer keeping a rule stable unless the evidence against it is
clear — rules that flip every iteration never converge.

Reply with EXACTLY this format:
RULES
1. <rule>
2. <rule>
...
RATIONALE
<one line per changed/kept-despite-conflict rule>
