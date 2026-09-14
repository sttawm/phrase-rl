# Minimal rule-distillation prompt, v2 (on-average criterion)

Differs from prompts/distill_minimal.md (frozen at tag icra2027) in one clause:
rules are judged by their expected effect across tasks, not by consistency on
every task. A rule may lose on some tasks if it gains more, or more often, on
the others. Used for pi0.5/LIBERO sealed-set-v2 draws 2 and 3 (2026-09-15).

---

You are given evidence from a robot-manipulation benchmark: for each task,
a set of candidate instruction phrasings and a score for each phrasing
(higher is better).

{{evidence_file}}

Write a rulebook of 10-20 numbered rules for rewriting any incoming task
instruction so that the robot is most likely to succeed. Judge each rule by
its expected effect on average over the tasks where it would apply: a rule
is worth including when the score gains it produces outweigh the losses,
in size and in frequency, even if it loses on some tasks. Prefer rules that
change wording over rules that only forbid changes, and keep rules general
enough to apply to instructions and objects not in the evidence. Output only
the numbered rules.
