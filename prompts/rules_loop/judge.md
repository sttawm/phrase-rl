You are auditing a rulebook for rewriting instructions given to a robot policy —
against the rewrites it produced, and against the measured effect of each of its
rules.

You audit the WHOLE rulebook at once, not one rule at a time, so that you can see
rules interacting: two rules pulling against each other, or one only paying off
when another has already fired, are findings a per-rule view cannot produce.

Read the working file before answering. It covers every rule in one place:

{{eval_file}}

It contains the rulebook's text and a sample of base → rewrite pairs from
applying the WHOLE rulebook, with each rewrite's measured score. A `.json` twin
sits beside it if you would rather compute over it.

There is deliberately NO per-rule score. A single rule is worth well under 1pp,
and resolving that would take ~8,700 measurements per rule — three orders of
magnitude beyond what a run can afford. Any per-rule number we could produce
would be noise, and acting on it would churn the rulebook at random. So judge
rules on ADHERENCE (below) and on what you can see in the rewrites themselves;
judge the BOOK on its measured delta. If you want a controlled contrast for one
specific rule, propose it as an experiment — that is what the planning step is
for, and spending the budget where you have a hypothesis beats spreading it
thinly across every rule.

RULEBOOK UNDER AUDIT:
{{rules}}

## What to report per rule

Two things that are genuinely different, never conflated:

- **adherence** — did the applier actually do what the rule said? Judge this from
  the rewrite samples, not from the numbers.
- **performance** — when it WAS obeyed, did the rewrites look better or worse to
  you, and did the BOOK's measured delta move? There is no per-rule number; say
  so plainly rather than inventing one, and flag which rules you would want a
  controlled experiment on.

A rule can have high adherence and poor performance (it was followed and that was
a bad idea) or the reverse (it is a good idea nobody executed). Those call for
opposite fixes, so state both even when one seems obvious.

Then say **what to do about the rule, and how** — concretely enough to act on. Not
a label. If it should be simplified, give the simpler wording or say which clause
to cut. If it should be narrowed, say to which input kind or which condition. If
it should be dropped, say what evidence would have to appear for it to earn a way
back. If the measurement is too thin to call, say what would settle it. Write
this as prose; the reader is another model deciding what the next rulebook should
contain, and it can weigh a real argument better than a category.

## Suggestions

After every rule is covered, propose **new phrasings to measure** — experiments
that would collect evidence the current bank lacks. These are for the phrase
bank, not for the rulebook: each line is a wording to go and test, plus what its
outcome would tell us. They do not need to correspond to rules one-for-one, and
if the evidence genuinely settles everything you looked at, it is correct to
propose none rather than invent filler.

## Format

Reply with these two delimiter lines verbatim, and nothing before the first:

===RULE NOTES===
1. <adherence, performance with numbers, and what to do about it and how>
2. <...>
(one block per rule; every rule covered; prose, as long as it needs to be)
===SUGGESTIONS===
<one phrasing experiment per line: what to measure and what it would tell us>
(may be empty)
