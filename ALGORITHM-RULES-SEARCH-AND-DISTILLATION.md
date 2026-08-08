# Rules search and distillation

An LLM writes a rulebook for rewriting task instructions. The rulebook is applied
by a rephraser model, the rewrites are scored, and the measurements come back to
the LLM so it can revise. Repeat until validation stops improving.

Run it **twice**: first on training data, then a few times on val8. The output
is **per-model rulebooks, never merged**: each rephraser ends with a
training-derived ruleset (round 1) and a training+val8-derived ruleset (round 2).

---

## The loop

```python
corpus_file = llm.assemble_corpus_stats(corpus_inputs)   # agent-authored, once
scored_bank = load_scored_phrases(exclude=SEALED)        # everything ever measured

def generate_phrases(task):
  return (vlm.do_oracle_search()          # upper bound
        + vlm.generate_adversarials(n=2)  # lower bound
        + vlm.generate_naturals(n=7))     # in between

scored_bank += score([generate_phrases(t) for t in tasks
                      if not covered(scored_bank, t)])

base_pool             = snapshot(scored_bank)      # frozen: see (1)
train_split, val_held = split(base_pool)           # by task
train_eval            = random_sample(train_split, n=SAMPLE_N, seed=FIXED)

# the unrephrased instructions, scored ONCE -- bases are frozen, so these never
# change; every iteration's deltas are read against them
base_mean = {split: mean(score(eval_set)) for split, eval_set in
             [("train", train_eval), ("val_held", val_held), ("val8", val8)]}

rules_per_model = {}
for rephraser in [qwen, claude, gemini]:           # shared bank, per-model rules: see (2)
  session    = new_conversation()                  # distiller + judge + planner: see (3)
  rules      = initial_no_rules_prompt()           # EMPTY: models inherit phrases, not rules
  best       = None        # (rules, val, rules_eval_summary) -- high-water mark
  last       = None        # (rules, val, rules_eval_summary) -- what we just tried
  since_best = 0

  for it in count():
    if since_best >= PATIENCE: break

    if it > 0:                                     # iteration 0 MEASURES the
      rules = session.distill_rules(               # no-rules prompt itself: see (9)
          prev_rules         = best.rules,         # see (4) for the two-argument split
          last_attempt       = last,               # rules + val + rules_eval_summary
          corpus_file        = corpus_file,
          evidence_file      = write_evidence(scored_bank, train_split))

    # apply the whole rulebook, and each rule alone: see (5)
    rewrites    = rephraser.apply(rules, train_eval, traces)
    train_score = mean(score(rewrites))
    per_rule_perf = {r: mean(score(rephraser.apply_only(r, sample, traces)))
                        - base_mean["train"]
                     for r in parse_rules(rules)}
    audit = session.judge(rules, per_rule_perf, rewrites)   # see (6)
    rules_eval_summary = {                         # the user-facing object, as in v0:
        "per_rule_adherence": audit.adherence,     #   did the applier obey each rule?
        "per_rule_perf":      per_rule_perf,       #   measured single-edit delta each
        "suggestions":        audit.suggestions}   #   experiments to run next
    scored_bank += rewrites

    val_held_score = mean(score(rephraser.apply(rules, val_held, traces)))
    val8_score     = mean(score(rephraser.apply(rules, val8,     traces)))
    val            = mean(val_held_score, val8_score)     # see (7)
    plot(train_score, val_held_score, val8_score, vs=base_mean)

    last = (rules, val, rules_eval_summary)
    if best is None or val > best.val:
      best, since_best = last, 0
    else:
      since_best += 1

    # probes may re-list a phrase already in the bank; that re-measures it on
    # fresh contexts and the measurements COMBINE -- see (10)
    scored_bank += score(session.plan_new_phrases(
        rules_eval_summary.suggestions, train_split), draw=it)

  rules_per_model[rephraser] = best.rules

# OUTPUT: one rulebook per (rephraser, round) -- six artifacts total. No merging:
# each model's rules are tuned to its own rule-following capacity (see (2)), and
# a combined book would reintroduce exactly the capacity mismatch that killed v2.
#   round 1 -> rules_r1[model]   (training-derived)
#   round 2 -> rules_r2[model]   (training + val8-derived, starting from rules_r1[model])
return rules_per_model
```

---

## Design notes

**(1) Bases are frozen at run start.**
The bank keeps growing with the loop's own rewrites. If bases were sampled from
the live bank, iteration N's base instructions would partly *be* iteration N−1's
outputs, and the comparison would drift instead of holding still. Same reason the
sample seed is fixed: every iteration-to-iteration difference must be a rules
contrast, not a resampling artifact.

**(2) One bank, three rulebooks.**
A phrase's score is a property of (phrase, task, policy) — it does not depend on
which model wrote the phrase — so the bank is shared and grows monotonically
across passes; later passes inherit earlier ones' exploration. Rules are *not*
shared, and **every model starts round 1 from an empty rulebook**: models inherit
*measurements*, never each other's rules. Rule-following capacity differs enough
that Qwen could not execute the v2 rules at all. Run the strongest applier last,
against the fullest bank.

**(3) One conversation per pass.**
The distiller, judge, and planner share a session, so each sees every rulebook,
rationale, and audit it has already produced. Rule *application* is deliberately
stateless — phrases must not contaminate each other, and statelessness is what
lets applies run in parallel.

**(4) Two matched (rulebook, measurements) pairs.**
This is the entire regression mechanism.

| pair | role |
|---|---|
| best rulebook + its `rules_eval_summary` | what to **build from** |
| regressed rulebook + its `rules_eval_summary` | what to **avoid** — present only after a regression |

A rulebook is never shown beside another rulebook's numbers. Building from the
best stops one bad iteration becoming the base for every later one; showing the
regressed pair stops the inputs being identical to the previous iteration's,
which would make the distiller re-derive the same failing revision forever. And
because both pairs are matched, the distiller can diff them directly — the same
rule measured under both is far more informative than one that appears only in
the loser. No history file is needed: the shared session already holds every
earlier rulebook verbatim.

**(5) Each rule is applied alone as well as together.**
A rule's effect is then a paired single-edit contrast against the same bases,
rather than an attribution guess over rewrites where several rules fired at once.
`parse_rules` reads the `===RULES===` section only — the rationale also contains
numbered lines, and counting those would invent phantom rules and corrupt every
measurement.

**(6) Adherence and performance are separate questions.**
Did the applier obey the rule, and was being told to do it a good idea? A rule can
have high adherence and poor performance, or the reverse, and those imply opposite
fixes. The auditor reports both per rule, then proposes experiments — suggestions
come only after every rule is covered.

**(7) Two validation sets, early-stop on their average.**

| | |
|---|---|
| `val_held` | held-out tasks from the training pool — does a rule generalize in-distribution? |
| `val8` | the sim tasks, the only split with real rollout ground truth |

Both are reported; the average drives early stopping so neither one's noise alone
can end a run. Everything is also charted as a delta against `base_mean` — "did
the rules beat saying nothing" is the decision-relevant view, since the absolute
level drifts with sample difficulty.

**(11) Every phrase carries what kind of input it is.**
`original` (the canonical training instruction), `natural` (a fluent rewording),
`adversarial` (awkward/ornate/indirect), `rephrased` (a rulebook's output),
`search` (turned up by phrasing search or proposed as a probe), plus `unknown`
for measurements that predate the labelling and are not guessed at. A `rephrased`
row also carries `base_kind` — the regime it was rewritten *from*.

This is load-bearing rather than bookkeeping: our own ladder puts rules at
+5.5..+6.3pp on adversarial inputs and +0.4..+2.8pp on natural ones. That
interaction is the finding, and a distiller that cannot see the label averages
the two regimes and tunes for neither. Per-rule single-edit deltas are therefore
reported broken down by regime, and a rule that helps one and not another is
reported as conditional, not weak.

**(10) Every phrase carries how well-measured it is, and can be re-measured.**
The bank stores `n_ctx` (scored contexts behind the estimate) and `n_meas` (how
many separate measurements). Re-measuring **accumulates**: `n_ctx` adds and the
channels become n-weighted means, each draw using different CRN contexts so the
second measurement is independent information rather than a replay of the first.
The evidence table exposes `n_ctx` for exactly this reason — a difference between
two thinly-measured phrases may be noise, and the planner is told it can re-list
such a phrase to buy significance instead of treating a weak reading as settled.

**(9) Iteration 0 measures the no-rules prompt.**
The scaffold prompt is evaluated before any distillation, so the curve has an
anchor, `last_attempt` starts with real numbers instead of nothing, and — because
`best` starts there — early stopping returns the no-rules prompt if no distilled
rulebook ever beats it. (This anchor is the rephraser *with an empty rulebook*;
`base_mean` is the *unrephrased* instruction. Both are meaningful and they
differ — the v11 run measured that gap at several points.)

**(8) Probes keep the bank from collapsing.**
Left alone, the bank fills with whatever region the current rules produce, and the
distiller stops seeing alternatives. The planner proposes phrases in regions the
evidence does not cover. The scores that come back are honest either way — bad
probes enter as evidence that they are bad.

---

## Costs and budgets

Scoring is the **proxy** throughout the loop. Real rollouts are spent only at
round boundaries, on a few iterations, to confirm what the proxy selected.

```
forward passes  = phrases_scored × F × C
phrases_scored  = 3·SAMPLE_N + SINGLE_EDIT_N·(R+1) + PROBES        (R = rule count)
LLM calls       = 5 + 3·SAMPLE_N + SINGLE_EDIT_N·R
```

`SINGLE_EDIT_N` sets the iteration length — it multiplies both terms. Tune it
first, `SAMPLE_N` second; probes are noise by comparison. Traces are loaded once
and reused for every application; they are never regenerated.

All scoring is CRN-seeded: the same (task, layout, rep) draws the same policy
noise for every phrase, so phrase contrasts are not swamped by decode variance.

## Splits and what may be reported

Round 1 early-stops on val8; round 2 trains on it (each model's round-2 pass
starts from its own round-1 rulebook). Either way val8 is spent as an evaluation
instrument — **the sealed set certifies the final rulebooks, separately per
model.**

Report *both* val8 numbers: the round-1 rules (proxy-distilled, val8 seen only
through early stopping) and the round-2 rules (distilled against val8 rollouts).
The gap between them is exactly what real rollout data bought.
