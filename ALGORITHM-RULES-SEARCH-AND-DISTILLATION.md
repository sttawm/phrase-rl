# Rules search and distillation

An LLM writes a rulebook for rewriting task instructions. The rulebook is applied
by a rephraser model, the rewrites are scored, and the measurements come back to
the LLM so it can revise. Repeat until validation stops improving.

Run it **twice**: first on training data, then a few times on val8. Combine the
two rulebooks at the end.

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

rules_per_model = {}
for rephraser in [qwen, claude, gemini]:           # shared bank, per-model rules: see (2)
  session    = new_conversation()                  # distiller + judge + planner: see (3)
  rules      = initial_no_rules_prompt()
  best       = None        # (rules, val, eval, audit) -- high-water mark
  last       = None        # (rules, val, eval, audit) -- what we just tried
  since_best = 0

  while since_best < PATIENCE:

    rules = session.distill_rules(                 # see (4) for the two-argument split
        prev_rules    = best.rules if best else rules,
        last_attempt  = last,
        corpus_file   = corpus_file,
        evidence_file = write_evidence(scored_bank, train_split))

    # apply the whole rulebook, and each rule alone: see (5)
    rewrites    = rephraser.apply(rules, train_eval, traces)
    base_mean   = mean(score(train_eval))          # the unrephrased instructions
    train_score = mean(score(rewrites))
    per_rule    = {r: mean(score(rephraser.apply_only(r, sample, traces))) - base_mean
                   for r in parse_rules(rules)}
    eval        = write_eval(rules, per_rule, rewrites, base_mean, train_score)
    audit       = session.judge(rules, eval)       # adherence AND performance: see (6)
    scored_bank += rewrites

    val_held_score = mean(score(rephraser.apply(rules, val_held, traces)))
    val8_score     = mean(score(rephraser.apply(rules, val8,     traces)))
    val            = mean(val_held_score, val8_score)     # see (7)
    plot(train_score, val_held_score, val8_score, vs=base_mean)

    last = (rules, val, eval, audit)
    if best is None or val > best.val:
      best, since_best = last, 0
    else:
      since_best += 1

    scored_bank += score(session.plan_new_phrases(audit.suggestions, train_split))  # see (8)

  rules_per_model[rephraser] = best.rules

final_rules = combine(rules_per_model)
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
shared: rule-following capacity differs enough that Qwen could not execute the v2
rules at all. Run the strongest applier last, against the fullest bank.

**(3) One conversation per pass.**
The distiller, judge, and planner share a session, so each sees every rulebook,
rationale, and audit it has already produced. Rule *application* is deliberately
stateless — phrases must not contaminate each other, and statelessness is what
lets applies run in parallel.

**(4) `prev_rules` and `last_attempt` are different things, and both are needed.**
This is the entire regression mechanism.

| | |
|---|---|
| `prev_rules` | where to **build from** — the best rulebook so far |
| `last_attempt` | what **just happened** — possibly a regression |

Without the first, one bad iteration becomes the base for every later one and the
search random-walks away from its own best point. Without the second, after a
rollback the inputs would be *identical* to the previous iteration's and the
distiller would re-derive the same failing revision forever. No separate history
file is needed — the shared session already holds every earlier rulebook verbatim;
what it cannot supply is the structured measurement of the attempt that just
failed, which is what `last_attempt` carries.

After a regression the eval file describes a **different rulebook** than the one
printed above it. That pairing is deliberate and the prompt says so.

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

Round 1 early-stops on val8; round 2 trains on it. Either way val8 is spent as an
evaluation instrument — **the sealed set certifies the final rules.**

Report *both* val8 numbers: the round-1 rules (proxy-distilled, val8 seen only
through early stopping) and the round-2 rules (distilled against val8 rollouts).
The gap between them is exactly what real rollout data bought.
