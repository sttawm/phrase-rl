# This should be run in two rounds
# Initially, it should be run on the training data (a mix of simulation tasks and
# training tasks). ROUND 1 VALIDATES ON TWO SETS: held-out training tasks
# (in-distribution generalization) and val8 (the only split with real rollout
# ground truth, so the only one that can confirm what the proxy selected).
#
# Once those rules are generated, it can be run a few times on the val8 split,
# without a validation split.
#
# In the end, the two sets of rules can be combined.
#
# This whole thing runs once per rephraser model (Qwen, Claude, Gemini),
# sharing one scored-phrase bank -- see the three-passes note below.
#
# COST: scoring is the proxy (sampled context/frame points) throughout the loop.
# Real VLA rollouts are spent only at round boundaries, on a few iterations, to
# confirm what the proxy selected. Forward passes = phrases_scored x F x C, where
# phrases_scored = 3*SAMPLE_N + SINGLE_EDIT_N*(R+1) + PROBES. The single-edit term
# is what sets the iteration length -- tune SINGLE_EDIT_N first, probes last.
#
# NOTE ON VAL8: round 1 early-stops on it (jointly with the held-out training
# tasks), round 2 trains on it. Report BOTH val8 numbers -- the round-1 rules
# (proxy-distilled, val8 seen only through early stopping) and the round-2 rules
# (distilled against val8 rollouts). The gap between them is exactly what real
# rollout data bought, which is worth knowing. Neither is clean held-out by the
# end: the sealed set certifies the final rules.

corpus_file = llm.assemble_corpus_stats(corpus_inputs)   # agent-authored, once
scored_bank = load_scored_phrases(exclude=SEALED)        # everything ever measured

def generate_phrases(task):
  upper_bound = vlm.do_oracle_search()
  lower_bound = vlm.generate_adversarials(n=2)
  in_between  = vlm.generate_naturals(n=7)
  return upper_bound + lower_bound + in_between

# only pay for phrases the bank does not already cover
task_phrases = [generate_phrases(t) for t in tasks if not covered(scored_bank, t)]
scored_bank += score(task_phrases)

# BASES ARE FROZEN HERE. The bank keeps growing with the loop's own rewrites, so
# sampling bases from the live bank would make iteration N's bases be iteration
# N-1's outputs -- the comparison would drift instead of holding still.
base_pool = snapshot(scored_bank)

# TWO validation sets in round 1, answering different questions:
#   val_held -- held-out tasks from the training pool: does a rule generalize
#               in-distribution, to tasks the distiller never saw?
#   val8     -- the sim val tasks, the only split with real rollout ground truth
# Report both, and their average. Early-stop on the average so neither one's
# noise alone can end the run.
train_split, val_held = split(base_pool)

# fixed for the whole run, so every iteration-to-iteration difference is a rules
# contrast rather than a resampling artifact
train_eval    = random_sample(train_split, n=SAMPLE_N, seed=FIXED)
val_held_eval = val_held
val8_eval     = val8

# all scoring is CRN-seeded: the same (task, layout, rep) draws the same policy
# noise for every phrase, so phrase contrasts are not swamped by decode variance.
# Scene descriptions (traces) are loaded once and reused for every application --
# they are expensive and are never regenerated.

# THREE PASSES, one per rephraser: Qwen, Claude, Gemini. The scored bank is
# SHARED and grows monotonically across passes -- each pass adds its rephraser's
# outputs, so later passes inherit the earlier ones' exploration. The RULES are
# not shared: rule-following capacity differs enough that Qwen could not execute
# the v2 rules at all, so each model gets rules tailored to it.
# Run the strongest applier last, so it distills against the fullest bank.
rules_per_model = {}
for rephraser in [qwen, claude, gemini]:
  session = new_conversation()     # distiller + judge + planner share it, so each
                                   # sees every rulebook and audit it has produced
  rules      = initial_no_rules_prompt()
  best       = None                # (rules, val, eval, audit) -- the high-water mark
  last       = None                # (rules, val, eval, audit) -- what we just tried
  since_best = 0

  while since_best < PATIENCE:

    # ---- distil -----------------------------------------------------------
    # prev_rules and last_attempt are DIFFERENT THINGS and the loop needs both:
    #   prev_rules   = where to build from  -> the best rulebook so far
    #   last_attempt = what just happened   -> possibly a regression
    # Without the first, one bad iteration becomes the base for every later one
    # and the search random-walks away from its own best point. Without the
    # second, after a rollback the inputs would be IDENTICAL to the previous
    # iteration's, and the distiller would re-derive the same failing revision
    # forever. Together they are the whole of the regression machinery -- no
    # separate history is needed, since the shared session already holds every
    # earlier rulebook and audit verbatim.
    rules = session.distill_rules(
        prev_rules    = best.rules if best else rules,
        last_attempt  = last,               # rules + val + per-rule deltas + audit
        corpus_file   = corpus_file,
        evidence_file = write_evidence(scored_bank, train_split))   # full spread, CSV

    # ---- apply and measure on the training sample -------------------------
    # Each rule is applied INDIVIDUALLY as well as all-together, against the same
    # base phrases, so a rule's effect is a paired single-edit contrast rather
    # than an attribution guess over phrases where several rules fired at once.
    rewrites    = rephraser.apply(rules, train_eval, traces)
    train_score = mean(score(rewrites))
    base_mean   = mean(score(train_eval))          # the unrephrased instructions
    per_rule    = {r: mean(score(rephraser.apply_only(r, sample, traces))) - base_mean
                   for r in parse_rules(rules)}    # RULES section only, never RATIONALE
    eval        = write_eval(rules, per_rule, rewrites, base_mean, train_score)
    audit       = session.judge(rules, eval)       # adherence AND performance, then suggestions
    scored_bank += rewrites                        # honest scores, good or bad

    # ---- measure on both validation sets ----------------------------------
    val_held_score = mean(score(rephraser.apply(rules, val_held_eval, traces)))
    val8_score     = mean(score(rephraser.apply(rules, val8_eval,     traces)))
    val            = mean(val_held_score, val8_score)
    plot(train_score, val_held_score, val8_score)  # each also as a delta vs base_mean

    # ---- the only place regression is handled -----------------------------
    last = (rules, val, eval, audit)     # the next distil sees this, pass or fail
    if best is None or val > best.val:
      best, since_best = last, 0
    else:
      since_best += 1                    # next distil builds from `best`, informed by `last`

    # ---- widen the bank ---------------------------------------------------
    # keeps the phrase bank from collapsing onto whatever the current rules emit:
    # the planner proposes probes into regions the evidence does not cover
    probes = session.plan_new_phrases(audit.suggestions, evidence_file, train_split)
    scored_bank += score(probes)

  rules_per_model[rephraser] = best.rules

# combine at the end: per-model rules plus whatever is common to all three
final_rules = combine(rules_per_model)
