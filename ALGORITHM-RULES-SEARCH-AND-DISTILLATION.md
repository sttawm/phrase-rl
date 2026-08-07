# This should be run in two rounds
# Initially, it should be run on the training data (a mix of simulation tasks and
# training tasks). ROUND 1 VALIDATES ON VAL8: val8 is where we hold real rollout
# ground truth, so it is the only split that can confirm what the proxy selected.
#
# Once those rules are generated, it can be run a few times on the val8 split,
# without a validation split.
#
# In the end, the two sets of rules can be combined.
#
# COST: scoring is the proxy (sampled context/frame points) throughout the loop.
# Real VLA rollouts are spent only at round boundaries, on a few iterations, to
# confirm what the proxy selected. The proxy is cheap enough to score thousands
# of phrases per iteration, which is what makes the val signal resolvable at all:
# a val split of ~8 tasks x 24 layouts of rollouts is +-3.5pp, larger than the
# effects we are chasing.
#
# NOTE ON VAL8: round 1 early-stops on it, round 2 trains on it. Either way val8
# is consumed as an evaluation instrument -- after this, only the sealed set can
# report a number.

rules = get_initial_no_rules_prompt()
corpus_summary = vlm.make_corpus_summary_files()

def generate_phrases(task):
  upper_bound = vlm.do_oracle_search()
  lower_bound = vlm.generate_adversarials(n=2)
  in_between = vlm.generate_naturals(n=7)
  return upper_bound + lower_bound + in_between

task_phrases = [generate_phrases(task) for task in tasks]
train_split, val_split = split(task_phrases)   # round 1: val_split IS val8

train_split_eval = random_sample(train_split, n=SAMPLE_N)
val_split_eval = val_split

# all scoring is CRN-seeded: the same (task, layout, rep) draws the same policy
# noise for every phrase, so phrase contrasts are not swamped by decode variance
scores_train = score(train_split)
scores_val = score(val_split)

val_scores = []
rules_eval_summary = None

# early stopping on best val, with patience -- a plain divergence test would fire
# on a single noisy iteration. We keep the best-val rules, not the last ones.
best_val, best_rules, since_best = -inf, None, 0

while since_best < PATIENCE:
  # distill_rules must reconcile conflicts between prev_rules and what the data
  # now shows -- it has prev_rules, the corpus summary, every scored phrase, and
  # per-rule evidence from the last round. It decides which side of a conflict
  # wins, and emits the reasoning plus any follow-up experiments as suggestions.
  rules = llm.distill_rules(prev_rules=rules, corpus_summary, scores_train, rules_eval_summary)

  # Eval on training data.
  #
  # Each rule is applied INDIVIDUALLY as well as all-together, against the same
  # base phrase, so a rule's effect is a paired single-edit contrast rather than
  # an attribution guess over phrases where several rules fired at once.
  #
  # Returns:
  #   - rephraser_score: the success rate of the vla, or the proxy's estimate
  #   - evaluated_phrases: the vla's success-rate (or proxy) for the rephraser's generated phrases
  #   - rules_eval_summary:
  #     - per_rule_adherence: summary of each rule's adherence
  #     - per_rule_perf: single-edit effect of each rule (rule-only phrase vs base)
  #     - suggestions: suggestions for further phrase experiments to perform
  train_score, evaluated_phrases, rules_eval_summary = eval(rules, train_split_eval)
  scores_train += evaluated_phrases

  # Eval on validation data
  val_score, _, _ = eval(rules, val_split_eval)
  val_scores.append((rules, val_score))

  if val_score > best_val:
    best_val, best_rules, since_best = val_score, rules, 0
  else:
    since_best += 1

  plot(train_score, val_score)

  # keeps the phrase bank from collapsing onto whatever the current rules emit:
  # the distiller proposes probes into regions it has little evidence about
  new_phrases = llm.plan_new_phrases(rules_eval_summary.suggestions)
  scores_train += score(new_phrases)

return best_rules
