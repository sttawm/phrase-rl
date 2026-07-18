# Reward-Function Repair Plan (v7 Track 1)

Goal: a reward whose FINE axis (ranking among already-relevant candidates)
predicts rollout success — validated offline BEFORE any training run uses it;
style-level (claims about register/structure repair, not object-noun lookup);
argmax-safe (GRPO exploits the top of the ranking — v5/v6 precedent).

Constraint carried from the style-claim decision: new-task data may VALIDATE
the judge (transfer exams) but never TRAIN it and never enter the RL reward
loop — OOD-task wins must remain attributable to style transfer.

## Step 0 — Freeze the exam first (pre-registration; $0, Mac)
Lock the tests before evaluating any candidate, so repair cannot overfit them.
- 0a. **Gate-Zero pair registry**: ~8-10 phrase pairs with measured rollout
  deltas >= ~10pp at n>=288/phrase and CI excluding zero, harvested from the
  existing eval12/screen parquets: spoon +/- "center" (-12.8pp), the wheel
  family ("on the black tire" 29.2 / "center of the black tire" 9.7 / "center
  of the tire" 0.0), coke "red can"->"red cola can" (+9.7), stack (+6.3),
  eggplant, ramekin "vertically inside" (pending Step-1 audit). Artifact:
  `results/analysis/gate_zero_pairs.json`.
  NOTE: each completed sampled-k8 screen adds 8 distinct (phrase -> success)
  cells per task — fold them in as they land; they are the cheapest new pairs.
- 0b. **Regret exam**: per task with >=3 measured phrases, the phrase->success
  table; metric = top-1/top-3 regret under each candidate's argmax; cap <=6pp
  with clustered-bootstrap CI lower bound.
- 0c. **Transfer exam**: leave-task-out across the 4 native tasks; later
  out-of-sample rows from >=1 new task (Step 5) — validation only.
- 0d. **Normalization control**: case/punctuation-normalized duplicates must
  rank identically (tokenization-keying detector).

## Step 1 — Ramekin cross-pipeline audit (pod4, ~$2, ~1h)
Score repair's "vertically inside the white bowl" and the clean phrase under
BOTH pipelines on the same 24 episodes x12. Decides whether the ramekin
inversion (repair 63.9 > clean 31.9) is a real phrasing effect (enters 0a) or
an artifact (corrects every headroom denominator).

## Step 2 — Candidate matrix (all offline; $0 GPU; ~half day)
- C1 current 4f ensemble (baseline to beat)
- C2 relative+floored gate (>= group median AND > frozen-passthrough floor)
  + raw grip fine term
- C3 = C2 with winsorized grip (clip tails; kills the 19.4pp stack regret
  outliers a policy would chase)
- C4 rank-blend fine = w*rank(gate logit) + (1-w)*rank(grip), w in {.25,.5}
- C5 = C2/C3 + MODIFIER-ECHO PENALTY: penalize spatial/adverbial qualifiers
  imported from the hostile source and absent from the nominal ("center",
  "exactly", "precisely", "vertically", "gently", ...) — the style-level term;
  also tested standalone as an add-on to C1.
- C6 reference row only: gated chunk-MLP (already measured 0.793/0.778/0.604 —
  no new engineering unless it surprises).
- Per-task gate calibration variant for whichever leads.

## Step 3 — Run the exams; select or kill ($0)
Selection rule (pre-registered): pass ALL Gate-Zero signs -> among passers,
lowest max-regret with CI clearance -> tie-break on LTO transfer.
Artifacts: `results/analysis/reward_bakeoff_v2.json` + memo addendum.
**FAIL branch** (no candidate passes Gate Zero): the offline fine-axis path is
dead. Options escalate to the project lead: (a) coarse-gate + echo-penalty
only (style shaping, modest ambition), (b) guardrailed rollout reward
(held-out layouts, sealed tasks, memorization audits — carries the
memorization risk the style claim excludes), (c) writeup-first on banked
results. No silent fallback.

## Step 4 — Within-group replay on logged GRPO groups ($0)
Apply the winner to logged v6-B groups: fraction of groups whose argmax
changes; gate pass-rate distribution; degenerate-group fraction (near-zero
advantage spread). If groups are too homogeneous for the gate to split,
the composition needs re-tuning before any launch.

## Step 5 — Out-of-sample validation rows (pod4, ~$3-5)
1-2 improvable tasks (from the Track-2 screen): ~40-60 frozen rollouts with
success labels, feature extraction, score with the winner, report
pairwise/regret out-of-sample. Validation only — never training data.
(Grip/gate readouts are sim-validated — grip scored 1.0 on eggplant sim;
learned-head candidates must NOT lean on sim rows, per the inversion caveat.)

## Step 6 — Integration + reference re-pin (~half day eng + $1)
Server: dual mode returning ensemble logit AND grip per phrase (l2 path
exists). Trainer: reward-transform module (gate composition, penalty) with
unit tests. Re-pin val-40 references under the new reward (orig, frozen
greedy, frozen ERT honest, passthrough) BEFORE any training consumes it;
new criterion lines on the charts. 5-step smoke run on the B pod after the
v6 stop rule fires.

## Step 7 — v7 launch decision (with the project lead)
Design A: v6 recipe frozen (n=16, tags, dropout, stratified 2/2/4), reward
swapped, KL-adaptive beta (target ~0.3 @ step 300), per-group n_unique +
degenerate-group logging, echo telemetry with pre-registered aborts.
~$55-70, 2 days. Primary endpoint: paired ID-4 contrast (style-attributable
headroom ~7pp); secondary: OOD-4 transfer (style generalization).

## Timeline / cost
Day 0: Steps 0-2 (~$2) | Day 1: Steps 3-4 ($0) + 5 (~$5) | Day 2: Step 6 +
decision | Days 2-4: v7 run (~$55-70). Total ~$65-80.
