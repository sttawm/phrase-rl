# v12: magnitude-reward, multiplicity-collapsed advantage tuning

Status: SPEC, ready to run. v11 continues until swapped — pausing it is a
decision, not a side effect. v11 is resumable from `v11_step_0150` on origin.

Supersedes the first draft of this file (2026-08-08). Three of its
recommendations were wrong and are corrected below, each with the measurement
that overturned it.

---

## What v11 measured

153 steps, no measurable change in rollout success: trend **−0.0103 pp/step**,
95% CI [−0.033, +0.012], p=0.39, over 13 nat24 cells.

**1. The policy climbed its reward and the gain did not transfer.** Paired
against the original instruction on the same contexts and CRN draws, the
candidate margin closed from 2.905 → 2.391 nats (t = −2.64): the policy recovered
**17.7% of the gap to the unrephrased instruction in ~100 steps**, and the
original still beats roughly three quarters of its rewrites. Meanwhile the
verifier channel improved 0.55–0.85 while gripper error moved 0.005. Under the
calibrated proxy's own coefficients the verifier gain alone predicts **+6.0pp**
of rollout success; measured −1.1 ± 1.3pp. That gap is the Goodhart signature
`EXPERIMENT.md` line 63 predicted.

The transfer calculation assumes coefficients fitted ACROSS PHRASES also describe
changes induced BY TRAINING. They need not — and its failure is precisely the
evidence for off-manifold optimisation.

**2. `cand_blend_mean` cannot move.** `rank01` emits average ranks, so any convex
blend of two rank01 vectors sums to n/2. With P=17 the statistic reduces to
`(8.5 − blend_rank_of_the_ORIGINAL)/16` ∈ [0.469, 0.531] — a 6%-wide window
reporting one fact. It sat at 0.4852 → 0.4867 for 161 steps and I read it as
proof of no learning. **Retire it.**

**3. `rank01` manufactures gradient from floating-point noise.** Identical
strings, identical frames, identical draws differ in the last bits (batched GPU
reductions are not bit-deterministic); `rank01` promotes ~1e-7 to a full rank
step. Of 1042 duplicate sets at step 150, **306 (29.4%)** carry a spurious
spread, and every nonzero spread is an exact multiple of a rank step
(0.0156 = 0.25/16, 0.0469 = 0.75/16, 0.0625 = 1/16) — the signature of
tie-breaking, not of differing frames or seeds.

**4. Collapse is real but is NOT why val is flat.** Median 6 unique candidates of
16 at step 150 (min 1, mean duplicate fraction 0.579). But v10 ran its whole life
at dup_rate 0.185 → 0.314 with identical flags and model, and its val was equally
flat (v10_pol +0.0037 pp/step t=1.56; v10_adv −0.0036 t=−0.63). **Duplication
varied ~2× across runs and the downstream metric did not move.** The first draft
of this spec leaned on collapse as the cause. That was wrong.

---

## Design

### Reward: calibrated proxy LOGIT

    reward = 0.4445 * z + 11.3193 * (-grip)        # intercept dropped:
                                                   # GRPO subtracts the group mean
Never the sigmoid — on training frames grip ≈ 0.04 sits on the flat end and the
probability pins at 0.999. Clip the logit to a sane range.

Within task, against real rollout success (8 tasks, 254 phrases, 36 rollouts
each), and on a degenerate group (16 candidates identical to within 1e-7):

| reward | Pearson | Spearman | max abs advantage on a DEGENERATE group |
|---|---|---|---|
| c4b rank01 w=0.25 (v11) | +0.478 | +0.442 | 1.790 |
| raw-std w=0.25 | **+0.549** | +0.445 | **2.016** |
| **proxy logit** | +0.533 | **+0.460** | **0.903** |

Chosen for the two properties that bear on training dynamics: best **Spearman**
(a ranking-consumed reward needs rank fidelity), and it is the only form that
**refuses to invent signal**. Fixed coefficients keep ε at ε until GRPO's
`(r − mean)/(std + 1e-6)`, where the floor caps it. Both group-adaptive forms
re-normalise by the group's own spread, so ε is stretched back to unit scale —
`raw-std` is marginally WORSE than `rank01` here, which is why it is an arm and
not the default despite winning Pearson.

Its coefficients also weight the channel that transfers (grip) **25.5×** the
channel that is exploitable (verifier) — inverting c4b's rank weighting, under
which a 0.25-weighted consistent signal beat a 0.75-weighted noisy one.

`--blend-w 0.75` is rejected: within group it reverses (raw-std 0.492 at w=0.75
vs 0.549 at w=0.25). The `fine_grid` case for it is a cross-task, large-gap
effect the update never sees.

### Generation: keep `sample_single`, keep K=16

The first draft proposed list-prompting for K=8 unique. Both halves are dropped:

- **K=16 stays.** Its justification was collapse, which finding (4) falsified.
  Best-of-K by the current reward degrades with K (+3.27pp at K=4 → +2.07pp at
  K=16) but never goes negative, and that penalty only bites when the update
  concentrates on the top — i.e. under RAFT or `--max-pos-per-ctx`, neither of
  which we use.
- **`sample_single` stays**, and this is the load-bearing reason: it samples from
  the prompt being updated, so signed full-group advantages are an unbiased
  on-policy estimator with **no importance correction needed**. List mode samples
  from `p_list` while updating `p_single`, which would require the computed
  weight `exp(log p_single − log p_list)` and its variance. Since duplication is
  not the problem, that machinery buys nothing.

### Update: signed advantages over the full group

All 16 candidates, ranked, positives and negatives. Not positives-only (winner's
curse: imitating the argmax of K noisy estimates at ~73% pairwise accuracy), not
top-K. Errors distribute across the ranking instead of concentrating on the pick.

### Budget: multiplicity-collapsed scoring buys C=10 for free

Score the **distinct** strings, then `np.repeat` rewards AND grips back to the
full 16-multiset **before** the reward transform. This is exact, not an
approximation: identical text has identical channels by construction. Collapsing
without re-expanding would change the group statistics and silently alter the
objective.

At mean duplicate fraction 0.579, E[unique] ≈ 7.7 of 16:

    v11:  16 slots x C=5  x F=4 = 320 evals/group
    v12: 7.7 slots x C=10 x F=4 = 308 evals/group      # same cost

Why C=10 — pairwise accuracy at F=4 on the band a GRPO group actually occupies
(closest pairs), from `fine_grid_all.json`:

| C | 5–10pp pairs | 15+pp pairs |
|---|---|---|
| 4 | 71.7 | 63.9 |
| 8 | 76.2 | 67.5 |
| **10** | **78.2** | 67.7 |
| 16 | 78.6 | 70.1 |
| 20 | 79.8 | 70.3 |

C=5→10 buys ~**+5pp** on close pairs; saturation begins only after C=10. (An
earlier claim that C=5→10 is saturated came from reading the `far_15+` column —
far pairs are exactly what a group of near-identical candidates never contains.)

The proxy was fitted at F=4, C=10–24 (median 18, F×C ≈ 72). C=10 also closes most
of that calibration gap. Both channels are means, so the linear predictor's
EXPECTATION is budget-independent and ranking stays valid at any C — but the
probability calibration is not, and must never be read as a success rate at
training budget.

### Skip uninformative groups

GRPO normalises every group to unit advantage variance, so a group with no real
spread contributes gradient as large as one with signal. Skip when the **raw**
channel spread is below a floor. No reward form can rank a group that has
nothing to rank.

### Fix the ratio: sum, not mean

`apply_update` computes `ratio = exp(mean_logp - old_lp)` = `true_ratio^(1/n)`.
For an 8–12 token phrase a genuine 2× ratio becomes ~1.07, inside the [0.8, 1.2]
clip, so `--replay-clip` never binds and replayed samples (up to 6 reuses) are
effectively uncorrected. Cosmetic today, load-bearing the moment a ratio carries
weight. Fix with this change.

---

## Configuration

    bash scripts/run_arm_v12.sh

**Amendment (2026-08-11, user decision, pre-step-10 restart):** image
conditioning restored -- `--prompt-family bplusimg` (prompt B+ text with the
camera frame as the first user content part). This makes the v11 comparison
two-variable (reward AND image); accepted knowingly ("who knows, it may
help"). Val cadence 10 (was 20), aligned with the checkpoint stride; the val
probe reports natural-input and adversarial-input series on the same frozen
contexts. The bplus-generated step-0 cells were discarded and regenerated
under bplusimg (a policy conditioned differently at step 0 is a different
step-0 policy; the paired-judge logic requires measuring THAT one).

Deltas from `run_arm_v11.sh`:

    --reward-blend proxy_logit      # NEW; was c4b
    --reward-contexts 10            # was 5 (paid for by collapse-scoring)
    --collapse-duplicate-scoring    # NEW: score distinct, re-expand before blend
    --min-group-spread 0.02         # NEW: skip noise-only groups
    --ratio-mode sum                # NEW: was an implicit mean
    (drop --no-gate)                # gate ON

Unchanged: `--gen-mode sample_single`, `--n-candidates 16`, `--update-rule grpo`,
beta 0.15, lr 7e-6, kl-abort 1.2, F=4, save-every 10, source-mix 0.25/0.5/0.25.

**Registered arms** (one variable each, run only if the primary is ambiguous):
- **A — raw-std reward:** `w·zscore(z) + (1−w)·zscore(−grip)`, w=0.25. Wins
  Pearson (+0.549) and is the only form with no task ranking backwards, but is
  the worst on degenerate groups.
- **B — list K=8 + importance correction:** tests diversity directly; requires
  `exp(log p_single − log p_list)` clipped, and `iw_clip_frac` logged.

---

## Telemetry v11 lacked

Per step: `n_unique_cands`, `dup_frac`, `reward_std_within_group` (raw channel
units, not post-transform), `grip_delta` and `verifier_delta` **separately** (the
Goodhart signature is the gap between them), `cand_margin_logit` and
`cand_margin_grip` (paired against the original — the only v11 training metric
with real dynamic range, t = −2.64), and **candidate text + per-candidate raw z
and grip to a parquet**. `replay.pt` preserved text and blended reward for a
50-step window by luck, not design; that is what made the ε-amplification
diagnosis possible at all.

Delete `cand_blend_mean` or rename it `..._IS_A_RANK_DO_NOT_READ_AS_PROGRESS`.

---

## Falsification (pre-registered)

**Step-60 review (2026-08-12, recorded; training continues per user order):**
ground truth 7 cells 44.79/45.31/42.71/41.67/40.62/39.06/39.58 -- trend
**-1.08pp per 10 steps, 95% CI [-1.37, -0.79]**, projecting ~29 at step 150
(passthrough floor 37.5). Kill criteria: Goodhart signature **TRIPPED** --
fixed-val verifier delta +0.60 (>0.3) while grip WORSENED (+0.0016 error);
the entire blended-reward climb is verifier-channel. Spread floor NOT tripped
(0.19-0.84, 2 groups skipped total); margin IS falling (policy climbs its
reward). Interpretation: v12's defenses removed the rank-noise pathway and the
policy still Goodharts -- via genuine verifier exploitation, faster and more
damaging than v11 (flat -> actively negative), while keeping grip flat
(physically plausible text that games the learned channel). The verifier
network itself is the exploit surface; fixed coefficients cannot leash it.
Direct evidence for uncertainty-penalized scoring + exploit hard-negatives
(v13 plan).

**Amendment (2026-08-12, user order): the step-60 kill criteria are ADVISORY
for this run** -- report any trip, but train through the night regardless
("I want to see what happens"). The 36h autostop guard remains the only hard
stop.

**Kill by step 60 if:**
- `verifier_delta` improves > 0.3 while `grip_delta` < 0.01 — Goodharting again
- `reward_std_within_group` < 2× measured reward noise — the reward cannot rank
  its own candidates, and no update rule repairs that
- `cand_margin_logit` fails to fall — the policy is not even climbing its reward

**Judge at step 0 AND step 150** on nat24 with **768 episodes per cell, not
192**, both cells on the SAME episode set (CRN: identical layouts, reps, seeds)
so the comparison is paired and shared episode noise cancels. At 192 the
per-cell s.e. is 3.56pp and the minimum detectable slope is 22.6pp/1000 steps
-- v11 could have been improving at +20pp/1000 and we could not have seen it.
The step-0 judge matters for three reasons: without it the difference s.e. is
sqrt(3.6^2 + 1.8^2) ~= 4.0pp and the baseline's noise vetoes the endpoint
measurement (with both at 768: ~2.5pp, so +5pp ~= 2 sigma, tighter still under
pairing); it validates the assumption that v12's step-0 policy equals v11's
(any prompt/sampling delta silently breaks the inherited 42.19); and it runs
concurrently with early training since nothing gates on it (the step-60 kill
criteria are telemetry-only). ~5 GPU-h.

**Success:** nat24 rises >= 5pp over the MEASURED step-0 judge (42.19 is the
inherited reference, not the comparator), trend CI excluding zero.

---

## FINAL VERDICT (2026-08-13)

Paired 768-episode judges, identical CRN episode sets: **step 0 = 44.40,
step 150 = 39.71 -- a 4.7pp DECLINE** (bar was >= 49.4). Training made the
rephraser worse on natural inputs than not training at all, ending 2.2pp above
the no-rephraser floor (37.5). Reward climbed throughout (fixed-val
original-input greedy -0.65 -> +0.05; every unit verifier-channel, grip flat).
Mechanism identified and quantified: IDENTITY COLLAPSE -- near-verbatim copy
rate of sampled candidates rose 13% -> 47% by step 120 (mean input-similarity
0.68 -> 0.82); copying pays on the 25% original tier and, under GRPO
mean-centering, is the zero-risk move everywhere once copies dominate the
group mean. The magnitude reward, collapsed scoring, spread floor, and sum
ratio all functioned as designed -- the failure came through the training
DISTRIBUTION, not the reward transform. Tail to step 158 archived
(v12_latest). Successor: v13 (originals removed, 0/0.67/0.33) launched
2026-08-13 with copy_rate as first-class telemetry.

## v13 FINAL VERDICT (2026-08-15)

v13 (originals removed, 0/0.67/0.33; single variable vs v12) -- paired
768-episode judges, same construction: **step 0 = 44.40, step 140 = 40.36:
-4.0pp** (v12: -4.7pp at step 150; difference +0.65 is within pairing noise).
Judge ran at 140 not 150: a stray autostop stopped the pod 2 steps short and
its host had no free GPU; the 140 phrases were already minted (deterministic),
amendment documented. Copy rate stayed contained the whole run (~0.02-0.12
with benign near-canonical spikes vs v12's ramp to 47%); mid-run the curve
recovered to -1.0 of baseline (step 70) before sagging back to the 40-41 band.

**Conclusion across the v10-v13 arc:** rank-noise amplification (v12 fixed),
identity collapse (v13 fixed), and input-mix confounds are each ELIMINATED as
the cause -- and the outcome is unchanged. The proxy's correlation with
rollout success does not survive optimization pressure via any pathway we can
close from the outside; the exploit surface is the learned verifier itself
(v13's reward climbed on all tiers while truth fell, with honest rewriting).
Direct motivation for reward-hardening (uncertainty-penalized ensemble
scoring, exploit hard-negatives) before any further RL spend; the rules loop
(#27) remains the productive path meanwhile.

## Resumability (standing order)

Checkpoints carry full state: `adapter_model.safetensors`, `optimizer.pt`,
`replay.pt`, `rng.pt`, `trainer_state.json`.

| run | on origin | resume |
|---|---|---|
| v10 | `v10_step_0010/0020/0030/0230/0240/0250/0260`, `v10_best_val`, `v10_latest` (step 260) | `ckpt_archive.sh unpack v10_latest` → `run_arm_v10.sh` |
| v11 | `v11_step_0010` … `v11_step_0150` | `ckpt_archive.sh unpack v11_step_0150` → `run_arm_v11.sh` |

v10 steps 0040–0220 are **lost** — rotated locally before ever being pushed.
`scripts/ckpt_drain_v11.sh` prevents recurrence: it verifies each checkpoint
against `git ls-tree origin/main` before dropping the local copy. Verify by
ORIGIN, never by a push exit code — `git push` exits 0 when nothing was staged,
which is how v11 steps 0140/0150 were logged "pushed" while origin held 0130.
