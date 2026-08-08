# v12: list-prompted, advantage-weighted rephrase tuning

Status: SPEC. v11 keeps running until this is ready; pausing v11 is a decision,
not a side effect of starting v12.

---

## Why v12 exists

v11 ran 153 steps and produced no measurable change in rollout success (trend
−0.0103 pp/step, 95% CI [−0.033, +0.012], p=0.39). Four measurements taken on
2026-08-08 explain that, and each one names a change below.

**1. The policy optimised the reward hard, and the gain did not transfer.**

| absolute metric (lower = better) | first 20 steps | last 20 steps |
|---|---|---|
| `cand_loss[nominal]` — verifier channel | 1.027 | **0.481** |
| `cand_loss[benign]` | 1.391 | **0.540** |
| `grip[nominal]` — gripper channel | 0.0670 | 0.0625 |
| `grip[benign]` | 0.0871 | 0.0874 |

The verifier channel improved by 0.55–0.85; the gripper channel moved 0.005.
Under the calibrated proxy's own coefficients the verifier gain alone predicts
**+6.0pp** of rollout success. Measured: −1.1pp ± 1.3. That is Goodharting the
verifier — the failure `EXPERIMENT.md` line 63 predicted.

Note the transfer calculation assumes coefficients fitted ACROSS PHRASES also
describe changes induced BY TRAINING. They need not. Its failure is the evidence
for off-manifold optimisation, not a refutation of the proxy.

**2. `cand_blend_mean` cannot move and must never be read as progress.** It is
the mean of a within-group rank; it is pinned near 0.5 by construction. It sat
at 0.4852 → 0.4867 for 161 steps and told us nothing.

**3. The candidate set collapsed.** From `replay.pt` at step 150 (408 groups):
median **6 unique candidates of 16**, minimum 1, mean duplicate fraction 0.579.
The modal output is frequently the input instruction verbatim.

**4. `rank01` amplifies floating-point noise into gradient.** Identical strings
scored on identical frames with identical draws differ in the last bits, because
batched GPU reductions are not bit-deterministic. `rank01` promotes that ~1e-7
difference to a full rank step — 1/16 of the reward range, an amplification of
~1e6. Of 1042 duplicate sets, **306 (29.4%)** carry a spurious reward spread,
and every nonzero spread is an exact multiple of a rank step (0.0156 = 0.25/16,
0.0469 = 0.75/16, 0.0625 = 1/16 …), which is the signature of tie-breaking
rather than of differing frames or seeds.

---

## The four changes, and why they are one change

**(a) Reward: calibrated proxy LOGIT, not c4b ranks.**

    reward = 0.4445 * z + 11.3193 * (-grip)      # intercept dropped: GRPO
                                                 # subtracts the group mean
Never the sigmoid — on training frames grip ≈ 0.04 sits on the flat end and the
probability pins at 0.999. Use the logit, clipped to a sane range.

Against real rollouts, within task (the only comparison the update makes):

| reward | mean ρ | n-weighted | tasks ranking BACKWARDS |
|---|---|---|---|
| c4b (v11) | +0.282 | +0.271 | 3/8 |
| **proxy logit** | **+0.319** | **+0.350** | **1/8** |
| gripper only | +0.278 | +0.249 | 3/8 |
| verifier only | +0.240 | +0.320 | 3/8 |

The correlation gain is not significant (+0.037, Wilcoxon p=0.95). The reasons to
switch are structural: a magnitude reward is Lipschitz, so finding (4) disappears
— ε in, ε out — and the fitted coefficients weight the channel that transfers
(grip) 25.5× the channel that is exploitable (verifier), inverting c4b's rank
weighting, under which a 0.25-weighted consistent signal beat a 0.75-weighted
noisy one.

Cost: magnitudes lose rank01's outlier robustness. Clip the logit.

**(b) Generation: prompt for K=8 UNIQUE rephrasings (`--gen-mode list`).**

List mode already dedupes (`unique = dedupe(parsed)`; `<6 unique` → parse-fail,
skip), so uniqueness is enforced at generation rather than hoped for. K=8 sits
above that floor.

**(c) Update: signed advantages over the FULL group, importance-corrected.**

Not positives-only (winner's curse: imitating the argmax of K noisy estimates at
~70% pairwise accuracy), not top-K. All 8 candidates contribute, ranked.

This is the subtle part. Positives-only imitation is proposal-agnostic — it is
rejection sampling, valid under any sampling distribution. **Signed advantages
are a policy gradient**, so they assume samples come from the policy being
updated. List mode samples from `p_list` while the update targets `p_single`.

Here, unusually, the correction is exactly computable:

    w = exp( log p_single(y) - log p_list(y) )      # clipped

`log p_single(y)` is already computed for every candidate in `apply_update`;
`log p_list(y)` is the generation log-prob, capturable at sampling time.

PREREQUISITE: the ratio must use **summed** log-probs. `apply_update` currently
computes `ratio = exp(mean_logp - old_lp)`, which is `true_ratio^(1/n)` — for an
8–12 token phrase a genuine 2× ratio becomes ~1.07, inside the [0.8, 1.2] clip,
so `--replay-clip` never binds. Harmless today; load-bearing the moment a ratio
carries weight. Fix mean→sum WITH this change, not after.

**(d) Budget: C=10 contexts, and skip uninformative groups.**

The proxy was fitted at F=4 with C=10–24 episodes (median 18), i.e. F×C ≈ 72.
v11 trains at F=4, C=1–5 → F×C = 4–20, so the reward is fed inputs 3.6–18×
noisier than the calibration assumed. Both channels are means, so the linear
predictor's EXPECTATION is budget-independent and ranking stays valid — but the
probability calibration does not, and it must not be read as a success rate at
training budget.

| config | unique cands | C | F×C | pairwise accuracy |
|---|---|---|---|---|
| v11 today | ~6 of 16 | 5 | 20 | ~69% |
| list K=16 | 16 | 5 | 20 | ~69% |
| **v12: list K=8** | **8** | **10** | **40** | **~73%** |

Same ~320 evaluations per group.

Also: GRPO normalises every group to unit advantage variance, so a group scored
at C=1 contributes gradient as large as one scored at C=5 despite carrying a
fifth of the information. Skip groups whose reward std is below a floor — they
are noise-only, and under (a) they are also where ε/(ε+1e-6) still bites.

---

## Configuration

    bash scripts/run_arm_v12.sh

Deltas from `run_arm_v11.sh`:

    --gen-mode list --n-candidates 8       # was sample_single, 16
    --update-rule grpo                     # signed, full group (NOT raft)
    --reward-blend proxy_logit             # NEW; was c4b
    --reward-contexts 10                   # was 5
    --min-group-std 0.02                   # NEW: skip noise-only groups
    --importance-correct list_to_single    # NEW: computed w, clipped
    --replay-clip 0.2                      # now actually binds (sum log-probs)
    (drop --no-gate)                       # gate ON: prompting for diversity
                                           # pushes toward semantic drift

Unchanged: beta 0.15, lr 7e-6, kl-abort 1.2, save-every 10, val-every 20,
source-mix 0.25/0.5/0.25, F=4.

---

## New telemetry (v11 was blind to its own failure)

Per step, all of these:

- `n_unique_cands` and `dup_frac` — (3) must not recur
- `reward_std_within_group` — the quantity that decides whether advantages carry
  signal; if it approaches the reward's measurement noise, nothing can be learned
- `iw_mean`, `iw_p95`, `iw_clip_frac` — importance-weight distribution. If
  p_single and p_list diverge, weights go heavy-tailed and this must be visible
  at step 5, not inferred from a flat curve at step 150
- `grip_delta` and `verifier_delta` SEPARATELY — the Goodhart signature is the
  gap between them, and v11 only surfaced it because the tiers were logged
- **candidate text + per-candidate raw z, grip** to a parquet. `replay.pt`
  happens to preserve text and blended reward for a 50-step window; that was
  luck, not design.

Retire `cand_blend_mean` as a progress metric, or rename it
`cand_blend_mean_IS_A_RANK_DO_NOT_READ_AS_PROGRESS`.

---

## Falsification

Pre-registered, so v12 is not judged after the fact.

**Kill it if, by step 60:**
- `dup_frac` > 0.3 — list prompting failed to fix collapse
- `iw_clip_frac` > 0.5 — the p_list→p_single gap is too wide to correct, and the
  update is effectively uncorrected off-policy
- `verifier_delta` improves > 0.3 while `grip_delta` < 0.01 — Goodharting again,
  with a different reward
- `reward_std_within_group` < 2× the measured reward noise — the reward cannot
  rank its own candidates; no update rule fixes that

**Judge it at step 150** against v11's own trend, on nat24 with **768 episodes
per cell, not 192**. At 192 the per-cell s.e. is 3.56pp and the minimum
detectable slope is 22.6pp/1000 steps — v11 could have been improving at
+20pp/1000 and we could not have seen it. 768 halves the s.e. This is the change
that makes the whole comparison meaningful and it is cheap next to training.

**Success:** rollout success on nat24 rises ≥ 5pp over the step-0 baseline of
42.19 with the trend's 95% CI excluding zero.

---

## Resumability (standing order)

Every run stays resumable forever. A checkpoint dir carries the full state:
`adapter_model.safetensors`, `optimizer.pt`, `replay.pt`, `rng.pt`,
`trainer_state.json`.

| run | commit | on origin | resume |
|---|---|---|---|
| v10 | see `results/experiments.json` | steps 0010–0030, 0230–0260, best_val, latest | `scripts/ckpt_archive.sh unpack v10_latest` then `run_arm_v10.sh` (has `--resume`) |
| v11 | `4930b7e` (launch), running through `94bb556` | steps 0010–0150 + latest | `scripts/ckpt_archive.sh unpack v11_step_0150` then `run_arm_v11.sh` |
| v12 | this spec | — | — |

`scripts/ckpt_drain_v11.sh` archives each new checkpoint, verifies it against
`git ls-tree origin/main`, and keeps only the newest two locally. Verify by
ORIGIN, never by a push exit code: `git push` exits 0 when nothing was staged,
which is how v11 steps 0140/0150 were reported "pushed" while origin still held
0130.
