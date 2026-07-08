# Teaching a small model to rephrase for a frozen robot policy

*Working paper draft. [EXPERIMENT.md](EXPERIMENT.md) holds the full protocol and
lab notebook; numbers are canonical in `results/**/metrics.json`. This is the
narrative.*

## The idea

A vision-language-action (VLA) policy maps a camera image + instruction to robot
actions, and the *phrasing* matters: "put carrot on plate" and "place the carrot
onto the plate" are the same request but the frozen policy responds differently.
CoVer exploits this at test time by generating rephrasings and picking good ones
with a trained verifier. We ask a sharper question: **can we train a small model to
*produce* the rephrasings a specific frozen policy prefers, using only that policy's
own action loss as reward?** If so, a tuned 9B model should out-phrase a frontier
model for this policy — because "what phrasing helps this policy" isn't in the
frontier prior; it has to come from the reward.

## The reward

The frozen policy is π0, a flow-matching VLA fine-tuned on BridgeV2. Flow-matching
policies give no likelihood, but they have a denoising loss. For ground-truth action
`a*` (28-dim: 4 steps × 7 DoF), sample noise `ε ~ N(0,I)` and flow time `τ ∈ (0,1)`,
form `xτ = τ·ε + (1−τ)·a*`, and the policy predicts velocity `u = ε − a*`. The
instruction `ℓ` enters only as conditioning:

```
L(ℓ) = E_{ε,τ} ‖ v_θ(xτ, o, ℓ, τ) − u ‖²          reward R(ℓ) = −L(ℓ)
```

Lower loss = the policy denoises `a*` better under that phrasing = better reward. We
estimate `L` with `K` noise draws and score **every rephrasing of a context against
the same K draws** (common random numbers), so differences reflect wording, not
sampling luck — the variance-reduction trick behind "Diffusion Classifier." Within a
group we use group-normalized advantages `A_i = (R_i − mean)/(std + ε)`.

**Length can't game it.** `ℓ` affects `L` only through attention; there's no term
that scales with token count (`L` is an MSE over the fixed 28-dim action, not a sum
over the phrase). Empirically, phrase-length/loss correlation is ~0.

## Does the reward actually discriminate? (the OpenVLA worry)

We tried this instinct once before, on OpenVLA, and it collapsed — a run drifted to a
single generic sub-goal ("grab X") and the val action metric barely moved. Our
**leading hypothesis**: OpenVLA's language grounding is weak enough that its action
loss carries almost no signal about *phrasing* — so the only gradient the optimizer
could find pointed back toward OpenVLA's own training-distribution templates (hence
the collapse to "grab X" specifically, not random degeneration). If phrasing barely
moves the reward except by resembling training data, RL can only rediscover that
data. (A related contributor: OpenVLA's *discretized* action head may be too coarse
to respond to wording at all.) π0's continuous velocity loss and stronger VLM
backbone should carry real phrasing signal — but that's a hypothesis, so we gate on
it (Phase 0b) rather than assume it.

*Terse test to confirm the diagnosis:* run Phase 0b unchanged on OpenVLA — same
contexts, score N rephrasings by OpenVLA's action loss, measure split-half rank
reliability. If it's flat on OpenVLA but reliable on π0 (below), the failure was the
reward carrying no phrasing signal, not the general idea. A sharper variant: check
whether OpenVLA's reward correlates with *resemblance to its training templates*
rather than with semantic fit — if so, that's the collapse attractor made explicit.

## Phase 0b — is the reward reliable?

**Setup.** 250 held-out BridgeV2 contexts (context = camera frame + instruction +
executed action). 32 rephrasings each from Gemini 3.1 Pro, Gemini 3.5 Flash, and
base Qwen3.5-9B; scored by frozen π0 under 16 shared draws, on two π0 checkpoints.

**Result: reliable, decisively.** Split the 16 draws in half, re-rank independently:
the two rankings agree at median Spearman **ρ ≈ 0.95** (signal share ≈ 0.98). At the
draw budget we can afford per training step, ~98% of phrase-to-phrase variation is
reproducible signal. Length/loss correlation ~0.

![Ranked rephrasings for three example contexts](results/charts/phase0b_ranked_phrases.png)

**What the policy prefers.** Each row: one frame, its 32 rephrasings ranked by π0
flow loss (shorter bar = preferred), **original Bridge instruction in orange.** The
headline result: **the original instruction is worse than the median rephrasing in
71% of contexts, and the single best in only 5%.** Top row is typical — "opened the
drawer" (terse Bridge label) ranks 32/33. Bottom row is one of the ~5% where the
original wins. π0, trained on paraphrase-augmented data, systematically prefers
cleaner phrasings. **Best-of-32 beats the original in 96–98% of contexts, cutting
loss a median 31–33%** — that's the headroom the project targets.

**Findings that shape next steps.**
- *Which checkpoint:* the paraphrase-trained π0 has ~25% smaller phrase spread than
  the plain one, but group-normalized advantages cancel absolute spread — only
  ranking reliability reaches the gradient, and it's equal (ρ≈0.95). So we **use the
  paraphrase (rephrase-FT) checkpoint**: it's the policy we roll out, avoiding a
  train/eval mismatch.
- *Noise band:* discriminability collapses near the clean-action end; we concentrate
  training draws where the signal lives (~25% fewer forwards, same fidelity).
- *Generator size:* base Qwen's reward spread is 75–100% of Gemini's — diverse
  enough, no larger generator needed. What it lacks is *direction*, which is the
  reward's job.

## Phase 0c — phrasing moves behavior, and the reward has a blind spot

We rolled out the frozen policy in the SIMPLER simulator under every rephrasing
(1,290 episodes: 4 tasks × ~33 phrasings × 10 shared initial states), and scored
the same phrasings offline on matched real Bridge contexts.

**Phrasing strongly moves task success** — the clean, all-in-sim result. On the
carrot task, success ranges **0.1 to 0.7** across phrasings of the same request on
identical initial states. The failures are interpretable: phrases whose *goal*
drifted ("put the carrot **by** the plate", "maneuver the spoon **past** the
towel") fail in exactly the way their words say.

**And that exposed a real blind spot in the reward.** The offline flow loss
*prefers* goal-drifted phrases: they score **better** (mean loss 0.081) than
clean rephrasings (0.095) while succeeding **less** (0.32 vs 0.58). Notably the
blind spot is specific to *goal* drift: object *renames* ("the orange vegetable")
succeed at the highest rate of any class (0.65) and get no reward discount. The
mechanism: teacher-forced loss measures how well the policy predicts the
demonstrated trajectory, which mid-episode is dominated by reach-and-transport
motion — "toward the plate" and "by the plate" demand nearly identical actions
until the final centimeters. Style fits; goals barely register. This drove the raw
reward↔success correlation negative (pooled ρ=−0.10); excluding judged goal-drift
(the exact Phase 2 gate) removes the effect (pooled ρ=+0.28, weakly positive at
our n=10-seed measurement precision).

**Design consequence.** This is a *confirmed reward-hacking axis*: naive RL on flow
loss would learn to emit trajectory-plausible, wrong-goal phrases. Phase 2 therefore
uses a **faithfulness-gated reward** — candidates judged unfaithful to the original
instruction are excluded/penalized before advantages are computed; flow loss ranks
only within the faithful set. (One more honest note: in sim, the *canonical*
original instructions are strong — the policy saw them verbatim in training — so
the offline finding that "originals rank poorly" reflects Bridge's messy labels,
not a universal advantage of rephrasing.)

## Step-0 — the reward survives the real training distribution, and reasoning source doesn't matter much

Before training we re-validated everything on the *actual* generation setup (CoVer's
verbatim template) with three arms: Gemini (the frontier baseline), base Qwen writing
its own inline reasoning, and base Qwen given a cached Gemini reasoning trace
(two-prompt flow).

![Step-0 A/B](results/charts/step0_ab_verdict.png)

Three results, one per panel: **reliability holds** on the true training distribution
(split-half ρ = 0.95–0.96 for all three arms — the 0b gate re-passes); **oracle
headroom is unchanged** (best-of-32 beats the original instruction by ~32% in all
arms); and **reward spread is healthy and nearly identical** across arms. On every
measurable-without-a-judge axis, giving the 9B model a frontier reasoning trace
changes nothing — its own inline reasoning produces candidates the reward finds
equally rich. The remaining discriminator (goal-drift rate per arm) awaits API
credits; unless it shows a large gap, **inline conditioning stands as primary**: it
keeps exact prompt parity with the baseline and requires no frontier call at
deployment — strictly cheaper to deploy than CoVer.

**Training-loop status.** The full RL stack (generate → faithfulness gate → CRN
score → advantage-weighted update) has been validated end to end: the trainer loads
2,000 contexts, attaches a 29M-parameter LoRA, generates and parses candidates, and
— by design — refuses to train ungated: when the judge API became unavailable it
checkpointed at step 0 and paused rather than optimize an unprotected reward. First
training curves follow once the gate is back.

## Next

- **Phase 2 (primary):** advantage-weighted tuning of Qwen3.5-9B from base, with
  the faithfulness-gated flow-loss reward. For a clean control, the tuned model and
  the frontier baseline share **CoVer's verbatim rephrase prompt** — the comparison
  isolates weights, not prompt design. Teacher-SFT warm-start and trace-conditioning
  as ablations.
- **Phase 4** remains the behavioral proof: roll out the *trained* generator in sim
  — all-in-sim, no domain confound.
- **Phase 1–2:** SFT + advantage-weighted tuning of Qwen3.5. Leaning toward
  RL-from-base-Qwen as the primary (cleaner claim: no frontier teacher), with
  teacher-distillation as an ablation.
