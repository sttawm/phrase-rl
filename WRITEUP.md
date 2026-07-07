# Teaching a small model to rephrase for a frozen robot policy

*Working paper draft. Companion to [EXPERIMENT.md](EXPERIMENT.md), which holds the
full protocol, every parameter, and the lab-notebook record. Canonical numbers
live in `results/**/metrics.json`; this document is the narrative.*

## The idea in one paragraph

A vision-language-action (VLA) policy takes a camera image and a natural-language
instruction and outputs robot actions. The *phrasing* of that instruction turns
out to matter: "put carrot on plate" and "place the carrot onto the plate" are
the same request, but the frozen policy does not respond to them identically.
Recent work (CoVer) exploits this by generating many rephrasings at test time and
selecting good ones with a separately trained verifier. We ask a sharper question:
**can we train a small model to *produce* the rephrasings that help a specific
frozen policy — using nothing but that policy's own action loss as the reward?**
If yes, a 9B model tuned this way should beat a frontier model at phrasing-for-this-policy,
because "what phrasing helps this policy" is not knowledge a frontier model has a
priori — it has to come from the reward.

## The reward, explicitly

The frozen policy here is π0, a flow-matching VLA fine-tuned on the BridgeV2 robot
dataset. Flow-matching policies do not output a probability we could read off as
"how much the policy likes this instruction." What they *do* have is a training
loss. Given a ground-truth action chunk `a*` (28 numbers: 4 timesteps × 7 DoF), the
policy is trained to denoise it: sample noise `ε ~ N(0, I)` and a flow time
`τ ∈ (0, 1)`, form the noised action `xτ = τ·ε + (1 − τ)·a*`, and predict the
velocity `u = ε − a*` that points back to the clean action. The instruction `ℓ`
enters only as conditioning. So for one context `(o, a*)` and phrasing `ℓ`:

```
L(ℓ) = E_{ε,τ} ‖ v_θ(xτ, o, ℓ, τ) − u ‖²        (frozen π0's own flow loss)
reward  R(ℓ) = − L(ℓ)
```

A phrasing is **good** if, conditioned on it, the policy predicts the denoising
direction more accurately — lower loss, higher reward. We estimate `L(ℓ)` with `K`
noise draws and, crucially, score **every rephrasing of a context against the same
K draws** (common random numbers), so loss differences reflect wording, not
sampling luck:

```
L̂(ℓ) = (1/K) Σ_k ‖ v_θ(x_{τ_k}, o, ℓ, τ_k) − u_k ‖²      (shared {(ε_k, τ_k)})
```

Within a group of rephrasings of one instruction, we convert rewards to
group-normalized advantages `A_i = (R_i − mean_j R_j) / (std_j R_j + ε)` for training.

**Why this reward can't be gamed by length.** The phrasing `ℓ` affects `L` *only*
through what the policy attends to — there is no term in the loss that scales with
token count. `L` is a mean-squared error over the fixed 28-dim action, not a sum
over the phrase. So a longer or shorter sentence carries no built-in advantage, and
we confirm it empirically below: the correlation between phrase length and loss is
~0. (This matters because reward hacking usually finds the cheapest degenerate axis;
length is the obvious one, and it's closed off.)

Two safeguards frame the whole approach. The common-random-numbers estimator is the
variance-reduction trick that makes "Diffusion Classifier" work in the image domain.
And we **measure the reward's reliability before trusting it** (Phase 0b), because
an earlier attempt of ours on a different policy failed precisely here — see next.

## Why we don't trust this reward for free — the OpenVLA cautionary tale

Before π0, we tried the same instinct on OpenVLA: let a planner propose sub-goal
phrasings and reward them by OpenVLA's action loss (cross-entropy / L2 on its
discretized actions). It failed in an instructive way. The action metric was
**flat** — the validation L2 sat at the same value at *every* RL step, regardless of
what the planner emitted. The reward simply could not tell good phrasings from bad,
so the policy gradient was chasing noise: one run collapsed to a single generic
sub-goal ("grab X") within ~50 steps; a higher-entropy run degenerated into
code-like garbage. The lesson was not "this idea is wrong" but "an action-loss
reward is worthless unless it actually *moves* with the wording — verify that first."
OpenVLA's binned action head was too insensitive to instruction phrasing to provide
signal. Phase 0b is the check we should have run then, now run first.

*(We don't have OpenVLA's raw logs in this repo for a side-by-side figure; the
contrast here is qualitative, from those earlier runs.)*

## Phase 0b — is the reward real?

**Setup.** 250 held-out BridgeV2 contexts (a context = one real camera frame + its
instruction + the action the robot actually took). For each, we generated 32
rephrasings from three sources — Gemini 3.1 Pro, Gemini 3.5 Flash, and the 9B model
we plan to train (Qwen3.5) — and scored every rephrasing with frozen π0 under 16
shared noise draws, on two π0 checkpoints.

**The reward is reliable — decisively.** Splitting the 16 draws into two halves and
re-ranking the phrases independently, the two rankings agree with median Spearman
**ρ ≈ 0.95** (signal share ≈ 0.98). At the modest number of draws we can afford per
training step, ~98% of the phrase-to-phrase variation is reproducible signal, not
noise. The OpenVLA failure mode is absent. The reward also has **no correlation
with phrase length** (~0.00), so the optimizer cannot cheat by just making phrases
longer or shorter.

![Ranked rephrasings for three example contexts](results/charts/phase0b_ranked_phrases.png)

**What the policy actually prefers.** The figure shows three contexts. Each row is
one camera frame and the 32 rephrasings ranked by how well π0 predicts the true
action under them (shorter bar = policy prefers). The **orange** bar is the
*original* Bridge instruction.

The striking, honest result: **the original human instruction is worse than the
median rephrasing in 71% of contexts, and is the single best phrasing in only 5%.**
Top row is the common case — "opened the drawer" (a terse, past-tense Bridge label)
ranks 32nd of 33, while "Extend the drawer" / "Slide out the drawer" score far
better. Middle row shows a mid-pack original. Bottom row is one of the ~5% where the
original ("flip pot upright in sink") is already best. Bridge's instructions are
often terse or ungrammatical, and π0 — trained on paraphrase-augmented data —
systematically prefers cleaner, more explicit phrasings.

This is the headroom the whole project targets: **the best-of-32 rephrasing beats
the original instruction in 96–98% of contexts, cutting the action loss by a median
of 31–33%.** Something worth learning to generate clearly exists.

**Two findings that shape the next phase.**

- *The paraphrase-trained π0 is more phrasing-invariant, but that does not change
  which checkpoint to use.* We scored two π0 checkpoints: one fine-tuned with
  paraphrase augmentation ("rephrase-FT"), one without ("plain"). The paraphrase one
  shows ~25% *smaller* phrase-to-phrase spread — its training partly washed out the
  sensitivity we exploit. We initially read this as "train against the plain
  checkpoint for a bigger signal," but that reasoning is wrong: because advantages
  are **group-normalized** (`A_i = (R_i − mean)/std`), the absolute spread cancels —
  only the *ranking reliability* survives into the gradient, and that is equally high
  (ρ ≈ 0.95) for both checkpoints. So the choice is made on a different ground:
  **use the rephrase-FT checkpoint**, because it is the policy we actually roll out
  in SIMPLER (Phase 0c/4). Optimizing phrasings for the same policy we evaluate
  avoids a train/eval mismatch.
- *Noise level matters.* The reward discriminates phrasings well at moderate noise
  but collapses near the clean-action end. In training we concentrate scoring draws
  where the signal lives, getting the same fidelity for ~25% fewer forward passes.

**Where the 9B model stands today.** Before any training, Qwen3.5-9B's rephrasings
are already about as diverse as the Gemini models' (its reward spread is 75–100% of
theirs) — so we do not need a larger generator. What it lacks is not diversity but
*direction*: it does not yet know which of its phrasings this policy prefers. That
is what the reward is for.

## Status and what's next

Phase 0b passed its go/no-go gate, so the reward is trustworthy. Two threads follow:

- **Phase 0c (running):** does a lower flow loss actually translate to higher
  *task success*? We roll out π0 in the SIMPLER simulator under each rephrasing and
  correlate success rank against flow-loss rank. This is the one link the offline
  loss cannot establish on its own.
- **Phase 1 (next):** distill a frontier teacher's rephrasings into Qwen3.5 (SFT),
  then tune it with the flow-loss reward so it learns to produce policy-preferred
  phrasings — the headline experiment.
