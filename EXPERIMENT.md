# phrase-rl: Advantage-Weighted Rephrase Tuning Against a Frozen VLA

**Status:** Phase 0 complete (0b GO: reward reliable ρ≈0.95; 0c: phrasing moves sim success 0.1–0.7, goal-drift is a confirmed reward-hack axis → faithfulness gate mandatory, gate-pass reward↔success ≈0 (early +0.28 figure retracted as a rank-pooling artifact)). Phase 1 teacher data complete (2000×16+traces). **Phase 2 stack built + adversarially reviewed** (commit e6be370): CoVer-verbatim generation → drift-only gate (majority-of-3, fail-closed) → CRN scoring (τ≥0.25) via cross-venv file IPC → phrase-token-only advantage-weighted LoRA update w/ KL anchor; resumable. NEXT: pod smoke test → step-0 baseline → primary RL run. **Last updated:** 2026-07-07.

## Hypothesis

CoVer ([arXiv:2602.12281](https://arxiv.org/abs/2602.12281)) improves VLA rollout success by generating instruction rephrases with a fixed frontier VLM and selecting instruction–action pairs with a separately trained contrastive verifier at test time. We instead **train the rephrase generator itself**, using the frozen downstream VLA's action loss as reward, and test whether:

1. Advantage-tuned rephrasing lowers offline VLA action loss for both single phrases and 16-phrase candidate sets.
2. **(Headline)** A small tuned model beats a frontier model at phrasing: Qwen2.5-VL-7B after advantage tuning produces better rephrases (by frozen-VLA action loss, later by rollout success) than Gemini zero-shot — evidence that "what phrasing helps this policy" is not in the frontier prior and must come from the reward.

Note CoVer never compares against ranking by the policy's own loss; it goes straight to a learned verifier. Using policy loss as a training signal is the novelty here — and the main risk (see Phase 0b).

## Models

| Role | Model | Notes |
|---|---|---|
| Frozen VLA reward | `juexzz/INTACT-pi0-finetune-rephrase-bridge` | π0 on BridgeV2 + paraphrase augmentation; LeRobot PyTorch; ~3.3B; chunk size 4, delta EE actions |
| Frozen VLA (ablation) | `juexzz/INTACT-pi0-finetune-bridge` | Non-rephrase checkpoint — tests dependence on paraphrase-augmented reward model |
| Frozen trace/teacher VLM | Gemini 3.1 Pro (batch API) | Replaces CoVer's GPT-4o (discontinued). **Prompt-parity decision, corrected 2026-07-07 (2nd revision):** CoVer's full prompt = system persona (`reference/cover_rephrase_prompt.txt`) + a user-turn template (`reference/cover_rephrase_user_template.py`) that mandates **inline boot-time reasoning** — image description → instruction meaning → noun/verb/adjective substitution analysis → N rephrases, all in ONE call (GPT-4o, temp 0.8, image attached, few-shot). **Primary path: both the frontier baseline and trainable Qwen use this verbatim template** — reasoning is inline self-generated CoT, not a separately cached artifact. The headline comparison isolates weights with zero prompt confound. Our *cached Gemini traces* (separate-call design) feed the **cached-trace ablation** (c = o, ℓ, r vs CoVer-style inline). Deployment = one boot-time call by the tuned model itself, image + instruction in, reasoning + list out |
| Trainable phrase model | `Qwen/Qwen3.5-9B` (natively multimodal, Feb 2026) | LoRA. **Thinking mode DISABLED everywhere** (`enable_thinking=False` in generation and, later, training): Qwen3.5 thinks by default, which is ~4× slower, leaks numbered lines from the reasoning block into parsed candidates, and would contaminate the advantage-weighted log-prob objective with reasoning tokens. All 0b/0c/eval numbers are no-think numbers. Revisit later: a *minimal/budgeted* thinking variant as an ablation (see Ablations). Fallback model `Qwen/Qwen3-VL-8B-Instruct`; 27B/32B only as post-signal scale-up (32B ≈ 73GB FP16, ~4× slower generation) |
| Optional baseline | CoVer verifier (`cover_verifier_bridge.pt`, ~312MB) | Test-time selection over our candidates; separates "better candidates" from "better selection" |

π0.5 note: our reward/eval model is π0 (Bridge/SIMPLER). π0.5 only appears in CoVer's PolaRiS extension; all machinery here (flow-matching scoring, CRN) transfers unchanged if we later swap it in.

## Data

BridgeV2 tuples `x = (o_t, l, a*_t)` (image, original instruction, ground-truth action chunk). For each, cache one Gemini call: reasoning trace `r` + 16 teacher rephrases. Training context for the phrase model: **`c = (o_t, l)`** (prompt-parity decision — CoVer's verbatim scaffold, no trace in the primary path; `c = (o_t, l, r)` is the trace ablation; the SFT ablation uses the cached 16-lists).

- **Scale:** 2,000 train + 500 val contexts (pilot). Test contexts held out untouched.
- **Cost:** ~1k input + ~600 output tokens per context → **~$13 total** with Gemini 3.1 Pro batch mode. Scaling to 10k later is <$50.
- Timestep choice: `t` here indexes **which frame of a demo episode becomes a context** — not when rephrasing happens (that's once, at boot, from the first observation). **Sample `t` uniformly** (no evidence yet that any episode phase matters more); log `t` per context, and have 0b report reward discriminability vs episode position as a free byproduct — reweight later only if that shows e.g. early "reach" chunks are phrase-insensitive. Optional refinement (unchanged): condition the phrase on the episode's *first* frame but score against action chunks at multiple later `t` (matches deployment on the input side, covers the episode on the reward side).

## Reward: π0 flow-matching loss with common random numbers

π0 has no likelihood; training loss is a denoising-velocity MSE, stochastic in the noise `ε` and flow time `τ` that LeRobot's `Pi0Policy.forward()` samples internally. To use it as a phrase score:

- Patch forward to accept **fixed (ε, τ) draws**.
- Score all 16 phrases of a context with the **same K draws** (common random numbers), K ≈ 8–16, τ stratified over [0,1]; average.
- `R_i = -L_VLA(a* | o, y_i)`; within-group advantage `A_i = (R_i - mean) / (std + eps)`.
- Loss magnitude varies strongly with τ, so also compute a **per-draw z-scored** variant (z-score the 16 phrases' losses within each (ε, τ) draw before averaging) so high-variance τ bins don't dominate the ranking; Phase 0b picks whichever has better split-half reliability. 0b should also report which τ band discriminates phrasing best (cf. Diffusion Classifier, arXiv:2303.16203 — the same conditional-denoising-loss scorer in the image domain; mid-range noise carried most signal there).

## Training

**Primary path is advantage-weighted tuning directly from base Qwen3.5 — no teacher distillation — with TRACE-CONDITIONED generation (user decision 2026-07-08: CoVer-parity, frontier reasoning at boot like CoVer's GPT-4o call; deployment = one Gemini trace call + local tuned Qwen). Inline self-reasoning is the ablation (86-step pilot run banked: results/checkpoints/phase2_inline_pilot on pod). Traces currently = the 2,000 old-prompt Gemini-pro teacher traces; regenerating in CoVer format is optional (~\$2.5 flash-lite / ~\$14 3.5-flash).** Rationale (2026-07-07): base Qwen already captures most of the reward headroom (0b: Qwen best-of-32 oracle gain 20.5% vs Gemini 26.7% — the teacher adds only ~6 pts of ceiling), and skipping the teacher makes the headline claim cleaner ("a 9B model with *no* frontier supervision, only the policy's reward, out-phrases the frontier model"). Teacher-SFT warm-start becomes an ablation (does a higher-ceiling start beat RL-from-base?).

1. **Advantage-weighted tuning (primary):** generate 16 candidates per context from the *current* model using **CoVer's verbatim template** (image + instruction → inline reasoning → numbered list; identical to the frontier baseline so the eval isolates weights), pass a **faithfulness gate** (majority-of-3 LLM judge; gate targets **goal/action drift only** — object renames like "the red fruit" are explicitly encouraged by CoVer's few-shot examples and stay in, but get logged; unfaithful candidates are excluded from the advantage group — mandatory per 0c), score survivors with frozen π0 (rephrase-FT ckpt) + CRN, update under the single-phrase prompt (loss applies to phrase tokens only, not the inline reasoning):
   `L = -Σ_i max(A_i, 0) · (1/|y_i|) · log π_θ(y_i | c, p_single) + β·KL-to-base anchor + λ·list-format aux`
   **Token-mean normalization** (the `1/|y_i|`): raw summed log-prob scales gradient contribution with phrase length, so long positive-advantage phrases would dominate updates. Positive-only advantages first; signed advantages as ablation. KL anchor to base Qwen (or to the SFT model when warm-started). This is advantage-weighted rephrase tuning, not exact GRPO.
2. **Log during training, not just eval:** duplicate rate, mean pairwise embedding similarity, object/target preservation, generic-phrase rate, format-failure rate. Known failure mode: collapse onto Bridge template phrasing ("put X in Y") — the rephrase-FT reward checkpoint was trained on a fixed paraphrase dictionary the optimizer may simply rediscover.
3. **Teacher-SFT warm-start (ablation, not primary):** split each Gemini 16-list into 16 single-phrase examples `(c → y_i)`, SFT Qwen, *then* advantage-tune. Tests whether a higher-ceiling starting distribution beats RL-from-base. Teacher data already collected (Phase 1).

### Source augmentation + splits (2026-07-08)

- **Source augmentation (primary, p=0.5, UPDATE-PROMPT ONLY — user decision 2026-07-08):** candidate generation always conditions on the *original* (max pool quality); the `[original phrase]` slot in the **update/p_single prompt** — which IS the inference-time prompt — is swapped for a random teacher rephrase with prob 0.5, so the tuned model learns to produce a good phrase from arbitrary input phrasings. Gate anchored to the original; reward unchanged (flow loss vs a*). **Deployment = ONE phrase from the tuned model via the single-phrase prompt** (no 16-list, no verifier — that's the contrast with CoVer's generate-and-select; list mode remains only for best-of-N eval rows). Caveat logged: cached traces were written for the original instruction and occasionally quote it — a mild leak of the original phrasing into augmented contexts; acceptable, revisit if it shows in metrics.
- **CoVer splits: nothing to adopt.** Their eval is SIMPLER rollouts (no Bridge test split); their verifier trains on a private preprocessed sample dump with a plain random 10% sample-level validation split. Our episode-level hash split is stricter (prevents same-episode frame leakage). One documented caveat: the released `cover_verifier_bridge.pt` was presumably trained on most of Bridge train, so for the "CoVer verifier selects among our candidates" ablation, overlap with our val/test contexts is possible and unfixable on our side.

### Reward v2 candidates → REWARD BAKE-OFF (2026-07-08; clarity note: 0b ρ≈0.95 = split-half RELIABILITY (instrument precision); 0c ρ≈+0.28 = gate-pass VALIDITY vs rollout success — precise ruler, weakly aligned target)

Score ALL candidate rewards offline against the same 0c rollout-success labels (LOTO-CV over tasks); winner becomes the RL-v2 reward. Candidates:
0. Raw flow loss (baseline, ρ=+0.28 gate-pass).
0b. **Learned τ-weighting w(τ)** — tiny linear head over per-τ-band losses (we store the full per-draw matrix). Evidence: τ<0.25 uninformative; stack improves to +0.40 on late-τ only → weighting may be task-conditional.
3. **CoVer verifier as reward:** decode π0's action under phrase y (ODE ~10 steps), verifier-scores (ORIGINAL instruction, decoded action) — behaviorally grounded, drift-resistant by construction (could subsume the gate). GOODHART CAUTION: CoVer used it for best-of-N selection; optimizing thousands of steps against a frozen verifier invites adversarial phrases — mitigate with KL/gate/rollout-validated checkpoints; run as an arm, not a silent swap. Shares decoding machinery with decoded-L2.

1. **Decoded-action L2 vs success:** integrate the flow ODE (~10 steps) → action chunk → per-dimension-normalized L2 vs a* (dataset-stats normalization; gripper dim separate). Correlate with 0c rollout success on the same 50 matched contexts/labels. Hypothesis: closer to executed behavior than denoising loss → may beat gate-pass ρ≈0.28. ~1–2h GPU, offline.
2. **Learned reward calibrator:** thin model over policy-internal features we already store (per-τ-band losses, across-draw variance, decoded-L2, length, rename flag) → predict rollout success. Train on rollout-labeled phrases (~129 now; ~260 after 0c-redo resumes), leave-one-task-out CV. If it beats raw flow loss: reward for RL-v2 + checkpoint-selection metric. Framing: a cheap, policy-internal learned verifier — CoVer's verifier idea, rebuilt from reward features + ground-truth success.

Val note: in-training val = offline flow-loss on 40 val contexts every 100 steps (mean + best-of-16 vs original). Rollout eval = Phase 4 only (cost: one rollout-val ≈ hours). Optional: small rollout probe (1 task × 8 phrases × 10 seeds ≈ 25 min) at each best-val checkpoint.

### Reward bake-off rounds 1–2 (2026-07-09, results/bakeoff/*.json)

Labels: 129 CoVer-template phrases × 4 tasks (10 seeds; 40 at 25 seeds), gated (3.5-flash judge; flash-lite judge REJECTED — misclassified target renames as drift). Key numbers (LOTO mean, gated): raw flow −0.06; flow-z +0.01; τ-band +0.01; **decoded-L2 +0.08 (only cross-task-consistent arm)**; rank-avg combo +0.03; ridge −0.19 (overfits).

**Findings:**
1. The CoVer-template generator is nearly drift-free (0–3%/task vs 35–65% for the old list-prompt) — the gate barely binds for this generator; drift-poisoning does NOT explain the weak correlations.
2. **Carrot anti-correlation (ρ=−0.65) mechanism:** the set is 24/32 renames; flow slightly prefers renames (π0 is paraphrase-trained; renames predict demonstrated actions fine) while sim success punishes them (0.29 vs 0.45 clean — renamed referents ground worse in sim). Opposite direction from the original-set rename finding (renames 0.65 success there) → class-level effects are task/set-dependent; the offline reward does not track fine-grained within-task sim success and can anti-correlate on rename-heavy sets.
3. Consequence: temper Phase-4 expectations for flow-RL; elevates the CoVer-verifier arm and a possible rollout-in-loop fine-tune; Phase 3 offline eval remains internally consistent (same-domain reward).

RL v1 post-mortem (same date): KL leash β=0.04 too weak — monotonic drift KL 0.3→2.0, parse-fail 1%→58% by step 465; best ckpt step 200. v2 = restart from best ckpt, β=0.15, lr 5e-6, rolling-KL circuit breaker (exit 6). v1 artifacts: results/charts/phase2_v1_training.png + raw log.

## Phases

- **0a — Infra + plumbing:** CoVer repo env; load both INTACT checkpoints via LeRobot; extract 2k+500 BridgeV2 contexts; fixed-noise scoring patch.
- **0b — Sensitivity gate (HARD go/no-go):** ~200 val contexts × 32 rephrases (sensitivity phases use 32 for more rank points; training groups stay at 16), CRN scoring. **Primary phrase source: candidates sampled from the trainable VLM itself** (base model, list prompt, thinking mode disabled — thinking is 4× slower, pollutes list parsing, and would complicate log-prob computation in training) — that is the distribution the reward must discriminate during RL. Secondary arms: Gemini 3.1 Pro (the SFT teacher; spread comparison measures teacher-distribution headroom) and Gemini 3.5 Flash (frontier-tier comparison: open-9B vs flash vs pro as phrase sources). Per phrase: `L̂_i = (1/K) Σ_k ‖vθ(Aᵏ; o, y_i, τ_k) − u*_k‖²` over the K shared draws; split draws into two fixed halves (same halves for all phrases) → two estimates per phrase → Spearman across phrases per context. **The gate is reliability at training-budget K (≈8–16), not asymptotic**: split-half correlation at fixed K measures SNR at operating conditions (Spearman-Brown: ρ ≈ σ²_signal/(σ²_signal + 2σ²_noise/K)); needing K≫16 for stable ranks = reward unaffordable per update = no-go on the same grounds. ~~Duplicate-phrase negative controls~~ are vacuous under CRN: identical text + identical draws + deterministic forward = bit-identical losses (verified in smoke_pi0), so the noise floor is measured by split-half disagreement, which subsumes them. Report the between-/within-phrase **variance decomposition**, plus the **phrase-length ↔ loss correlation** (a hackable axis if present). **Go** iff phrase ranking reproduces across halves and spread clears the duplicate-control floor. Plain-English version of the gate: every phrase is measured twice (draws 1–8 vs 9–16, a free byproduct of the same K forwards — no extra compute); if the two rankings of the 32 phrases agree, the spread is signal; if the second ranking reshuffles the first, the spread was noise (raw variance alone can't distinguish these — noise creates spread even between identical phrases). Also report discriminability vs episode position `t` (informs harvest weighting). Division of labor vs 0c: 0b asks "does the instrument repeat its own readings" (free, 200-context breadth, catches instrument failure); 0c asks "do the readings predict success" (expensive, 4 instructions). Running only 0c is ambiguous on a null result — corr ≈ 0 can't separate broken-reward from phrasing-doesn't-matter. Run both INTACT checkpoints (differential sensitivity is itself a finding). This is exactly the non-discriminative-reward failure mode from the earlier OpenVLA GRPO runs — catch it before training anything.
  Note what 0b does and doesn't establish: it calibrates the *instrument* (can CRN flow loss resolve phrases above its noise floor), not whether phrasing matters behaviorally (CoVer shows behavior moves, but through a verifier over sampled instruction–action pairs — consistent with rephrases merely widening the action proposal pool) and not whether lower loss ⇒ better rollouts (risk 3; 0c and Phase 4).
- **0c — Rollout sensitivity (SIMPLER), sanity check not a gate.** Reframed 2026-07-07: the training reward is *real-frame* flow loss and rollouts are *sim*, so any "reward predicts success" correlation must cross the real→sim gap — intrinsic to offline-real + sim-eval, unfixable without a real robot. So we split 0c into a clean part and a caveated part:
  - **(A) In-sim phrasing sensitivity [CLEAN, headline].** Roll out all ~32 rephrasings + the original per task in sim (shared episode_ids across phrases = paired). Ask: does sim success vary across phrasings, and **do rephrasings beat the original instruction in sim?** No real/sim confound — phrases and success are both in sim; directly tests the project thesis. Readouts: per-phrase success spread, frac rephrasings beating original, split-half reliability over seeds. Money plot: original vs rephrasing-distribution success per task.
  - **(B) Real-reward vs sim-success correlation [CAVEATED bonus].** Spearman(−flow loss on matched real Bridge contexts, sim success). Positive = reward transfers even across the domain gap (nice); **null is uninterpretable** (broken reward vs. domain-specific phrasing) so we don't gate on it. 3 tasks only (eggplant has no Bridge counterpart).
  - Budget: 10 shared episode_ids × ~33 phrases × 4 tasks ≈ 1,300 episodes; prefer mid-range-success tasks (floor/ceiling can't discriminate). The real behavioral proof is Phase 4 (trained model in sim — also all-in-sim, also clean); 0c just de-risks before Phase 2.
- **1 — Teacher data:** Gemini 3.1 Pro batch (~$13, 2k train contexts, traces + 16-lists). Feeds the Gemini eval baselines and the teacher-SFT *ablation* — no longer a required predecessor to training.
- **2 — Advantage-weighted tuning (PRIMARY / headline):** RL Qwen3.5-9B from base against frozen π0 (rephrase-FT ckpt) + CRN, as in Training §1. Teacher-SFT warm-start runs as an ablation arm (Training §3), not the main path.
- **3 — Offline eval matrix** (below).
- **4 — SIMPLER rollouts** via CoVer repo, only on a clear Phase 3 win.

## Phase 4 eval protocol — CoVer comparability (audited 2026-07-09)

Read their code (github.com/cover-vla/cover-vla) + paper v2. Their Table 3 protocol is NOT a
nominal-instruction eval; matching it exactly requires four deliberate choices:

1. **Instructions are ERT red-team, not nominal.** Every number in Table 3 uses adversarial
   phrasings as the user instruction (carrot: "Balance the carrot on the ceramic platter.";
   eggplant: "Arrange the eggplant neatly in the yellow bin."; spoon: "Set the spoon exactly
   in the middle of the towel."; stack: "Arrange the lush green element atop the yellowish-
   orange element."). Their π₀ baseline = π₀ *fed these directly* (ID avg 41.5). This is
   exactly our deployment problem: the comparable run feeds the red-team instruction to OUR
   tuned rephraser and executes its single greedy output.
2. **Initial states: reset seeds 1000–1049 cycled** (50 unique states; 100 trials/task = 2
   passes), NOT INT-ACT's `episode_id` enumeration. Reported ±std is over 3 repeat runs.
3. **Horizon 150 steps, TimeLimit ignored** (stock envs truncate at 60/120 — successes after
   that still count for them). Break on success (`terminated`).
4. **Same checkpoints we already use**: π₀+CoVer rows = `INTACT-pi0-finetune-bridge`;
   rephrase rows = `INTACT-pi0-finetune-rephrase-bridge` (ours). Chunk-4 replan + INT-ACT
   BridgeSimplerAdapter post-processing match our runner already.

Their test-time cost per chunk boundary: 8 instructions × 5 action samples = 40 candidates,
verifier-scored (two-stage; winner's instruction persists). Our tuned arm = ONE phrase, zero
test-time overhead. Comparison rows: their π₀ 41.5 / π₀+CoVer 57.0 / π₀(rephrase)+CoVer 65.5
(ID avg) vs our `redteam-direct` (baseline replication) and `redteam→tuned-rephrase`.

Consequence: our nominal-instruction rollouts (0c, 4-arm eval) stand on their own but are NOT
comparable to any CoVer table row. `phase0c_rollout.py --cover-protocol` implements (2)+(3).

## Offline eval matrix (held-out contexts, frozen-VLA action loss)

| # | Condition | List metrics where applicable |
|---|---|---|
| 1 | Original instruction | — |
| 2 | Gemini with **CoVer's verbatim prompt** (`reference/cover_rephrase_prompt.txt`) — same scaffold as the tuned model | — (PRIMARY frontier baseline: identical prompt to ours ⇒ 2 vs 6 isolates weights/reward) |
| 2b | Gemini with our strict-preservation prompt + trace | secondary — measures how much prompt design alone moves the frontier baseline |
| 3 | Gemini 16-list | random / mean / oracle |
| 4 | Qwen SFT single | — |
| 5 | Qwen SFT 16-list | random / mean / oracle |
| 6 | Qwen advantage-tuned single | — |
| 7 | Qwen advantage-tuned 16-list | random / mean / top-k / oracle |

Oracle = argmin ground-truth action loss over the 16 (not deployable; measures candidate-set headroom).

**Headline comparisons.** Attribution chain (SFT is distilled from Gemini, so each step isolates one factor):
- **2 vs 4:** distillation loss (frontier → 7B imitation).
- **4 vs 6:** pure reward-signal contribution.
- **2 vs 6:** tuned 7B vs frontier zero-shot — the headline claim if 6 wins.
- **3-oracle vs 7-oracle:** whose candidate set has more headroom.

## Sequencing decision (user, 2026-07-09)

Finish the flow-vs-L2 A/B as-is (Gemini cached traces). Pick the winner by val
rollouts. Then rerun the winning recipe with QWEN'S OWN reasoning as update
context (self-traces, frozen base, cached offline; loss stays phrase-only) —
kills the Gemini-trace train/deploy mismatch. ERT inputs slot into that run.

## v3 (planned): red-team-input RL via adapted ERT (audited 2026-07-09)

ERT = arXiv:2411.18676 (Karnik et al.), code at Improbable-AI/embodied-red-teaming. Loop:
GPT-4o generates N "correct-but-challenging" instructions (image-conditioned; system prompt
enforces task-faithfulness softly), best-of-5 CLIP-diversity selection, policy rollouts score
difficulty, top-k hardest fed back as examples for round k+1. Key facts for us:

- **CoVer only used ERT(k=0)** — single-pass GPT-4o, no refinement loop, no rollout feedback
  (Bridge tasks were never in ERT's release; they generated their own). Their 33/task
  "ert_rephrases" are DEFENSIVE test-time rephrases of the red-team instruction, not ERT output.
- **R(π,c) is a black-box scalar** in Algorithm 1 — the rollout-success sort key swaps for our
  offline CRN flow loss with no structural change. Agent-recommended form: per-episode DELTA
  (loss under candidate − loss under nominal instruction), isolating the instruction effect
  from state/multimodality — exactly our CRN advantage structure, reversed (keep HIGH delta).
- **ERT has no faithfulness check** (prompt-only); an adversarial loss-proxy drifts to
  semantically-wrong instructions faster than rollout feedback would → our 3-class gate
  (exclude goal_drift) is the missing component, run on selected candidates.
- Refinement rounds that ERT could only afford in fast sims (~26k rollouts/round on CALVIN)
  become a few GPU-hours of batched π0 forwards offline.

Recipe: 2000 train contexts × N=10 candidates (generator: local Qwen $0 first; selection does
the work), CLIP-diversity via the PAPER's pairwise-cosine (repo's `.mean()` is buggy), keep
high-delta gate-clean candidates, K=1–2 rounds. Train: red-team instruction as INPUT →
rephrase → flow/L2-to-a* reward. Eval on CoVer's 4 ERT instructions, strictly held out.
Optional proxy calibration: vanilla rollout-scored ERT on the 4 SIMPLER tasks vs offline
delta-loss rank correlation (~50 rollouts/instruction).

## Ablations

| Ablation | Question |
|---|---|
| **Teacher-SFT warm-start then tune** (vs primary RL-from-base) | Does distilling Gemini's higher-ceiling distribution first beat pure RL-from-base? 0b says teacher has ~6 pts more oracle headroom — worth testing, not assuming. |
| **Cached-frontier-trace conditioning** (c = o, ℓ, r) vs primary CoVer-style inline reasoning | Does a *frontier* model's cached scene reasoning beat the tuned model's own inline reasoning as conditioning? 2,000 Gemini traces cached. |
| Signed vs positive-only advantage | Does downweighting bad phrases help? |
| List-prompt vs 16× single-prompt sampling + dedupe | Cleaner diversity source? |
| Non-rephrase π0 as reward | Dependence on paraphrase-augmented reward model |
| Minimal thinking budget | Does a short, capped `<think>` budget improve candidate quality enough to justify the generation cost and the training-objective complications? (Baseline everywhere: thinking off) |
| CoVer verifier selecting among our 16 | Better candidates vs better selection |

## Compute

One pod, **2× ~40–48GB GPUs** (e.g., 2×A40 — cheaper than 1×A100 80GB):
- **GPU A — reward server:** frozen π0 (~7GB bf16); 16 phrases × 16 draws = 256 forwards per context, batched. Second process, no networking.
- **GPU B — phrase model:** Qwen-7B LoRA + gradient checkpointing; HF `generate` for sampling (vLLM later if throughput bites).

## Results so far

Narrative + figures: **[WRITEUP.md](WRITEUP.md)**. Canonical numbers: `results/**/metrics.json`. This section keeps only the lab-notebook facts.

### Phase 0b — sensitivity gate: **GO** (2026-07-07, commit e4c9ee1)

250 val contexts × ~92 phrases (original + 3 arms × 32, deduped) × K=16 shared draws, both INTACT checkpoints, 1× RTX A6000. Metrics: `results/phase0b/metrics.json`; charts: `results/charts/phase0b_{rephrase,plain,ranked_phrases}.png`; per-context: `results/phase0b/per_context_*.parquet`.

| Arm | ρ split-half (z), rephrase ckpt | ρ, plain ckpt | spread rephrase / plain | length-loss corr |
|---|---|---|---|---|
| gemini_pro (3.1-pro-preview) | 0.956 | 0.956 | 0.0087 / 0.0114 | ~0 |
| gemini_flash (3.5-flash) | 0.958 | 0.955 | 0.0089 / 0.0107 | ~0 |
| qwen (Qwen3.5-9B base, no-think) | 0.954 | 0.951 | 0.0065 / 0.0102 | ~0 |

Lab-notebook facts (rationale/narrative in WRITEUP):
- Signal share ≈0.98 at training-budget K → reward reliable; OpenVLA failure mode absent.
- τ-band: discriminability collapses for τ<0.25 (clean-action end; lerobot τ=1=noise), plateaus ≈0.8 for τ≥0.3 → concentrate training draws at τ≥0.25.
- Rephrase-FT π0 has ~25% *smaller* phrase spread than plain, but advantages are group-normalized so absolute spread cancels — ranking reliability (ρ≈0.95) is equal for both. **Use rephrase-FT as reward** (it's the rollout policy; avoids train/eval mismatch). Corrects an earlier "prefer plain for bigger signal" note.
- Original Bridge instruction worse than median rephrase in 70.8% of contexts, single-best in 5.2%; best-of-32 beats original in 96–98%, median 31–33% loss reduction.
- Qwen spread 75–100% of Gemini arms → no larger generator needed (27B deferred).

### Phase 0c — rollout sensitivity: phrasing moves success; reward has a confirmed drift-hacking axis (2026-07-07)

1,290 SIMPLER episodes (4 tasks × ~33 phrases × 10 shared episode_ids, rephrase-FT π0) + CRN scores on 50 matched real Bridge contexts (3 tasks) + LLM faithfulness judge. Metrics: `results/phase0c/metrics.json`; raw: `results/phase0c/raw/`.

- **(A) in-sim [clean]: CONFIRMED.** Success spans 0.1–0.7 across phrasings of one task on identical initial states (std 0.17–0.18 vs binomial floor ~0.15). Sim-canonical originals are strong (carrot orig 0.5, only 12% of rephrases beat it) — unlike offline 0b where originals rank poorly.
- **(B) reward↔success [caveated]: negative until goal drift is removed.** Pooled ρ=−0.10 (carrot −0.25, spoon −0.29, stack +0.21). Re-judged 2026-07-07 with the 3-class majority-of-3 judge (clean/rename/goal_drift, = Phase 2 gate semantics): **goal_drift (45/129 phrases): success 0.32, loss 0.0807 — reward-favored AND success-poor (the hack class). clean (69): success 0.58, loss 0.0953. rename (15): success 0.65 (highest!), loss 0.0938 — renames are behaviorally fine and get no reward discount**, so the drift-only gate (renames allowed, CoVer-parity) is validated. Gate-pass correlations: carrot −0.02, spoon −0.06, stack +0.27; per-task mean +0.06. ~~pooled +0.28~~ RETRACTED 2026-07-09: rank-scale artifact (unnormalized within-task ranks concatenated across different-size tasks); normalized pooled = +0.04. Corrected claim: gating removes the anti-correlation; residual fine-grained validity ≈ 0.
- **Consequence for Phase 2: faithfulness-gated reward is mandatory** — compute advantages among faithful candidates only (LLM judge or equivalent), penalize drift; otherwise RL will exploit the confirmed axis and generate plausible-trajectory wrong-goal phrases. (Generalizes the "log object/target preservation" plan item from monitoring to reward.)
- Judge noise note: flash-judge verdicts vary slightly across runs (spoon 14 vs 15/32); training-time gate should use majority-of-3 or a stricter deterministic check.
- Late-τ/late-episode reward variant: mixed evidence (stack ρ→+0.40 late, carrot unchanged; n_ctx 5–9/band) — parked.

### Step-0 / 0b-redo (2026-07-08): CoVer-template distributions re-pass the gate

120 val ctx (250 for gemini arm) × 32, K=16, rephrase-FT ckpt. cover_gemini ρ=0.954/oracle 31.8%; cover_qwen_inline ρ=0.955/32.2%; cover_qwen_trace ρ=0.962/32.2%; spreads 0.0102–0.0108. Judge-free axes: trace ≈ inline → interim primary = inline (prompt parity, no frontier call at deploy). Drift-rate comparison pending Gemini credits (3rd depletion; judging burned prepay). RL stack validated to the gate: trainer checkpointed step 0 and exited 3 on GateUnavailable (fail-closed works). Infra fixes: publish() rebase-before-push (race killed overnight chain); env injection into tmux sessions (RunPod bashrc early-return ate GEMINI_API_KEY).

## Data artifacts

| Artifact | Where |
|---|---|
| Code, results, metrics, charts, experiment log | GitHub `sttawm/phrase-rl` (results/ committed; data/ gitignored) |
| 250 val contexts (`data/contexts_val_0b.parquet`, images+actions) | local Mac `data/` + pod volume; regenerable deterministically via `extract_contexts.py` |
| Rephrase arms: `rephrases_val_0b` (Gemini pro + traces), `rephrases35flash_val_0b` (flash + traces), `qwen_rephrases_val_0b` (Qwen, full) | pod volume `/workspace/phrase-rl/data/`; pro+flash also local; qwen only partial (25 ctx) local |
| CRN scores: `scores_0b_{rephrase,plain}.parquet` (370k rows each) | local `data/` + pod volume |
| Frozen models (INTACT ×2 ~13GB, Qwen3.5-9B ~18GB, PaliGemma tokenizer) | pod volume `/workspace/hf_cache` (re-downloadable) |
| Pod envs: `.venv` (INTACT-era lerobot fork, reward), `.venv-gen` (transformers 5, generation) | pod volume `/workspace/phrase-rl/` |

Network volume: RunPod US-KS-2, 200GB — everything on it is reproducible from git + HF + the local copies.

## Risks

1. **Reward non-discriminative** (Phase 0b gates this). Prior evidence: CE/L2 reward on OpenVLA was flat across text variation → mode collapse.
2. **Reward hacking → template collapse** (diversity metrics during training; KL anchor).
3. **Offline loss ↔ rollout success link unproven** — Phase 0c tests a small slice early; Phase 4 is the real test; CoVer's verifier-based selection is the fallback comparison. Related: reward/offline eval use real Bridge frames, rollout eval is sim (SIMPLER) — if they disagree, keep "loss doesn't predict success" and "real↔sim gap" distinguishable.
4. ~~Repo housekeeping~~ resolved 2026-07-03: fresh repo initialized here; stray parent repo removed (backup at `~/.dev-stray-git-backup`).

## Regime change @ flow~500 / l2~300 (2026-07-10, user)

Margin creep diagnosed as under-training: batch = 2 contexts/update (noisy
advantages) + double-braking after v1 (lr halved AND beta quadrupled), while val
best-of-16 margin (+33%) vs mean (+8%) shows large un-captured headroom. Change,
both arms simultaneously (A/B fairness): contexts-per-step 2->6, lr 5e-6->1e-5.
beta=0.15 + KL breaker 1.2 unchanged as guardrails. Charts: step axis is now
~3x more data per step after this point.

## Contingency (user, 2026-07-10, standing order): rollout-reward arm

If the proxy-reward RL hits a wall (margins plateau after the cps=6/lr fixes, or
deployment evals stop improving vs base): switch to a ROLLOUT-REWARD run.
Design: contexts = sim initial states of the VAL split (4 tasks x ep 0-24, incl.
eggplant); reward = success rate over ~2 reps x ~12 candidates (~15 min/step);
NO separate val during training — report training success directly; final
evaluation ONCE on the sealed test tier (ep 25+, CoVer protocol). Rollout server
mirrors the score-server file-IPC protocol, INT-ACT venv. Caveat for writeup:
trains on eval-adjacent tasks (ERT instructions still held out).

## Speed package @ flow~690 / l2~490 (2026-07-10, user: batches tiny, profile, cut judge)

Profile at cps=6: step 283s; scoring only 20-30s (8%) — the rest is generation +
judge + update (now instrumented: gen_sec/judge_sec/score_sec/update_sec per step).
Changes: (1) per-stage timers; (2) L2 ARM DROPS THE GATE (--no-gate) — hypothesis:
decoded-action L2 punishes goal drift natively (wrong goal -> different trajectory
-> large L2), making the judge redundant there; flow KEEPS the gate (0c: flow
reward actively prefers drift). NOTE: the A/B now differs in reward AND gating —
documented deliberately. (3) flow judge trimmed (verdict max tokens 1000->400).
Batch-size context: typical GRPO updates are 256-1024 prompts x 8-16 samples;
ours is 6x~14 — next lever after timers is batched multi-context generation.

## Eval state-split registry (CORRECTED 2026-07-11)

FINDING (source audit of put_on_in_scene.py): episode_ids index a FINITE per-task grid
(~12-24 layouts; spoon = 12 xy pairs x 2 quats = 24) and WRAP BY MODULO — ids 25+ alias
ids 0-23. There is no unseen-state tier: val (ids 0-24) already covered every layout.
All earlier "sealed state range" designations are void.

- VAL: episode_ids 0-23 (the full grid), x6 repeats, CRN-pinned decode noise.
- TEST = HELD-OUT TASKS, never states. Final eval: chosen checkpoints on held-out
  tasks + CoVer's ERT instructions (also unseen by training).
- Contrast-variance floor: ~96 (task,state) cells on the 4 ID tasks. Going below ~±4pp
  on paired contrasts requires ADDING TASKS, not states or reps.
- Note: CoVer's "50 reset seeds" also sample (with replacement) from these same grids.

### TASK-TIER REGISTRY v4 — VAL-8+screen / TEST-15 incl CoVer trio (user ruling 2026-07-18)

Amends v3 on one point, USER RULING 2026-07-18: the CoVer OOD trio
(redbull_on_plate, zucchini_on_towel, tennis_in_basket) moves to the SEALED
TEST tier — reserved for the direct head-to-head against CoVer's published
OOD ERT rows. NOTHING touches them before the one-shot final eval: no ERT
generation, no screening rollouts, no reference scoring. VAL grows from the
unused native _clean inventory instead (members picked by the Track-2
improvability screen, disclosed as improvability-selected per dissent (f));
TEST = 12 sealed natives + the trio = 15. v3 text below retained for history.

### TASK-TIER REGISTRY v3 — VAL-11 / TEST-12 (user decision 2026-07-16, SUPERSEDED on the trio by v4 above)

Supersedes v2 and the 10/4/6 split below. Rationale: the verifier reward trains
on Bridge only — no arm trains on sim tasks — so all registered tasks are eval
capital; sealed tier = every never-touched task, maximized.

- VAL (11, selection + development): the 8 TOUCHED native tasks — the 4 ID
  Bridge tasks (spoon, carrot_on_plate, stack, eggplant: also the reward-model
  diagnostic tasks; the reward was SELECTED on their labels, so checkpoint
  selection weights the other 7 val tasks and reports this quartet as a
  labeled, mildly-circular column) + the 4 spent val tasks (coke_can_on_plate,
  carrot_on_keyboard, coke_can_on_ramekin, carrot_on_wheel) — plus CoVer's 3
  OOD tasks (redbull_on_plate, zucchini_on_towel, tennis_in_basket; envs
  imported from CoVer's release or reimplemented; object-novel/
  receptacle-familiar composition, report labeled as such).
- TEST (12, SEALED): every never-touched native task. Names deliberately not
  enumerated (composition-gaming note, 2026-07-15). Eval scripts refuse these
  tasks unless FINAL_EVAL=1. One shot: all finalist + static arms in a single
  run, nominal + ERT inputs, paired CRN, ×12. Pre-computed static rows stay
  SEALED-UNREAD until trained arms join the table.

### TASK-TIER REGISTRY — 10/4/6 split (designed 2026-07-13, SUPERSEDED 2026-07-16)

Supersedes the provisional 6-variant list that stood here (PutCokeCanOnPlate was
spent as VAL by the 2026-07-14 night suite; the rest are reassigned below).

SOURCE OF TASKS (audited 2026-07-15): INT-ACT's ManiSkill2_real2sim fork,
`custom_scenes/put_on_in_new.py` @ eeb04c7 — 18 registered "-v2" (clean) combos
+ the 4 original Bridge tasks = 22; our 20 = all of them minus the 2 near-dup
"sponge_larger" variants. widowx_<obj>_on_<recv>_clean names wrap the -v2 env ids.
The battery is FINITE: adding combos beyond it = a new env subclass with
hand-specified xy/quat layout grids (~40 lines, asset library ~8 sources x ~7
targets). Success is analytic sim geometry+contact (source-on-target: xy within
target top-surface extent, z above, source contacts only target/robot); ramekin
and wheel targets use the looser src-IN-target override; nut_on_wheel tweaks
z tolerance. No demos or learned detector involved — every registered task has
a well-defined success signal, same rule for all arms.

- TRAIN (10): the 4 ID Bridge tasks (spoon_on_towel, carrot_on_plate, stack_cube,
  eggplant_in_basket) + pepsi_on_plate, cube_on_plate, coke_can_on_keyboard,
  carrot_on_ramekin, eggplant_on_sponge, nut_on_wheel.
- VAL (4): coke_can_on_plate, carrot_on_keyboard, coke_can_on_ramekin,
  carrot_on_wheel. First use: 2026-07-14 task-generalization night suite
  (nominal inputs, 5 arms — L2@280 the only arm above original, 32.8 vs 28.3).
  Reusable for model selection; NEVER for headline claims.
- TEST (6, SEALED — no arm, no eval, no phrase generation may touch these until
  the one final eval): nut_on_plate, orange_juice_on_plate, eggplant_on_keyboard,
  carrot_on_sponge, coke_can_on_wheel, small_plate_on_green_cube.
  Protocol: single shot, nominal + ERT inputs, all finalist arms in one run,
  paired CRN contrasts, x12 reps. Opening this tier requires updating this line
  with the date and commit of the eval that spent it.

## Rollout-reward arm LIVE (2026-07-12) — the wall fired

Both v3 arms plateaued per the tripwire (flow: 8 vals flat at 6-9%; L2: 5+ vals
under its +5.0% best). Standing order executed: pod 2 handed over from L2
training (archived @280 best, +5.0% margin, 49.0% deployed) to ROLLOUT-REWARD RL.
Reward = SIMPLER success rate (2 CRN-pinned reps/candidate), contexts = the 96
(task, layout) val cells, NO val pass (train-on-val per user; success is read
directly off the training log; final eval ONCE on held-out task variants).
Config otherwise unchanged (32 cands, top-8, accum 4, beta .15, lr 7e-6).
~40 episodes/step ≈ 5-6 min/step, 48 steps/epoch. Flow arm continues on pod 1
as the proxy-reward representative.

## Hot-optimizer probe on flow arm @ ~step 490 (2026-07-12)

Plateau shape = fast-then-flat (not slow-climb) + KL drifting at 0.55 → hypothesis:
reward-information ceiling, not lr. Cheap falsification since flow is the spare arm:
clip 1.0->2.0 (grad norms ran ~2x clip — lr alone would be absorbed) + lr 7e-6->1e-5.
If margins break out of the 6-9 band: optimizer was binding (revisit rollout arm's lr
too). If KL runs hot with flat margins: information ceiling confirmed; the rollout
arm (reward = the target metric itself) is the real escape route.

## v4 on pod 1: GRPO update rule x flow reward (user, 2026-07-13)

Confounder check: the proxy "info ceiling" verdict assumed the update rule was
adequate — but positive-only weighted SFT (RAFT-style) can imitate good phrases,
never repel bad ones. v4 = standard GRPO (full group, SIGNED z-advantages,
negatives pushed down; single on-policy update per batch so no importance clip
needed), same flow reward, same recipe otherwise, G=16 for full-group cost,
FROM SCRATCH. If GRPO breaks the base-parity band: the ceiling was the update
rule, not the reward. Flow v3 archived (best_val@40, deployed 48.9 = parity).
NOTE: pod-3 eval loop keeps watching results/checkpoints/phase2 — its
"tuned_flow" arm label means THE GRPO ARM for pairs f<step> from here on.

## v5: GRPO x ROLLOUT reward (user redirect, 2026-07-13) + fair-eval protocol

User: GRPO should test the rollout reward (not flow — v4 killed at launch), and
positive-only updating was a mistake in general. v5 = pod 2 from scratch:
GRPO (full group, signed advantages) x SIMPLER-success reward.

FAIR-EVAL PROTOCOL (rollout RL trains in the same sim it is evaluated in):
- TRAIN: layouts 0-17 per task (72 cells), nominal instructions, reps 0-1,
  noise seeds offset +1000 (disjoint from all eval seeds).
- VAL (continuous): ERT instructions x ALL 24 layouts x fresh-seed reps —
  reported SPLIT: seen layouts (0-17) vs HELD-OUT layouts (18-23). The 18-23
  column is the honest generalization number; a seen/held-out gap = layout
  overfitting, quantified.
- TEST (once, end): the 6 untouched task variants, nominal + ERT instructions.
- Contamination audit trail: training never sees ERT phrasings, layouts 18-23,
  eval noise seeds, or the held-out tasks.

## Reward-function decision (2026-07-16)
v5 arms A/B run to completion on the LOCKED 1-frame reward (5-seed calibrated
both_nodiff ensemble, `verifier_reward_ensemble.json`) — unchanged mid-run for
curve comparability. The native-4f ensemble (`verifier_reward_ensemble_4f.json`,
5 seeds, calibrated T=1.75-2.7) is HELD for a later from-scratch run; until then
it serves only as selection/eval scorer (CoVer-select arm, checkpoint picking).
Basis: bridge-val 4f-agg AUC 0.972 vs 0.959 (hard 0.954 vs 0.937); study-exam
pooled pairwise 0.89 vs 0.87 (ensemble+calibrated, all three evidence readings
higher). Exams: results/overnight/raw/exam_4f_results.txt, charts reward_scorers_*.

## v6 run spec (accumulating; 2026-07-17)
- REWARD: native-4f calibrated 5-seed ensemble (`verifier_reward_ensemble_4f.json`)
  — the held reward from the 2026-07-16 decision; v6 is the "later run" it was held for.
  (Score server: --verifier-ensemble results/checkpoints/verifier_reward_ensemble_4f.json)
- INPUTS: mixed-source conditioning — nominal / benign rephrase / hostile ERT-style
  (`ert_train_sources.parquet`, stage-1 partial 2,124; retry pending credit top-up)
- TRACES: ERT-derived for hostile-source steps (`ert_train_traces.parquet`, stage 2
  pending credits) — leakage firewall: trace generator never sees the nominal phrase.
- PROTOCOL: from-scratch (input-distribution + reward change); v5 runs to its
  performance stop under the locked 1f reward for clean attribution.
- SUCCESS CRITERIA (pre-registered): tuned ERT->greedy beats frozen ERT->greedy
  (honest, ERT-derived-trace variant) within ~100 steps; x12 ERT rollouts separate
  from frozen greedy 40.2%.
- CONDITIONING CONSISTENCY (2026-07-17): the p_single update conditions on the SAME
  source that generated the candidates — both arms. Arm A's v5 asymmetry (16-list
  farmed from nominal, update conditioned on source) is eliminated; in v6 the list
  prompt is source-conditioned as well. Trace lookup is tier-aware: (episode,t,variant)
  -> ERT-derived trace for hostile sources, nominal trace for nominal/benign.
- v6.1 (launch amendment, same night): INPUT-SLOT DROPOUT p=1/3, orthogonal to the
  tier mix — the prompt's instruction slot becomes "infer the task from the context"
  while the tier-matched trace carries the task (traces quote their source phrase, so
  this drops the SLOT, not the information; targets the slot-keyed echo circuits).
  Gate/reward anchored to the true instruction; generation and update share the
  dropped prompt. Per-step telemetry: input_dropout_rate. From-scratch relaunch.
- VAL-PROBE REWARD PARITY (2026-07-17, staged for next natural restart): run_val
  scores at the same 4-frame evidence as training. New git-canonical
  `results/phrase_artifacts/contexts_val_multit.parquet` (val40 x 4 quartile frames,
  same generator/seed as the train multit); trainer merges it into the reward frames
  map via --val-reward-frames-map (missing file -> loud warning + old 1f behavior).
  Rationale: judge pairwise 0.769 at 4f vs 0.687 at 1f — val probes were the noisier
  judge. NOTE: the val reward series LEVEL steps at the deploy boundary (4f-averaged
  calibrated logits vs 1f); within-series comparisons resume after it.
- v6.2 (hot amendment, 2026-07-17 late, arms resumed at ~A52/B61): STRATIFIED step
  plan — 8 contexts/step = exactly 2 nominal / 2 benign / 4 ERT (ERT quad = 2 short-
  register + 1 rename + 1 stance variant); input-slot dropout = exactly one slot per
  tier (3/8 = 37.5%). Replaces iid tier draws; kills zero-tier steps, halves gradient
  mixture variance, delivers the half-CoVer-register hostile share deterministically.
  grad-accum 4->6. Same objective; resumed in place (no from-scratch restart).
- v6.3 (hot amendment, same night, resumed ~A65/B75): TIER-CONDITIONING TAGS —
  the prompt's instruction slot is prefixed with its regime ([input: original
  wording] | [input: paraphrased] | [input: adversarially reworded] | [input:
  withheld]). Rationale: regime is TOLD, not inferred from surface statistics
  (kills the style-fingerprint pathway); constant-at-deployment for the 100%-
  hostile benchmark, so distribution knowledge, not per-example oracle —
  DISCLOSE in writeup. Val probes tagged truthfully; candidate echo-strip added;
  eval gen takes --input-tag (v6.3 ckpts screened with "adversarially reworded").
  CORRECTION (same night): the first v6.3 hot-swap silently failed (graceful
  SIGTERM exit ~8-10 min vs a 5-min wait; the old untagged process kept the
  GPU). Redone with a full-exit wait: BOTH arms resumed from step_0040, so
  TAGS ARE ACTIVE FROM STEP 41 on A and STEP 51 on B (the true resume states
  were A@40 / B@50 from latest/; the abandoned untagged redo steps were dropped
  from telemetry by resume dedupe). Ladder screens are step-gated: VAL_TAG_FROM
  =41 for A, =51 for B (A v6step_0020/0040 and B v6step_0020..0050 = pre-tag).
  VAL SCALE BREAK AT STEP 60 (root cause found, no bug): commit 0ede0b3
  ("val probes at 4-frame parity", STAGED) activated at the v6.3 restarts.
  Vals <=40 and val40_references_4f.json scored the 4f ENSEMBLE at 1 FRAME
  (val episodes were absent from the frames map; silent fallback); vals >=60
  score at TRUE 4 frames. The step-60 "drops" (orig 0.847 -> 0.617, identical
  across arms) are this scale change, NOT the tags and NOT contamination —
  the earlier CONTAMINATION note is WRONG on the orig/double-tag mechanism
  (res["instruction"] was never tagged; gen input was single-tagged and
  correct). The val-probe echo-strip added that night remains as hardening.
  References rebuilt at true 4f: val40_references_true4f.json; criterion
  comparisons valid for vals >=60 vs the new refs. best_val_margin poisoning
  concern likewise withdrawn (orig shift was uniform per era).
  ARM A TERMINATED (cost decision, ~step 80): v6 continues as arm B only.
  Rationale: A ran 2.6x slower per dollar (batched-update parity fallback),
  v5-A was the copy-drift arm, list-mode showed tag-era parse-fail uptick,
  and the A/B contrast had delivered its finding (conditioning consistency).
  ALL A checkpoints archived locally (adapter_archive/, incl. final state +
  optimizer — resumable). Val fleet: nominal chain pod and pod4 self-remove
  when their queues drain; v6-A ladder self-removes after screening
  v6step_0060. Steady state: B RL pod + v6-B ladder.
  QUARANTINE (2026-07-18 dawn, panel-verified byte-level): the 37.6% nominal
  x12 headline is VOID as a reward-transfer verdict — (a) nominal_eval_assets
  .parquet (8b3ae02) had a 3-cycle TRACE MISALIGNMENT (carrot<-eggplant<-stack
  <-carrot): 288/2304 episodes ran wrong instructions (stack 0.0% = artifact);
  fixed in nominal_eval_assets_v2.parquet (script now requires v2); (b) v5-B
  s260 ECHOES the nominal input byte-exactly on all 7 clean tasks — the +0.54
  verifier "climb" = converging to emit the original (score = originals ref).
  Echo-Goodhart, not fine-axis transfer failure. Ex-bug: tuned 43.0% vs 42.6%
  battery (deficit vanishes; parity PROVISIONAL, cross-pipeline). (c) The
  40.2% battery anchor is RELABELED: base_greedy generated from ERT inputs —
  it is frozen ERT->greedy HOSTILE REPAIR, not frozen-greedy-on-clean; no
  clean-input frozen rollout anchor exists yet (pending fixed frozen leg).
  Ladder boundaries (pre-registered, vs 40.2 repair anchor, +/-3-4pp): STRONG
  >=38, WEAK 33-38, NULL 27-33, HARMFUL <27; verdict checkpoint >=step 120;
  mandatory echo byte-audit before interpreting any leg.
  V6-B END OF RUN (user cost decision, 2026-07-18 night): training stopped at
  step ~192 (of the >=300 protocol) after the migration; rationale = the flat
  triptych (mode/distribution/oracle all unresponsive to verifier gains
  through 180) made steps 192-300 marginal. Full step-190 state (adapter +
  optimizer + trainer_state) archived locally (adapter_archive/B_v6_final_s190).
  Fleet reduced to pod4 (A6000) for the exam + selection-exam workstream.

### V7 RUN SPEC (authorized 2026-07-19; launches after selection-exam confirmation)
Single treatment change from v6-B: the FINE reward. reward = C4b rank blend
(0.25*rank01(ensemble logit) + 0.75*rank01(-grip_err), within candidate
group) — exam-selected (66/68 native signs vs production's 51/68; grip-rank
ceiling 67/68; 25/75 chosen over w=0 for pre-registration cleanliness,
diversity against grip's per-task pathologies (carrot rho -0.12), and
two-axis Goodhart resistance). Blend applies to TRAINING ADVANTAGES ONLY;
all telemetry/val metrics stay on component scales (logit + grip) so every
existing reference remains valid; grip_by_tier + val grip components added.
Recipe otherwise FROZEN from v6-B: from scratch, ckpt phase2_v7, n=16
sample_single, stratified 2/2/4, dropout 1/3, tier tags, beta 0.15, lr 7e-6,
grad-accum 6, val/save cadence identical, tripwires identical. Host: pod4
A6000. Gates before launch: selection exam (C4b as best-of-8 picker on
native sampled cells) + 2-step smoke. Gemini NOT required (reuses v6's
4,708-source ERT corpus); needed later only for keeper-task eval assets.

## SFT-17k: supervised ERT->GT reconstruction (user pivot, 2026-07-19)
Motivation: the GT-convergence result (tuned v6 ERT outputs drift toward
Bridge GT phrasing, +0.134+/-0.027 overlap vs frozen, 5-sigma) plus the
reward=GT-proximity-detector diagnosis imply the direct move: skip RL and
supervise the mapping hostile->canonical on ALL Bridge train-split
instructions. User: "give the VLM the ERT phrase and merely fine-tune it to
reconstruct the original task phrase. No RL."
DATA: 17,297 unique clean train-split instructions (60k episodes ->
19,658 uniques -> minus val/test-split instruction keys, 5<=len<=120);
3 hostile variants each, generated LOCALLY by frozen Qwen3.5-9B
(pod4_gen_variants.py; styles: synonym/attribute injection, stance/manner+
indirect reference, CONCISE 40-70ch; CoVer-release shots anchor the genre).
Eval hostile sets remain Gemini/CoVer-authored => held-out attack
distribution. Generation v1 bug: thinking-mode ate the token budget (0/48
kept) — v2 uses enable_thinking=False + `ERT: "` prefill (the
apply_template discipline). Filters: per-style length windows, !=GT,
refusal/format rejects, and a SPATIAL-FLIP reject (variant introducing the
antonym of a GT-stated relation, e.g. right->left, is a goal change, not a
rewording — caught in smoke).
TRAIN (sft17k_train.py, PRE-COMMITTED, no checkpoint selection): LoRA v6
geometry (r16/a32/dropout0, same 7 targets) on Qwen3.5-9B; 2 epochs, cosine
1e-4, bs16 x accum4, loss on target tokens only; text-val = held-out 5% of
PAIRS (never SIMPLER). Conditioning: WRAPPER user turn + "Canonical:"
prefill, continue_final_message, enable_thinking=False — reused byte-
identical at eval (sft17k_generate_eval.py). Artifact = final/; best_val/
archived per repo policy but NOT used for selection.
EVAL PROTOCOL (ruling, 2026-07-19): NO SIMPLER val phase — nothing to
select. One full x12 ERT eval on the 8 TOUCHED val tasks vs the existing
frozen anchors (repair 40.2, sampled 33.7, oracles), CRN-paired; sampled
k=8 companion for the distribution view. The SEALED test (12 natives +
CoVer trio) stays unspent: it fires ONCE with all finalists (frozen, v6,
SFT-17k, any v7/v8) per registry v4. If SFT-17k beats frozen repair on the
touched-8, it enters that final one-shot as a finalist.

## SFT-17k amendments + provenance discovery (2026-07-19, user design round)
PROVENANCE (verified, HF card): the frozen pi0 IS the INTACT suite's
rephrase-finetune (arXiv 2506.09930) — BridgeV2 fine-tune WITH instruction
paraphrasing per the PUBLIC dictionary rail-berkeley/OXE_paraphrases.
Bridge slice extracted + committed: results/phrase_artifacts/
oxe_paraphrases_bridge.parquet — 6,463 Bridge keys x median 38 paraphrases
= 235,764 benign phrases pi0 plausibly saw. Covers 5,506/17,297 (32%) of
our SFT train GTs; 2 of 4 native SIMPLER nominals are keys (spoon-towel,
carrot-plate) — their clusters are literal pi0 training inputs. Reframes
the robustness story: pi0 saw ~38 benign paraphrases/instruction and still
breaks on ERT => ERT sits outside the paraphrase manifold; GT-convergence
result unchanged (GT = densest seen mode). results/phrase_artifacts/
bridge_train_uniques.parquet = the 17,297 inventory (committed; matches
pod4 count exactly).
OVERFIT GUARDS (user concern): (1) trainer holdout switched to GT-GROUPED
split — text-val now scores reconstruction of instructions never seen as
targets, a true memorization probe (pair-level split only measured new-
variant-of-seen-target); (2) eval attack distribution is held out by
construction (train attacks = local Qwen styles; eval attacks = Gemini/
CoVer-authored); (3) SIMPLER rollout vs frozen is the binding test, per
user. Optional lever (PREPARED, not folded into v1): benign-arm data
(paraphrase->GT pairs from the dictionary, downsampled) would add input
diversity per target + pin nominal-input no-op behavior; v1 stays
hostile-only per the launched spec — one-command rerun if wanted.
ICL BASELINE ARM (user proposal, built): sft17k_icl_baseline.py — frozen
Qwen + retrieval (char-3gram cosine over 17,297 GTs, matched THROUGH the
235k paraphrase clusters, credit-to-GT) + reworded->canonical demos
(benign from dictionary, hostile from SFT pairs) + optional Gemini trace
+ "Canonical:" prefill. Train-split content only. Control question: if
frozen weights + inventory access ~= SFT-17k, the value was knowledge,
not weights; also memorization-proof by construction. Runs on the same
touched-8 protocol as SFT (greedy + sampled k=8).
TRACE-SFT VARIANT (user question, assessed, HELD): SFT on the full CoVer-
style prompt (self-trace in context, loss on final phrase tokens only)
is the natural v2 if bare-wrapper SFT shows weak transfer — matched
train/deploy conditioning, reasoning available. Cost: trace generation
for ~50k examples (~10-20h A6000) or image extraction for 17k episodes;
also loses the single-pass deployment advantage (bare SFT repairs in one
forward pass; CoVer needs a trace round-trip). Decision deferred to v1
reconstructions + rollout verdict.

## ERT construction provenance — image usage per source (2026-07-19, user Q)
- Native-4 eval ERTs (cover_ert_instructions.json): verbatim CoVer release;
  CoVer took them from the ERT framework (INTACT lineage): HAND-CRAFTED
  linguistic variations (verb shifts, negation, referential appearance,
  commonsense cues) — scene-aware via human familiarity with the fixed
  SIMPLER scenes, not model-generated from image input.
- Keeper-quartet eval ERTs (valtier_ert_bank.json): Gemini-authored (prior
  session; producer script not retained). Text style is scene-referential
  ("on the left", "red-labeled cylinder", "long orange vegetable with green
  leaves" — matches real SIMPLER assets), so scene-grounded in effect;
  whether the frame itself was in the request is unverifiable from
  surviving artifacts.
- RL-training hostile corpus (ert_train_sources.parquet, 4,708; also
  gen_ert_val40): Gemini BATCH, NO image part — but prompt includes
  "Scene context: {trace[:400]}" where the trace came from the image
  (teacher pass) => indirectly image-grounded through trace text.
- SFT-17k variants (pod4_gen_variants.py): pure text-only — no image, no
  trace. The ONLY fully ungrounded ERT source in the project (documented
  deviation; eval attack distribution stays grounded => honest gap).
- Contrast, CoVer's own machinery: deployment repair VLM IS multimodal
  (scene image + instruction -> reasoning + variants, their Sec 4.3);
  their VERIFIER training augmentation was text-only GPT-4o (128
  variants/instruction) — precedent for text-only training-side variants.
Implication: the train/eval grounding gap is real but narrow — referential-
appearance attacks ("orange vegetable with green leaves") resolve in text;
the true blind spot of a text-only reconstructor is pure deixis ("the one
on the left"), which appears in the quartet bank. Watch v1 failures for
exactly that signature before reaching for image-conditioned v2.

## How OUR ERT sources were scene-grounded (2026-07-19, user Q, verified)
Chain for the 4,708 RL-training sources (and val40): each context is an
(episode_index, t) FRAME from a real Bridge episode; the teacher pass
(gemini-3.5-flash, CoVer template, cover35_teacher_train.parquet, 2000
contexts) received that actual camera frame + instruction and wrote a
trace VERBALIZING the scene (objects, colors, layout, distractors); the
hostile batch then received trace[:400] as text. Grounding = whatever
scene facts the teacher verbalized; the hostile generator never saw
pixels. Verified example of what this bought: GT "put the blue
rectangular block on top of the tower" -> ERT "Grasp the azure cuboid
currently crowning the wooden spire and deposit it horizontally upon the
pale, isolated block resting on the table surface" — "pale isolated
block" is a REAL distractor knowable only from the frame.
SAME EXAMPLE, the flip side: the trace said the blue block was ALREADY
on the tower (mid-episode frame), so the generator redirected the goal
to the distractor block — a goal-INCONSISTENT hostile input (not a
rewording). For RL inputs this was tolerable-by-design (reward pulled
toward the episode's true actions regardless of what the input claimed);
for SUPERVISED pairs it would be label noise. Corollary: the SFT-17k
text-only generator structurally CANNOT do this (no scene knowledge =
nothing real to redirect to; it can only rename/decorate the GT's own
content) — text-only is narrower in attack coverage but SAFER in goal
preservation, complementing the spatial-flip filter (which catches
antonym flips, not object redirection).
Keeper-quartet: traces verified image-fed (gemini_redteam_assets passes
the task frame's image_png); the bank ERT instructions' own generation
inputs remain unverifiable (producer not retained).

## SFT v1 = the DEAD-SIMPLE experiment (user directive, 2026-07-19)
User: "Get all of the bridge tasks, and ideally the rephrases as well from
the pi0 bridge rephrase finetuning. Then generate red-team rephrases using
text-only (no image). Then do SFT (no reasoning). Then evaluate vs frozen
Qwen (no reasoning). Bells and whistles later (Qwen reasoning; Gemini
frame traces for ERT and at inference)."
Mapping: (1) tasks = the 17,297 train-clean uniques (val/test-split keys
stay excluded — cross-experiment hygiene, costs 12%); (2) pi0-seen
rephrases FOLDED IN: scripts/merge_sft_pairs.py adds OXE-paraphrase->GT
pairs as style 3, downsampled to 4/key (~21k) to keep a hostile majority
(~51k styles 0-2) — supersedes the earlier "v1 stays hostile-only" note;
(3) red-team generation already text-only/no-image (running); (4) SFT
already no-reasoning (enable_thinking=False, bare wrapper); (5) COMPARATOR
CHANGE: primary baseline = frozen_bare — the BASE model under the
IDENTICAL no-reasoning wrapper (sft17k_generate_eval --include-frozen),
so the treatment is weights-only. The existing CoVer-style frozen repair
anchor (with trace, 40.2) stays as context, not the primary comparison.
Chain v2 on pod4: genvar -> pairs to git -> merge -> train (GT-grouped
holdout spans both styles) -> greedy recon smoke (SFT + frozen_bare).

## SFT-17k v1 TRAINED + smoke verdict (2026-07-20)
Run: 73,304 merged pairs, 2 epochs, 2,180 steps, ~4.5h A6000 (fla-core
fast path + non-reentrant grad ckpt after 2 OOMs on the linear-attn torch
fallback; 22GB steady). Text-val (GT-grouped, 864 unseen instructions):
best 0.663 (~epoch 1), final 0.706 (mild epoch-2 drift; best_val/
archived, final/ = the pre-committed artifact). Adapters archived locally.
SMOKE (greedy, val-8 ERT, both arms): frozen_bare ECHOES 5/8 hostile
inputs unchanged; SFT echoes 0/8. SFT nails all in-vocabulary tasks:
spoon/carrot-plate/stack/eggplant PERFECT Bridge register ("put the green
block on top of the yellow block" from "lush green element atop the
yellowish-orange element"), coke-on-plate preserves deixis correctly
("take the can on the left and put it on top of the yellow plate on the
right"). The 3 imperfect cases are ALL OOV-receptacle tasks — audit:
keyboard/mouse pad/ramekin/wheel have ZERO hits in the 17,297 GTs +
235,764 paraphrases (tire: 3) — keyboard->"mouse pad" (wrong neighbor),
ramekin->"the white object" (graceful degrade), wheel->NAILED by pure
language reasoning. The keeper quartet is thus an unintended OOD-object
probe: in-vocab repair vs out-of-vocab repair split cleanly.
GREEDY x12 EVAL LAUNCHED (run_sft_eval.sh, sfteval tmux): SFT arm then
frozen_bare, 2,304 eps each, results auto-commit to results/val_screens/
sfteval12_{sft,frozen_bare}.parquet. Read against: frozen+trace repair
40.2, clean ceiling 41.8, v6 RL tuned 37-39.6, passthrough anchors.

## SFT-17k x12 INTERIM (sft arm landed 2026-07-20; frozen_bare running)
SFT greedy x12: POOLED 31.1% — but pooled is the wrong statistic; the
result is bimodal along the vocabulary axis:
- NATIVES 56.3% = best ever recorded on this suite (v6 RL s80/s120
  47.3/45.0; frozen+trace sampled 40.5). spoon 59.7 (v6: 25.0!),
  eggplant 94.4 (81.9), carrot-plate/stack at par.
- QUARTET 5.8% vs v6 29.4 — catastrophic, mechanism identified per-task:
  ramekin: v6's 64.2% phrase mapped ramekin->"white bowl" (correct
  in-vocab neighbor); SFT emitted "the white object" (vague) -> 3.8%.
  keyboard: v6's 22.9% phrase KEPT the OOV noun ("black keyboard"); SFT
  invented "mouse pad" (wrong neighbor; "mouse" = only desk-adjacent noun
  in corpus, 4 OXE keys) -> 3.1%. wheel: SFT 10.4 actually beats v6 3.1.
  keyboard/wheel/ramekin/tire have ZERO keys even OXE-WIDE -> a broader
  dictionary slice does NOT fix this; the lesson is behavioral: resolve
  the description, DON'T force the noun into training vocabulary (v6's
  image+trace conditioning got this right; bare text SFT learned
  always-in-vocab and pays for it exactly where the vocab ends).
VERDICT SO FAR: supervised canonicalization CONFIRMED where the canonical
target exists (natives +9-11pp over the best RL arm) and ANTI-confirmed
on OOV receptacles (information destruction). The hypothesis test split
cleanly along its own assumption boundary.
PRE-REGISTERED PREDICTION (before frozen_bare lands): frozen_bare echoes
5/8 -> its quartet should score near echo/passthrough (~15-30%), natives
mixed; pooled plausibly ABOVE SFT's 31.1 — if so, the bare-vs-bare
primary is a frozen win on pooled with SFT dominant on natives, and the
honest headline is stratified, not pooled.

## CORRECTION (user caught, 2026-07-20): protocol mixing in the interim table
The interim comparison listed v6 greedy-x12 rows beside frozen+trace
SAMPLED (33.7) — implying v6 RL > frozen, which was never the finding.
Record: same-protocol sampled = TIE (tuned +1.5+/-1.6 vs frozen 33.7);
greedy tuned 37.2-39.6 vs frozen repair 40.2 (older battery suite, cross-
suite caveat) = frozen nominally ahead. The temperature tax is asymmetric
(frozen -6.5pp sampled vs tuned -2.0) so cross-protocol reads are extra
misleading for exactly that frozen row. Frozen+trace GREEDY x12 on the
current suite has never been measured -> leg queued after frozen_bare:
completes the all-greedy same-suite ladder passthrough -> frozen_bare ->
frozen+trace -> SFT -> v6 RL. Protocol-clean claims that survive: SFT
natives 56.3 vs v6 greedy natives 45.0-47.3 (+9pp, same protocol/suite);
quartet 5.8 vs v6 29.4-31.1 (same protocol/suite).

## OOV-hope trace probe (user request, 2026-07-20): Qwen vs Gemini on tire+ramekin
results/analysis/trace_probe_oov.json. VERDICT: the resolution the SFT
arm destroyed EXISTS in Qwen 9B's own image-grounded reasoning — 2/2:
ramekin -> "white ceramic bowl" (verbatim the noun class that scored
64.2% under v6), wheel -> "black tire with a silver rim". Gemini traces
resolve the same nouns with cleaner structure + explicit replacement
lists ("white bowl, white ramekin, ceramic cup" / "tire, wheel, hub").
Qwen's gap is FORMAT, not content: 1/2 outputs degenerated into a
repetition loop inside the CoVer template's word-list section
(extract_trace -> None); the ramekin trace also echoed template
placeholders. Implication: trace-conditioned v2 has a concrete mechanism
(trace injects the resolved noun; SFT formats canonically) and the trace
prompt should be a MINIMAL two-section scene+referents prompt (or Gemini
traces), not the full CoVer template, for 9B reliability.

## SFT-17k PRIMARY VERDICT: bare-vs-bare x12 complete (2026-07-20)
SFT 31.1 vs frozen_bare 25.8 pooled; CRN-PAIRED +5.3pp +/- 2.3 (n=192
cells) — the user-spec'd primary comparison is an SFT WIN (>2 sigma).
Stratified: natives SFT 56.3 vs 38.5 (+17.8); quartet SFT 5.8 vs 13.0
(-7.2, entirely ramekin). Per-task: SFT wins 7/8 (+1.7..+40.6; eggplant
94.4 vs 53.8, spoon 59.7 vs 40.6); loses ONLY ramekin -38.2 (frozen echo
of "hollow white ceramic cup-like container" scores 42.0 — pi0 parses
that circumlocution! — vs SFT's "the white object" 3.8).
PRE-REGISTERED PREDICTION OUTCOME: WRONG in the informative direction —
predicted frozen_bare quartet ~15-30 via echoes; actual 13.0 with echoes
mostly FAILING (keyboard circumlocution 0.3, deictic coke-plate 1.0).
pi0's descriptive-language competence is OBJECT-DEPENDENT (ramekin
circumlocution 42.0 vs keyboard circumlocution 0.3) — echo is not a
reliable OOV fallback either; RESOLUTION (v6's "white bowl" 64.2, "black
keyboard" 22.9) beats both echo and vague canonicalization.
Ladder (greedy x12, current suite): frozen_bare 25.8 < SFT 31.1 < v6 RL
37.2-39.2 (v6 generated under trace+image conditioning — not bare;
frozen_trace leg launching to complete the ladder). QUEUED on freed GPU:
trace leg then sampled leg (chained).

## V2 DIRECTIVE (user, 2026-07-20): jump directly to Qwen image-grounded traces
No templated/text-derived intermediate. v2 training traces = Qwen-generated
from (initial-scene frame + one sampled hostile variant per GT), SHORTENED
2-section prompt (probe: CoVer template breaks the 9B; minimal prompt is
reliable), all 17,297 v1 instructions (set held fixed for v1->v2
comparability), trace dropout ~50% at train, referent-mapping deployment
format. Existing Gemini traces cover only 1,353/17,297 uniques (7.8%) —
hence generate, not reuse. Pipeline: frames (scripts/extract_frames_17k.py,
CPU/network, t=0, one representative episode per instruction preferring
hash-train, temp-download+delete, shard-resumable) NOW alongside rollouts
-> Qwen trace gen (GPU, after eval queue) -> v2 retrain (~4h) -> v2 eval
in ITS OWN native trace format. Gate stays: v2 proceeds only if the
sft_trace / sft_selftrace arms move the quartet.

## SPLIT-LOGIC CORRECTION (found 2026-07-20 building the frame extractor)
The SFT inventory/exclusion (pod4_gen_variants + bridge_train_uniques)
excluded the first 10% of episodes BY FILE ORDER; the real Bridge split is
HASH-based (episode_split_u: val 0-5%, test 5-10%, train >=10%). Quantified:
1,893 hash-val/test instruction keys (1,640 exclusive to val/test episodes)
ARE in SFT training; the 2,360 excluded keys were an arbitrary file-order
set. IMPACT: no live claim depends on it — SIMPLER eval is a different
domain; verifier val/test episode integrity is episode-level and governed
by rl_train_exclusions.json (hash-correct, regenerated); SFT text-val is
internally consistent. Ledger language "minus val/test-split instruction
keys" in the SFT-17k entries is hereby corrected to "minus a file-order 10%
(split-logic error, no downstream effect)". v1 instruction set kept as-is
for v1->v2 comparability; frame extractor uses the CORRECT hash split for
representative-episode preference.

## V2 spec amendments (user, 2026-07-20): no dropout; copy-nouns is the point
TRACE DROPOUT REMOVED — v1 already exists as the no-trace artifact; v2
trains 100%-traced for its single deployment config (trace->reconstruct).
Copying from the trace is the MECHANISM, not a failure mode: division of
labor = nouns from the (image-grounded) trace, relation/goal from the
variant text (a t=0 frame cannot know the goal), register from the
trained prior. Only degenerate case: trace carrying the full answer
sentence (collapses v2 into echo-the-trace = frozen relation-parsing =
the measured 25.8). Referents-only trace format enforces the split.
Train-time trace noise (Qwen misnaming objects on real frames) is the
organic anti-blind-copy regularizer; no artificial safeguard added.

## Trace-source comparison protocol (user-confirmed 2026-07-20)
No training on Gemini traces anywhere. Trace SOURCE is compared at EVAL
(same v1 adapter, same 8 tasks, cached-Gemini vs Qwen self-trace
conditioning — controlled, zero API cost). Trace TRAINING is Qwen-only
(v2, 17.3k). Reserve design if v2 underperforms while the Gemini-trace
gate passed: paired small-adapter ablation on the 1,353-instruction
intersection (Gemini traces cached; Qwen traces free) — identical
coverage, trace source the only variable. No new Gemini spend in any
branch.

## SAMPLED k8 verdict (both arms, 2026-07-20 evening)
SFT 27.7 vs frozen_bare 22.7 pooled; paired +5.0+/-2.2 (n=192) — pooled
win replicates greedy's +5.3. NOT every axis: natives SFT 48.2 vs 27.2
(+21, dominant, wins all 4 + wheel); quartet SFT 7.2 vs 18.1 (frozen
wins ALL 3 OOV-vocab tasks: ramekin -26.6, coke-plate -11.5, keyboard
-6.8). Best-of-8 oracles pooled SFT 42.7 vs 36.2, but per-task quartet
oracles: frozen ramekin 70.8/coke-plate 41.7/keyboard 22.9 vs SFT
54.2/8.3/8.3 => ramekin is a MODE ARTIFACT (SFT draws contain "white
bowl" at 54.2; greedy just picked badly) while keyboard and coke-plate
are DISTRIBUTION-DEEP (SFT's best of 8 still ~8) — vocabulary
destruction, not decoding luck. Temperature tax symmetric (-3.4/-3.1),
unlike the RL arms. Sharpens the v2 case: traces target exactly the
distribution-deep pair; gate leg (gemini-trace arms) rolling now.

## LADDER COMPLETE (greedy x12, current suite, protocol-clean, 2026-07-20 night)
frozen_bare 25.8 < SFT-v1 31.1 < v6 RL 37.2-39.2 < FROZEN_TRACE 40.5.
frozen_trace (frozen Qwen + cached Gemini trace + image, greedy) tops the
ladder and reproduces the old cross-suite 40.2 anchor on the current suite
(40.5). Stratified: natives 49.5 (SFT-v1 still wins natives, 56.3, +6.8);
QUARTET 31.5 (ramekin 63.5 / keyboard 22.9 / coke-plate 27.4 — trace
conditioning rescues OOV, concept PROVEN). Two sharpened conclusions:
(1) v6 RL trained FROM this conditioning and landed BELOW it (37-39 vs
40.5) — RL was net-negative vs its own frozen baseline, hardening the v6
postmortem; (2) v2's target decomposes cleanly: SFT-v1's natives edge +
trace-carried OOV rescue. GATE LOGIC FIXED: concept gate now keys on
frozen_trace (PASS, 31.5>=15.8) — sft_trace failing to READ traces (v1
trained traceless) would be evidence FOR v2, not against; sft_trace stays
as a diagnostic arm. Gate redeployed: waits for parallel tracegen
TRACES-DONE (no double-run), then best_val smoke, then v2 training.

## Arm naming convention (user, 2026-07-20 night)
Reporting names (artifact/arm strings unchanged until the queue drains —
waiters key on the short names):
  frozen_bare        = frozen Qwen, bare wrapper, no trace
  sft                = SFT-v1 adapter, bare wrapper, no trace
  frozen_trace       -> FROZEN_GEMINI_TRACE (frozen + cached Gemini trace + image)
  sft_trace          -> SFT_GEMINI_TRACE (v1 adapter under same conditioning; diagnostic)
  frozen_selftrace   = FROZEN_QWEN_TRACE (frozen + Qwen self-trace)
  sft_selftrace      = SFT_QWEN_TRACE (v1 adapter + Qwen self-trace)
  v2                 = SFT retrained WITH Qwen traces (native trace format)
Val suite unchanged: 8 SIMPLER tasks = 4 CoVer natives + keeper quartet
(coke-plate/carrot-keyboard/coke-ramekin/carrot-wheel _clean, registry-v4
prescreen survivors). natives-vs-quartet = stratification, not a suite
change; it happens to align with in-vocab vs OOV receptacles.

## Critic amendments to the next-phase plan (panel + adversarial critic, 2026-07-20 night)
1. PICKER EXECUTABILITY: grip/logit are a*-derived — the 25/75 picker cannot
   score quartet/sealed tasks at deployment. verifier_features.py has a
   documented DEMO-FREE 'deploy' mode; the $0 desk check (deploy-mode
   features as picker on banked sam8 tables, stratified, greedy included in
   the candidate set) runs BEFORE any deployment selection code.
2. DPO DEMOTED: current preference capital is ~8-10 CI-clear pairs from 70
   phrases — too thin. Confirmed lexical facts fold into the v2/v3 SFT
   corpus as targeted supervised pairs instead; preference harvesting waits
   for minted tasks (~30 unused source x target combos).
3. GENERALIZATION: confirmed preferences are pi0-idiolect lexical facts
   about ~8 objects; transfer to the sealed set requires the META-rule
   (concrete familiar nouns > category nouns; don't force OOV nouns), which
   minted-task harvests can test pre-one-shot.
4. V7 RL: demoted to DEAD-UNLESS-GATE — if the 25/75 argmax does not beat
   greedy on-policy in the desk check, RL against that reward cannot beat
   SFT (same optimum, $60 vs $5). If ever run: from-scratch per the
   pre-registered spec, or a disclosed 2-arm init ablation.
5. V2 READ PRE-REGISTERED (before any arm lands): PRIMARY = v2 vs
   frozen_selftrace, CRN-paired (weights-only, trace source controlled) —
   NOT vs frozen_gemini_trace 40.5 (different trace source). SECONDARY =
   v2 natives vs SFT-v1 56.3; v2 quartet vs frozen_gemini_trace 31.5
   (cross-source context). TRACE-HEALTH GATE (critic): PASSED — 8,500/17.3k
   audited: 0.0% template echo, 0.1% repetition loops, 0.0% missing
   referent sections, median 383 chars; retrain authorized.
6. SEALED ONE-SHOT AUTHORSHIP: test-task ERTs must NOT be Qwen-authored
   (SFT/v2 would face their own training corruption style on the decisive
   eval). Budget ONE small eval-time Gemini batch (~$5-10, flash,
   thinking_budget=0) at one-shot time for test ERTs + gemini-trace arms —
   held-out authorship is the hygiene that made every SFT claim honest.
   Requires user sign-off on the spend at endgame time.

## sft_gemini_trace diagnostic landed + v2 training launched (2026-07-21 ~06:15)
sft_gemini_trace (v1 adapter under Gemini-trace+image conditioning it never
trained with): pooled 33.8 | natives 50.0 | quartet 17.6. Reading: v1 can
PARTIALLY read traces — quartet +11.8 over its bare 5.8 (coke-plate 30.9 vs
5.9, ramekin 28.5 vs 3.8 — trace nouns get through; keyboard 1.4 — vocab
forcing still wins there) — but pays a NATIVES TAX of −6.3 (50.0 vs 56.3):
the off-distribution conditioning degrades its canonicalization. Net +2.7
pooled. Exactly the train/deploy-mismatch prediction; strengthens the v2
thesis (train WITH traces natively → quartet rescue without the tax).
CHAIN REPAIR: the auto-fired v2 train OOM'd against the still-running
trace-arm rollout workers (3×6.76GB — bigger than planned) — V2-CHAIN-
FAILED-TRAIN; meanwhile sft_gemini_trace completed+banked and the
selftrace GENERATION completed (both arms' phrases written). Relaunched
serial on the freed GPU: v2 train (alone) -> selftrace rollout leg.
Trace-health gate re-confirmed on the full 17.3k set implicitly (8.5k
audit clean; generation finished without errors).

## NEXT-PHASE PLAN — panel proposals (recorded 2026-07-21 for future reference)
Source: 3-lens proposal panel + adversarial critic (workflow wf_c5b3cf82-efd;
full text in the session transcript). Critic amendments recorded in the
"Critic amendments" entry above; they modify items marked [amended].

LENS A — exploit the oracle gap, no retraining:
A1 VOCAB-ROUTER COMPOSITE ($0): pre-register a noun-coverage routing rule
   (in-vocab -> SFT bare; OOV -> trace arm); compute composite from banked
   CRN-paired parquets. Projects ~43.9 pooled vs ladder top 40.5. Risks:
   winner's-curse (pre-register rule text incl. threshold before computing;
   tire has 3 corpus hits); self-trace may undershoot cached-Gemini traces.
   First: results/analysis/vocab_router_composite.py, rerun when self-trace
   arms land. Default no-retrain finalist for the sealed test.
A2 RETRO-SELECTION EXAMS ON BANKED POOLS ($0-8) [amended: must use the
   demo-free 'deploy' feature mode — grip/logit are a*-derived and cannot
   score quartet/sealed tasks]: (a) picker-transfer check on SFT/frozen
   pools (0.62 capture was validated on v6-era pools only); (b) a*-free
   picker search: decode-spread, text-MBR centrality (predicted to fix the
   ramekin mode artifact), OOV-noun-preservation, frozen log-prob. Gate for
   everything downstream incl. v7.
A3 UNION-POOL BEST-OF-N ARM (~$10-20, one x12 leg): pool ALL banked phrases
   per task (incl. v6 bank's ramekin->"white bowl" 64.2), pick offline with
   the certified picker (NEVER with banked success labels — oracle leakage),
   byte-audit picks, roll once. Expected 42-46 pooled; its residual vs
   oracle = cleanest measure of what only execution-grounded learning buys.

LENS B — beyond the training distribution:
B1 ROLLOUT-RACED PHRASEBOOK (~$5): per-task successive-halving bandit over
   structured families (referent alternates from trace replacement lists,
   cube/block-class noun swaps, register variants) vs real CRN rollouts.
   Calibration cell FIRST (stack + ramekin must reproduce the two known
   preferences for ~$1). Output: measured better-than-GT headroom + test of
   the lexical meta-rule (concrete familiar nouns > category nouns).
   Winners are val-suite capital, not headline claims; confirm top-1 on
   held-out layouts 18-23.
B2 EXPERT ITERATION (ReST-style, ~$70-120, GATED): sample from v2, verify
   vs the ANALYTIC sim success rule (not a learned detector — v5/v6 Goodhart
   channel structurally absent), distill winners with margin weighting + v1
   replay. Only mechanism whose ceiling is pi0's competence, not the Bridge
   distribution. Fund only if the $10 yield probe (32 contexts x 8 samples
   x 2 reps) shows >=25% contexts with a non-GT-register winner beating the
   nominal anchor. Task space: mint new source x receptacle combos (~25-30
   untouched); mandatory seen/held-out layout split.
B3 REFERENCE-FREE BEHAVIORAL PICKER ($0 first step): mine banked decode
   arrays for a*-free features (cross-seed action variance, gripper
   decisiveness), benchmark on the frozen yardsticks (68-pair sign exam,
   selection cells). Selection-time use ONLY — never promote to a training
   reward (re-opens the r=-0.93 failure). Merges with A2's search.

LENS C — scientific endgame:
C1 SELECT-8 DEPLOYMENT ARM: reward-as-selector (the workstream's positive
   result: reward fails as RL objective, works as picker). Offline replay
   first on banked pools; stratified pre-registered read; confirmation
   rollout only after the natives gate (>=1/3 capture) passes. [amended:
   deploy-mode features required beyond natives]
C2 ICL KNOWLEDGE-VS-WEIGHTS CONTROL: run the already-built
   sft17k_icl_baseline greedy x12; pre-registered interpretation rule
   (ICL >= SFT-2pp natives => knowledge; <= frozen+5 => weights; between =>
   decomposition). Elevates the natives record to a mechanism claim.
C3 SEALED ONE-SHOT PROTOCOL: fire once, only after v2 + select-8 verdicts;
   <=6 arms (anchors, frozen_bare, frozen_qwen_trace, SFT-v1, v2, select-8);
   in-vocab/OOV stratification drawn EX ANTE by an automated vocabulary
   audit of sealed-task nouns (falsifiable prediction, not hindsight);
   written predictions filed before FINAL_EVAL=1; full dry-run on a val task
   with byte-level echo/trace-alignment audits (v5 trace-misalignment is the
   cautionary precedent). [amended: test ERTs must NOT be Qwen-authored —
   one small Gemini batch ~$5-10 at eval time, user sign-off required]

MERGED PRIORITY (with critic amendments): (0) in flight: v2 + self-trace
arms + pre-registered v2 read; (1) A2/B3 deploy-mode desk check ($0) — also
the v7 gate; (2) A1 router ($0); (3) A3 union-pool arm (one leg); (4) B1
phrasebook racing (~$5, calibration first); (5) C2 ICL control; (6) B2
expert iteration iff yield probe passes; (7) C3 sealed one-shot. v7 RL:
dead-unless-gate per critic; if ever run, from-scratch spec or disclosed
2-arm init ablation. DPO on current data: demoted (8-10 CI-clear pairs is
too thin); lexical facts fold into v3 SFT corpus instead.

## ICL leg queued + benign-input eval axis (user, 2026-07-21)
ICL arm (panel C2, built as sft17k_icl_baseline.py) queued behind the
selftrace2 leg: trace-FREE form (clean vs the bare ladder), pre-registered
read per C2 (ICL >= SFT-2pp natives => knowledge; <= frozen_bare+5 =>
weights; between => decomposition). Retrieval inventory note (user Q):
targets are the 17,297 Bridge GTs; the 235,764 pi0-seen benign paraphrases
participate as retrieval bridges and demo pairs — and SFT-v1/v2 training
likewise used BOTH (hostile styles 0-2 + benign style 3).
BENIGN-INPUT EVAL AXIS (user, gated on v2's verdict): all rollout evals to
date use HOSTILE (ERT) inputs + nominal anchors; pi0's trained benign-
paraphrase distribution has never been an eval INPUT. Planned leg: feed
benign paraphrases (OXE dict entries where the task instruction is a dict
key — spoon-towel and carrot-plate have real clusters — light paraphrases
elsewhere) through each finalist arm; tests the NO-HARM property (repair
must not degrade already-benign inputs; the passthrough/echo question in
its benign form). Build after the v2 verdict.

## frozen_selftrace landed (2026-07-21 18:43): the Gemini trace-quality premium
frozen_selftrace (frozen + Qwen self-trace): pooled 34.5 | natives 46.7 |
quartet 22.3. vs frozen_gemini_trace 40.5/49.5/31.5 => GEMINI PREMIUM =
6.0pp pooled, concentrated in the quartet (-9.2) and three tasks: ramekin
-17.4, keyboard -15.6, spoon -12.5. Qwen self-traces carry the concept
(quartet 22.3 still ~4x SFT-bare's 5.8) but lose real ground on trace
quality. v2's pre-registered primary bar = 34.5 (paired). Ladder update:
SFT-v1 31.1 < frozen_selftrace 34.5 < v6 RL 37.2-39.2 < frozen_gemini 40.5.

## B4 (user proposal, 2026-07-21): explicit rule distillation via frontier reasoning
Claude analyzes (i) the 17.3k GT corpus structure + 235k paraphrases and
(ii) the banked phrase->success evidence (70+ measured cells, 68 powered
contrast pairs, sampled pools) and authors an EXPLICIT pi0-phrasing rule
set. RETRODICTION GATE ($0, before any rollout): the rule set must predict
the measured winner on the 68 powered pairs at ~>=60/68 (context:
production reward 51/68, exam blend 66/68). Application forms: (a) rules
as prescriptive system prompt for frozen Qwen + self-trace (deployable);
(b) Claude-authored phrases directly (ceiling). Testing: val-8 =
development only (rules derive from its evidence); honest test = minted
tasks, then the rules arm enters the sealed one-shot as a finalist per C3.
Caution: rules cannot resolve appearance descriptions without grounding —
deployable form is rules + self-trace, not rules alone. Generalizes by
structure (the answer to the task-space concern); also a writeup artifact
regardless of outcome.

## B4 amended (user, 2026-07-21): no retrodiction gate; structural generalization
Retrodiction gate DROPPED (circular: rules derived from the pairs cannot
be validated on them; a held-out split would halve the learning signal).
ALL evidence feeds the reasoning (corpus, paraphrases, 68 pairs, sampled
pools, trace forensics). Generalization guard is STRUCTURAL: rules must be
task-agnostic and executable — functions of (input, trace, corpus stats),
banned from naming val objects (rule-text inspectable before any spend).
Evaluation: ONCE on minted tasks (Qwen-authored ERTs acceptable for the
development read, flagged), then the rules arm joins the sealed one-shot
as a finalist. Deployable form confirmed: rules + self-trace.

## B4 data manifest (pinned 2026-07-21)
Inputs to the rule-distillation reasoning — ALL of: (1) 17,297 GTs +
235,764 OXE paraphrases (distribution structure, unlabeled); (2) the 68
CI-separated contrast pairs (top evidentiary tier); (3) every greedy x12
(task, phrase, success) cell at n=288; (4) the sampled pools per-draw WITH
success — INCLUDING all best-of-8 oracle winners (user-confirmed in);
(5) mechanism case families (ramekin/keyboard/cube-block numbers) + sample
self-traces. Statistical discipline encoded in the analysis: oracle
winners are winner's-cursed (24-ep SE ~10pp) — evidence tiering is
CI-pairs > n=288 cells > pooled draw patterns > single oracle ranks;
oracle phrases inform rules via STRUCTURE, not raw rank.

## V2 VERDICT (x12 landed 2026-07-21 21:26)
sft_v2: pooled 37.1 | natives 57.4 | quartet 16.8.
PRIMARY (pre-registered, paired vs frozen_selftrace 34.5): +2.6 +/- 2.1
(n=192) — a LEAN, not a certified win (1.2 sigma).
SECONDARY: natives 57.4 vs SFT-v1 56.3 — the record HELD under trace
conditioning (train/deploy match killed the -6.3 mismatch tax, as
designed); quartet 16.8 vs frozen_gemini 31.5 / frozen_selftrace 22.3 —
rescue only PARTIAL (+11.0 over v1's 5.8 but short of what traces offer).
Per-task: spoon +29.9 (63.9), keyboard +9.4 (16.7, the carried noun
paying off), wheel +9.4 (15.6 = task record), eggplant 94.4 (ties
record); ramekin -21.5 (24.7 — the "white object that looks like a cup"
hesitation priced) and coke-plate -19.4 (10.1) are the two residual
canonicalization overrides. Ladder: SFT-v1 31.1 < frozen_selftrace 34.5
< SFT-V2 37.1 ~ v6 RL s120 37.2 < s80 39.2 < frozen_gemini 40.5. v2 is
the strongest FULLY-SELF-CONTAINED arm (no Gemini anywhere). Mode-vs-
distribution question for ramekin/coke-plate goes to the queued sampled
face-off; sft_selftrace diagnostic rolling now.

## B4 prompt-ready file descriptions (user request, 2026-07-21)
Draft descriptions for the learning-task prompt, one per input file
(results/analysis/b4_rules_inputs/):

01_bridge_gt_instructions.txt — "The complete list of 17,297 unique
instructions from the robot policy's training data, one per line. This is
the phrasing distribution the policy was trained to follow. No labels."

02_oxe_paraphrases_SAMPLE60keys.csv — "A sample of 60 instruction
'clusters' from the policy's finetuning augmentation: each row pairs a
training instruction (gt) with one of its ~38 benign paraphrases that the
policy also saw during training. 235,764 such paraphrases exist in total."

03_contrast_pairs_139.json — "139 statistically confirmed preference
facts. Each record is two phrasings of the SAME task, both executed many
times in simulation; 'better' beat 'worse' by at least 6 percentage
points of rollout success with z>1.96 under conservative (layout-
clustered) errors, each phrase measured over >=200 episodes. These are
the highest-confidence facts about which phrasings the policy executes
better."

04_greedy_cells_all_arms.csv — "A chart of rollout successes for 8
simulated tasks. Each task is identified by its nominal instruction; each
row is one alternative phrasing of that task with its measured success
rate over at least 288 rollouts (up to 1,152 where measurements were
pooled). 73 phrasings total. Compare phrasings within a task to see what
wording the policy executes best."

05_sampled_pools_per_draw.csv — "Individual sampled phrasings (8 random
draws per task from various generator models) with their measured success
over 24-96 rollouts each; 435 rows. The best draw in each pool is flagged
(is_pool_oracle). CAUTION: per-row success is noisy (SE ~10pp at n=24)
and pool winners are selected on noise — use these rows for the STRUCTURE
of what wins, never for exact rankings."

06_mechanism_cases.md — "Curated case studies with measured numbers:
side-by-side phrasings of the same task showing large success differences
and the apparent cause (e.g. concrete noun vs vague category word;
keeping vs replacing an unfamiliar object name; short canonical form vs
long descriptive form)."

07_selftraces_val8.md — "What the deployed system actually receives at
test time alongside the hostile instruction: a short model-written scene
description plus a mapping from the instruction's object descriptions to
plain object names. Rules must be executable given ONLY the hostile
instruction, this trace, and the training corpus."

## v2 checkpoint probe verdict (quartet x12, 2026-07-21 23:20)
v2 best_val (step 600) quartet 15.0 vs final 16.8 — paired -1.7 +/- 2.0:
NO rescue; the earlier-checkpoint hypothesis is settled negative with
rollout evidence. Ramekin's "white cup" gained only +3.8 (28.5 vs 24.7;
"cup" is a weaker neighbor than v6's "bowl" at 64.2), and best_val's
LONGER coke-plate phrasing lost -8.3 (1.7 vs 10.1 — Case 6's length
penalty exactly). keyboard/wheel flat. final/ stays the v2 artifact;
the quartet gap is a phrasing-choice problem (bowl > cup > "cup-like
object"), i.e. selection/rules territory, not training-duration.

## SEALED TEST-SET PLAN — draft v1 (user directive 2026-07-21 night)
Rules arm goes STRAIGHT to the sealed one-shot (no val-8 rollout spend —
val evidence is its training data; minted-task step waived by user
decision). One firing, all finalists together, FINAL_EVAL=1.

ARMS (candidate list — FREEZES after the overnight legs land):
| # | arm                       | decoding | eps    | notes                          |
|---|---------------------------|----------|--------|--------------------------------|
| 1 | originals (nominal input) | greedy   |  4,320 | ceiling anchor                 |
| 2 | ERT passthrough           | greedy   |  4,320 | floor anchor                   |
| 3 | frozen_selftrace          | greedy   |  4,320 | best no-training self-contained|
| 4 | sft_v2 (native trace)     | greedy   |  4,320 | trained finalist               |
| 5 | RULES + selftrace (Qwen)  | greedy   |  4,320 | b4_phrasing_rules.md as system |
| 6 | RULES + selftrace (Qwen)  | k=8 x1   |  2,880 | distribution + bo8 oracle      |
(optional 7: sft_v2 sampled k=8, +2,880 — decide at freeze)
15 tasks (12 sealed natives + CoVer trio) x 24 CRN layouts; total ~24.5k
eps ~ 2.5 pod-days single pod (clone a second pod: ~1.3 days, ~$1 setup).

ASSETS NEEDED PER SEALED TASK (authored AT eval time, none exist yet):
- nominal instruction (env registry — exists)
- ERT instruction: ONE Gemini batch (~$5-10, flash, thinking_budget=0) —
  held-out authorship per critic amendment; USER SIGN-OFF REQUIRED
- initial-frame render + Qwen self-trace (existing machinery)
- vocabulary audit: automated noun-coverage of each sealed task vs the
  17,297+235k corpus, computed BEFORE any success number is read (the
  in-vocab/OOV stratification is a pre-registered prediction)

PRE-REGISTRATION CHECKLIST (all BEFORE any test asset is generated):
[ ] arm list frozen (after ICL + sampled face-off land, ~morning)
[ ] written predictions filed (per-arm, per-stratum)
[ ] rules-arm generation SMOKE on val-8 (generation only, zero rollouts —
    verifies Qwen can follow the prescriptive prompt; output eyeballed)
[ ] dry-run of the full harness on ONE val task, FINAL_EVAL unset, with
    byte-level echo + trace-alignment audits (v5 quarantine precedent)
[ ] Gemini batch sign-off + key rotation check
[ ] second-pod decision (clone pod4 per the documented ~45-60min path)

## ICL arm dropped (user, 2026-07-22): superseded by the B4 rules arm
The C2 knowledge-vs-weights control is waived — the rules arm IS the
knowledge-based approach going to the sealed test; its rollout budget goes
there instead. Face-off waiter rekeyed to the sft_selftrace marker
(pulls the three-arm sampled face-off ~3h earlier, ~08:15).

## sft_selftrace diagnostic (2026-07-22 02:02): v1+qwen-trace MATCHES v2
sft_selftrace (v1 adapter, phase4 image+selftrace conditioning): pooled
36.6 | natives 56.7 | quartet 16.6 — vs sft_v2 native 37.1/57.4/16.8:
all deltas within noise. TWO consequences: (1) the Gemini-trace mismatch
tax (-6.3 natives) does NOT appear with Qwen self-traces (56.7 vs 50.0)
— the tax was trace-content/length-specific, not trace-per-se; (2) v2's
traced RETRAINING bought ~nothing measurable over v1-with-traces-at-
inference on this suite. v2 retains the deployment edge (text-only native
format, no image in the rephrase prompt; v1+selftrace needs phase4
image conditioning) — but the writeup must state the marginal honestly.
Freeze implication: sft_v2 stays the trained finalist on deployment
profile; v1+selftrace is its no-retrain twin.

## Sealed test-set plan v2 (user matrix, amended 2026-07-22)
| # | trace  | rephraser        | decoding | eps   | question answered              |
|---|--------|------------------|----------|-------|--------------------------------|
| 1 | —      | originals        | greedy   | 4,320 | ceiling anchor                 |
| 2 | —      | passthrough      | greedy   | 4,320 | floor + no-trace reference     |
| 3 | gemini | frozen Qwen      | greedy   | 4,320 | trace premium (with #4)        |
| 4 | qwen   | frozen Qwen      | greedy   | 4,320 | deployable no-training arm     |
| 5 | gemini | v6 RL (native)   | greedy   | 4,320 | RL vs its own conditioning     |
| 6 | qwen   | sft_v2 (native)  | greedy   | 4,320 | trained finalist               |
| 7 | gemini | RULES -> Qwen    | greedy   | 4,320 | rules under best traces        |
| 8 | qwen   | RULES -> Qwen    | greedy   | 4,320 | rules, fully self-contained    |
| 9 | gemini | RULES -> Gemini  | greedy   | 4,320 | rules ceiling (executor probe) |
CUT: qwen x RL (redundant; v6 not a deployment candidate). ABSENT BY
DESIGN: gemini x sft_v2 (measured mismatch config, not deployable).
Sampled k=8 for 1-2 headline arms decided at freeze (post face-off).
Total ~38.9k greedy eps ~ 3.5 pod-days serial / ~1.8 with cloned pod.
All prior checklist items stand (vocab audit ex-ante, predictions filed,
dry-run, Gemini batch sign-off — now covers ERTs + eval-time traces +
rules-execution calls).

## Sealed plan v2 addendum: #10 oracle arm (user, 2026-07-22)
| 10 | — | ORACLE best-of-16 nominal rephrases | 2-stage | ~10,080 | headroom above nominal |
Design: frozen Qwen samples k=16 rephrases OF THE NOMINAL (temp 1.0, bare
prompt, no trace); stage 1 screens all 16 at x1 (5,760 eps); stage 2
confirms each task's winner at x12 (4,320 eps) — the CONFIRMED number is
the reported oracle (winner's-curse honest). Reads: #10 vs #1 = better-
than-nominal headroom out-of-sample (the cube>block question); #10 vs
best of #4-9 = residual selection/search headroom. Measurement arm, not
a competitor. Bonus: the stage-1 pool + successes = free out-of-sample
selection-exam material for the offline pickers. New total ~49k eps
~ 4.5 pod-days serial / ~2.3 with second pod.

## Checklist update (2026-07-22): Gemini key rotated + batch signed off
User rotated the Gemini key and provisioned it (file-transfer protocol,
never in logs). Sign-off for the ~42-call sealed batch: GRANTED. Batch
still fires only at eval time per pre-registration. Post-batch key
rotation recommended once the one-shot completes.

## Rules-arm generation smoke: PASS (2026-07-22, checklist item 3)
Qwen follows the B4 rules with high fidelity on all 8 val inputs — and
makes the winning move on ALL THREE tasks the trained arms kept failing:
ramekin -> "pick up the soda can and place it upright inside the ceramic
bowl" (the exact v6-64.2% shape: Rule 7 upright-inside + Rule 1 trace
noun); coke-plate -> "put the soda can on the plate" (deixis dropped per
Rule 8/Case 6 — the move v2 never made); keyboard -> "put the carrot on
the keyboard" (noun kept). Bonus: stack -> "put the teal-green cube on
the yellow cube" — T5 implemented the cube>block preference from the
trace's own scene noun without the pair ever being named. Eggplant maps
to "yellow dish rack" (trace's visual name; historically a 92.7% cell —
watch it). Wheel takes the trace's "tire" (record-holder said "wheel" —
T5 tension, trace wins by design).

## Rules smoke — inferential status caveat (user, 2026-07-22)
The smoke's val-8 "winning moves" are DEVELOPMENT-FIT, not generalization:
the rules were authored from val's own answer key (rule SELECTION carries
val fit even where rule FORM is task-agnostic). What the smoke proves is
executor compliance only (9B follows the 315-line prompt faithfully) —
the performance claim belongs exclusively to the sealed test. Predictions
file must flag GreenCubeOnPlate as the one sealed task where a val
lexical fact (cube>block) legitimately reaches the arm via the trace
mechanism.

## Three-arm sampled face-off COMPLETE (2026-07-22 07:25) — val program closed
sampled k8: frozen_selftrace 34.8 (zero temperature tax) > sft_v2 32.0
(-5.1 tax) > v2_best_val 29.8. CHECKPOINT QUESTION CLOSED BOTH WAYS:
best_val loses on mode (greedy quartet 15.0 vs 16.8) AND distribution
(sampled quartet 11.5 vs 15.0; its ramekin pool still never commits to
"bowl" — best draw 45.8 'picked and transfer the can into the cup').
bo8 oracles 44.5-47.6 for all three arms — far under the search's
emerging per-task numbers (ramekin r1 leader 72.2@n36). Freeze decisions
retro-validated (no sampled twins for v2 family). Val-side evidence
program is now COMPLETE; remaining work is search rounds + sealed
execution.

## SEARCH PHASE COMPLETE (2026-07-22 10:43) — the headroom law
CONFIRMED on held-out layouts 18-23, n=72/phrase, same-split contrasts:
- RAMEKIN: "place the red coke can inside the white bowl" 95.8% vs
  nominal 36.1% on the same split — +59.7pp, near-ceiling, task
  effectively SOLVED by phrasing alone. (Runner-up "into the white
  bowl" form 83.3.) Every arm's full-grid ramekin (24-64) is dwarfed.
- COKE-PLATE: search-best 84.7 vs nominal 83.3 — TIE. No better-than-
  nominal phrase in 48 candidates across 3 rounds.
THE LAW: better-than-nominal headroom is inversely proportional to
nominal quality — enormous where the nominal is bad (ramekin), absent
where it is good (coke-plate). Phrase optimization's entire value
concentrates on bad-nominal tasks.
Method verdict: LLM-guided search (3 rounds x 16 + confirmation, ~$8
rollouts, one night) found the 95.8 phrase; its structure is exactly the
rules' T-family prediction (single-clause place-frame, color+concrete
noun, inside-relation, no orientation word needed). Round-trip lessons
fed back: "set" and "please" wrongly banned (neutral), "upright"
unnecessary, adjective TYPE matters (white>>ceramic), lexeme ladders
(bowl>cup>>dish; coke-token worth 15-30pp; "the coke" >= "the coke can").
Cross-split level shifts (~20pp) mean ONLY within-split contrasts are
valid — all claims above are same-split.
V7 IMPLICATION: a prize exists (bad-nominal tasks) but search+distill
just demonstrated it's reachable for ~$8/task without RL; v7's reward
also cannot represent rollout-defined winners by construction. The v7
gate question is now "can RL find what search finds, cheaper or more
generally?" — prior strongly no. Search scoreboards = fresh CI-grade
pairs for rules-v2.

## FREEZE AMENDMENT (user, 2026-07-22 ~14:40, PRE-MEASUREMENT): rules-v2 substitution
No sealed rollout has run; only assets exist (authored once, canonical,
committed). User directive: complete the 8-task search first, feed its
lessons into a RULES-V2 (fresh-context re-run of the B4 runner with the
search scoreboards + corrected case evidence added to the inputs folder),
then evaluate rules-v2 on the sealed set. Amendment is legitimate because
it precedes any sealed measurement; documented here for the record.
Arms #7/#8/#9 use rules-v2. OPTIONAL +1 leg (decide at firing): rules-v1
at #8's config alongside v2 — directly prices the search->rules feedback
loop (~$3). The staged rules leg was INTERCEPTED post-asset-commit;
pod1 search resumed. Known search->rules corrections queued for the v2
inputs: "set"/"please" wrongly banned (neutral); "upright" unnecessary;
adjective TYPE (white>>ceramic); lexeme ladders (bowl>cup>>dish, coke
token +15-30pp, "the coke" >= "the coke can"); single-clause preference
confirmed; cross-split level-shift discipline.

## PRE-ROLLOUT FINDING (2026-07-22 21:40): rules quality is EXECUTOR-BOUND
Audit of generated sealed phrases: rules_v2 under the QWEN executor
reproduces v1's failure modes on ~5/12 tasks ("purple object", "block on
the dish", "fastener", coke->"soda can") despite the v2 rules explicitly
repairing each — while the GEMINI executor, given the IDENTICAL v2 rules
file, executes them correctly ("put the coke can on the keyboard", "put
the carrot in the white bowl", no category words). The 9B cannot follow
the longer v2 rule set's override logic (brand-token precedence, fallback
ladder) in one pass; the frontier model can. Echoes CoVer's boot-time-
frontier design choice. Consequence: pod2 leg order updated to measure
rules_gemini (#9) tonight alongside rules_v2_selftrace — the #8-vs-#9
gap is now a primary read (executor premium), pre-registered here BEFORE
any sealed rollout of either arm.

## FREEZE AMENDMENT 2 (user, 2026-07-22 ~22:30, PRE-MEASUREMENT): rules-v3 substitution
The in-flight rules_v2_selftrace leg was killed mid-roll with NO result
committed or read; no sealed rollout of ANY rules arm exists. Rationale:
the purification boards showed rules-v2's evidence base was partially
confounded (dish/ceramic/category bans overstated by stacked-edit frames;
ramekin-true-name toxicity −41.7 is what survives). User directive: finish
the certification suite (xcert + purification), author RULES-V3 on the
certified evidence, then run all rules arms on sealed. Rules-independent
arms (sft_v2, anchors, frozen x2, v6, oracle) proceed tonight — their
phrases are frozen artifacts unaffected by rule revisions.

## 2026-07-22 late — grip⊕full-L2 reward exam addendum (POST-HOC EXPLORATORY; frozen bakeoff unchanged)

User question: did any examined reward blend gripper error with full-action-chunk
L2? Answer: no — norm_l2 was only a verifier input feature + norm control. Addendum
(results/analysis/reward_exam_l2_addendum.py, same frozen features/pairs/exams):

| candidate | signs | max top-1 regret | pooled ρ |
|---|---|---|---|
| C4b (frozen winner) | 66/68 | 2.1 | 0.493 |
| GRIP_pure | 67/68 | 2.1 | 0.528 |
| **L2_pure** | **36/68 (coin flip)** | **41.0** | **−0.024** |
| GL_75_25 (25% L2) | 65/68 | 11.8 | 0.451 |
| GL_50_50 | 52/68 | 12.6 | 0.254 |
| GL_25_75 | 39/68 | 36.1 | 0.038 |
| ZL_25_75 (C4b w/ L2 in grip's seat) | 38/68 | 41.0 | 0.029 |

**Full-chunk L2 carries ~zero phrase-quality signal** (ρ=−0.02) and ANY admixture
degrades grip (even 25% L2 costs 2 signs and 10pp regret on stack). Failure mode:
L2 prefers verbose "exactly in the middle of the towel" phrasings on spoon (arm
path matches a* while grip catches the real difference) — the same verbose family
v5's echo-Goodhart chased. Implications: (1) v7-w0 stays PURE grip-rank (adding L2
would only dilute); (2) prediction for the resurrected phase2_v2_l2 arm's ladder
eval: noise-guided RL — drift, not targeted gain (filed pre-eval).

## 2026-07-22 late — purification consolidated into PHRASE-SEARCH.md

All five *_pure1 boards folded in (per-scene purified tables + revised
cleanest-evidence table + cross-cutting rewrite). Headlines: ramekin TRUE-NAME
−41.7 is the largest clean single edit; dish/ceramic/cloth cliffs were confound
artifacts; category words fine (basin +16.7 over bowl); teal-green −33.4 = visual
fidelity must stay inside corpus vocabulary; clause verbs place≫put +19.4,
pick-up≫take +22.2; set≥put (+13.9 spoon); stack verb −19.4; onto harmless +2.8.
xcert/xcert2 certification (n=144/phrase/task) supersedes on overlap when it lands.

## 2026-07-22 late — pair-measurement standard upgraded to all-24-layouts

Per user: the 18/6 layout split only serves ACTIVE SEARCH (selection needs
untouched layouts). Pre-registered minimal pairs get no benefit — so all
future purification/minimal-pair boards run all 24 layouts ×6 (n=144, paired
SE ≈ 5.9pp), i.e. xcert mode, via file naming xcert_pure_*. The three round-2
boards (wheel/carrotplate/eggplant) upgraded before first run. Completed
n=36 pure1 boards stand; after xcert lands, any pure1-unique cells worth
certifying get topped up to 24×6. Also fixed search_worker.sh empty-shard
bug (2-phrase boards + NW=3 → KeyError 'task' → whole board failed; now
shards = min(NW, n_phrases)).

## 2026-07-23 — v7 GREENLIT (user: "one last shot"): C4b RL, v6 recipe + 50% phrase dropout

User spec: redo RL with C4b; "50/25/25" mix (= v6's 50 hostile-ERT / 25 nominal /
25 benign, unchanged); phrase dropout 0.5 (v6: 0.3333) to force trace usage;
train with the cached Gemini teacher traces (cover35, same as v6); val = reward
curve on the ERT val set (rephrases_val_0b, every 20 steps) + sampled rollout
probes on the val-8 task contexts (every 25 steps) — v6-identical instrumentation.
Arm B (on-policy GRPO), from scratch in phase2_v7. Harness pre-existed
(run_arm_v7.sh: --reward-blend c4b --blend-w 0.25; score-server grip sidecar;
within-group rank blend); only delta applied: --input-dropout 0.5.

Queue: launches on the first pod freed after the certification queue drains
(pod1 or pod3, est. ~04:30); sealed legs on pod2 unaffected.

Pre-registered expectations: unlike v5/v6 (reward climbed while probes diverged
— fine-axis pathology), C4b's exam profile (66/68 signs, 2.1pp max regret,
rho 0.49) predicts ERT-val reward and probe success should move TOGETHER.
Watch-fors: (a) trace-copy degeneracy amplified by 0.5 dropout — acceptable if
probes hold (noun-copying is the desired mechanism); (b) echo-style verbose
drift — C4b carries no echo penalty, but the L2-addendum showed grip (unlike
L2) does not prefer the verbose family; (c) KL runaway (kl-abort 1.2 armed).

## 2026-07-23 — v7 reward-noise analysis (user idea): frames >> decode draws; spec now F=8, k=4

User asked whether more pi0 decode samples would sharpen the reward (flow
sampling is cheap-ish). Desk analysis on the stored exam features (per-draw
grip_err, CRN pairing, 400 adjacent-rank candidate pairs across contexts):
- Frame-to-frame disagreement carries ~62% of the reward-difference variance;
  decode-draw noise ~38% (already suppressed by common-random-numbers pairing).
- Adjacent-pair "coin-flip zone" (|gap| < 1 SE): F=4,k=4 (v6 prod): 89%.
  Doubling draws (k=8): 86% — nearly nothing. Doubling frames (F=8): 62%.
  Both: 52%.
- The user's own follow-up ("roughly equivalent to using more frames") is the
  right frame: only frames touch the dominant component (different t, a*).
- Also noted: the bakeoff's 67/68 grip signs averaged over ~9 episodes/task on
  top of 4 frames; training rewards get no episode averaging — per-context
  signal is far noisier than the exam headline suggests, which is why F matters.

DECISION: v7 trains with REWARD_FRAMES=8, k_decode stays 4 (verifier feature
slots are calibrated to 4; frames dominate anyway). Cost ≈ 2x score time
(~+50% step time) — quality over step count for the one-shot run. Requires an
8-octile rebuild of contexts_train_multit / contexts_val_multit (builder
multi_t_contexts --points 8; running locally; shipped to the training pod
before launch; trainer linspace-subsamples so the 8-frame tables remain
compatible with any REWARD_FRAMES <= 8).

## 2026-07-23 — v7 frames finalized: REWARD_FRAMES=16 (user), k_decode=4

Extended decomposition: coin-flip zone 89% (F=4) -> 62% (F=8) -> 48% (F=12) ->
43% (F=16); achievable floor ~25% (F=64). F=16 captures ~72% of the possible
improvement. Cost per user's timing notes (scoring = 30% of v6 step):
step = 0.7 + 0.3*4 = 1.9x v6 — ~5-6 min/step, 130-160 steps by tomorrow
evening (v6 peaked s80). Launcher default now REWARD_FRAMES=16; 16-octile
context tables building locally (multi_t_contexts --points 16, train+val),
to be dropped in at data/contexts_train_multit.parquet / contexts_val_multit
.parquet on the training pod (linspace subsampling keeps them valid for any
F<=16). Final v7 deltas vs v6: C4b reward (w=0.25) · input-dropout 0.5 ·
REWARD_FRAMES=16.

## 2026-07-23 ~03:30 — user trade: certification-first; v7 deferred to ~07:30

Per user: pod1 stays on certification after its clause boards instead of
flipping to v7. Disjoint filters at next board boundary — pod1:
xcert2_wheel + xcert_{eggplant,keyboard,ramekin} + xcert_pure_carrotplate;
pod3: xcert_{cokeplate,spoon,stack,wheel} + xcert_pure_{wheel,eggplant}.
Certification ETA ~08:00 (was ~11:30); v7 launches on pod1 after its share
(~07:30; 16-frame tables already staged). Clause certification 6/8 in:
structure free except keyboard −11.8 (~3σ). Sealed leg 1 (sft_v2) landed
03:04: pooled 24.6, in-vocab carries / OOV craters per vocab-audit strata.

## 2026-07-23 ~05:00 — grip-screen proposal (user): corpus-scale phrase comparison via the 67/68 reward

Idea: use the exam-validated grip channel (rank01(-grip_err), CRN-paired) to
compare phrase edits on the TRAINING corpus itself — 2000 Bridge episodes x
16 frames (contexts_train_multit16), each with its GT instruction. Text-only
edit family per instruction (put→set, on→onto/on-top-of, clause-wrap,
telegram, please, category-noun map); paired per-context grip deltas.
PRE-REGISTERED VALIDATION: sign agreement vs the certified rollout cells
(clause family x8 + six-edit matrix); screen is DISCOVERY ONLY — rollout
certification stays ground truth. Pilot 200 episodes (~20 min GPU) slotted
after pod3's certification tail; full screen opportunistic behind sealed
legs. Script: src/phrase_rl/phrase_grip_screen.py (writing now).

## 2026-07-23 — grip-screen epistemic status CORRECTED (user challenge)

The earlier "pre-registered validation vs certified cells" framing overclaimed:
certified cells live on SIMPLER scenes, the screen on Bridge scenes — agreement
conflates proxy validity with domain shift, and novel screen effects have NO
possible rollout ground truth. Status: the screen is SUGGESTIVE-TIER evidence
(rules-inputs file-05 rung: structure, not proof). Proxy validity itself rests
on the exam (67/68, bridge-real features). Added: ANCHOR SCENES — Bridge
episodes that mirror val-8 tasks (carrot-plate, spoon-towel, eggplant-basket);
on these the screen's per-scene partial order is compared to the certified
rollout order (transfer check on matched tasks, explicitly narrow). Pilot
design: 12+ scenes, image-grounded phrase sets authored by Claude from the
actual frames, per-scene grip scoreboards = partial-order chains.

## 2026-07-23 11:43 — sealed legs 1-2 scoreboard (12 tasks x 24 layouts x 12)

| arm | pooled | note |
|---|---|---|
| originals (nominal) | 36.1 | 2 bad-nominal tasks (ramekin 14.6, coke-wheel 12.2 both BELOW their ERT) |
| passthrough (raw ERT) | 26.6 | hostile tax -9.5 vs originals |
| sft_v2 | 24.6 | BELOW passthrough pooled: big OOV losses (pepsi -33.6, coke-wheel -22.6 vs passthrough) vs in-vocab wins (+10..+16) |

Remaining: frozen_selftrace (running, ~16:15), frozen_gemini, v6_rl, oracle_pool,
then rules-v3 legs post-authoring.

## 2026-07-23 20:21 — sealed leg 5: frozen_gemini_trace 31.0

originals 36.1 > frozen_gemini 31.0 > passthrough 26.6 > sft_v2 24.6 >
frozen_selftrace 23.7. Gemini-trace premium +7.3 (val said +6.0 — transfers).
First rephraser above passthrough. Beats the NOMINAL on bad-nominal ramekin
(46.2 vs 14.6) = the val rescue mechanism on sealed. Still loses where it
rewrites good inputs (pepsi 19.1 vs passthrough 52.4 — rule-1 damage).
Next: v6_rl leg (pod2, ~01:00), then oracle_pool.

## 2026-07-23 ~21:15 — grip sweep approved: SPLIT DESIGN

User: Gemini cover35 traces (2000/2000 coverage verified) ground the edits;
split design chosen over flat F=16: leg A = 2000 eps x F=8 (pooled corpus
tables; between-episode variance dominates), leg B = 100 eps x F=16
(resolvable per-scene partial orders). 11 edit categories incl. trace-true
add_color and noun-family swaps (substitutability table). ~18h on pod3,
launching at wheel-RESULT. Output: suggestive-tier; nominations only.

## 2026-07-23 ~22:00 — RULES-V3 FROZEN (pre-sealed-rules-legs)

Derived via fan-out workflow (7 agents): 3 drafts (maximalist / minimalist /
template-first) -> adversarial executor red-team on the 7 historical failure
ERTs -> synthesis. Backbone = maximalist (best red-team grade); all surviving
patches applied. Sealed-blind by construction (agents restricted to
PHRASE-SEARCH.md + rules-inputs; sealed files banned). Reviewed by Claude
against the certified matrix: zero contradictions. File:
results/analysis/b4_phrasing_rules_v3.md (~16.5k chars; Gemini-primary).
Key mechanisms: Step-0 triage (pass-through vs rebuild, aggressiveness scales
with input badness), two-template bank with fixed clause verbs, noun ladder
(brand-carry + visual brand-guess exploiting wrong-brand-free; closed family
table; corpus-absence rename with color license; category fallback-to-guess),
adjective licenses, difficulty-gated clause count, geometry-driven relation.
Sealed legs next: rules_v3_gemini_executor (API), rules_v3 x Qwen x
{selftrace, gemini-trace}. Executor-premium read preserved (#8 vs #9).

## 2026-07-23 ~22:40 — rules-v3.1 amendment (PRE-MEASUREMENT): corpus vocabulary appendix

User: make the corpus-absence test executable. Appendix added (150 common
words with counts + 354 present-tier; <5 occurrences excluded as noise) with
the explicit guard PRESENCE NEVER OVERRIDES A BAN (frequency != outcome).
1.2c heuristic replaced by exact lookup. No rules-leg rollouts had run;
Gemini-executor phrases regenerated under v3.1. Amendment is sealed-blind
(appendix derives from training corpus only).

## 2026-07-24 ~00:45 — plans logged (user session)

**v8 sketch (rules-primed RL; PLANNED, not launched):** policy prompt =
rules-v3.1 + trace + ERT (identical conditioning to the rules_v3_selftrace
sealed arm = its step-0 baseline); inputs 100% hostile ERT; reward C4b F=16
unchanged; KL anchors to rules-primed behavior. Rationale: executor-bound
failures (soda/rubber-tire class) are exactly what the reward punishes → RL
as the bridge between rules and a 9B executor; RL learns the coin-flip
arbitrations rules can't encode. Decision point: current v7 val at step
60/80 — flat → swap; climbing → v8 queues.

**PRE-REGISTERED: sealed sampled twin for rules_v3 gemini executor** —
declared before any rules-leg result is read. k=8 Gemini samples per sealed
ERT (temp ~0.9), rolled 24 layouts x3 (n=72/phrase). Purpose: pipeline
variance + greedy-vs-pool position. Runs after the three greedy rules legs.

**CoVer 3-task port:** still deferred; revisit post-matrix iff CoVer release
ships zucchini/tennis assets.

**v7 rollout evals:** post-hoc on named checkpoints (best_val/s80/s120) per
v6 protocol; in-loop probes select which checkpoints earn rollout spend;
probe phrases may be grip-screened as a leading indicator.

## 2026-07-24 ~01:30 — PRE-REGISTERED: gemini_bare baseline arm (user)

Declared before ANY rules-leg result is read. Same executor/machinery as
rules_v3_gemini (model, temp 0.2, parsing, trace+ERT conditioning); prompt =
RULES-V3 preamble + output contract with ALL rules/tables/examples removed
(results/analysis/gemini_bare_baseline_prompt.md). Isolates rules-effect from
strong-executor-effect. Phrases banked (ph_sealed_gemini_bare.parquet):
baseline independently finds white-bowl rescue but deletes brands (red can,
blue can), emits teal block + tire. Leg queued on pod3 after rules leg 2.
Prediction: rules arm > bare on brand/OOV tasks by the certified magnitudes;
bare ~ frozen_gemini_trace elsewhere.

## 2026-07-24 ~02:15 — paired analysis: sft_v2 vs frozen_selftrace (sealed, existing data)

Paired over the shared 288-cell grid: +0.87pp pooled, cell-bootstrap CI
[-2.7,+4.2], task-cluster CI [-4.4,+6.5], cells 85/85/118 — pooled effect
indistinguishable from zero. BUT per-task: SFT +14..+18 on the in-vocab
stratum (ramekin +18.1, nut-plate +16.3, eggplant-sponge +16.0, cube-plate
+14.2) and -9..-16 on OOV (eggplant-keyboard -16.0, carrot-sponge -13.2,
nut-wheel -11.1, coke-keyboard -9.0). The val vocabulary-redistribution
story transfers to sealed: SFT != null effect; SFT = +-15pp redistribution
netting ~0 on this OOV-heavy set.

**QUEUED (post-hoc robustness, motivated by the greedy read):** sampled
twins of both arms — k=8 samples/task/arm, identical conditioning, rolled
24 layouts x3 (n=72/phrase, ~7h/arm) — measures whether the redistribution
is a mode-effect or a distribution-effect. Runs on pods freed after the
rules+bare legs and the pre-registered gemini-rules sampled twin.

## 2026-07-24 ~03:00 — sealed leg 6: v6_rl 30.1 (in-vocab 40.6 / OOV 22.6)

Statistically tied with frozen_gemini_trace (31.0/42.1/23.0), marginally
below on every stratum. CLEAN NEGATIVE: v6's RL polish (+~5 on val) buys
~0 on sealed — the pipeline's entire sealed value is the Gemini trace.
Remaining: oracle leg (pod2, ~05:00 UTC) -> leg3 chain; rules leg 1 grinding
on pod3 (~08:30 UTC at mfs pace) with L40S-pause option pending user.

## 2026-07-24 ~05:30 — SEALED RULES LEG 1 COMPLETE: rules_v3+Gemini 31.1

Pooled 31.1 (in-vocab 37.2 / OOV 26.7) = dead heat with frozen_gemini 31.0
but mirror-image strata: rules +3.7 OOV (0.6 below the ORIGINALS' OOV) and
-4.9 in-vocab. Mean recovery ratio 1.02x (vs 0.97x), 3 nominal-beating
rescues (coke-wheel 2.31x, ramekin 1.90x, eggplant-keyboard 1.49x). Brand
rule validated: pepsi 37.5 vs frozen 19.1. Autopsy cells: nut-plate 17.0
(adjective tax), carrot-ramekin 27.8 vs frozen 46.2 (uninstructed rewrite
beat the textbook one). No arm beats originals pooled; rules closed ~half
the hostile gap. gemini_bare twin pending (the rules-isolation read).

## 2026-07-24 ~06:30 — PRE-REGISTERED: two executor arms (user)

(1) rules_v3_gemini_pro: gemini-pro-latest, thinking_budget=16384, max_out
4000 (the measured flash arm ran thinking_budget=0, max 100 — now labeled
"flash, no reasoning" wherever reported). Phrases banked; differ from flash
on ~7/12 tasks with visibly tighter adjective-license discipline.
(2) rules_v3_claude_agent: Claude subagents as executor, RULES-V3.md + LIVE
corpus grep access (executes 1.2c/appendix as real lookups), one fresh agent
per task, sealed-reads banned. Phrases to be banked on workflow completion.
Both declared before any of their rollouts; combined leg queued after
gemini_bare on pod3.

## 2026-07-24 ~07:30 — sealed oracle pool: nominal-rephrasing headroom is ~ZERO

oracle_pool (16 Qwen rephrases of each NOMINAL, x24x1): mean-of-pool 29.5
(random rephrasing hurts, -6.6 vs originals), best-of-16 37.8 optimistic
(n=24/cell) vs originals 36.1 — the ERT-free rephrase ceiling is the
originals, +-noise. Sharp contrast with val (bo8 44.5-47.6 = real headroom).
Notably the pool NEVER finds the ramekin rescue (best 16.7 vs rules 27.8,
frozen_gemini 46.2): rescues need scene/trace knowledge, not phrase lottery.
Stage-2 confirm deferred (headline is null). Pod2 batch COMPLETE; leg3
(rules_v3 gemini-trace -> Qwen) auto-launching.

## 2026-07-24 ~11:00 — sealed leg: rules_v3 -> QWEN executor 31.5 (OOV CROWN 28.5)

Qwen+rules 31.5 (35.7/28.5) edges Gemini-flash+rules 31.1 (37.2/26.7):
executor premium INVERTED at this tier; per-task the two trade +-18 while
tying pooled. OOV 28.5 beats the ORIGINALS' 27.3 = first arm above nominal
on the OOV stratum. Anomaly for the record: Qwen's rule-violating "rubber
tire" phrase won nut-wheel (25.7 vs correct-phrase 8.0) — certified tire
penalty was carrot-sourced; task-conditionality strikes the closed family
table. Rules>no-rules at Qwen executor: +5.5 OOV. Pending: pro + agent +
bare executors.

## 2026-07-24 ~13:30 — sealed: deployable rules arm 27.3 (RULES TRIO COMPLETE)

rules_v3_selftrace (Qwen+Qwen, deployable) 27.3 (26.9/27.6) vs twin
frozen_selftrace 23.7 (29.7/19.4): +3.6 pooled, +8.2 OOV, -2.8 in-vocab —
the FLATTEST strata profile on the board (rules erase the vocab cliff).
eggplant_keyboard 46.9 = best score of ANY arm on any-arm basis for that
task (originals 25.3). Clears passthrough where its twin couldn't. Rules
gains now consistent across all 3 configs: OOV-led +5.5..+8.2, small
in-vocab tax, pooled +0.5..+3.6. Bare baseline rolling; executor pair next.

## 2026-07-24 ~19:30 — PRE-REGISTERED: gemini_pro_bare (user) + pair leg now 3 arms

Pro-with-reasoning NO-rules twin (same prompt as flash bare, gemini-pro-latest,
thinking 16384): the matched isolation for rules_v3_gemini_pro. Folded into
the chained executor-pair leg (36 phrases, one pass): pro+RULES,
claude_agent+RULES, pro-bare. Declared pre-rollout.

## 2026-07-24 ~21:00 — SEALED ORACLE SEARCH (user): headroom estimation protocol

Adaptive phrase search ON the sealed tasks to estimate the phrasing ceiling.
EXPLICIT FRAMING: outputs are ORACLE/ceiling numbers (test-set-adaptive),
never comparable to the blind arms; all blind arms measured+frozen before
this begins; rules-v3.1 frozen earlier. Round-1: 10 boards x16 seeded from
the 11 arms' measured per-task phrases + certified-law recombinations,
exploiting sealed-specific inversions (purple-object 46.9 on eggplant-kbd;
clause+into on ramekin 46.2; telegram dominance on plate tasks; pepsi's
verbose 52.4). Search grade 0-17x2 (n=36); leaders confirm at 24x6.
Skipped: 2 saturated tasks. Runs on pods 2/3 post-tournament.

## 2026-07-24 ~22:00 — EPISTEMIC REFRAME of the corpus sweep (user challenge)

The sweep's hypothesis space (11 edit categories, family maps) was chosen
post-val by Claude — so the sweep tests GENERALIZATION of the val-derived
laws across the corpus, NOT offline discoverability (question-selection is
val-polluted even though measurements are independent). The clean
discoverability test is a BLIND-DISCOVERER protocol: fresh agents with
corpus + traces + screen instrument only (PHRASE-SEARCH/rules/val results
firewalled) derive their own laws; grade against certification (overlap =
discoverable; misses = rollout-only knowledge; inventions = nominations).
Designed, not yet run — awaiting user go.

## 2026-07-24 ~23:00 — sampled-twin gate (user)

Row-15 sampled twin = k=8 temperature draws of the GEMINI rules pipeline
(pre-registered; referent updated to the pro executor when run) — NOT a
sampled Claude-agent arm (none declared). Per user: GATED on explicit
go-ahead; removed from the auto-queue. Row-16 twins (sft_v2 /
frozen_selftrace) remain queued as post-hoc robustness.

## 2026-07-24 ~23:10 — row-16 twins also gated (user)

sft_v2 / frozen_selftrace sampled twins now ALSO gated on explicit user
go-ahead. No sampled-twin work runs without it; freed pod capacity goes to
sealed-oracle search rounds/confirms.

## 2026-07-25 ~01:00 — sealed: flash-bare 27.5 -> RULES EFFECT +3.6 at matched executor

gemini_bare (flash, no rules) 27.5 vs flash+RULES 31.1: rules worth +3.6
pooled at identical executor/traces/temp; bare itself is only +0.9 over
passthrough. Agent leg auto-launched (pod3); pro pair mid-flight (pod2).

## 2026-07-25 ~02:00 — QUEUED (build+run gated on user go): sandboxed agent arm

rules_v3_claude_agent_sandboxed: headless Claude (claude -p / Agent SDK) in
a container whose filesystem holds ONLY RULES-V3.md + corpus txt + the task
input (clean-room mount, no repo, network = API only, access-audited).
Model: Fable 5 at the user's "Ultra" effort tier (map to harness xhigh at
build time; confirm with user). 12 invocations -> one 12-phrase sealed leg.
PRE-REGISTERED PREDICTION: phrases ~= the measured instruction-isolated
agent arm (its decision notes suggest the soft firewall held). Do NOT build
or run until explicit go-ahead.

## 2026-07-24 — Row 14 confirm round authored (oracle headroom, estimation stage)
Search phase complete: 10/10 boards (n=36/phrase, layouts 0-17x2). Confirm boards
authored per pre-registration: top-3 GENUINELY-NEW phrases per task (search winners
already measured in a sealed x12 leg are excluded — their layout-18-23 cells are
extracted from the leg parquets instead; rollouts are deterministic per
phrase+layout grid). Confirm = held-out layouts 18-23 x12 reps (n=72/phrase),
run by the search worker's *confirm* case on pod4. FINAL headroom estimates use
ONLY held-out cells: selection on 0-17, estimation on 18-23. Search-grade numbers
never enter the final estimate. Caveat carried: per-phrase held-out estimates are
unbiased, but taking the max across confirmed phrases retains a small winner's
curse; report per-phrase CIs alongside the max.

## 2026-07-25 — Rows 12/13 land: rules isolation REPLICATES at pro tier
pairA_pro_x12 (n=6912, both arms same leg, 24x12 all layouts):
  rules_v3_gemini_pro  31.6 pooled (38.2 in-vocab / 26.8 OOV)
  gemini_pro_bare      27.8 pooled (40.2 / 18.9)
Paired rules effect +3.79pp pooled, +7.9 OOV, -2.0 in-vocab — near-identical to
the flash pair (+3.6 / +8.5 / -3.2). Executor gradient at fixed RULES is FLAT
(pro 31.6 ~ Qwen 31.5 ~ flash 31.1): the rules do the work; reasoning tier adds
~0. Bare gradient also flat (pro 27.8 ~ flash 27.5): 16k-token reasoning without
rules does not rediscover corpus law — confirms the phrase-level finding at
outcome level. Largest effects: eggplant_on_keyboard +27.8, pepsi +17.0,
coke_wheel +13.9, coke_keyboard +9.7 (all OOV). Casualties: orange_juice -11.5
(OOV — the one anomaly, phrase-level autopsy pending), cube_on_plate -11.1
(in-vocab tax, same site as flash pair's -11.1 — suspiciously identical, check
shared phrase choice).

## 2026-07-25 — Row 14 COMPLETE: confirmed oracle headroom (held-out estimation)
Protocol: selection on layouts 0-17 (n=36/phrase); estimation on virgin layouts
18-23 (n=72/phrase); comparators = the SAME held-out cells from sealed x12
parquets (deterministic grid). 10-task means:
  confirmed oracle      44.9   (per-task adaptive best phrase)
  per-task best arm     38.9   (itself an oracle-ish comparator: argmax over 12 arms per task)
  nominal (originals)   28.3
Headroom read: +6.0pp above even per-task-best-arm selection; +16.6pp above
nominals — phrasing alone, no weights touched. Ceiling framing stands: phrase
chosen adaptively per task; not a blind arm.
Standouts (held-out): pepsi 77.8±4.9 ("place pepsi on plate"); ramekin 61.1±5.7
(corpus-absence rename); coke_keyboard 47.2 vs best-arm 31.9 (drop "can").
Winner's-curse cases now visible: eggplant "purple object" 55.6(sel)->40.3(conf);
small_plate 5.6(sel)->0.0(conf) — floor task, no phrasing rescue.
Two tasks where search found NOTHING above existing arms/nominal on held-out:
nut_on_plate (nominal 40.3 > oracle 37.5), carrot_on_sponge (frozen_gemini_trace
47.2 > oracle 41.7). Chart ceiling line = 44.9.

## 2026-07-25 — Row 11 lands: executor tournament COMPLETE, gradient totally flat
rules_v3_claude_agent (Fable, max effort, corpus grep access): 31.0 pooled
(35.3 in-vocab / 27.9 OOV). Final executor ladder at fixed RULES-v3:
  pro+16k 31.6 ~ Qwen-gem-trace 31.5 ~ flash-no-think 31.1 ~ Claude-agent 31.0
Agent produced byte-identical phrases to pro on 8/12 tasks (deterministic grid
=> identical outcomes). The 4 divergences: pepsi +11.5 (agent grepped "yellow
plate"=0 and DROPPED the unsupported attribute — caution won), nut_on_plate
-14.2 ("gray hexagon" rename backfired vs plain handling), nut_wheel -3.8,
small_plate -0.3. Net -0.58pp ~ 0: grep-driven divergence is a wash.
VERDICT: the rules are the artifact. Executor reasoning tier, agency, and
corpus access add nothing detectable beyond them (range 0.6pp across four
executors spanning no-think flash to max-effort agent).

## 2026-07-25 — PRE-REGISTERED row 17: nominal-input RULES arm (row 18 DROPPED)
Row 17: nominal + gem-trace => Gemini-pro + RULES (with reasoning) => pi0.
UNCHANGED production prompt (rules-v3.1); input = the 12 nominal instructions
instead of ERT. Measures the polish/deployment question: does the pipeline
leave good input alone (step-0 triage) and capture corpus-law headroom above
nominal (ramekin-class renames)? Comparator = originals arm, same cells, free.
Predictions filed: pass-through on most in-vocab tasks; renames on
corpus-absent nouns; pooled >= nominal iff unforced-error rate stays low.
Row 18 (nominal => pro-bare) DROPPED (user 2026-07-25): bare prompt has no
pass-through concept — would measure forced rewriting of good input; not
deployment-relevant. Leg: 12 rows x24x12 (~4h), pod5 after round-2 confirms.

## 2026-07-25 — Row 17 generation notes (pre-rollout)
First generation pass TRUNCATED (max_output_tokens=100 includes thinking tokens
at 16384 budget -> fragments like 'put'); discarded wholesale, regenerated at
cap 20000 + fragment guard. Corrected phrases (pre-rollout observations):
0/12 byte-identical pass-throughs — prediction #1 REFUTED at generation time
(nominals all violate the template bank, e.g. missing articles, so step-0
rebuilds; pass-through bar is stricter than predicted). Ramekin rename fires
("white bowl" — oracle-confirmed family); brand kept on pepsi; size word
dropped on small_plate; orange_juice "box" collapse RECURS on clean input.

## 2026-07-25 — Round-2 confirms land: FULL 12-task oracle coverage
12-task confirmed oracle 50.1 (held-out) vs nominal 35.4 vs per-task best arm
45.9. Round-2 boards behaved as predicted: eggplant_sponge near-ceiling (80.6
best — full color-combo template, echoing the certified combo interaction),
cube 72.2 (clause phrase, +6.9 over nominal held-out ~1sigma). NOTE the
held-out-only slice REORDERS some pairs vs the pooled full-grid estimates
(pro rules gap -0.5 held-out vs +3.8 pooled; Qwen pairs compress): n=72/task
cells are noisy and layout-heterogeneous. FRAMING DECISION: pooled full-grid
numbers (n=6912/pair) remain the primary intervention estimates; the held-out
slice is for oracle comparability only. Cube best-arm on held-out = frozen_
selftrace 87.5 (!) — above its own oracle confirm; single-cell excursions at
n=72 are exactly why we don't headline this stratum.

## 2026-07-25 — QUEUED: oracle_confirmed full-grid leg (user directive)
User: oracle belongs on the main scoreboard at all 24 configurations. The 12
confirm-winner phrases (frozen) roll as a standard x12 leg (arm=oracle_confirmed,
n=288/task). Framing: full-grid number carries mild selection bias (phrases
chosen on 0-17 evidence; n=216 re-roll washes selection LUCK but keeps adaptive
layout-fit). Unbiased held-out reference stays 50.1. PREDICTION: full-grid
pooled lands 48-53. Bonus validation: its 18-23 cells must reproduce the
confirm numbers exactly (determinism check). Chart label carries the asterisk.
Queue: after row-17 leg on first restarted pod.

## 2026-07-25 — LOSS RECORD: v7 weight snapshots steps 60-280
Old L40S pod terminated with its volume; local mirror had silently stopped
syncing after step_0040 and was never re-verified (process failure, mine —
violates the standing always-archive order). LOST: adapter snapshots 60-280
(~2.3 days L40S). KEPT: full measured record — real-rollout curve to step 200
(v7_rollout_curve.jsonl), proxy vals + all probe phrases to step 200
(phase2_v7_record/train_log.jsonl), snapshots 20+40 (l40s_mirror).
REMEDIATION (mandatory for every future run): checkpoint-sync watcher that
pushes each snapshot to the git archive ON CREATION and round-trip-verifies;
liveness of the sync checked at every fleet status pass, not just at setup.

## 2026-07-25 — CORRECTION to loss record: FULL v7 RECOVERY
The old L40S /workspace was a network volume; it survived termination and the
new pod (103.196.86.46) attached it. First probes hit mfs cold-mount metadata
lag and read empty — false alarm. RECOVERED: all snapshots step_0000..0240 +
train_log through step 244. Actual loss: ~4 steps of momentum. The mirror-lapse
lesson stands (remediation unchanged: sync watcher + liveness checks); archive
of all 13 snapshots to git in progress before any other use of the pod.

## 2026-07-26 — Row 17 lands: rules on ORIGINAL phrasing = vocabulary redistribution, not free lunch
rules_pro_nominal 34.8 pooled (42.1 in-vocab / 29.6 OOV) vs originals 36.1
(48.3 / 27.3). Paired -1.30pp pooled (~1 sigma, statistically nil). Structure
is the familiar redistribution: OOV +2.3 with three big corpus-law rescues
(eggplant_keyboard +17.4 — reaching oracle level; coke_wheel +16.0; ramekin
+13.2 — rename family works from clean starts) offset by in-vocab tax
(nut_plate -18.7, coke_keyboard -14.9, cube -11.5 vs its 75% nominal).
Notable inversion vs certification: +color on keyboard was certified +11 in
ERT-repair context but costs -14.9 applied to a clean nominal — the license
is CONTEXT-dependent, not task-dependent. RULES-V4 DESIGN INPUT: on clean
input, corpus-absence rename should be the ONLY active rule (conditional
rewriting captures the +13..+17 rescues and skips the tax). Deployment
verdict: unconditional rewriting of good input is not safe; input-aware
gating is the path.

## 2026-07-26 — Oracle full-grid leg lands: 48.2 (prediction 48-53 CONFIRMED)
oracle_confirmed_x12: pooled 48.23 (in-vocab 50.3 / OOV 46.8). Held-out
estimate was 50.1 — the two agree within ~1 sigma; honest summary "oracle ~ 49".
Selection bias worry was overblown: full-grid came in BELOW held-out.
DETERMINISM DOCTRINE REVISED: the 18-23 cross-check reproduced exactly on only
5/12 tasks; deviations = 1-4 flipped episodes per 72 (max 5.6pp). Rollouts are
exact WITHIN a pod (same GPU/driver); across pods there is a small machine-
noise term (cuDNN/numeric drift flipping borderline trajectories). Cell-dedupe
and comparator-extraction remain statistically valid (unbiased, ~independent)
but are NOT byte-exact across pods. All prior cross-pod "exact" claims should
be read as "exact within-pod, ±2-5pp across-pod at n=72".
Ladder rebuilt full-grid: ordering now matches the main scoreboard (held-out
slice anomalies were slice noise, as suspected). Charts: oracle bar replaces
the dashed ceiling line.

## 2026-07-26 — Comprehensive rollout-val backfill (user: "backfill everything")
Regenerate greedy+sampled phrases from ALL 15 archived v7 adapters (step_0020..0300)
and roll both at equal 192-ep budgets (greedy 24x2, sampled 2 samples x24x1).
Writes results/analysis/v7_curve_backfill.jsonl (own file; make_v7_progress.py
merges live+backfill per step, prefers backfill). pod5 only (L40S GPU-saturated
training). Setup: venv-gen rebuilt torch/transformers/peft (wiped in restart);
adapters scp'd L40S->pod5 via relay (rsync absent both pods). Self-healing
orchestrator (run_backfill_day.sh): waits for setup markers, stops live rval,
runs backfill resumably (skip steps already in curve), resumes live rval for
320+ after. ~11h/pod; STEP_FILTER sharding ready if more pods added.
DECISION LOG: sampled = 192 (RVAL_SAMPLES=2, user 2026-07-26), not the old 5x
nor 384; pooled sampled-mean is the checkpoint-selection metric.

## 2026-07-26 — v7b: beta-sweep fork (user-approved)
Hypothesis from v7's KL panel: policy over-regularized — KL collapsed to ~0.04
(abort 1.2, beta 0.15) exactly as rollout plateaued; KL/grad-norm peaked at
steps 100-150 = the rollout peak (58.3), then both decayed and rollout flatlined
at ~56. v7b tests it: warm-start POLICY from v7 step_0140 (greedy-rollout peak)
via --init-adapter (fresh optimizer/step), KL reference stays = frozen base
(unchanged), beta 0.15 -> 0.05. Everything else identical (C4b, F=16, 25/25/50,
0.5 dropout, grpo, lr 7e-6, probe-samples 4). scripts/run_arm_v7b.sh.
JUDGMENT RULE: evaluate on REAL rollouts, not proxy — lower beta lets the policy
chase the C4b proxy harder, which could widen the known proxy->real gap
(best-of-16 rising while rollout flat). If v7b's rollout climbs above v7's 58.3
plateau, beta was the brake; if only its proxy/win-rate rise, it's reward-hacking.
Fork point step_0140 provisional (backfill still running; re-point via FORK_CKPT
if a better checkpoint emerges). v7 proper stopped at ~step 355 (plateaued;
snapshots archived); L40S repurposed to v7b.

## 2026-07-27 — v7 backfill COMPLETE: best-checkpoint analysis (15/15, greedy+sampled from weights)
Full curve, n=192/condition/checkpoint (SE ~3.6pp, so ~2pp diffs are within noise):
  best greedy   step 120/140 (58.3)
  best sampled  step 200 (54.7)
  best mean     step 260 (55.7)
VERDICT: v7 PLATEAUED. Mean rollout sits 53-56% across steps 100-300 with no
statistically distinguishable peak (all within ~1 SE). Greedy peaks early
(120-140) then eases to ~56-57; sampled is noisier and slightly favors LATER
checkpoints (200, 260) — i.e. late training tightened the distribution a hair
(sampled 45->54) without moving the argmax. Deploy-greedy -> ship step 140;
deploy-sampled -> step 200. Fork point for v7b (step 140) confirmed as the
greedy peak with strong sampled (52.1); within noise of 200/260 so a sound
choice. This plateau (+ the collapsed KL) is exactly what motivated the v7b
beta=0.05 fork.

## 2026-07-27 — PRE-REGISTERED: step-100 gate for v7b + v7c (lr arm)
v7b evidence at step 58: KL 0.16 (4x v7a), win-rate 15% (program best), ERT
proxy improving — but rollout still on v7a's plateau (56.8; peak ref 58.3).
GATE at v7b step ~100: (a) if v7b rollout > ~58.3, beta was causal — continue
v7b. (b) else swap L40S to v7c = same fork (v7 step_0140), beta=0.05, lr
7e-6 -> 2e-5 (scripts/run_arm_v7c.sh, ready) — the last optimizer knob.
(c) If NEITHER v7b nor v7c converts proxy gains to rollout > 58.3 by ~step
100-150 of its run, the optimizer family is exhausted: bottleneck = reward
signal; L40S goes to v8a (real-rollout reward, already designed).
KL-interpretation note (user obs + refinement): v7a's rollout peak coincided
with its KL/grad-norm peak (steps 100-150). v7b tests whether KL is lever or
symptom: KL now 2x v7a's best-ever WITHOUT rollout following (yet) — leaning
symptom-of-productive-learning; the fork adjudicates.

## 2026-07-27 — GATE AMENDED (user): v7d (F=32 noise arm) jumps ahead of v7c (lr)
User proposal: attack reward noise at the fork — F=32 or k_flow 16. Decomposition
evidence (frames-vs-draws): between-frame variance = 62% of reward-diff noise ->
F=32 is the dominant-axis cut; k=16 the minor axis (deferred; optional later).
Design logic: if noise is the bottleneck, lr-up AMPLIFIES it while F=32 fixes the
instrument for every later arm -> noise arm first. Amended order at v7b step-100
gate (if v7b rollout <= ~58.3): v7d = fork step_0140, beta=0.05, REWARD_FRAMES=32
(scripts/run_arm_v7d.sh, ready; step ~1200s/it, 100 steps ~33h) -> then v7c (lr)
-> then v8a. Gate criteria unchanged: judge on real rollout > 58.3.

## 2026-07-27 — F-sweep exam SCOPED (reward-rollout correlation vs frames)
User q: do we have corr(reward, rollout) at F=4/16/32? Answer: F=4 only —
C4b spearman 0.493, 66/68 signs (l2 addendum table); F=16 was adopted on the
variance-decomposition MODEL, never re-examined; F=32 unmeasured. Sweep is
buildable: panel + ground truth all in git (results/overnight/raw: 27 phrases,
4 t/episode = F4; phrase_success_table; gate_zero_pairs). Original extractor was
pod-era ad-hoc (lost); reconstruct from sim_contexts_extract.py sibling, extend
t-grid to 16/32, extract on pod5 idle (small: 27 phrases x ~10 eps), rerun exam
per F. Deliverable: spearman(F) + signs(F) -> evidence for the v7d gate
(if corr plateaus by F=16, v7d premise dies cheaply).

## 2026-07-27 — CAVEAT pinned (user): exam metrics are in-the-limit figures
The bakeoff/addendum sign agreements (C4b 66/68) and spearman (0.493) are
computed on EPISODE-AVERAGED rewards: 10-20 episodes x 4 t x 8 draws (up to
~640 evaluations) per phrase before the comparison. They are the reward's
asymptotic discrimination, NOT its per-decision accuracy. GRPO's operating
point is ONE context x F frames x k_decode draws — where the decomposition
estimated ~43% adjacent-pair sign-flips at F=16. F-SWEEP DESIGN AMENDED: report
per-F metrics at BOTH aggregation levels — (a) in-the-limit (all episodes),
(b) per-single-context (the training-relevant operating point). Curve (b) vs F
is the primary v7d gate evidence; (a) tracks whether the reward's ceiling
itself improves with frames.

## 2026-07-27 — Gate menu addition: v7e (context-averaged reward), from F-sweep chart
fsweep_model.png reading: F=32 buys +8.5pp single-context resolution (57->65.5,
flattening curve, floor 75); but the LIMIT (97%) is reached through AVERAGING —
which frames cannot buy. v7e: score each GRPO candidate on 2 contexts (parent +
1 same-task context), reward = mean. Same 2x scoring cost as v7d; attacks
context idiosyncrasy AND frame noise. Gate menu now: v7b (running) -> v7d (F=32)
vs v7e (2-context) vs v7c (lr) -> v8a. Choice between d/e informed by the
measured F-sweep (task #7) when it lands.

## 2026-07-27 — F x C grid CORRECTED (v2) — context axis dominates; v7e favored
Grid v1 had a t-column sampling bug (drew F shared columns from the sparse
per-task t-UNION -> NaN holes, discarded data; its 60-67% numbers are VOID).
v2 draws F frames WITHIN each selected episode. All cells carry the full banked
draw averaging (8 flow / 4 decode) — same as the 97% asymptote; draws constant.
MEASURED (68 pairs, B=400): C=1 row is FLAT in F (57.1/56.9/58.0 at F=1/2/4)
— at GRPO's single-context operating point, frames buy ~nothing in the banked
range. Context axis: 58 -> 85 (C=1 -> 10 at F=4). Iso-cost: F1C4 65.9 >> F4C1
58.0. VERDICT: measured evidence favors v7e (context-averaged reward) over v7d
(F=32) at equal cost; also resolves the user's 67-vs-97 question — no cliff:
corrected curve ~58 (C=1) -> 85 (C=10) -> 97 only at the complete zero-variance
banked sample (10-60 eps/task). fc_grid.png v2 replaces v1; fsweep_model.png's
frame-axis projections inherit the v1-adjacent assumptions — treat as
superseded pending the measured extraction (task #7).

## 2026-07-27 — BLEND x AGGREGATION grid: the bakeoff optimized the wrong regime
5-blend F x C grid (same CRN draws, 68 pairs, B=400): at C=1 — GRPO's actual
operating point — 100% ENSEMBLE scores 71.5% vs C4b's 58.6 (+12.9pp); ordering
INVERTS with aggregation (C=20: grip 92.4 > C4b 90.9 >> ens 74.0, matching the
bakeoff's full-sample verdict 51/68 for ens). Reading: ensemble = low-variance,
BIASED (errors systematic, ceiling ~75); grip = high-variance, low-bias
(averaging keeps paying). The frozen bakeoff graded at full aggregation and
crowned grip-heavy C4b — correct for that regime, never measured at C=1.
CANDIDATE ARM v7f: flip --reward-blend to ensemble-heavy (blend-w 1.0 or 0.75)
— zero new infrastructure, +13pp per-decision sign accuracy by this grid.
CAVEATS before full trust: (a) 68 pairs are large-gap (median 25pp) — verify on
fine-gap pairs; (b) verifier's episode-hash split means exam episodes may
overlap verifier TRAIN -> ens C=1 advantage could be partly memorization; run a
held-out-episode check before launching v7f.
FULL-BRIDGE census (53,192 eps, 19,974 instructions): >=2-ep instructions cover
65% (34,812 eps; sample-2k was 40%) -> v7e viable at scale from the full set;
anomaly to check: top instruction has 14,532 episodes (possible placeholder).

## 2026-07-27 — PRE-REGISTERED v7e-strong (user cell: grip F=2 C=16)
User: "grip at C=16, F=2 is 86% — why not try that?" Arithmetic agrees and
overturns the naive many-cheap-steps lean once FIXED step costs (gen+update
~325s) are included: signal-per-GPU-hour ~2.8-3.1x v7's config (C4b F16C1 =
1.0x; the optimum is flat-topped over F*C ~12-32). SPEC: grip-pure rank01
reward, averaged over C=16 same-instruction contexts x F=2 frames/context;
everything else v7b-like (beta 0.05, grpo, sample_single). Pool: full-Bridge
census gives 214 instructions with >=16 eps (~16k episodes) EXCLUDING the
empty-string label (14,532 unlabeled eps — anomaly resolved: placeholder).
Trade accepted: instruction diversity narrows 1361 -> 213 (canonical head).
Semantics note: cross-context reward optimizes scene-AGNOSTIC phrase quality
(mild tension with scene-color licenses; arguably right for repair).
Step est ~1150-1400s. Judged on real rollouts vs v7 58.3 / v7b. Queue: build
club context tables (bg) + trainer multi-context scoring patch; launch at the
v7b step-100 gate on the L40S.

## 2026-07-27 — Fine-discrimination reward exam DESIGNED (user: closer pairs)
Gate-zero pairs are coarse (median 25pp). Harvested from val phrase-search
ground truth: 99 distinct phrases (n>=36) on the 4 Bridge-native tasks ->
351 pairs at 5-10pp gap + 202 at 10-15pp (analysis set, confidence-weighted
ordering truth: results/analysis/close_pairs_val.json) + 260 pairs at 0.5-5pp
(calibration bucket — a clean reward should score ~50% there). Plan: score all
99 phrases on club contexts for these instructions (C<=10 eps x F<=4 frames,
~16k phrase-frames ~7h pod5 idle, standalone score-server), then re-run the
F x C x blend grid on close pairs. Purpose: does the C-axis dominance and the
blend inversion hold in the fine-gap regime v7e actually operates in?

## 2026-07-27 — Fine-discrimination exam EXTENDED to OOV via sim-grounded a* (user)
User: evaluate reward on OOV tasks using successful rollouts' executed actions
as a*. Mechanism pre-exists (sim_contexts_extract.py, 07-09 note; eggplant exam
contexts were built this way) and phase0c already has --record-dir /
--record-success-only. OOV harvest: 155 phrases n>=36 on the 4 OOV val tasks ->
717 pairs 5-10pp + 562 pairs 10-15pp. COMBINED exam: ~254 phrases, ~1068
fine-gap pairs across 8 tasks (native: club/Bridge a*; OOV: sim-grounded a*).
Caveats carried: sim a* = pi0's own successful behavior (self-consistent
estimand, mild successful-layout selection bias) — for predicting SIM rollout
success this REMOVES the real->sim confound; calibration bucket = 0.5-5pp pairs
(clean reward should score ~50%). Runs on pod5 idle after the v7e gate.

## 2026-07-27 — Val-8 reference frame for RL (user): provisional chart + rolled leg queued
Purpose: give RL rollout numbers comparators on THEIR OWN tasks. Provisional
(banked boards, mixed n 36-180): Orig pooled 47.7 / Oracle 65.1 on the dev-8.
On the 4-task rval probe subset: Orig 54.8 / Oracle 68.6 — so v7/v7b greedy
(56-58) sits ~+2 over nominal with ~12pp measured headroom below oracle; the
RL's stack_cube (35) crushes its nominal (6.9) — RL's clearest per-task win.
QUEUED: rolled reference leg (Orig + canonical-ERT Adv + Oracle-best, 8 tasks
x 24 layouts x 12 reps, sealed protocol) on pod5 after the fine-exam scoring;
Adv phrases to be drawn from the ERT registry per task at leg-build time.

## 2026-07-27 — GATE EXECUTED: v7b -> v7e (pre-registered rule)
v7b at step 100: rollout 56.8/56.8/56.3 (steps 25/50/75) — below the 58.3 bar.
VERDICT: beta=0.05 unfroze the policy (KL 4x, win-rate 15% program-best) and
TIGHTENED the distribution (sampled closed onto greedy at step 75; samples are
genuine variants, not self-copies) but did NOT raise the argmax. Beta was a
distribution-sharpness lever, not the bottleneck. v7b findings banked; adapters
archived via v7bsync. L40S swaps to v7e-strong: grip-pure reward averaged over
C=16 same-instruction contexts x F=4 frames (86-89% measured per-decision sign
accuracy), fork step_0140, beta 0.05 retained, 8-task probes from birth,
ensemble-z tripwire logged.
Also: val-8 reference leg assets authored (ph_val8_reference.parquet — orig/
ERT-adv/oracle-best x 8 tasks; ERT provenance-matched to sealed stage_gemini);
leg queued on pod6 post-ship.

## 2026-07-28 — v7b closing datum: step 100 = 55.2 greedy / 51.0 sampled
Final point confirms the gate: greedy DECLINED over the run (56.8 -> 55.2) and
the step-75 sampled convergence partially reverted (56.25 -> 51.0; partly
noise). v7b complete record: 100 steps, 4x policy movement, rollout flat-to-
down. Beta verdict stands.

## 2026-07-28 — Val-8 reference leg LANDS (n=288/task/arm) + CORRECTION
Rolled reference (pod6): Oracle 54.7 (58.6 iv / 50.9 oov), Original 40.6
(53.7 / 27.4), Adversarial 34.5 (41.3 / 27.7). CORRECTS the banked provisional
(Orig 47.7 / Oracle 65.1 — board estimates were winner/mix biased, regressed
at full n exactly like the sealed confirm round taught). CORRECTION to the
prior read: v7 best checkpoints (42.2-43.1 val-8 pooled) sit +1.6..+2.5 ABOVE
the true Original (40.6), not below the provisional 47.7 — v7 modestly beats
nominal on val-8, ~12pp below Oracle. Other findings: Adv costs -6.1 pooled but
~0 on OOV (27.7 vs 27.4 — can't break what's already broken; all Adv damage is
in-vocab -12.4). Star datum: ramekin Adv 66.7 vs Orig 11.1 — the ERT attack
accidentally performs the corpus-absence rename ("white ceramic dish") and
SEXTUPLES the nominal: the rename law confirmed from a hostile direction.

## 2026-07-28 — Val-8 checkpoint backfill COMPLETE (4/4)
v7 on the full val-8 (greedy, n=192-216/pt): step 120: 42.2, 140: 42.2,
200: 43.1, 260: 43.8. Flat 42-44 band — all +2..+3 over Original (40.6),
~11 under Oracle (54.7). Best: step 260 (within noise of all four).

## 2026-07-28 — Fine-discrimination grid, OOV half (sim-grounded, 1132 pairs)
Buckets: fine 5-10pp (120 pairs, conf>=0.8), med 10-15 (578), calib 0.5-5 (434,
unfiltered). FINDINGS: (1) C-axis dominance HOLDS on fine pairs (ens 57->72
along C at F=1); frames help more here than on Bridge coarse pairs (F1->F3
+4-6pp). (2) Blend at C=1: ens100 57.3 > c4b 54.4 > grip 54.0 — inversion
direction holds but margin +3, not +13. At F3C10: c4b 80.2 > grip 76.4 > ens
74.8 — the BLEND wins high-aggregation on OOV (unlike Bridge-coarse where grip
won); C4b vindicated on this half. (3) Calibration bucket: grip ~50-53, ens
~53-60 — NOTE the honest benchmark is ~55-60 (mean ordering-confidence of the
bucket), not 50: ens is near-optimal there, grip UNDER-resolves resolvable
ties. Native half (sim-grounded recordings pending on pod5) will separate
OOV-effect from sim-grounding-effect. v7e reads: C-dominance premise CONFIRMED
in the fine regime; grip-pure choice partially challenged (blend > grip at
high C here) — revisit at v7e's first-probe gate with native half in hand.

## 2026-07-28 — ops: pod git repair, checkpoint backlog pushed, v7e sync live
Root cause found for the recurring pod "could not detach HEAD" wedges: scripts scp'd
to pods before their local commits landed become untracked-vs-tracked collisions, which
block `git pull --rebase`, which makes every ckpt-sync loop log PUSH-DEFERRED forever.
Repair (both pods): backup-if-differs -> rm collision files -> rebase -> chunked push.
Landed: L40S backlog of 7 commits pushed (v7a step_0320/0340 snapshots, v7b adapters
20-80, metrics.json now tracking v7e); pod5 deferred result commits pushed after
union-dedup jsonl conflict resolution. v7b confirmed complete at step_0080 (trained ==
archived). v7esync session launched on L40S (sed-clone of sync_v7b.sh): v7e step_0000
+ step_0020 archived within minutes of launch. Chart relabeled v7a per user (all
val-8 backfill points are the beta=0.15 run; v7b/v7e never val-8-evaluated yet).
Prevention rule reaffirmed: commit+push locally FIRST, pull on pod — never bare scp
of tracked files.

## 2026-07-28 — v7f: C=16 -> C=10 fork of v7e (pre-registration)
User decision after fine+coarse grid review: fork v7e to reward-contexts=10.
Evidence: coarse grid (68 gate-zero pairs, uncapped pools) shows C 8->10 is the
steepest segment on the axis (grip F=4: 82.2->86.8) while 10->16 buys only +2.1pp
(86.8->88.9); fine grid (5-10pp pairs) shows 10->16 flat (77.0->77.1 c4b) though
half its tasks cap at 10-11 eps. Theory: C is a variance knob and unbiased per-step
noise averages out across SGD steps; F*C=40 sits at the top of the optimal 12-32
band vs v7e's 64. Cost: scoring ~1935s of the 2215s step scales with C -> step
~26min vs ~37min, +48% steps/GPU-day. F stays 4 (F axis is steep: F4->F2 costs
5.9pp at C=10). Blend stays grip-pure pending the native-half exam adjudication
(fine grid shows 50/50 within 0.6pp on fine pairs and better on 10-15pp pairs —
lever reserved for a later arm if native half confirms).
Fork point: phase2_v7e/step_0020 (latest v7e snapshot; its 20-step C=16 prefix is
provenance-noted, benign — same estimator, lower variance). Everything else
identical to v7e: grip-pure blend-w 0.0, F=4, beta 0.05, lr 7e-6, probe8 probes,
ensemble-z Goodhart tripwire. Launcher: scripts/run_arm_v7f.sh. Judged on real
rollouts (v7f_rollout_curve.jsonl, 8-task pooled greedy+sampled, n=192 each).
v7e retired at step ~27 (first rollout point: 40.89 greedy / 42.71 sampled, n=384/192
— indistinguishable from fork reference v7a-140 at 42.19; too early for a verdict,
absorbed as v7f's baseline lineage).

## 2026-07-28 — correction: fine-grid F=5 row was computed on a partial rescore
The first F{1,3,5}xC{..16} grid ran on a mid-flight pull of the pod6 rescore
(12,373 rows): ramekin had zero 5-frame episodes scored yet and the F=5 row was a
different task mix (charted with a caveat). The scorer finished at 20,020 rows —
every episode now carries 6-8 frames, all F rows share identical 63-ep pools.
Superseding F=5 numbers (fine bucket): C=10 c4b 79.4 / grip 79.3 / 50-50 77.7;
C=16 grip 79.9 / c4b 79.6 — the 10->16 hop stays ~flat on close pairs at every F,
while all-F coverage confirms saturation at C~8-10. Strengthens the v7f C=10 call.
Med bucket unchanged in shape (ens-lean leads: 75/25 69.9 at F5C16 vs grip 61.2).

## 2026-07-28 — early-repair backfill (pre-registration) + pod7 split pipeline
User question: did v7a's repair ability (adversarial input -> ~42% rollout) develop
in the first ~100 steps (where proxy/rollout correlation was 0.94 and the greedy-ERT
proxy climbed steeply), or gradually? Current repair coverage is a single point
(ckpt 140 = 41.9 vs raw-Adversarial 34.5).
Leg: dev8adv-early on pod5 — checkpoints 0000-0100 (every 20), ERT input
(contexts/traces_probe8_adv), greedy+sampled 192 each, appended to
v7_dev8adv_backfill.jsonl (charts auto-update). Hypothesis (user): most of the
repair jump happens by step ~100. Baseline anchor: step-0000 = fork init (SFT
policy) — may sit BELOW 34.5 if naive rewriting hurts adversarial inputs.
Infra: pod7 (A6000) render stack live (vulkan manifest repair + libegl1 per
playbook; INT-ACT 19G + pi0 hf-cache 6.1G shipped from pod5; venv symlink audit
clean). Split pipeline: scripts/v7_dev8_gen_only.sh (pod5, has .venv-gen +
adapters) stages phrase parquets in results/phrase_artifacts/dev8q_* via git;
scripts/v7_dev8_roll_only.sh (pod7) polls git for them and rolls. Late-polish
checkpoints 0280-0340 -> pod7; early-repair 0000-0100 -> pod5.

## 2026-07-28 — CORRECTION + finding: repair numbers were contaminated; repair DECLINES with training
Bug: phase0c_rollout resumes/accumulates on its --out parquet ("resume: N episodes
already recorded"). The dev8adv runs (pre-cleanup-patch runner) therefore averaged
stale rows from earlier rolls into their recorded pooled numbers. The tell: n=432/528
instead of the designed 192. Polish rows are clean (n=192 exact, fresh files).
Recovery: append order preserves run boundaries -> row-order segmentation, verified
structurally (each clean slice = exactly 8 phrases x 24 layouts; 260 additionally by
input-phrase filter; 4 of 260's greedy rewrites were identical to 140's and their
reused episodes are valid — frozen executor, same phrase+layout).
Corrected: repair-140 = 39.58 greedy / 41.15 sampled (was 41.9/38.82);
repair-260 = 34.90 greedy / 36.46 sampled (was 38.64/38.41).
RETRACTIONS: "repair-140 fully repairs the attack / lands on the polish curve" — at
140 repair recovers 66% of the polish gap (34.5 -> 39.6 vs polish 42.2); by 260
repair is GONE (34.9 ~= raw adversarial 34.5) while polish rose to 43.75.
FINDING (1-sigma-ish per point, but consistent across greedy+sampled and concentrated
in coke_can_on_plate 45.8->20.8): later v7a training traded repair ability for
polish. The greedy-ERT proxy (panel 1) stayed flat through this decline — another
deaf-gradient blind spot. Checkpoint selection must now weigh both conditions;
the early-repair leg (0-100, tonight) will locate the repair peak.
Guards landed: out-parquet rm in dev8 runner (SKIP_SAMPLED patch), roll-only script
patched + pod7 session relaunched before any roll (no pod7 contamination).

## 2026-07-28 — BUG (user-caught): eval-time generation omitted tier tags entirely
--tier-tags training (v6.3+, so v7a/v7b/v7e/v7f) never shows the policy a bare
instruction: every input is one of "[input: original wording]" / "[input:
paraphrased]" / "[input: adversarially reworded]" / "[input: withheld] infer the
task from the context". But BOTH eval-time generators built prompts untagged:
scripts/gen_ckpt_phrases.py (all dev8 backfills, both conditions, 4-task curve)
and run_probes in phase2_train (trainer-emitted probes -> rval curves, incl. v7e's
40.89 point). Every rollout eval of checkpoint-generated phrases has therefore
probed the policy OUT-OF-DISTRIBUTION. run_val (proxy panel) tagged correctly all
along — the proxy/rollout divergences get a third possible cause, and the
repair-decline finding (39.6->34.9) is CONFOUNDED: later checkpoints may be more
tag-reliant rather than worse at repair. Sealed-test rows unaffected (fixed phrase
arms, no policy generation; rules/baseline arms use Gemini/Qwen prompts, untagged
by design).
Fixes: gen_ckpt_phrases now tags via GEN_TAG (default "[input: original wording]",
adv evals pass "[input: adversarially reworded]", GEN_TAG="" for pre-tag ckpts) and
strips model-emitted tag prefixes from outputs; run_probes tags nominal probes when
tier_tags is on. v7f restarted (step 2, ~50min lost) so its entire probe record is
tagged-mode. Untagged-mode rows stay in their jsonls as internally-consistent
untagged measurements; tagged redos write v7_dev8_tagged.jsonl /
v7_dev8adv_tagged.jsonl. Redo queue: late 0280-0340 (tagged from the start),
early-repair 0000-0100 (tagged), then re-eval polish 0120-0260 + repair 0140/0260.
Training-side audit (user request): tier->tag->source mapping verified correct in
pick_source_and_trace — every return path pairs the label with matching content,
fallbacks relabel properly (ert->benign->nominal). One footnote: a missing ERT
trace falls back to the NOMINAL trace for a hostile source (leakage-firewall
exception, frequency unknown; acceptable, logged here for the record).

## 2026-07-29 — early-repair curve (tagged): FLAT — no early jump (user hypothesis refuted for 0-100)
Tagged greedy repair, val-8, n=192/pt: steps 0/20/40/60/80/100 =
35.94/36.46/36.46/36.46/35.94/36.46 — six points spanning ONE episode (69 vs 70
of 192). Greedy adversarial-input behavior is static through the first 100 steps
and its small +1.5-2pp offset over raw-Adversarial (34.5) exists at step 0 (SFT
init). The only remaining window for genuine repair learning is 100->140; tagged
redo of 120/140/200/260 rolling (chainB). Late-polish leg complete on pod7:
280/300/320/340 all exactly 40.62 greedy (frozen behavior, == Original line).
Polish early (pod6, in flight): 0/20/40/60 = 39.06/40.62/42.71/39.58 — wobbles
around Original, no significant departure; per-task vectors show real zero-sum
trades under the flat pooled (plate 8->29, ramekin 46->12 between 20 and late).

## 2026-07-29 — overnight program v2 (user directive: pods 5-8 all night)
1) Tagged matrix completes (repair-redo pod5, polish-redo pod7, repair-late pod8,
   native-half pod6). 2) DEEPEN decision checkpoints at 2x reps greedy-only
   (sampled skipped: its variance is phrase-draw-dominated, episode reps do not
   reduce it): pod7 polish 0040/0120/0140/0200/0260, pod8 repair 0120-0260 ->
   *_deep jsonls, n=384/pt (SE 3.6->2.5). 3) HIGH-FIDELITY CLUB RANKING (user:
   "definitely do this"): all 213 >=16-ep club instructions scored at F=4 x C=20
   (grip 92.4% / c4b 90.9% coarse cell) on common per-task club contexts (CRN),
   pod6 score server after native-half; outputs club_phrase_ranking.parquet +
   top3 json. 4) Shutdown authorization: pods idle after all work may be stopped
   (podstate backup + sync-liveness verification first); pod5 stays (v7f rval
   host), L40S stays (v7f training).

## 2026-07-29 04:10 — tagged repair 120/140: v7a NEVER learned repair
Tagged repair-140 = 35.94, bit-equal to the step-0 init; repair-120 = 38.54 (a
within-noise wiggle). Full curve 0-140: 35.9/36.5/36.5/36.5/35.9/36.5/38.5/35.9
— a 2.6pp band around 36.2, indistinguishable from init at every point. The
untagged 39.6@140 was an out-of-distribution artifact. Combined with polish
(pinned at the Original line): tagged v7a's entire deployable value — a ~+1.5pp
n.s. offset over raw adversarial input and Original-equivalence on nominal input
— existed at initialization. The proxy reward's rich learning dynamics (best-of-16
climb, win-rate, KL) produced no measurable real-rollout movement in-distribution.
Raises the stakes for v7f: measured-reward training must move REAL tagged rollouts
where v7a's C=1 reward could not. Repair 200/260 + polish 140-260 redo rolling.

## 2026-07-29 06:05 — tagged repair curve COMPLETE (0-260): never learned, ends below baseline
Final: 0/20/40/60/80/100/120/140/200/260 = 35.94/36.46/36.46/36.46/35.94/36.46/
38.54/35.94/34.90/34.38. Flat ~36 through 120, declining to 34.4 by 260 — BELOW
the raw-Adversarial reference (34.5). v7a never acquired repair ability and late
training degraded adversarial handling to slightly worse than no rewriting.

## 2026-07-29 06:55 — NATIVE HALF ADJUDICATES: grip-heavy wins both buckets on-distribution
fine_exam_features_native.parquet (8,220 rows; oracle-best trajectories, 4 ID val
tasks, 70 eps with >=5 frames). Native fine bucket F=5: C=16 c4b 81.1 / grip 79.8
>> 50/50 76.5 / 75/25 73.8 / ens 71.0; C=10: c4b 76.0 / grip 75.4. Native MED
bucket: GRIP LEADS (63.9 at F5C16 vs c4b 62.4, ens-lean <60) — the OOV half's
med-bucket ensemble advantage does NOT replicate on native contexts; it was
provenance-specific (sim-grounded a* OOV frames). Verdict: v7f's grip-pure reward
stands adjudicated on the distribution it trains on; blend lever closed. Combined
(all) grid computing. Fine-discrimination exam substantively COMPLETE.

## 2026-07-29 ~11:00 — deepened points CONFIRM a genuine mid-run polish peak
2x-rep deepening (n=384, SE 2.5): polish 40 = 44.53, 120 = 45.05, 140 = 45.05 —
three consecutive independent ~+4.4pp elevations over Original (40.6), jointly
strong evidence of a REAL tagged polish peak at checkpoints 40-140 that decays to
Original-parity by 200. Revises "never left the Original line": v7a earned ~+4pp
of genuine polish mid-run and lost it with continued training. Fork@140 (v7e/v7f
lineage) sat at the peak. Note on rep mechanics: sim execution is deterministic
per (phrase, layout, rep-seed) — rep-2 draws a fresh seedset, so deepening
genuinely doubles n (and explains both frozen 1-rep vectors and deep-vs-shallow
value shifts). v7a-best reference line (v7f chart) auto-rose to 45.05.

## 2026-07-29 ~15:10 — reward-guided phrase search over training corpus (design)
User intent: oracle-search-style exploration (which phrases are good + which
phrase-CHANGES help), powered by the measured high-fidelity reward instead of
rollouts. Funnel: (1) candidates = 8 policy samples (v7a-140 fallback; pod5 had
no extracted v7f adapter — provenance noted) + original, for the 100 biggest-club
instructions (~900 phrases); (2) screen all at F=1 x C=4 (~65% coarse cell, cuts
obvious losers, ~1.6h); (3) finals: top-3 + original per instruction at F=4 x
C=10 (~86-92% band, ~7h overnight); (4) edit analysis: categorize winner-vs-
original diffs (color adjectives, noun renames, verb changes, length) and rank
transformations by reward delta — rules-derivation methodology on training data.
Baseline layer (as-written club ranking at F4xC<=20) finishing separately.

## 2026-07-29 16:45 — deep repair-120 CONFIRMS checkpoint-120 as v7a's best (user call)
Deep (n=384): repair-120 = 39.06 (+4.6 over Adversarial 34.5, ~1.8sigma) — the
only repair point that STRENGTHENED under deepening (140 regressed to 36.98,
260 to 36.72 = init band). With deep polish-120 = 45.05 (+4.4, ~1.8sigma),
checkpoint 120 is the sole checkpoint above reference in BOTH conditions at
doubled precision. Its repair edge concentrates in coke_can_on_plate (41.7 vs
21-25 at neighbors) + the matrix's only nonzero keyboard repair (4.2). Deep
polish 200/260 both 42.71 (+2.1, decay only partial through 260). Deployment
recommendation: v7a step_0120 is the checkpoint of record.

## 2026-07-29 19:50 — v7f first adversarial point: repair signature appears
v7f adv step 20 (tagged, n=192): 39.06 — +4.6 over raw-Adversarial (34.5),
matching v7a's best-ever repair (deep ckpt-120, also 39.06) at fork+20. KEY:
keyboard repair = 25.0 (v7a: 0.0 across its entire repair matrix save one 4.2)
— the corpus-absent-rename repair the low-resolution reward could never select
for. Ramekin preserved at 62.5. Single point, ~1.3sigma pooled; keyboard column
qualitatively new. Prediction registered: if the high-fidelity reward is doing
what the measurement program says, the adv curve should climb while v7a's never
moved. Step-40 rolling.

## 2026-07-29 ~21:20 — v7f adv-40 = 41.67: SURPASSES v7a's all-time repair at fork+40
v7f adversarial curve: step 20 = 39.06, step 40 = 41.67 (climbing; keyboard
repair 25.0 replicated at both). 41.67 > v7a's best-ever repair (39.06, deep
ckpt-120) and nominally above the Original-phrasing reference (40.6). The
high-resolution-reward-enables-repair hypothesis is confirming by trend: two
elevated, climbing points with a stable qualitative capability (corpus-absent
rename) v7a never produced. Polish meanwhile Original-parity (25/50 = 41.2/40.4)
— the repair-polish gap has nearly closed at fork+40.

## 2026-07-30 00:20 — pod8 deep late-polish REWRITES the decay story (seedset artifact)
Deep (2-rep, n=384) polish late: 280/300/320 = 43.23, 340 = 44.27 — vs the
shallow rep-1 flatline of 40.62. Four independent checkpoints all shifting +~3-5
with rep-2 = the rep seedsets differ SYSTEMATICALLY in difficulty (CRN reuses the
same per-(layout,rep) seeds across checkpoints, so rep-1-only rows share a
harder-than-average draw). REVISED tagged polish story: rises 39->44.5 by step
40, then holds a +3-4pp plateau over Original through 340 — no mid-run peak, no
late decay. Checkpoint-120 remains the balanced BOTH-conditions best (its repair
39.1 stands alone; repair still degrades late). Caveats now attached to all
rep-1-only comparisons vs multi-rep references (references averaged 12 seedsets);
v7f's rep-1 rollout points likely UNDERSTATE its level vs references by the same
mechanism. Deepening verdict updated: it caught a systematic bias, not just
variance — worth it.

## 2026-07-30 ~02:55 — v7f adv-60 = 42.19: three consecutive climbing repair points
Curve: 20/40/60 = 39.06/41.67/42.19 (1-rep; 2-rep re-rolls queued). Keyboard
repair 25.0 at all three (stable capability); cokeplate repair rising (21->37.5).
+7.7 over raw-Adversarial at step 60, above the Original reference. The
high-resolution-reward-enables-repair prediction is holding as a trend.

## 2026-07-30 ~03:40 — CORRECTION: v7a initialized from BASE QWEN, not SFT
run_arm_v7.sh has no --init-adapter: v7a cold-started (fresh zero-init LoRA =
exactly base/frozen Qwen at step 0). Earlier ledger/chart language calling the
v7a step-0 "the SFT init" was wrong (user caught it). Therefore the tagged
step-0 values ARE the frozen-Qwen-rewriter baselines: repair 35.94, polish
39.06 (n=192). Chain of baselines, adversarial condition: pass-through 34.5 <
frozen-Qwen rewrite 35.94 < v7a-140/fork-init 36.98 < v7f current 42.19.

## 2026-07-30 ~07:10 — freeform loop complete (4 rounds); 2-rep adv re-rolls land
2-REP ADV (n=384): step 20 = 38.28, 40 = 40.36 (rep-1 read ~1pp high; climb
shape holds: +2.1 from 20->40; keyboard repair persists at depth, 14.6 vs the
rep-1 25.0 — real but smaller). vs frozen-Qwen 35.9: +4.4 at step 40, at depth.
FREEFORM INSIGHT LOOP: holdout testing REFUTED the elaborate hypotheses its own
early rounds proposed ("precise high-intent verbs" -0.94pp, "rich spatial
descriptions" -1.2pp) and converged on template conservatism: keep canonical
nouns, simple truncated directions, avoid elaboration. Independently replicates
the boards' originals-win pattern ON CLUB (in-corpus) instructions — the
enhancement rules (renames, adjectives) only pay off out-of-vocab, exactly the
sealed-test contrast. Transforms grid (running next) will put CIs on this.

## 2026-07-30 ~09:05 — TRANSFORM RULES TABLE (49 instr x 16 single-change probes)
On IN-CORPUS (club) instructions, essentially every transform hurts or is
neutral. Neutral: preposition variants (+0.000 n.s.), polite "please" prefix
(-0.0005 n.s. — politeness is free). Significantly harmful: rename_visual
(-0.025**, the WORST), elaborate (-0.015**), destination_first (-0.013**),
add_spatial (-0.008**), verb_synonym (-0.005**), color_adj (-0.007*).
THE MASTER RULE, now quantified from three independent methods (boards, freeform
holdout, transform grid) and matching the sealed 2x2: phrasing interventions are
CONDITIONAL ON CORPUS MEMBERSHIP — in-vocab, the canonical template is the
optimum (touch nothing); out-of-vocab, visual renames are the biggest win
(+42% rel sealed OOV). rename_visual being worst in-corpus and best OOV is the
cleanest statement of the whole project's thesis.
ADV 2-REP UPDATE: 20/40/60 = 38.28/40.36/38.54 — the rep-1 monotone climb
softens to an elevated band (+2.4..+4.4 over frozen-Qwen 35.9) with slope within
noise; steps 80/100 decide the trend. Keyboard repair persists (14.6).

## 2026-07-30 ~19:30 — STRATA DECOMPOSITION: v7f fully repairs hostile input on familiar tasks
v7f repair (2-rep, tagged), in-vocab stratum: steps 20/40 = 54.7/55.2 vs
Original-phrasing 53.7 and adversarial pass-through 41.3 — >=100% recovery of
the adversarial damage on in-corpus tasks, AT or ABOVE original-level. Steps
60/80: 51.6 (~96% recovery). OOV stratum: repair 21.9-25.5 vs pass-through 27.7
(slightly negative pooled; task-mixed — keyboard uniquely repaired 14.6-25 vs ~5,
ramekin's accidental-rename benefit partially undone). The pooled 38-40 band
was averaging a SOLVED stratum against a neutral one. This is the two-component
method's core evidence: gate + repairer => canonical-level performance from
hostile input on familiar tasks.

## 2026-07-30 ~19:30 UTC — pol-80 recovered (38.02); Goodhart trigger half-armed; PREREG Amendment 1 (composed router, zero new exposure)
- **v7f polish step 80 = 38.02** (n=384, 2-rep). The roll completed on pod6's polcatch
  session but its inline merge crashed before writing; the episode parquet was still
  in data/dev8_g_out.parquet (384 rows, 8 phrases — verified) and the point was
  recovered without re-rolling. Polish trend now 41.67 -> 40.36 -> 38.80 -> 38.02
  (steps 20/40/60/80): monotone drift down, consistent with grip-proxy Goodharting.
- **Trigger status**: the pre-agreed fork condition was "pol-80 AND pol-100 both
  <= 38.5 while proxy climbs". pol-80 fired (38.02). Awaiting pol-100 (gen queued,
  trainer at step ~109). If it also lands <= 38.5 -> fork from polish-peak (step 20)
  with tier-conditional edit-cost.
- **PREREG Amendment 1 filed**: arm D = composed router (in-vocab -> arm B v7f cells,
  OOV -> arm C rules cells, gate = existing zero-count noun audit). Analysis-only:
  pooled from already-planned per-task cells, no new sealed rolls, routing table
  frozen before any arm B/C repair results are seen. This formalizes the user's
  proposal (v7f as in-distribution canonicalizer + rules for OOV) as a testable
  sealed endpoint. Prediction: arm D > max(arm B, arm C) on repair pooled.

## 2026-07-30 ~20:00 UTC — USER-CAUGHT ROUTER HOLE: input-side gate misroutes hostile renames; output-gated cascade (PREREG Amendment 2)
User: "Shouldn't renaming of an in-corpus noun to an out-of-corpus noun route to v7f?"
Correct — and yesterday's routing bullet had it backwards. Audit (census = zero-count
content nouns over the 2,000 training contexts, verified vs raw word-boundary counts):
- ALL 8 val-8 hostile inputs contain zero-count nouns ("root vegetable", "eating
  utensil", "nightshade", "input device"...) — incl. all 4 in-vocab tasks. An
  input-side gate is stratum-blind under adversarial input and would send the
  rename-attack cases (v7f's proven best stratum) to the rules path.
- v7f OUTPUT re-gates perfectly: census-clean on all 4 in-vocab tasks at every
  step 20/40/60/80/100 ("root vegetable"->"carrot" etc.). On true-OOV objects it
  self-renames to visual analogs (ramekin->"white bowl", wheel->"tire", coke->"red
  can") EXCEPT keyboard, which it keeps verbatim (no corpus analog).
- Per-task repair cells (passthrough x12 vs v7f 2-rep): IV big wins — carrot/plate
  15.6->50.0, stack 16.0->37.5(s40), eggplant 83.3->91.7; spoon 50.3->41.7 the one
  IV loss. OOV mixed: keyboard 4.5->14.6 (+), wheel floor (1.7->0), coke/plate
  37.8->31.2 (-), ramekin 66.7->56.2 (-) — conservatism law again: rewrite pays
  only where passthrough fails; both negative cells had competent passthrough.
- Design consequence: gate the OUTPUT, not the input -> Amendment 2 arm D-deploy
  cascade (verbatim-hit passthrough / v7f / output-census fallback to rules).
  Composition from arm B/C per-task cells; zero new sealed exposure. Not designing
  finer gates off 8 noisy 2-rep cells (~±7pp SE).

## 2026-07-30 ~20:20 UTC — Arm D simplified to USER'S SERIES PIPELINE (Amendment 3); keyboard styling probe launched
User proposal adopted: route EVERYTHING through v7f; census the OUTPUT; clean ->
keep; retained OOV noun -> rules applied to v7f's output. Strictly better than
Amendment 2's branch-to-rules-on-raw-input because the rules lexicon needs
canonical names to key on and only v7f recovers them from hostile paraphrase
("black input device" -> "keyboard"). Sealed rules-v3 phrases confirm rules keep
keyboard/wheel verbatim and rename only analog-having objects (ramekin->"white
bowl", juice->"box") — the same renames v7f emits on its own — so the fallback's
only live effect is styling on keyboard-class outputs. Launched val-8 probe
(pod6 tmux seriesprobe): "Place the carrot on the keyboard." vs "put the carrot
on the black keyboard", 24 layouts x 2 reps each -> series_fallback_probe.jsonl.
Dropped front passthrough gate is safe at the frozen arm-B window (polish >=
passthrough at steps 20-40) but couples benign-traffic safety to checkpoint
choice — noted as the one caveat if later checkpoints are ever deployed.

## 2026-07-30 ~21:00 UTC — SEAT DECISION DATA: v7a-120 vs v7f as the pipeline rewriter (Amendment 4)
Q (user): seat v7f or v7a in the series pipeline?
- In-vocab repair stratum is a TIE: v7a-120 = 55.2, v7f-40 = 55.2, v7f-20 = 54.7
  (frozen Qwen 52.1) — full canonicalization of hostile in-vocab input is NOT
  v7f-specific; it is base competence + ~3pp RL polish, equal in both runs.
- Polish decisively v7a-120: 45.05 (replicated at 140) vs v7f best 41.67; series
  pipeline routes ALL traffic through the rewriter so this carries full weight.
- Pooled repair inside noise: v7f-40 40.4 vs v7a-120 39.1 (SE ~2.5pp).
- v7f's one clear unique cell — keyboard 14.6 vs 4.2 — is neutralized by the
  pipeline: both seats emit canonical "keyboard", census catches it, fallback
  handles it either way.
- v7a-120 sealed emissions (both conditions, censused): canonical OOV names kept
  (keyboard), brand variants (soda/cola can); polish shows two grounding drifts
  (green->teal cube, coke->soda) — motivates unknown-adjective stripping in
  rules-v4-lite fallback.
- CAVEAT ledgered: exploratory routing tables used the 2k-RL-parent census;
  census-of-record = bridge-side counts (audit coverage source); recomputation
  pending on-pod (bridge_census.json) before any arm D reporting. Recommendation
  is census-independent.
RECOMMENDATION: seat A (v7a-120); v7f held as challenger; sealed table
adjudicates the seat for free (Amendment 4 secondary endpoint).

## 2026-07-30 ~21:15 UTC — SERIES PROBE RESULT: fallback styling is LOAD-BEARING (+14.6pp); pol-80 clobber repaired
- **series_fallback_probe (val-8 keyboard, 24x2 each)**: v7f raw output "Place the
  carrot on the keyboard." = 14.6 (exact reproduction of the curve cell — CRN
  determinism check passes); rules-styled "put the carrot on the black keyboard"
  = **29.2**. The Amendment-3 expectation that the fallback adds "only styling"
  is refuted in the useful direction: the styling DOUBLES success on the
  retained-OOV case (plausible mechanism: "black" gives the executor a visual
  anchor for a language-OOV object). Paired n=48/cell — suggestive, not final.
- CONSEQUENCE for sealed arm D: approximating fallback cells by arm A/B raw cells
  (Amendment 3 clause) is now known-CONSERVATIVE (understates arm D). Honest
  options: report conservative composition, or roll true fallback texts on the
  retained-OOV sealed tasks (~2-4 tasks x 24 x 12 per seat/condition = small NEW
  sealed exposure; requires explicit user sign-off under the discipline clause).
- **pol-80 clobber post-mortem**: refresh script scp'd pod curve copies OVER
  git-tracked files; a stale copy rode the pull autostash and my blanket
  `git add -A` amendment commit deleted the measured pol-80 row from origin.
  Restored from commit 6f5f7ac. Fix: scp now stages to untracked _scp_* files
  (gitignored); chart max-n merges git + staging; _merged tolerant of conflict
  markers. New footgun variant recorded in memory (autostash-clobber).

## 2026-07-30 ~22:00 UTC — USER DIRECTIVE: no fallback-text sealed rolls; sealed-leg status pass (pod8 Tier-A resurrected)
- **USER DECISION (final): do NOT roll the true fallback texts on sealed.** The
  sealed set stays sealed; arm D is reported via the conservative composition
  (raw arm A/B cells for fallback-routed tasks), with the val-8 series probe
  (14.6 -> 29.2) cited as evidence the composition understates arm D. No
  further sealed exposure beyond the prereg'd arms.
- Arm A (v7a-120) sealed status: ALIVE on pod5. Today's roll = data/seal_out.parquet
  (mtime-verified); ~540/3456 polish eps (~16%) at ~4.3 eps/min -> polish leg
  ~06:30 UTC, repair leg after (~19:00 UTC Jul 31 full arm A). NOTE: the
  sealed_out_w0/w1/w2 parquets on pod5 are Jul-26 RELICS of earlier legs —
  their shard means are NOT arm A results (nearly misread; mtime check saved it).
  Peek discipline: only the running partial mean (43.1 @ 540 eps) was observed,
  after all amendments were filed.
- Tier-A (pod8): found DEAD — the tmux session never survived the pod8 restart
  (container-layer wipe); no log, no process. Repaired the untracked-collision
  wedge (7 collisions, backup-diff-rm recipe, pulled to ab480ee), reinstalled
  tmux, relaunched run_tierA_sibling.sh -> rolling anchors arm (12x24x3) on the
  sibling ckpt. ~4h/arm. Monitor armed.
- Pod7: step-0 adversarial roll in progress (started 19:47); then step-0 pol,
  step-100 adv/pol. v7f trainer at step ~112.

## 2026-07-30 ~22:30 UTC — FULL IV/OOV STRATA DECOMPOSITION of every tagged checkpoint eval (user Q: "is the value in-vocab?")
Chart: results/charts/v7_strata.png. References (12-rep): originals IV 53.7 /
OOV 27.4; adv passthrough IV 41.3 / OOV 27.7; oracle IV 58.6 / OOV 50.9.
- REPAIR: gains are ENTIRELY in-vocab. IV: 52.1 (frozen) -> 55.2 (v7a-120 ==
  v7f-40). OOV: NO checkpoint of EITHER model ever beats passthrough 27.7 —
  best-ever OOV repair 25.5 (v7f 40-80); most points 19-23. RL repair = in-vocab
  canonicalization, full stop.
- POLISH: frozen Qwen is OOV-TILTED (OOV 34.3 > passthrough 27.4, but DAMAGES
  IV: 43.7 vs 53.7). RL REDISTRIBUTES value to IV: v7a-120/140 IV = 58.9 —
  above in-vocab originals (+5.2) and matching oracle IV (58.6) — while OOV
  drifts to 28-31 (still > passthrough, < frozen's 34.3). v7f polish drift is
  IV-concentrated (56.8 -> 50.5); its OOV never matches v7a's.
- Remaining oracle headroom is ~ALL OOV: oracle OOV 50.9 vs best model OOV 31.2
  (v7a polish) / 25.5 (repair). The OOV stratum is untouched by both RL runs —
  exactly the territory delegated to the rules fallback in the arm D pipeline.
- Caveats: 4 tasks/stratum (SE ~3.5-5pp/point), rep-1 rows carry ~-3pp seedset
  bias (marked on chart), val-8 stratum labels are audit-confirmed (every OOV
  task contains a zero-count noun per bridge census).

## 2026-07-30 ~23:05 UTC — CHECK-ME VERDICT: user's sealed evaluation plan is fully measured already (Amendment 5, frozen routing)
User proposal: sealed-evaluate v7a-120 + "rephrase; if output remains OOV,
re-route to rules with Gemini". CHECKS OUT, with a correction to MY earlier
framing: rules-v3 is NOT mechanical code — it is a frozen rewrite protocol
(b4_phrasing_rules_v3.md, noun ladder + brand table + corpus-vocab appendix)
EXECUTED BY gemini-pro (temp 0.2, thinking 16k). "Rules with Gemini" was the
correct name. Discovery: BOTH fallback legs were already rolled in the
scoreboard era — rules on ERT = pairA_pro_x12 arm rules_v3_gemini_pro (31.6
pooled, 2026-07-25); rules on nominal = rules_pro_nominal_x12 (34.8,
2026-07-26). So the entire arm D seat-A evaluation = arm A (rolling, lands
~Jul 31 eve) + existing cells. ZERO new sealed exposure of any kind.
Frozen routing (armD_routing_seatA.json): polish 5 KEEP/7 FALLBACK, repair
7 KEEP/5 FALLBACK. Fallback synergies visible in the phrases themselves:
coke_on_keyboard fallback restores the brand v7a genericized ("soda can" ->
"put the coke can on the black keyboard"; brand->generic is a certified
-47.2 error in rules-v3's own measurement table) and adds the color anchor
(+11.1 cert). Gate blind spot ledgered: brand-dropping is census-invisible
(pepsi repair emission "blue can" gates KEEP).

## 2026-07-30 ~23:50 UTC — v7f KILLED (step 114); v8 LAUNCHED (tag-free, cold start, beta=0.15)
USER SPEC: C=10 x F=4, grip-pure reward, NO TAGS, tier mix ert/original/benign
50/25/25 (= v7a's proven 0.25,0.25,0.5 nom/benign/ert), 50% TRACE dropout,
beta=0.15, cold start from frozen base Qwen, checkpoint evals every 10 steps.
Design notes (assessment):
- Tag-free is the arm-D-consistent training: the census gate externalizes
  regime detection, so v8 is the pure repairer the method deploys. Removes the
  input-slot dropout mechanism (needed the [withheld] tag) — replaced by NEW
  --trace-dropout 0.5 (phase2_train.py), which needs no tag signal and trains
  trace-optional rewriting (deploy may lack a scene trace).
- beta=0.15 responds to the fired Goodhart trigger (v7f polish 41.7->38.0 by
  step 80 at beta=0.05, drift IV-concentrated 56.8->50.5).
- Eval stride 10 both conditions, TAG-FREE generation (GEN_TAG="").
Execution: v7f final archive verified (newest-archived == newest-durable =
step 100; 101-114 forfeited, ledgered; Amendment 6 fixes arm B selection to
archived 20-100, step-100 pair still lands). L40S: train/score/v7fsync killed,
v8 launched via scripts/run_arm_v8.sh (cold start verified: fresh base-weight
load, no resume/init), v8sync archiver up. Pod6: v7fadvgen -> v8gen (stride 10,
tag-free). Pod7: current roller drains the v7f queue (step-0 pair mid-roll,
step-100 pair next — still wanted: true fork baseline + arm B selection);
swaps to scripts/eval_roll_loop.sh (unified v7f+v8 roller) at drain.
First v8 checkpoint ~step 10 in ~5-6h; first curve points shortly after.

## 2026-07-31 ~00:10 UTC — USER-CAUGHT SPEC MISREAD: v8 dropout is INPUT-SLOT dropout, not trace dropout; relaunched
"50% dropout of trace" meant the v6.1 mechanism: DROP THE INSTRUCTION SLOT
("infer the task from the context") so the model relies on the trace, which
quotes the source phrase. I had implemented the inverse (dropping the trace).
CORRECTION to my own earlier claim: input-slot dropout does NOT require the
[withheld] tag — tag-free, the dropped slot is a plain instruction; the tag was
only an annotation when --tier-tags was on. Launcher now --input-dropout 0.5
(v7f's rate), no trace dropout (--trace-dropout remains in the trainer as an
unused capability, default 0). v8 relaunched from scratch on L40S (~20 min of
weight-loading lost, no steps trained under the wrong config). All else
unchanged: cold start, tag-free, beta=0.15, grip-pure C=10 F=4, mix 25/25/50.

## 2026-07-31 ~00:50 UTC — LEAKAGE AUDIT of arm-E derivation evidence (user challenge: no sealed data in rules derivation)
Sweep of all 4 mining files + v3 for sealed markers (sealed/widowx_/InScene/x12/
ERT strings/sealed task phrases): ALL CLEAN except "pepsi" x1 in v3 itself =
the val-8 wrong-brand probe row (coke->pepsi -2.8, n=36), written pre-sealed.
freeform "holdout" verified INTERNAL to training data: hypothesis tests ran on
fresh bridge TRAINING instructions scored by grip proxy (records are reward
deltas, n~12) — no sealed, no val-8 contact. Added gate: the returned v4 draft
gets the same sealed-marker sweep before acceptance.

## 2026-07-31 ~01:20 UTC — pod7 repurposed to arm A REPAIR (Amendment 8); v7f val-8 queue closed
Killed the v7f 0000-polish roll mid-flight (user directive: step-0-pol +
step-100 pair dropped). Arm B selection final: step 40 (40.36 2-rep adv).
Pod7 now rolls arm A repair (12x24x12, ~13h, ARMAREP-UP 22:38 pod-clock), then
a cube_on_plate 24x3 drift re-anchor, then execs the unified v7f/v8 eval
roller. Pod5 continues polish (~32% at 22:37); watcher armed to intercept
pod5's runner after polish so repair isn't double-rolled (its polish merge will
be run manually from seal_out.parquet if the wrapper is killed pre-merge).
Arm A ETA compressed: both conditions ~12:00 UTC (was ~evening).
v8 heartbeat: step 1. Rules-v4 workflow: miners + draft done, verifiers running.

## 2026-07-30 22:55 UTC (verified against pod clocks) — TIMESTAMP ERRATUM
Ledger/amendment entries stamped "~23:05 UTC" through "2026-07-31 ~01:20 UTC"
(Amendments 5-8 and adjacent entries) were written with assumed clock times
that drifted ~2.5h ahead of reality; true times were 2026-07-30 ~20:30-22:45
UTC. Filing ORDER is unaffected and git commit timestamps are authoritative.
Also: the derive-rules-v4 workflow died when the user's machine slept (local
agents); resumed 22:50 UTC from cache (miners + draft cached; verifiers rerun).

## 2026-07-30 23:05 UTC — Tier-A CANCELLED (Amendment 9); pod8 idle pending arm E
Killed tierA + rulesv4 waiter sessions on pod8 (waiter would have tripped its
missing-phrases retry and exited; it gets relaunched after v4 phrases are
staged). No sibling data ever produced. Partial out-parquet removed.

## 2026-07-30 23:20 UTC — pod6 consolidated to all-in-one v8 eval worker (user fleet optimization)
v8_gen_roll_loop.sh on pod6 now does gen (30min) + both condition rolls (~3.6h)
sequentially per checkpoint — fits the ~6h stride-10 cadence with slack. Pod7
therefore has NO v8 duty after its arm A repair leg + drift anchor: at
ARMAREP-COMPLETE its exec'd roller gets killed (would race pod6), results
verified, podstate backup, then pod7 is a shutdown candidate (~12:00 UTC).
Steady state after arm E: L40S + pod6 ONLY.

## 2026-07-30 23:20 UTC — ARM E LIVE: rules-v4 derived (7-agent workflow), leakage-gated, generated, preflighted, ROLLING on pod8
Derivation: 3 isolated miners (boards 17 findings / transforms 16 / freeform 10)
-> draft in v3 protocol form -> 2 adversarial verifiers (6 evidence issues + 7
regression issues found) -> revision applied. Output leakage sweep: CLEAN
("pepsi" = v3 brand-table row carried verbatim; "sibling" = plain-English board
reference). b4_phrasing_rules_v4.md committed (36k chars).
v4's headline novelties vs v3 (all train-data-cited): REGISTER triage
(ornate rebuild vs training-style pass-through default, 144/213 unbeatable),
frozen families (fold/wipe/end-effector/move-to memorized basins), telegraphic
expansion as the one profitable rewrite target (articles inserted, 20/27
winners), articulated-object verb repair (24-27% win rates), two-clause
hard-put exception, strengthened corpus-noun frequency ladder.
Generation (gemini-pro-latest, temp 0.2, thinking 16k, sealed traces):
both conditions preflighted clean. NOTABLE DIVERGENCE for the record: v4
nominal KEEPS ramekin ("white ramekin") and orange juice where v3 renamed
(bowl/box) — training-conservatism overriding the rename lexicon on short
inputs; ERT condition still renames (ornate register rebuild). This is the
live test of prediction 3. Rolls: pod8 rulesv4 (ert leg started 23:18,
nominal after, ~7h each). Monitor armed.

## 2026-07-31 00:00 UTC — arm E nominal leg moved to pod6 (user: everything done by morning)
Pod6's v8 worker paused (no checkpoint existed yet; first v8 eval pair will
queue and drain tomorrow afternoon — accepted tradeoff). Pod6 rolls
rules_v4_nominal 12x24x12 now (started 00:00), then a cube_on_plate 24x3
drift re-anchor (pod6 is new to sealed rolling), then execs back to the v8
worker. Pod8 keeps the ERT leg; AT ITS DONE-ert EVENT (~08:40) its runner
gets killed so its own nominal iteration never double-rolls.
By-morning schedule (UTC): armE-ert ~08:40 (pod8), armA-polish ~09:30 (pod5),
armE-nominal ~10:30-13:00 (pod6), armA-repair ~12:30 (pod7) -> arm D
composition immediately after; three pods shutdown-ready by early afternoon.

## 2026-07-31 07:50 UTC — FIRST SEALED LANDINGS: arm A polish 31.97 (below originals — val-8 edge did NOT transfer); arm E ERT 33.30 (> v3 31.6, prediction 1 MET); arm D polish composed 35.18
- ARM A POLISH (v7a-120 rewriting sealed originals, 12x24x12): pooled 31.97 vs
  originals 36.1 (-4.1). The val-8 polish advantage (+4.5 over originals there,
  IV stratum at oracle ceiling) INVERTED on sealed. Per-task: huge wins
  cube_on_plate 84.0, eggplant_sponge 79.9; catastrophic cells nut_wheel 3.1,
  small_plate 1.0, coke_wheel 6.9. Prereg prediction 2 as written (>= v6_rl
  30.1, >= sft_v2 24.6) technically MET but the honest headline is the
  originals gap. Payload archived (v7a120_polish_x12).
- ARM E ERT (rules-v4 + gemini-pro on sealed ERT): pooled 33.30 vs rules-v3
  31.6 (+1.7, ~1.5 SE) — PREREG PREDICTION 1 MET. Notable cells: eggplant_
  keyboard 42.7, coke_wheel 33.7, pepsi 37.5.
- ARM D POLISH (composed, frozen routing): 35.18 — beats BOTH components
  (arm A 31.97, rules 34.78) but still below originals 36.1. Router wins:
  pepsi->rules +21.9, coke_wheel->rules +21.2, cube_on_plate KEEP +20.5 (vs
  rules 63.5), eggplant_keyboard->rules +9.7. Router losses: ramekin routed to
  rules 27.8 while arm A's kept-alien "white ramekin" phrase scored 47.9
  (-20.1 — the census gate sent the task to the weaker cell; kept-alien+color
  strikes again, echoing the val-8 keyboard lesson). Amendment-1 prediction
  (arm D > max components) holds on polish by +0.4 (inside noise) vs rules.
- Ops: pod5 polish interceptor fired correctly (repair double-roll killed <8min);
  pod5 untracked-collision wedge repaired (identical files rm'd, 2 commits
  rescued). Pod8 ERT landed early (07:35), duplicate nominal killed at ~07:40.
  Pods 5 + 8: 0 unpushed, payloads archived, podstate verified -> SHUTDOWN-READY.

## 2026-07-31 09:45 UTC — ARM E COMPLETE: rules_v4_nominal 37.18 BEATS ORIGINALS (36.1) — first sealed arm to clear no-rewrite; pod6 drift anchor validates
- **rules_v4_nominal = 37.18** (12x24x12): +2.4 over rules-v3 nominal (34.78,
  PREREG PREDICTION 2 MET), +1.1 over originals 36.1 — the first rewrite arm
  ever to beat the no-rewrite baseline on sealed, and above the old arm-C
  composite prediction (36.8). Combined with rules_v4_ert 33.30 (+1.7 over v3),
  rules-v4 sweeps both conditions vs v3: the training-data-mined conservatism
  + register triage + corpus-noun ladder GENERALIZED to sealed.
- Prediction 3 (gains concentrated OOV) checkable per-task later; the nominal
  divergences (kept ramekin/juice) land inside a winning arm.
- **Cross-pod drift: EXCLUDED.** Pod6 re-anchor cube_on_plate originals 68.06
  (n=72) vs pod-era reference 68.4 (n=576) — delta 0.3pp. Pod7's anchor comes
  with the repair leg.
- Ops: pod6 rebase-wedge (timeout-killed mid-rebase) repaired; a
  conflict-marker version of sealed_rules_v4.jsonl had reached origin —
  union-resolved and pushed clean (all rows preserved). v8 worker already
  rolling v8_adv step 20 (first v8 eval!). Arm E's frozen-routing arm D uses
  v3 cells per Amendment 5 — a v4-fallback composition variant would be a
  separate clearly-labeled post-hoc analysis, not the prereg arm D.

## 2026-07-31 12:40 UTC — CLOSING TABLE COMPLETE: arm A repair 30.12; arm D repair composition FAILS its prediction (28.33 < both components); rules-v4 alone is the best sealed method on BOTH conditions
- ARM A REPAIR (v7a-120 rewriting sealed ERT, 12x24x12): 30.12 pooled.
- ARM D REPAIR (composed, frozen routing): **28.33 — BELOW arm A alone (30.12)
  AND rules alone (31.57). Amendment-1/5 prediction REFUTED on repair** (it held
  on polish: 35.18 > 31.97/34.78). Anatomy of the failure — the census gate made
  systematically wrong repair calls: KEPT v7a's weak census-clean phrases
  (coke_wheel "tire" 6.2 vs rules 33.7; pepsi brand-dropped "blue can" 19.1 vs
  37.5 — the ledgered gate blind spot realized), and felled back AWAY from
  v7a's strong cells (orange_juice armA 31.9 -> rules 16.7; ramekin held KEEP
  correctly at 46.2). Output-census routing is not a reliable quality signal.
- **HEADLINE OF THE SEALED PROGRAM: rules-v4 + Gemini alone is the best method
  on both conditions** — ERT 33.30 (> armA-repair 30.12, > armD 28.33, > v3
  31.6) and nominal 37.18 (> originals 36.1 > armD-polish 35.18 > v3 34.78 >
  armA-polish 31.97). The mined-rules arm beat every learned-rewriter
  configuration on the sealed set. Per prereg discipline: reported as landed.
- Full table (sealed, pooled %): originals 36.1 | v7a-120 polish 31.97 /
  repair 30.12 | rules-v3 nominal 34.78 / ERT 31.6 | rules-v4 nominal 37.18 /
  ERT 33.30 | armD polish 35.18 / repair 28.33 | oracle 54.7-era ref.

## 2026-07-31 12:55 UTC — sealed program CLOSED operationally: pod7 anchor 68.06 == pod6 68.06 (CRN determinism, 0.0pp cross-pod delta); pods 5/7/8 all shutdown-ready
Pod7 drift anchor n=144 (its anchors filter matched 2 phrase-rows; pooled
identical to pod6's 72-ep cell — same seeds, same outcomes: the strongest
possible no-drift evidence). Pod7 handoff roller killed before any duplicate
v8 roll; 0 unpushed; podstate rebuilt. Fleet steady state reached: L40S
(v8 training, step ~32) + pod6 (v8 eval worker, rolling step-20 adv now).
Task #15 closed. Sealed table final as ledgered at 12:40.

## 2026-07-31 13:30 UTC — DATA BLEMISH: archived "polish x12" payload was actually 160 post-kill repair episodes
Pod5's wrapper rm'd seal_out.parquet at the top of its repair iteration; the
interceptor killed the roll ~160 episodes in; my post-kill archive therefore
captured those 160 repair rows, not the 3,456 polish rows (which were merged
into sealed_v7a120.jsonl BEFORE the rm — aggregates intact, per-task intact).
File renamed to v7a120_repair_partial160_pod5.parquet. Episode-level polish
raws are not materialized but are EXACTLY regenerable (CRN determinism) by
re-rolling ph_sealed_v7a120_polish.parquet if ever needed. Also surfaced:
anchors_x12 contains the ERT-VERBATIM passthrough baseline = 26.59 — the
repair condition's do-nothing floor (every repair arm clears it).

## 2026-07-31 13:50 UTC — scoreboard refresh confirms PREDICTION 3: rules-v4 gains are OOV-concentrated
Strata from the refreshed scoreboard: rules_v4_nominal OOV 30.9 and rules_v4_ert
OOV 30.7 are the two best non-oracle OOV cells ever measured (passthrough OOV
27.3/23.7; v3 29.6/26.8), while rules_v4_nominal in-vocab (46.0) actually sits
BELOW originals in-vocab (48.3) — the arm wins the pooled table on the OOV
side, exactly as prediction 3 filed. All three arm E predictions now confirmed.

## 2026-07-31 14:15 UTC — ARM F ROLLING: frozen Qwen + rules-v4, both conditions (pods 5+7)
Generation preflighted (frozen Qwen, gemini traces, rules-v4 prompt via the
same sft17k_rules_gen path as the v3-era rows). Preflight note (ledgered as
qualitative arm-F evidence): Qwen follows the protocol imperfectly where
gemini-pro was exact — emits "teal block" (violates the basic-shade rule AND
cube>block family override) and "black tire" (violates wheel>tire), spells
"coca-cola can" (non-corpus brand form). Well-formed throughout -> rolled
as generated. ERT leg pod5, nominal leg pod7 (started 14:11, ~10-14h each).
Amendment 10 predictions pending. Pod8 has nothing queued (shutdown blessed).

## 2026-07-31 15:20 UTC — ARM G staged + full self-driving wind-down armed (user stepping back)
Arm G (Claude Fable 5 + rules-v4) phrases generated by the agent applying the
protocol directly (register call, noun ladder vs appendix, adjective licenses,
geometry prepositions) — both conditions preflighted + pushed. Notable
independent-application divergences from arm E (ledgered as protocol-ambiguity
evidence): cube/plate ERT rebuilt minimal ("put the green cube on the yellow
plate" vs E keeping small/circular); juice ERT register-B call matched E
exactly ("set the citrus carton on the circular yellow plate").
CHAIN: pods 5/7 run armF leg -> armG leg -> push-with-retries -> verify
unpushed==0 -> runpodctl SELF-STOP (foreground per footgun memory); any
failure aborts the stop and leaves the pod for rescue. Pod8 self-stop issued
now (verified clean). End state: L40S (v8 train) + pod6 (v8 evals) only.
Landing order: armF legs ~00:00-04:00 UTC, armG legs ~10:00-14:00 UTC Aug 1,
then pods stop themselves.

## 2026-08-01 ~00:15 UTC — DESIGN PANEL: reasoning-then-rephrase GRPO (hypothetical; 4-agent workflow, no implementation)
Verdict (panel + critique, all claims file-cited): full reasoning-RL from cold
is dominated. Four measured blockers: (1) executor tournament at fixed rules —
thinking tiers span 0.6pp (pro+16k-think 31.6 ~ Qwen 31.5 ~ flash-no-think
31.1) with thinking_token_probe.json PROVING ~3k think tokens/case were spent;
(2) bare pair 27.8 vs 27.5 — 16k thinking does not rediscover the corpus law;
(3) the model's own reasoning prior is the measured weak link (self-trace
-7.3pp sealed); (4) CRITIQUE HEADLINE: the grip proxy is BLIND to reasoning
quality — step-0 inline-CoT vs gemini-trace pools score identically on the
proxy (rho .955/.962, oracle 32.2 both) while differing ~7pp at rollout; GRPO
cannot teach what the reward cannot see. Yield ranking: #1 distill-then-RL
(SFT on gemini-pro rules-v4 outputs/traces, then phrase-only GRPO); #2
rules-in-context + GRPO execution-fidelity (panel more skeptical than the
session's intuition: frozen +0.5pp prior, proxy doesn't measure fidelity);
#3 masked frozen-think 100-step diagnostic (~0pp expected); #4 full
reasoning-RL only as #1's continuation. Mechanics-if-built preserved in the
workflow output (two-segment sampling with forced "1. ", Dr.GRPO constant
normalizer vs 25x length dilution, segment-split KL beta_phrase .15 /
beta_think ~0-.05, think budget 256, degeneracy tripwires; gen 70s->~400s).
BONUS FINDING: an 86-step inline-reasoning RL pilot sits BANKED and UNEVALUATED
(results/checkpoints/phase2_inline_pilot, EXPERIMENT.md:45) — 2 checkpoint
evals through the tag-free path would answer this question with data.

## 2026-08-01 01:10 UTC — v8 ADVERSARIAL STEP-0 ANCHOR: 37.76 == steps 30/40/50; greedy outputs 7/8 IDENTICAL to base Qwen
The tag-free base-Qwen anchor (n=384, 2-rep) lands at exactly the checkpoint
band value. Phrase-level check: v8-50's greedy adversarial rewrites are
byte-identical to base Qwen's on 7/8 tasks (sole diff: carrot_on_wheel drops
the pick-up clause). CONCLUSION: v8 has produced no adversarial-side greedy
behavior change beyond one clause edit — the v7f pattern, now measured with a
protocol-matched anchor from birth. Step-20's 38.54 was transient. The
"RL trains rephrasing" claim on the adversarial side is now refuted in TWO
independent forks; polish anchor (rolling, ~02:15 UTC) adjudicates the rest.
Reward-side corroboration: win-rate ~3%, greedy proxy reward ~flat.

## 2026-08-01 03:30 UTC — BOTH ANCHORS IN: v8 is flat-to-negative on BOTH conditions; arm F ERT refutes its prediction (30.06)
- **v8 polish step-0 (base Qwen, tag-free, 2-rep) = 44.79** — ABOVE every v8
  checkpoint (41.93/44.01/41.67/42.71). Combined with adv step-0 = 37.76
  (7/8 phrases identical at step 50): v8 has improved NOTHING over its init;
  polish sits 0.8-3.1pp BELOW it. The earlier "44.01 above all baselines" read
  was an anchor artifact (tagged rep-1 references) — the user's insistence on
  true step-0 anchors was decisive a second time. Note in itself: tag-free
  base Qwen POLISH = 44.79 > passthrough originals 40.6 (+4.2) — the frozen
  model with a Gemini trace is already a strong polisher on val-8.
- Complete RL arc for the writeup: beta=0.05 (v7a) moves behavior but overfits
  val-8 (sealed inversion) and Goodharts (v7f); beta=0.15 (v8) pins KL~0.002
  and moves nothing (win-rate ~3%, proxy flat, outputs byte-frozen). The grip
  proxy is blind to what matters (panel finding) — with a strong KL anchor
  there is no usable gradient left. THE RULES ARE THE ARTIFACT.
- **ARM F ERT = 30.06** (frozen Qwen + rules-v4 on adversarial): REFUTES
  Amendment-10 prediction 1 (>= v3-twin 31.48); also below no-rules 30.96.
  The richer v4 protocol HURT the weak executor (-1.4 vs v3 rules) — protocol
  value inverts under unfaithful execution; the preflighted violations were
  the mechanism. Executor-fidelity is a first-class variable.
- RECOMMENDATION for the user (not executed): stop v8 training — 60 steps of
  measured no-movement at pinned KL; continuing spends GPU on a refuted
  configuration. Arm G (Claude+rules ERT) rolling on pod5 since 03:15;
  arm F nominal finishing on pod7.

## 2026-08-01 ~04:10 UTC — CORRECTION (user-caught): v7a ran at beta=0.15, not 0.05
run_arm_v7.sh line 56: --beta 0.15. The 0.05 settings were the v7b/c sweep
forks (which concluded "beta not the bottleneck"). The 03:30 entry's RL arc is
therefore WRONG about beta being the moved/flat separator. Corrected analysis:
the separator is the REWARD — v7a (moved) trained on the c4b blend w=0.25
(verifier-logit component, F=16, C=1); every grip-pure C-club run (v7f tagged
beta=.05; v8 untagged beta=.15) is flat. Tags are exonerated by v7f's tagged
flatness. Grip-pure proxy ranks originals above the whole candidate pool
(orig +0.617 vs best16 -0.22) and is quality-blind — no trainable signal.
Discriminating experiment specified (not run): v8b = v8 config + blend reward.
USER DECISION: v8 continues training as-is for now.

## 2026-08-01 04:55 UTC — ARM F COMPLETE: frozen Qwen + rules-v4 = 27.81 nominal / 30.06 ERT — the executor-fidelity result, brutal edition
Nominal 27.81: -8.3 vs originals (36.08), -9.4 vs the SAME RULES under
gemini-pro (37.18). On good inputs the weak executor's unfaithful rewriting is
actively destructive — license-to-rewrite without capacity-to-comply is worse
than no rules on both conditions (ERT 30.06 < no-rules 30.96). Full v4 ladder:
gemini-pro 37.18/33.30 >> frozen Qwen 27.81/30.06 — ~9pp executor swing on
identical protocol text. Amendment-10 predictions: (1) REFUTED, (3) CONFIRMED
(F << E). Claude arms (G) now adjudicate whether a strong-but-different
executor tracks gemini (prediction G ~ E) — both legs rolling (03:15 / 04:44).
Conflict-marker residue union-resolved from sealed_rules_v4_qwen.jsonl.

## 2026-08-01 16:10 UTC — ARM G ERT: Claude Fable + rules-v4 = 34.26 — best adversarial arm ever measured
Executor ladder on IDENTICAL v4 text (ERT): Claude 34.26 >= gemini-pro 33.30
(+0.96, ~1 SE) >> frozen Qwen 30.06. Amendment-11 predictions 1 (>= claude-v3
30.99) and 3 (tracks E; fidelity binds, not scale) CONFIRMED. Claude+v4 also
recovers 7.7pp of the ERT gap over passthrough 26.59 — the deepest repair on
record. Pod5 chain proceeding: arm H (bare Claude) rolling since 16:03, then
self-stop. Arm G nominal ~1h out on pod7.

## 2026-08-01 18:30 UTC — ARM G COMPLETE: Claude + rules-v4 = 37.73 nominal / 34.26 ERT — best arm on BOTH conditions; pod7 retired
Full executor ladder on identical v4 text:
  Claude Fable   37.73 / 34.26   (best both columns; nominal beats originals +1.65)
  Gemini-Pro     37.18 / 33.30
  frozen Qwen    27.81 / 30.06
Claude~Gemini within ~1 SE on both conditions; the cliff is executor fidelity,
not which strong model. Amendment-11 predictions 1+3 confirmed; OOV-signature
(p2) checked at scoreboard refresh. Ops: pod7's push wedge (append-append
rebase + stale arm-F rebase state) resolved by reset-to-origin + recompute-
from-raw (armG_out.parquet); nominal row recomputed, x12 archived, 0 unpushed;
pod7 STOPPED (verified-clean manual stop after its runner correctly aborted
self-stop on push failure — the safety gate worked as designed). Remaining:
arm H on pod5 (~05:00 UTC), then pod5 self-stops; L40S+pod6 on v8.

## 2026-08-02 ~02:30 UTC — SECOND CORRECTION to moved-vs-flat (user-driven): the REWARD is exonerated; the CORPUS is the prime suspect
User: the grip reward passed the fine-discrimination exam (~ensemble at CxF=40).
Decisive check: v7a's proxy vals have the SAME structure as v8's — orig ~0.62
above the whole candidate pool, greedy -1.72 -> -1.0, win 2.5-10% — identical
telemetry, yet v7a's rollouts moved (+15pp polish IV) where v7f/v8 are flat.
So neither beta (first correction) nor reward form separates them. Remaining
candidates, now ranked: (1) TRAINING CORPUS — v7a sampled parents from the
full 2,000 mixed set (1,361 unique instructions); v7f/v8 train on the 809
CLUB parents, the most-canonical least-improvable instruction population (the
conservatism law says precisely these have nothing to teach a canonicalizer);
(2) tags (secondary; v7f had tags and was still flat -> weak). Testable later
by a club-vs-mixed parent A/B at fixed reward.
COVERAGE MEASUREMENT (user request): the 2,000 traces span 2,000 UNIQUE
episodes in ~1,809 visual scene-clusters (dhash hamming<=6; mean 1.1 eps/
cluster, max 7) — i.e. our sample has ~no scene sharing, covering ~3.3% of the
60k episodes directly. CAVEAT: the 2k sample was drawn to be diverse, so this
UNDERSTATES the 60k's true scene-clustering; the real cluster count needs the
60k initial frames (first output of any future corpus-prep job).

## 2026-08-02 ~03:30 UTC — v9 LAUNCHED: v8 fork + group-level replay + 16 contexts (user-directed)
Implementation (phase2_train.py): ReplayBuffer stores whole scored groups
(context + candidates + deterministic rewards + fresh-time mean logprobs);
per update, 16 replayed groups mix with 16 fresh; replayed candidates train
under a PPO-clipped surrogate on the mean-logp ratio (eps 0.2); window 50
steps, max reuse 6, never replaying a parent present in the fresh batch;
buffer persists in latest/ (resumable). Fresh contexts doubled 8->16 (across-
context variance halved — user's choice over 32 rephrases, correct at equal
cost). Reward UNCHANGED (grip-pure C=10 F=4): user cites the CxF=40
discrimination exam; the v7a-val-structure check exonerated reward form.
Warm fork from v8@latest (~step 113 weights; v8 frozen — curves through
adv-110/pol-100; the in-flight pol-110 roll was lost in the worker handoff,
ledgered). Expected ~88 min/step, 32 effective groups (~2.4x throughput,
2x corpus coverage rate). Eval: pod6 v9worker (stride 10, tag-free, both
conditions). Open hypothesis this run also probes: if v9 moves where v8
didn't, throughput/diversity mattered; if flat, the club-corpus explanation
strengthens further.

## 2026-08-02 ~04:15 UTC — CONFOUND TRIANGLE ledgered; user's tag-theory adopted as v9's operative hypothesis; pod7 on the v7a-340 legs
Verified from v8's boot log: "parents 2000 -> 809" — the C=10 club reward
filter subsets training parents to 809 rows / 213 instructions for v7f/v8/v9
(v7a trained on all 2000 / 1361 instructions). So v7a-vs-v8 changed BOTH
corpus-subset and tags (beta equal); v7a-vs-v7f changed corpus and beta;
v7f-vs-v8 changed tags and beta — no pair isolates one variable. USER THEORY
(operative for v9): the point of v8 was removing tags, and tag-free training
plausibly needs LONGER to learn regime inference without tags; v9's replay
(2.4x effective updates) is the accelerated test. If v9 bends, tag-free was
slow-but-real; if flat, the corpus-narrowing suspect strengthens (future
single-variable probes: mixed-corpus tag-free via C-fallback reward, or
club+tags at beta .15).
Ops: pod7 restored (new addr; tmux+libvulkan reinstalled, restore.sh patched)
and queued on BOTH v7a-340 legs (waits for phrases from the pod5 gen chain;
pod5/pod8 stage-2 waiters will grep-skip when pod7 lands them first).
v9 chart = v8 chart (continuation fork at step 113, purple/orange v9 strata).

## 2026-08-02 ~05:00 UTC — v9 REDEFINED + relaunched: adaptive CxF over ALL 2,000 parents (user-caught oversight)
User caught that v7f/v8/v9 silently trained on 809/2000 parents (the C=10 club
filter). v9 rebuilt (~1 step discarded): --adaptive-contexts keeps ALL 2,000
parents; per parent, C = available same-instruction club contexts (<=10) and
frames scale to hold C*F ~= 40 (cap 16/context) — sparse parents run e.g.
C=1xF=16 (v7a's original geometry) while club parents keep C=10xF=4. Reward
budget per parent ~constant. Replay + 16 contexts + tag-free + beta .15
unchanged. 60k-EPISODE EXPANSION explicitly gated: traces exist ONLY for the
2,000 parents; the named next data project is scene-cluster trace generation
(+club/chunk-stats prep) before any parent beyond the 2k can be used.

## 2026-08-02 ~15:20 UTC — USER DECISION: Qwen reasoning tested at rules-v4 ONLY (no bare+think control)
The Qwen row mixes decoding modes by design: bare (no think), v3 (no think),
v4 (no think), v4+think (the new pair). INTERPRETATION CAVEAT for the writeup:
a positive v4+think delta is consistent with BOTH "deliberation improves
protocol fidelity" (the hypothesis) and "deliberation improves Qwen's
rewriting generally"; the disambiguating bare+think control was offered and
declined as unnecessary. Report the v4+think delta as suggestive of the
fidelity mechanism, not as isolating it. Termination scaffolding (FINAL:
marker + repetition_penalty, Amendment 15 note) remains part of the cell's arm
definition — the 9B cannot terminate open-form reasoning unaided.

## 2026-08-02 21:05 UTC — FLEET-WIDE OUTAGE: all five machines down (account-level, suspected billing)
Raw TCP probes (bypassing ssh) to every pod's port: connection refused on
pod5/pod6/pod7/L40S, timeout on pod8, while GitHub and runpod.io respond
normally from the same host — i.e. the ports are not listening, not a local
network fault. Five machines across regions/providers going dark together =
account-level event (credits/billing most likely).
Last push 18:39 UTC (v9 step_0014 adapter). SAFE IN GIT: v9 through step 14,
all landed sealed cells (claude_bare 29.14, qwen_bare_ert 24.19,
rules_v3_claude_nominal 34.69, plus arms A/E/F/G).
INTERRUPTED (progress lost, data intact on /workspace which survives a stop):
qwen_v4_think_ert ~57%+, rules_v3_qwen_nominal ~52%+, v7a340_polish ~34%+;
v7a340_repair never started.
RESTART RECIPE (when pods return): (1) bash /workspace/.podstate/restore.sh on
each (now installs tmux+libvulkan1+libegl1 and the ICD manifest); (2) rm -f
data/cellq_out.parquet data/armH2_out.parquet on each pod BEFORE relaunching —
the rollout script APPENDS to an existing --out parquet (n != 3456 is the
contamination tell); (3) relaunch the three interrupted legs + v7a340_repair
via scripts/roll_cell_queue.sh; (4) restart v9 from step 14 (the per-step
latest/ save, committed 0caefc0, deploys on this restart -> future
interruptions cost <=1 step); (5) re-arm watchers.
Pod-dependent monitors stopped to avoid hourly noise; will re-arm at restart.

## 2026-08-03 ~00:40 UTC — FLEET RECOVERY (pods 5/6/8 back; pod7 + L40S migrating)
Restored on all three: podstate, tmux, libvulkan1/libegl1, ICD manifest to
/usr/share/vulkan/icd.d (the loader-path variant), repo synced, and — per the
outage recipe — ALL stale out-parquets deleted before relaunch (cellq_out,
armH2_out, armF_out, armG_out, dev8_g_out) so no roll appends onto interrupted
data. Queues: pod5 = qwen_v4_think_ert (restarted from 0); pod8 = v3_qwen_
nominal -> v7a340_polish -> v7a340_repair (pod7's legs reassigned since pod7 is
still migrating); pod6 = v9 eval worker (stride 7). v9 training resumes when
the L40S returns — from step 14 (the last synced adapter), and that restart
deploys the per-step latest/ save so future interruptions cost <=1 step.

## 2026-08-03 ~14:10 UTC — TWO CELLS LANDED: v7a-340 polish 31.19 (transfer failure replicates); qwen+think ERT 19.56 INVALID (parser leakage)
- **v7a340_polish = 31.19** (IV 44.4 / OOV 21.7) vs arm A (v7a-120) 31.97 and
  originals 36.08. Amendment-17 primary question ANSWERED: 220 extra RL steps
  changed sealed polish by -0.8 (noise). The val-8 -> sealed inversion is a
  property of the RL policy family, not of an unlucky checkpoint: v7a's val-8
  polish rose 44.3 -> ... while sealed sat ~31-32 at BOTH checkpoints. Strata
  match arm A's shape (IV-heavy, OOV floor).
- **rules_v4_qwen_think_ert = 19.56 — DO NOT REPORT AS A RESULT.** Preflight
  audit of the rolled phrases: 5/12 are PROMPT FRAGMENTS, not instructions
  ("' followed by\".", "followed by the rewritten instruction. No commentary
  after.", "'\". (From final instruction)."). The FINAL-marker parser
  (Amendment 15 note) split on the marker's own restatement inside the model's
  deliberation, and the >=3-word fallback guard passed the garbage. The cell
  measures a parsing bug, not deliberation. Correct reading of the underlying
  question: UNRESOLVED. The 7 well-formed rows are also mixed (2 are verbatim
  ERT passthrough — the model reasoned and then declined to rewrite).
  Not charted; ledgered as void with cause.

## 2026-08-03 ~15:40 UTC — POST-RECOVERY AUDIT: v9 was dead ~35h (empty tmux shells); fleet re-verified process-by-process
Discoveries on the 15:00 pass (24h since last sweep):
- L40S: the 00:40 recovery relaunch created tmux sessions whose commands died
  instantly — the migration wiped /root, dangling .venv/.venv-gen symlinks
  ("bash: line 1 ... No such file or directory" appended to train.log). My
  "V9 LAUNCHED" check verified sessions, not processes; v9 sat dead ~35h.
  The pre-death run had completed step 16 but latest/ held step 14 (per-step
  latest, 0caefc0, was NOT active in that boot) -> steps 15-16 retrain (~1.7h).
- pod6: v9 eval worker was the pre-cadence-change instance (stride 10) — rolled
  step 10 then idled forever; step 14 (7-cadence) never matched stride 10.
- pod8: rules_v3_qwen_nominal ROLL FAIL at 08:37 after 7.9h (~92%, silent
  host-level kill #2 on this pod); queue advanced through a fast-failed/killed
  polish attempt to v7a340_repair, rolling clean since 09:40 (rm-per-attempt in
  roll_cell_queue.sh line 20 rules out append contamination; n=2220@15:13
  matches fresh rate exactly).
Actions: L40S venvs rebuilt (rebuild_both_venvs.sh, both import-verified); v9
resumed from latest/ step 14 with replay.pt intact, GPU 84%; per-step latest
NOW live (future deaths cost <=1 step); v9sync mirror recreated (pack+push per
new snapshot, idempotent via manifest). pod6 worker restarted on pulled
stride-7 script. rules_v3_qwen_nominal re-tasked to pod5 (fresh roll 15:01,
pod8 has now silently killed two long rolls). pod7: ALL-CELLS-DONE, unpushed=0,
data/ parquets pulled to results/pod_payloads/pod7_data_payload_20260803.tgz
-> STOP-READY. Consolidated watcher armed (repair/v3nom landings + v9 + pod6
worker liveness, 3-strike flake tolerance).
LESSON (now memory): relaunch verification = pgrep + log mtime + one progress
line, never tmux session existence. Session-exists-but-empty is the signature
of a dangling-venv relaunch.

## 2026-08-03 ~16:40 UTC — GRPO IMPLEMENTATION AUDIT (user-requested): solo pass + blind second reviewer, reconciled
VERDICT: no math-corrupting bug in the live v9 path. Verified correct: group-
standardized advantages (zero-variance-safe); signed full-group GRPO; PPO-clip
replay surrogate (textbook form both advantage signs; old_lp = true behavior-
policy logp, captured pre-optimizer-step); gen/update/replay prompt parity
EXACT (source mutated before storage; PNG roundtrip lossless); lora_dropout=0.0
live-verified (train/eval forward parity -> clean ratios); reward<->survivor
alignment through the club fan-out offset walk; logprob/KL indexing in both
single and left-pad batched paths (+ runtime parity check); replay buffer
prune/exclude/persist; atomic checkpoint swaps; torch+CUDA RNG restored on
resume; deterministic context schedule fast-forward.
FINDINGS (all fixes committed THIS entry, inert until next unavoidable restart):
1. Mid-window restart dropped in-flight accum grads (optimizer steps every 6th
   step; p.grad never checkpointed): each restart lost up to 5 steps' gradient
   — "restart costs <=1 step" was overclaimed (v9's 2 restarts each hit this).
   FIX: latest/ now persists grads.pt (~58MB bf16) + restores on resume.
2. numpy + python-random streams were neither seeded nor persisted (tier
   draws, dropout coins, club picks): restarts re-randomized them — unbiased
   but not stream-exact. FIX: rng.pt now carries both; fresh boots seed all 3.
3. All-failed step at an accum boundary returned before step()/zero_grad(),
   leaking a whole window into the next (double-size late update). Near-zero
   v9 exposure (needs 16/16 context failures + empty replay). FIX: boundary
   flush in the empty-items path.
4. DOC CORRECTION (since v7e): score_phrases re-expands EVERY context row by
   episode_index, so club contexts score at F=4, not the "club singles" the
   ledger claimed — C=10 was really 10 contexts x 4 frames = 40 frame-evals.
   The CxF=40 exam validated this ACTUAL reward (deployed = validated; within-
   group comparability never broken). v9's sparse-parent "deficit -> extra
   parent frames" compensation was a provable NO-OP under that expansion
   (duplicate rows re-expand to the parent's same frames; duplicates cancel in
   the mean) while paying ~4 frame-evals/row. FIX: dead compensation removed
   (CRN-safe: server draws are fixed per (k,seed,tau) config, not row-position)
   -> sparse parents score parent-only (v7a's C=1 F=4 shape, honest
   n_reward_contexts); frees ~36 wasted frame-evals per sparse-parent context
   AFTER next restart (should cut v9 step time noticeably).
5. blend_rewards: ANY NaN grip silently flips the whole group to pure logit
   rank. v9 (verifier mode) emits real grips — low risk — but the swap was
   invisible. FIX: blend_fallbacks counter in the step record.
Design notes (correct as-built, now ledgered): replay ratio is exp(delta token-
MEAN logp) — per-token geometric-mean ratio, so clip 0.2 is intentionally
tighter than a sequence-ratio clip; logged kl averages fresh+replayed
candidates (v8-vs-v9 per-step KL comparisons carry a composition caveat);
objective = mean over contributing candidates / grad_accum_groups (documented
deviation); two same-instruction replayed groups may co-occur (different
contexts; fresh-vs-replay collisions are excluded).

## 2026-08-03 ~17:30 UTC — v9b DECISION (user-approved): accum 6 -> 2 at the step-17 boundary restart
Context: user challenged the 6-step accumulation window ("gradients on stale
parameters?"). Resolution ledgered for the paper: the window is NOT stale-
gradient async (policy frozen across the window; gradients computed at and
applied to the same theta; = standard GRPO rollout-batch assembly, mu=1 —
nonstandard only in that the sim-reward delivers the batch in ~50min
installments). The REAL miscalibration: accum 6 was sized for v8's 128-cand
steps (~768/update); v9's 512-cand steps made it ~3,072/update at one update
per ~5h. v9b = accum 2: ~1,024 cands/update (calibrated scale preserved -> lr
7e-6, beta 0.15, KL-abort unchanged) at 3x policy-iteration cadence.
CUTOVER PLAN (user: "do the restart at the boundary as planned"): automated
watcher restarts the trainer the moment "latest saved at step 18" appears
(post-step-17 optimizer flush -> zero gradient loss; fallback trigger step 24).
The same cutover deploys the full audit package: grads.pt + np/py RNG persist
(lossless restarts), empty-boundary flush, no-op deficit removal (expect step
time ~50 -> ~30min; reward values PROVABLY unchanged), blend_fallbacks
telemetry. Ledger the executed cutover + KL-telemetry break annotation as
v9b@18 when it fires. NOTE for curve reading: per-step KL after v9b reflects
2-step windows (was 6) — per-update drift comparable only after x3 scaling;
the v8_progress chart's KL panel needs a v9b marker at 18.

## 2026-08-03 ~17:50 UTC — v9b AMENDED: accum 1 (user: "I am comfortable moving our accum to 1 ... I really want to speed up training")
One optimizer update per step: 512-candidate updates at every ~50min step
(~30min once the deficit removal lands) = ~6x the policy-iteration rate of the
accum-6 regime, ~2x the accum-2 plan. Update batch halves vs the calibrated
~1,024 — noisier updates, accepted for speed; lr 7e-6 / beta 0.15 / KL-abort
1.2 unchanged (abort is an absolute drift ceiling; still valid). Monitoring
note: logged per-step kl now reflects a full update every step (absolute
divergence will climb ~6x faster per step than v9-accum-6 telemetry — annotate
the chart at 18); eval stride 7 now samples every 7 UPDATES (was ~1.2) — drift
between eval points is correspondingly larger; watch the val-8 curves for
v7f-style polish drift, beta is the lever if it appears. grads.pt becomes
vestigial under accum 1 (never mid-window state; file simply not written).
Cutover mechanics unchanged: same watcher, same step-17 zero-loss boundary.

## 2026-08-03 ~18:40 UTC — v9b@18 EXECUTED: zero-loss boundary cutover, verified live
Watcher fired on "latest saved at step 18" (= step-17 optimizer flush under the
old accum-6 build -> no in-flight gradient existed to lose). Verification per
the process-discipline: "resumed at step 18" printed; --grad-accum-groups 1 on
the LIVE cmdline; [replay] buffer active: 286 groups loaded; tqdm 18/1000 in
flight; grads.pt absent (expected at a boundary); GPU 100%. v9b regime from
step 18: one 512-candidate optimizer update per step (~6x accum-6 iteration
rate), deficit-removal active (expect step time ~50 -> ~30min; first timed
step will confirm), np/py RNG + grads persistence live (restarts now lossless
and stream-exact). KL telemetry break annotated at 18 (per-step kl now
reflects a full update every step). Chart marker for v8_progress: v9b@18.

## 2026-08-03 ~23:20 UTC — Amendment 19 registered: rephrase-robustness eval; generation launching on pod5
K=16 natural rephrases/sealed task (gemini-pro-latest 2x8 temp 1.0, image, no
trace, knowledge-free prompt), roll layouts 0-11 x 1 rep on BOTH pi0 variants
(rephrase-augmented + plain finetune) queued overnight on pod5 behind v3nom.
Layout subset RECORDED (0-11); episode_id per row enables the subset-variance
check before committing to layouts 12-23. See PREREG Amendment 19.

## 2026-08-04 ~04:40 UTC — LADDER COMPLETE: rules_v3_qwen_nominal 32.23 (IV 34.9/OOV 30.3); robustness queue rolling
Final unfilled cell landed (pod5 DONE-UNPUSHED wedge -> recomputed from raw x12
locally, the standing recovery). Executor-fidelity pattern REPLICATES on the
nominal side: Qwen v3-nominal 32.23 >> Qwen v4-nominal 27.81 (+4.4 — richer
protocol hurts the weak executor in BOTH conditions), and both sit below
originals 36.08 (corpus conservatism: with a weak executor, rewriting originals
is net harm). Strata signature extreme: IV 34.9 vs originals-IV 48.3 (-13.4
damage where the corpus already wins) against OOV 30.3 vs 27.3 (+3.0). Both
condition-ladder charts + scoreboard refreshed — every planned cell now filled.
Rephrase-robustness queue opened on schedule: pi0rephrase rolling since 04:25.

## 2026-08-04 ~08:45 UTC — v9 VERDICT (early, via v9b cadence): greedy behavior FROZEN through step 35
POL 44.27 with byte-identical strata at steps 10/14/21/28/35; ADV frozen at
36.46 since 28 (drift band 36.5-40.1, anchor 37.8). The span covers replay
(16 groups/update), the accum 6->1 cutover, ~17 single-step 512-candidate
updates, and all-2000-parent coverage. Same conclusion as v8, now replicated
under 4x batch, 10x update cadence, replay, and the full corpus: tag-free GRPO
at beta=0.15 sharpens the sampling distribution around a mode it never
displaces. Program implication: the scaffold few-shots anchor the mode (see
the 08-03 prompt audit); the user's bare-template v10 design — no few-shots,
step-0 = the sealed qwen_bare_ert cell 24.19 exactly — is the indicated next
run. Step-42 point (~14:30 UTC) is the formality; v10 decision is the user's.

## 2026-08-04 ~13:50 UTC — REPHRASE-ROBUSTNESS ARM 1 (pi0rephrase): 26.30 — natural human variation is AS DAMAGING as adversarial rewording
Layout-matched (0-11) baselines from anchors raw: originals 34.38, ERT
passthrough 26.85. The K=16 natural-rephrase distribution scores 26.30 —
REGISTERED PREDICTION (a) REFUTED: natural variation does NOT sit between
canonical and adversarial; it lands AT the adversarial floor (-8.1 vs
originals). Per-rephrase distribution (24 eps each): median 16.7%, quartiles
0 -> 41.7, >25% of phrasings at literally ZERO while others hit 83-100 —
extreme phrase sensitivity inside plainly-natural wording ("Move the carrot to
the ramekin." = 0/24 on a task whose nominal scores ~28). Oddities: question
forms OUTSCORE statements (35.2 vs 25.9, n=9); "Drop the eggplant..." variants
thrive (91.7). Task split: cube_on_plate 72.4 / eggplant_on_sponge 78.1 robust;
plate_on_cube 0.5, coke_on_wheel 1.0 shattered. Program implication: the
rules/rewrite pipeline now has a NATURAL-variation deployment story, not just
an adversarial one — repairing ordinary human phrasing back toward 34 is the
use case. Arm 2 (pi0base) rolling since 13:25 -> lands ~22:30 (pace 4.4/min);
predictions (b) (larger gap for pi0base) and (c) (spread >> executor gap) then.

## 2026-08-04 ~15:10 UTC — v9 verdict FINAL: pol 44.27 x6 identical (steps 10-42); adv trendless (36.5-40.1 band)

## 2026-08-04 ~17:30 UTC — USER DECISION: v9 CONTINUES (long-horizon test)
User: "It may begin to move. Forty steps is still a tiny amount." The 10-42
freeze verdict stands as a description of steps <=42 only; the run now tests
the tag-free-needs-longer theory at ~50 steps/day (29min/step, one 512-cand
update each). Movement alarm to be added at next watcher re-arm: fire if any
new pol point != 44.27 or adv exits [35, 41] — so first sign of movement pings
immediately. v10-bare stays designed-and-shelved pending user call.

## 2026-08-04 ~19:55 UTC — v9c@58 VERIFIED + Amendment 20 launched
v9c (8 fresh + 8 replay, 256-cand updates): first restart at step 57 silently
kept the OLD 16/16 config — the in-script `git pull || true` swallowed a failed
pull against the degraded GitHub link and relaunched from the stale script.
LESSON (pattern update): deploy-at-restart requires verifying the ON-DISK
script BEFORE arming the restarter, and the LIVE cmdline after — a push is not
a deploy. Redeployed at the step-58 boundary; live flags verified 8/8.
Projected ~18min/step, ~80 updates/day. Telemetry break marker: v9c@58 (256-
cand kl/grip composition).
Amendment 20 in flight: 192 rules-v4 Gemini rewrites of the natural rephrases
generating on pod5 (greedy, family wrap); roll gated behind pi0base -> arm
rephrase16_rulesv4_gemini overnight. Registered prediction: pooled >= 32.

## 2026-08-04 ~20:30 UTC — v10 SPEC (user design, not launched): RL on CoVer-minus-rules
The de-ruled CoVer scaffold (persona + image + trace + format contract kept;
ALL phrasing guidelines AND the four few-shot examples removed; text frozen in
this session, to be committed as cover_prompt variant on build) becomes the
TRAINING prompt for the next RL run. Properties:
(1) Minimal delta from the v8/v9 lineage — the ONLY change is removed
knowledge/anchoring, so if v10 moves where v9 froze, the few-shot/rules
anchoring hypothesis is confirmed CAUSALLY; if v10 also freezes, the anchor is
deeper than the prompt (base-model corpus conservatism).
(2) The frozen-baseline re-eval the user proposed the same evening IS the
run's step-0: frozen Qwen through cover-minus-rules, rolled on sealed as
pre-registered anchors (both conditions) BEFORE training — the baseline cell
and the RL anchor are the same artifact.
(3) Higher-entropy step-0 output (no few-shot register anchor) = more
explorable within-group ranking signal for GRPO — the mechanism by which
removal could unfreeze training.
Pipeline otherwise inherits v9c economics (8 fresh + 8 replay, accum 1,
~18min/step); COLD start from base Qwen (fresh LoRA — warm-forking v9 would
carry the old prompt's conditioning). Launch, anchors, and amendment on user
go; v9c continues meanwhile.
