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
