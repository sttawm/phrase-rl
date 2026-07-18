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

### TASK-TIER REGISTRY v3 — VAL-11 / TEST-12 (user decision 2026-07-16, final)

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
