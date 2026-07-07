# phrase-rl: Advantage-Weighted Rephrase Tuning Against a Frozen VLA

**Status:** Phase 0a done; **Phase 0b PASSED (GO)** 2026-07-07 — median split-half ρ≈0.95 at K=16, signal share ≈0.98, no length-hack axis, oracle gain 31–33%; rephrase-FT checkpoint ~25% less phrase-sensitive than plain (see results/phase0b/metrics.json). τ finding: discriminability collapses for τ<0.25 (clean-action end) — concentrate scoring draws at τ≥0.25 in training. Next: Phase 1 (teacher data + SFT) and/or 0c rollout sensitivity. **Last updated:** 2026-07-07.

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
| Frozen trace/teacher VLM | Gemini 3.1 Pro (batch API) | Replaces CoVer's GPT-4o (discontinued). Generates cached reasoning traces + 16-rephrase teacher lists. **Trace is fixed CoVer-parity machinery, not an ablation axis** (decided 2026-07-07): CoVer's VLM does structured scene reasoning + rephrases at boot time and reports no trace ablation; we mirror that — traces cached for training, fresh frontier call at boot for deployment/Phase 4. The tested contrast is *RL over trace-conditioned generation* (eval matrix: base/SFT vs advantage-tuned, all trace-conditioned) |
| Trainable phrase model | `Qwen/Qwen3.5-9B` (natively multimodal, Feb 2026) | LoRA. **Thinking mode DISABLED everywhere** (`enable_thinking=False` in generation and, later, training): Qwen3.5 thinks by default, which is ~4× slower, leaks numbered lines from the reasoning block into parsed candidates, and would contaminate the advantage-weighted log-prob objective with reasoning tokens. All 0b/0c/eval numbers are no-think numbers. Revisit later: a *minimal/budgeted* thinking variant as an ablation (see Ablations). Fallback model `Qwen/Qwen3-VL-8B-Instruct`; 27B/32B only as post-signal scale-up (32B ≈ 73GB FP16, ~4× slower generation) |
| Optional baseline | CoVer verifier (`cover_verifier_bridge.pt`, ~312MB) | Test-time selection over our candidates; separates "better candidates" from "better selection" |

π0.5 note: our reward/eval model is π0 (Bridge/SIMPLER). π0.5 only appears in CoVer's PolaRiS extension; all machinery here (flow-matching scoring, CRN) transfers unchanged if we later swap it in.

## Data

BridgeV2 tuples `x = (o_t, l, a*_t)` (image, original instruction, ground-truth action chunk). For each, cache one Gemini call: reasoning trace `r` + 16 teacher rephrases. Training context for the phrase model: `c = (o_t, l, r)`.

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

**Primary path is advantage-weighted tuning directly from base Qwen3.5 — no teacher distillation.** Rationale (2026-07-07): base Qwen already captures most of the reward headroom (0b: Qwen best-of-32 oracle gain 20.5% vs Gemini 26.7% — the teacher adds only ~6 pts of ceiling), and skipping the teacher makes the headline claim cleaner ("a 9B model with *no* frontier supervision, only the policy's reward, out-phrases the frontier model"). Teacher-SFT warm-start becomes an ablation (does a higher-ceiling start beat RL-from-base?).

1. **Advantage-weighted tuning (primary):** generate 16 candidates per context from the *current* model (list prompt for diversity), score with frozen π0 (rephrase-FT ckpt) + CRN, update under the single-phrase prompt:
   `L = -Σ_i max(A_i, 0) · (1/|y_i|) · log π_θ(y_i | c, p_single) + β·KL-to-base anchor + λ·list-format aux`
   **Token-mean normalization** (the `1/|y_i|`): raw summed log-prob scales gradient contribution with phrase length, so long positive-advantage phrases would dominate updates. Positive-only advantages first; signed advantages as ablation. KL anchor to base Qwen (or to the SFT model when warm-started). This is advantage-weighted rephrase tuning, not exact GRPO.
2. **Log during training, not just eval:** duplicate rate, mean pairwise embedding similarity, object/target preservation, generic-phrase rate, format-failure rate. Known failure mode: collapse onto Bridge template phrasing ("put X in Y") — the rephrase-FT reward checkpoint was trained on a fixed paraphrase dictionary the optimizer may simply rediscover.
3. **Teacher-SFT warm-start (ablation, not primary):** split each Gemini 16-list into 16 single-phrase examples `(c → y_i)`, SFT Qwen, *then* advantage-tune. Tests whether a higher-ceiling starting distribution beats RL-from-base. Teacher data already collected (Phase 1).

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

## Offline eval matrix (held-out contexts, frozen-VLA action loss)

| # | Condition | List metrics where applicable |
|---|---|---|
| 1 | Original instruction | — |
| 2 | Gemini single rephrase (same prompt + trace as Qwen) | — |
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

## Ablations

| Ablation | Question |
|---|---|
| **Teacher-SFT warm-start then tune** (vs primary RL-from-base) | Does distilling Gemini's higher-ceiling distribution first beat pure RL-from-base? 0b says teacher has ~6 pts more oracle headroom — worth testing, not assuming. |
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
- **(B) reward↔success [caveated]: negative until drift is removed.** Pooled ρ=−0.10 (carrot −0.25, spoon −0.29, stack +0.21). Cause: ~40% of Qwen 0c phrases are goal-drifted ("by the plate", "past the towel"); **drifted phrases get LOWER flow loss (0.0794) than faithful (0.0948) while succeeding less (0.40 vs 0.56)** — teacher-forced loss rewards trajectory-style fit, and goal words barely change mid-episode action targets. Faithful-only: carrot/spoon −0.09, stack +0.27, pooled +0.20.
- **Consequence for Phase 2: faithfulness-gated reward is mandatory** — compute advantages among faithful candidates only (LLM judge or equivalent), penalize drift; otherwise RL will exploit the confirmed axis and generate plausible-trajectory wrong-goal phrases. (Generalizes the "log object/target preservation" plan item from monitoring to reward.)
- Judge noise note: flash-judge verdicts vary slightly across runs (spoon 14 vs 15/32); training-time gate should use majority-of-3 or a stricter deterministic check.
- Late-τ/late-episode reward variant: mixed evidence (stack ρ→+0.40 late, carrot unchanged; n_ctx 5–9/band) — parked.

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
