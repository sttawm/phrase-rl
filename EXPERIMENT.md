# phrase-rl: Advantage-Weighted Rephrase Tuning Against a Frozen VLA

**Status:** planning complete, Phase 0a next. **Last updated:** 2026-07-03.

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
| Frozen trace/teacher VLM | Gemini 3.1 Pro (batch API) | Replaces CoVer's GPT-4o (discontinued). Generates cached reasoning traces + 16-rephrase teacher lists |
| Trainable phrase model | `Qwen/Qwen3.5-9B` (natively multimodal, Feb 2026) | LoRA. **Phase 0a smoke test:** image input + LoRA via transformers/PEFT; fallback `Qwen/Qwen3-VL-8B-Instruct` (mature FT ecosystem, ~24GB LoRA). Shared weights, two prompt modes (single-phrase, 16-list). 27B/32B only as scale-up ablation after a positive result (32B ≈ 73GB FP16 — needs 80GB pod, ~4× slower candidate generation) |
| Optional baseline | CoVer verifier (`cover_verifier_bridge.pt`, ~312MB) | Test-time selection over our candidates; separates "better candidates" from "better selection" |

π0.5 note: our reward/eval model is π0 (Bridge/SIMPLER). π0.5 only appears in CoVer's PolaRiS extension; all machinery here (flow-matching scoring, CRN) transfers unchanged if we later swap it in.

## Data

BridgeV2 tuples `x = (o_t, l, a*_t)` (image, original instruction, ground-truth action chunk). For each, cache one Gemini call: reasoning trace `r` + 16 teacher rephrases. Training context for the phrase model: `c = (o_t, l, r)`.

- **Scale:** 2,000 train + 500 val contexts (pilot). Test contexts held out untouched.
- **Cost:** ~1k input + ~600 output tokens per context → **~$13 total** with Gemini 3.1 Pro batch mode. Scaling to 10k later is <$50.
- Timestep choice: `t` here indexes **which frame of a demo episode becomes a context** — not when rephrasing happens (that's once, at boot, from the first observation). Bias harvesting toward early frames to match the generator's deployment input distribution, but keep some mid/late-episode contexts: the boot-time phrase must aid action prediction across the whole episode, and first chunks are often generic "reach" motions that any phrasing predicts equally well — grasp/place chunks may be where wording discriminates. Optional refinement: condition the phrase on the episode's *first* frame but score it against action chunks at multiple later `t` of the same episode (matches deployment exactly on the input side, covers the episode on the reward side).

## Reward: π0 flow-matching loss with common random numbers

π0 has no likelihood; training loss is a denoising-velocity MSE, stochastic in the noise `ε` and flow time `τ` that LeRobot's `Pi0Policy.forward()` samples internally. To use it as a phrase score:

- Patch forward to accept **fixed (ε, τ) draws**.
- Score all 16 phrases of a context with the **same K draws** (common random numbers), K ≈ 8–16, τ stratified over [0,1]; average.
- `R_i = -L_VLA(a* | o, y_i)`; within-group advantage `A_i = (R_i - mean) / (std + eps)`.
- Loss magnitude varies strongly with τ, so also compute a **per-draw z-scored** variant (z-score the 16 phrases' losses within each (ε, τ) draw before averaging) so high-variance τ bins don't dominate the ranking; Phase 0b picks whichever has better split-half reliability. 0b should also report which τ band discriminates phrasing best (cf. Diffusion Classifier, arXiv:2303.16203 — the same conditional-denoising-loss scorer in the image domain; mid-range noise carried most signal there).

## Training

1. **SFT:** split each Gemini 16-list into 16 single-phrase examples `(c → y_i)` under the single-phrase prompt; small auxiliary loss on the list prompt to preserve clean numbered-list output.
2. **Advantage-weighted tuning:** generate 16 candidates per context (list prompt for diversity), score with frozen π0 + CRN, update under the single-phrase prompt:
   `L = -Σ_i max(A_i, 0) · (1/|y_i|) · log π_θ(y_i | c, p_single) + β·KL/SFT anchor + λ·list-format aux`
   **Token-mean normalization** (the `1/|y_i|`): raw summed log-prob scales gradient contribution with phrase length, so long positive-advantage phrases would dominate updates. Positive-only advantages first; signed advantages as ablation. This is advantage-weighted rephrase tuning, not exact GRPO.
3. **Log during training, not just eval:** duplicate rate, mean pairwise embedding similarity, object/target preservation, generic-phrase rate, format-failure rate. Known failure mode: collapse onto Bridge template phrasing ("put X in Y") — the rephrase-FT reward checkpoint was trained on a fixed paraphrase dictionary the optimizer may simply rediscover.

## Phases

- **0a — Infra + plumbing:** CoVer repo env; load both INTACT checkpoints via LeRobot; extract 2k+500 BridgeV2 contexts; fixed-noise scoring patch.
- **0b — Sensitivity gate (HARD go/no-go):** ~200 val contexts × 32 rephrases (sensitivity phases use 32 for more rank points; training groups stay at 16), CRN scoring. **Primary phrase source: candidates sampled from the trainable VLM itself** (base model, list prompt) — that is the distribution the reward must discriminate during RL; Gemini lists as a secondary arm (they're the SFT data; comparing the two spreads measures teacher-distribution headroom). Per phrase: `L̂_i = (1/K) Σ_k ‖vθ(Aᵏ; o, y_i, τ_k) − u*_k‖²` over the K shared draws; split draws into two fixed halves (same halves for all phrases) → two estimates per phrase → Spearman across phrases per context. **The gate is reliability at training-budget K (≈8–16), not asymptotic**: split-half correlation at fixed K measures SNR at operating conditions (Spearman-Brown: ρ ≈ σ²_signal/(σ²_signal + 2σ²_noise/K)); needing K≫16 for stable ranks = reward unaffordable per update = no-go on the same grounds. Include **duplicate-phrase negative controls** (same phrase twice in a group → empirical noise floor) and report the between-/within-phrase **variance decomposition**, plus the **phrase-length ↔ loss correlation** (a hackable axis if present). **Go** iff phrase ranking reproduces across halves and spread clears the duplicate-control floor. Run both INTACT checkpoints (differential sensitivity is itself a finding). This is exactly the non-discriminative-reward failure mode from the earlier OpenVLA GRPO runs — catch it before training anything.
  Note what 0b does and doesn't establish: it calibrates the *instrument* (can CRN flow loss resolve phrases above its noise floor), not whether phrasing matters behaviorally (CoVer shows behavior moves, but through a verifier over sampled instruction–action pairs — consistent with rephrases merely widening the action proposal pool) and not whether lower loss ⇒ better rollouts (risk 3; 0c and Phase 4).
- **0c — Rollout sensitivity + metric validation (SIMPLER):** the rollout analog of 0b, pulling a small slice of Phase 4's loss↔success question forward. 4 SIMPLER Bridge tasks (prefer tasks with mid-range baseline success — floor/ceiling tasks can't discriminate) × 32 Gemini rephrases each; **shared initial-state seeds across phrases** (the CRN analog for rollouts — success differences are paired per seed). No verifier, no best-of-N over actions: condition on one phrase, roll out, score success — isolates phrase quality from CoVer's action-selection confound. Readouts: (1) phrase-to-phrase success spread vs. binomial noise (permutation test over paired seeds); (2) split-half rank reliability over seeds; (3) **Spearman between flow-loss rank (real Bridge contexts, matching instruction) and success rank (sim)** — the reward-validation plot. Budget: **adaptive** — start at 10 shared seeds × 32 phrases × 4 tasks ≈ 1,280 episodes. At n=10 per-phrase success SE is ~0.16, so individual phrase pairs won't separate, but the pooled paired tests and the 32-point rank correlation retain power; escalate seeds (10 → 25 → 50) only on tasks/phrases where the readouts are inconclusive. Alternative cheap pass = extremes-only (top/bottom phrases by flow loss, many seeds each). Caveats: only 4 instructions (complements 0b's breadth, doesn't replace it); flow loss on real frames vs success in sim, so weak correlation is ambiguous (metric failure vs domain gap) while strong correlation is decisive. Interpretation: flow-sensitive + success-correlated → go; flow-sensitive but uncorrelated → reward measures the wrong thing, pivot before training; flow-flat but rollout-sensitive → policy-loss reward dead yet phrasing matters (CoVer-style/success-reward path); both flat → hypothesis dies cheaply.
- **1 — Data + SFT:** Gemini batch job (~$13); SFT Qwen.
- **2 — Advantage-weighted tuning** as above.
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
| No trace in context | Does frozen frontier reasoning help? |
| Signed vs positive-only advantage | Does downweighting bad phrases help? |
| List-prompt vs 16× single-prompt sampling + dedupe | Cleaner diversity source? |
| Non-rephrase π0 as reward | Dependence on paraphrase-augmented reward model |
| CoVer verifier selecting among our 16 | Better candidates vs better selection |

## Compute

One pod, **2× ~40–48GB GPUs** (e.g., 2×A40 — cheaper than 1×A100 80GB):
- **GPU A — reward server:** frozen π0 (~7GB bf16); 16 phrases × 16 draws = 256 forwards per context, batched. Second process, no networking.
- **GPU B — phrase model:** Qwen-7B LoRA + gradient checkpointing; HF `generate` for sampling (vLLM later if throughput bites).

## Risks

1. **Reward non-discriminative** (Phase 0b gates this). Prior evidence: CE/L2 reward on OpenVLA was flat across text variation → mode collapse.
2. **Reward hacking → template collapse** (diversity metrics during training; KL anchor).
3. **Offline loss ↔ rollout success link unproven** — Phase 0c tests a small slice early; Phase 4 is the real test; CoVer's verifier-based selection is the fallback comparison. Related: reward/offline eval use real Bridge frames, rollout eval is sim (SIMPLER) — if they disagree, keep "loss doesn't predict success" and "real↔sim gap" distinguishable.
4. ~~Repo housekeeping~~ resolved 2026-07-03: fresh repo initialized here; stray parent repo removed (backup at `~/.dev-stray-git-backup`).
