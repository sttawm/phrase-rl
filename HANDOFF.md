# HANDOFF — phrase-rl (written 2026-08-10)

For a fresh session with zero context. The project: improve a **frozen π₀ VLA**
purely by rephrasing task instructions. Three methods: (1) **LLM rules
distillation loop** — the best method and the paper's centerpiece; (2) SFT
back-translation; (3) GRPO RL on a rephraser — a **null result** (reward climbs,
ground truth flat; documented as such in the paper). An IEEE-conference paper is
in active drafting. Simulator: SimplerEnv/INT-ACT WidowX tasks.

---

## 1. In-flight work (do this first)

**Reward re-verification — analysis DONE, paper update NOT done.** The user
asked to "re-investigate the correlation between our proxy reward and actual
rollout success" using the enlarged bank. `scripts/verify_reward_fit.py` is
committed and reproduces everything; outputs in
`results/analysis/reward_verify_434.json`. Findings:

- Ground truth is now **434 phrases at n=36 rollouts** (254 `fine_exam_phrases.parquet`
  close paraphrases + 180 `sim_rollouts_natadv_{0,1}of2.parquet` natural/adversarial).
  All 434 join bank proxy channels (z, grip) at F×C=64.
- The 180 nat/adv were rolled AFTER the logistic was fitted → genuine held-out
  test. **The frozen fit ranks well held-out**: within-task Spearman 0.509 on
  nat/adv (vs 0.319 on the fine_exam it was fit on), pairwise ordering 79.6% on
  15+pp gaps, 70.7% on 10–15pp, 60.2% on 5–10pp.
- **Calibration is broken**: mean predicted 57.7% vs measured 23.5% on nat/adv.
  Never read sigmoid output as a success rate (paper already says this — keep it).
- **Do not refit on the pooled 434**: it drives the z coefficient to ~0
  (−0.0135) and WORSENS task-held-out CV Spearman 0.422 → 0.395, because the
  pooled fit chases cross-task base rates instead of within-task ordering. The
  frozen coefficients `sigmoid(8.123 + 0.4445·z + 11.3193·(−grip))` survive.
  Untried next idea: refit with per-task intercepts (fixed effects) so slopes
  learn only within-task signal.
- Discrepancy to understand before editing the paper: the paper's 79.6%
  fine-pair number comes from `results/analysis/fine_grid_all.json` →
  `grid["F4_C20"]["grip"]["fine_5-10"]`, computed from the exam FEATURE
  extractions (`fine_exam_features_{native,oov}.parquet`) at matched budgets;
  this script's bank-based 5–10pp number is 59.5%. Different context pools and
  budgets — not apples-to-apples. The paper's 92.4% "coarse pairs" is from a
  yet older 4-task coarse exam (`EXPERIMENT.md:1988`, grip C=20). The reward
  section currently stitches those two different exams into one sentence.
- **Candidate paper update** (present numbers in chat FIRST — the user verifies
  every printed number): replace the stitched sentence with the 434-phrase
  held-out validation (Spearman 0.51 on unseen nat/adv, ~80% pairwise at 15+pp,
  calibration caveat). Paper edits go through Overleaf (§8).
- **Self-agreement diagnostic (DONE, 2026-08-11 overnight):** scored the gt
  phrases a second time with a different context draw (seed 8007, frame
  resampling; verifier noise must stay pinned to the ensemble seed — the member
  MLPs encode those CRN draws). `scripts/selfagree_analysis.py` →
  `results/analysis/selfagree_434.json`: draw-to-draw pairwise agreement
  **95.3%** (z Spearman 0.961) vs proxy-vs-gt **68.9%** on the same confident
  pairs (231 phrases, 11 tasks; the 4 clean tasks whose contexts live only on
  e6's volume were skipped — optional gap-fill if e6's host ever frees).
  Conclusion: context-sampling noise is small; the proxy's remaining error is
  **model bias**, so retraining is the lever, not more contexts.

### 1b. Verifier-v2 retrain plan (designed with the user 2026-08-11, not built)

- Deep-Sets over (GT, measured) pairs: shared per-pair MLP → masked mean-pool
  frames → masked mean-pool contexts → score head. No transformer. Variable N
  via masked pooling (no special tokens); train-time random subsampling of
  episodes/frames (4..256) is the key augmentation for thin tasks.
- **Deploy as residual**: f = α·(current aggregate z) + g(frozen per-pair
  embeddings — tap the penultimate layer, NOT just the scalar logit — plus
  grip, t_frac); α init 1, g zero-init. Bit-identical fallback at init; the
  frozen verifier can't be overwritten; g's gain is one ablation number.
- **Loss: within-task pairwise ranking (RankNet on confident pairs), NOT
  success-rate regression** — contexts identify the task, so regression chases
  per-task base rates (exactly how the pooled 434 refit killed the z
  coefficient). One scalar per phrase at inference; sort. No calibration head
  (user decision): if evidence tables need readable numbers, map to within-task
  percentiles at write time.
- Stage 1 (if the frozen tap underperforms): retrain the pretext task
  (own/hard/easy instruction-vs-trajectory discrimination, free labels) on sim
  successful rollouts + Bridge demos, then freeze and refit g. Bridge stays in
  the pretext only for domain coverage of train-task scoring — it has no
  success labels.
- Validation: rotating 10/5 task splits; labels born later (the rules loop's
  own sim evals, n=18, weight by n) are the uncontaminated test set.
- **VERDICT (2026-08-11, rungs 1-2 DONE, both NEGATIVE):** trainer =
  `scripts/train_verifier_v2.py`; rich per-frame features (64-d penultimate
  embeddings + member logits, all 434 phrases, 15 tasks) =
  `results/analysis/v2_features_gt434.parquet` (extracted on e6). Six
  configurations (thin/rich features x mean/meanmax pooling x alpha/penalty
  variants) on TWO validation axes: task-held-out (Bridge-deployment bar)
  base 81.1 vs v2 75.9-76.9; phrase-held-out on known tasks (the sim loop's
  consumer) base 78.2 vs v2 77.1. Every residual that grew failed to
  transfer. Conclusion: **the frozen mean + fixed blend is the ceiling of
  anything built on the frozen verifier's outputs**; the proxy's remaining
  ~20% error (bias, per selfagree_434) lives below the encoder. Only rung 3
  (retrain the encoder pretext on sim rollouts) or a new signal source can
  attack it -- weigh against just running the rules loop (#27), which
  generates fresh labels as a side effect. Full numbers:
  `results/analysis/verifier_v2_rich_{mean,phrase}.json`.

## 2. The rules loop (the main algorithm)

Driver: `scripts/rules_loop_driver.py` (+ `rules_loop_jobs.py`,
`rules_loop_worker.sh`). Design doc: `ALGORITHM-RULES-SEARCH-AND-DISTILLATION.md`.
Per iteration, in this order:

1. `rewrites ← rephrase(instructions, rules_t)` — VLM rephraser conditioned on
   the current rulebook AND a per-input scene trace (`trace_for()` precedence:
   per-base → task → phrase → placeholder; unit-tested).
2. `scores ← score(rewrites, VLA)` — sim tasks by rollouts, train tasks by proxy.
3. `adherence ← judge(rules_t, rewrites, scores)` — separate LLM judge sees ALL
   rewrites; writes the adherence report (`rules_eval.md`, prompt `judge.md`).
4. `experiments ← propose_experiments(evidence, rules_t)` — the agent's
   `plan.md` picks its own probe tasks from `evidence_summary.csv` (the per-task
   coverage table doubles as the permitted task list; criteria: thin coverage,
   bunched scores, large spread on thin sampling).
5. `evidence ← evidence ∪ score(experiments)` — propose → run → use, in that
   order, all before distillation.
6. `rules_{t+1} ← distill(evidence, adherence, rules_t)` — every new rule must
   cite a testable measurement.
7. `v_t ←` mean validation score; patience early-stop; return best-v rulebook.

Invariants: bases frozen at run start (`base_pool`); **train-only banking**
(held-out scores are never banked); judge is a different call from the
distiller; corpus/vocabulary statistics are counted deterministically in code
(never by an LLM). Flags: `--distiller gemini` (inlines working files for API
backends), `--mock-scoring` (real agents, synthetic scores — GPU-free smoke
test). The agent workspace lives OUTSIDE the repo (a cwd inside a denied
subpath made the CLI die with EPERM).

**Eval design is an executable spec: `config/params.py`** — run
`python config/params.py`. Settled values: T_train=156, T_held=31, T_sim=8
(live1 splits); N per task: train 8 (evidence volume), train_held 16 (precision
4.2pp; never banked), sim 32 bases × n=18 rollouts (layouts 0–17 × 1 rep)
= 4,608 episodes/iter → 3.5pp resolution at σ_phrase=0.166 against a +3.2pp
gemini-sized effect; proxy grade F×C=64; base mix orig:nat:adv = 20:50:30;
~47.9 GPU-h per iteration. Measured constants with provenance live in the file
(0.42 min/episode, 2.7s+0.2s proxy costs, +4.3pp layout-half gap, replay
fraction 0.76). `KNOWN_ISSUES` there is honest — read it.

## 3. The scored bank (what it is, how to read it)

The bank is the loop's evidence store AND a scoring cache. All under
`results/analysis/`:

- **Worklist**: `bank_to_score.parquet` (5,006 rows: task, phrase, source) ∪
  `bank_generated.parquet` (2,879 generated rows; kind = natural/adversarial).
- **Proxy scores**: `bank_scores_{train_0of2,train_1of2,sim_0of1}.parquet` —
  columns `z` (verifier-ensemble logit), `grip` (gripper error), `n_ctx`;
  F=4, C=16 grade. 4,975 of 5,006 scored; the 31 unscored are val_held
  stragglers, deliberately skipped (no consumer reads them).
- **Rollout ground truth**: `fine_exam_phrases.parquet` (254) +
  `sim_rollouts_natadv_{0,1}of2.parquet` (180), gt composition standard =
  layouts 0–17 × 2 reps (n=36). `fine_exam` doubles as the val8 "original" tier.
- Join key is always `(task, phrase)`; drop nulls in z AND grip.
- `seed_bank()` in the driver rebuilds a run-local bank from the worklist
  (scores attach via merge) and **warns loudly if a frozen run-local
  `bank.parquet` predates its inputs** — use a fresh run id, never reuse
  `results/rules_runs/live1/`.
- `results/rules_runs/_rule_cache.parquet` is a kept ARTIFACT (66 rows, 19.7%
  pass-through, cited by `CachePolicy`); the code that produced it was removed.

## 4. Reward function — how to evaluate its fit

Proxy = two channels per (task, phrase): `z` and `grip`, scored at F frames ×
C episodes. Calibrated logistic (FROZEN, validated §1):
`success = sigmoid(8.123 + 0.4445·z + 11.3193·(−grip))`.

To (re-)evaluate: `.venv/bin/python scripts/verify_reward_fit.py` — joins the
434-phrase gt to the bank, prints within-task Pearson/Spearman, pairwise
ordering by gap bucket (5–10/10–15/15+pp), held-out metrics on nat/adv, a
refit + task-held-out CV, and calibration; writes
`results/analysis/reward_verify_434.json`. Budget-sweep grids (agreement vs
F,C) live in `fine_grid_all.json` (buckets calib_0.5-5/fine_5-10/med_10-15/
far_15+; variants ens100/z75g25/z50g50/c4b/grip).

Known truths: standardized by within-task SD the verifier carries ~2× the
gripper (z 0.846 vs grip 0.414) — the old "gripper is 25× the verifier" claim
was a raw-coefficient artifact and was removed from the paper. Ranking
generalizes to nat/adv; absolute calibration does not.

## 5. v12 RL plan (held task #36)

Spec: `ALGORITHM-RL-V12.md` (v10/v11 history: `ALGORITHM-RL.md`). Core changes
vs v11: reward = the proxy **LOGIT** `0.4445·z + 11.3193·(−grip)` (never the
sigmoid), clipped, replacing within-group ranks (rank magnitude amplified
ε-noise into gradient); keep `sample_single` K=16; C=10 funded by
duplicate-collapsed scoring (score distinct strings once, `np.repeat` back
before the transform); min-group-spread skip; ratio sum-not-mean fix; gate on;
judge at step 150 on nat24 with 768-episode cells. Pre-registered kill
criteria: verifier moves >0.3 while grip <0.01 = Goodhart; reward_std < 2×
noise floor; cand_margin not falling. v12 adds image conditioning and the
natural-derived traces (now generated:
`results/phrase_artifacts/traces_rules_v1.parquet`, 2,000 rows). Two spec
rationales need revisiting: the 25× claim is dead (§4), and §1 shows the logit's
ranking (what GRPO uses) is fine but its scale is uncalibrated.

Resumability: v11 checkpoints steps 0010–0240 and v10 0010–0030 + 0230–0260 +
best/latest are on origin/main as split-tar archives (v10 0040–0220 lost).
Telemetry: `results/analysis/{v10,v11}_telemetry/train_log.jsonl`. Temperature
sweep (`scripts/temp_sweep_qwen.py`, `results/analysis/temp_sweep.json`) proved
diversity collapse is training-caused, not sampling-caused — and NOT the cause
of flat val.

## 6. Determinism / CRN facts

Rollout decode noise is pinned by `crc32(task|ep_id|rep)` — phrase excluded, so
identical cells are comparable across arms. Post-CRN repeat-cell disagreement
9.2% (was 31.8%). Cell-level cache key: (task, phrase, episode_id, rep, seed);
**seed is not yet written to the rollout parquet** (task #31). `episode_id`
wraps modulo len(xy_configs)×len(quat_configs) (grids 4–360 per env; val8 grids
unverified — task #29). Layout halves 0–11 vs 12–23 differ +4.3pp
(executor-dependent); treatment arms are flat across halves. The 0–17/18–23
layout reservation applies to rollout RL only.

## 7. RunPod workflow (phrase-rl fleet TERMINATED 2026-08-15; NEW pi0.5/LIBERO fleet 2026-08-25)

**NEW FLEET (2026-08-25, running the pi05_libero four-tier eval — see
`results/experiments.json` id `pi05_libero_fourtier`):** four community pods
`lb1 dvgajobn39jl9w` / `lb2 lslpteu0p5r7ql` / `lb3 p7499qcw2nx2z9` (RTX 4090)
/ `lb4 xzaybguoege6m8` (RTX 3090), 60GB volumes. Stack = the SEPARATE public
repo `sttawm/interactive-vlas` (local clone `~/dev/interactive-pi`),
`pi05_libero/setup.sh` → `eval/run_fourtier.sh <shard>`; tmux sessions
`server` (OpenPI serve_policy :8000) + `fourtier`. Outputs
`/workspace/fourtier_<shard>.jsonl` (+`.boards.json` sidecar) are pulled back
over ssh and committed locally — pods hold NO git push token. SSH endpoints in
the session scratchpad `lb_ssh.json` (re-derive per-pod ip/port from the API).
Terminate when the four-tier analysis has landed in results/analysis/.

**The old eight phrase-rl pods are DELETED** — ids in git history are dead. Everything
canonical was salvaged first: all checkpoints/telemetry (v7 through v13
incl. v13_latest), all context-table archives (val8 + oov + oov5 + native in
results/analysis/data_archive/), cells/judges, and the Mac data/ masters.
Rebuilding a pod: scripts/bootstrap_trainpod.sh (train) or the eval-pod
recipe in the memory files (Vulkan: libegl1 + ICD manifest). CPU-only resume
via API was refused on full hosts — the console button is the fallback.

### (historical) RunPod workflow

- API key: `KEY=$(grep -m1 'RUNPOD_API_KEY' ~/.zshrc | sed "s/^[^=]*=//; s/[\"']//g; s/[[:space:]]*$//")`
  — never echo it, never put it in a URL or `ps`-visible position.
- List: `curl -s -H "Authorization: Bearer $KEY" https://rest.runpod.io/v1/pods`;
  stop: `POST /v1/pods/<id>/stop`. Pods (ALL STOPPED as of 2026-08-08):
  e1=g820s2yb35nggu, e4=xlnjfghj5v9j0o, e6=0gx0ylavw0ofzp, L40S=8u4ljy4ddrgoki.
- Pods sync via GitHub (scripts + git pulls), never scp of commands. Pod git
  remotes embed a GitHub token — **never print a pod remote URL**; the user
  still needs to rotate a previously leaked token.
- Long jobs: `setsid nohup ... &` (tmux-session-scoped jobs die when the tmux
  server tears down; when stopping a pod from inside tmux, run the stop
  FOREGROUND). Verify every push landed with
  `git ls-tree origin/<branch>` from local — never trust a drain log or exit code.
- Auto-memory (`~/.claude/projects/-Users-sttawm-dev-robotics-phrase-rl/memory/`)
  has hard-won pod repair recipes: Vulkan ICD manifest repair, libegl1
  requirement, pod-state backup before stop, venv-symlink audits, partial-clone
  fix for checkpoint-bloated .git. Read `MEMORY.md` there before pod surgery.
- Irreplaceable inputs are archived on origin:
  `results/analysis/data_archive/contexts_sim_val8.tgz.part-{aa,ab}` + manifest.
  `data/sim_traj_val8` (2.9GB) is NOT archived — regenerable.

## 8. Paper / Overleaf workflow

- Canonical = Overleaf git clone at `/Users/sttawm/dev/robotics/phrase-rl-paper`
  (remote `https://git.overleaf.com/6a7269aaf9f91194e71e1092`, branch `main`).
  Auth: `OVERLEAF_GIT_TOKEN` in `~/.zshrc`, wired into
  `~/.overleaf-git-credentials` (chmod 600) via a host-scoped credential
  helper — never in a URL.
- **`git pull` before ANY edit** (the user edits live in Overleaf and has
  drifted mid-session multiple times). Push after each section, then mirror
  with `cp main.tex ../phrase-rl/paper/main.tex` and commit the mirror in
  phrase-rl.
- Compile check: `pdflatex` twice in the clone; grep for "Output written" and
  undefined references; render pages with `pdftoppm` for visual checks.
  `\graphicspath{{./}{../results/charts/}}` lets the mirror compile in-repo.
- Figures are scripted: `scripts/make_results_translating.py` (main results),
  `make_grpo_progress.py`, `make_pi0_conditions.py`, `make_training_flow.py`
  (scripted teaser — NOT in the paper; user chose the hand-drawn
  `training_flow_compact.png`). Edit the script, rerun, never hand-edit a PNG.
- State: ~10 pages, compiles clean, "ROUGH DRAFT" banner in the title.
  Terminology: "in-distribution"/"out-of-distribution"/"OOD" everywhere; full
  words, no invented acronyms (RD/RTD banned); "arms"/"nominal" banned.
- The user prompts one section at a time; never rewrite unprompted sections;
  every number is verified in chat before it enters the tex.

## 9. Security / sealed-set discipline

- All secrets (`GEMINI_API_KEY`, `RUNPOD_API_KEY`, `OVERLEAF_GIT_TOKEN`,
  `HF_TOKEN`) are read from `~/.zshrc` by grep/pipe at point of use — never
  echoed, logged, or interpolated into visible commands.
- Sealed evaluation set: any new exposure (generation touching sealed phrases)
  requires a PREREG amendment documented BEFORE generation, `FINAL_EVAL=1`, and
  the phrases preflight-printed. The 192 sealed per-natural traces are
  **ungenerated**, pending "Amendment 3".

## 10. Do Not Retry (dead ends, with the failure)

1. **`cuted` for two-column balancing** — page-2 left column silently skipped
   entirely (even with cuted v2.0 installed locally). Fix was strip→`figure*`.
2. **Teaser on page 1** — `\dbltopnumber`/`\dbltopfraction` etc. are ignored
   for page 1; placing the `figure*` before `\maketitle` produced a lone float
   page. Only remaining route is `\twocolumn[]` surgery; user hasn't opted in.
3. **`AutoModelForCausalLM` for Qwen-VL** — silently loads the wrong module
   tree (no error, garbage behavior). MUST use `AutoModelForImageTextToText`.
4. **Refitting the reward logistic on pooled phrases** — kills the z
   coefficient and worsens held-out ranking (§1). Only try again with per-task
   intercepts.
5. **Row-existence resume in `score_bank.py`** — permanently poisoned 132 rows
   (null z). Resume must require non-null z AND grip (fixed; keep it).
6. **LLM counting corpus vocabulary** — inlining 17k instructions read-timed
   out; counting is done deterministically in code.
7. **Per-task full parquet reads in trace generation** — 3GB re-read × 6
   threads made the machine unusable. `gen_traces_rules.py` now single-pass
   preloads via pyarrow `iter_batches` + `nice -n 15` + ≤4 workers.
8. **tmux-backgrounded pod jobs / nohup'd pod-stop inside tmux** — die with
   the tmux server. `setsid nohup` for jobs; foreground the stop.
9. **`pkill -f pattern` over ssh** — matches its own ssh command line; bracket
   the first char: `pkill -f '[p]attern'`.
10. **Trusting push/drain exit codes on pods** — checkpoints silently missing;
    verify with `git ls-tree origin/<branch>`. Also `scp` into a pod repo
    before committing → untracked-collision rebase wedge.
11. **Scripted tex edits with guessed anchors** — line-wrap guesses corrupt
    nothing only because scripts must `grep` the exact anchor text and assert
    before writing. Never write without the assert.
12. **Session scratchpad for derived data** — the reward-verify join parquet
    vanished with the scratchpad between sessions (and v11 telemetry was
    nearly lost the same way). Derived artifacts go in `results/analysis/`.
13. **Trusting EXPERIMENT.md ledger for latest numbers** — it ends 2026-08-06;
    A28 landings live in `rephrase_robustness.jsonl` and commit messages.
14. **`--out` parquet reuse in rollout scripts** — the phase0c script
    accumulates onto an existing parquet; `rm` before each roll
    (n≠design is the contamination tell).

## 11. TODO, priority order

1. **Finish the reward re-verification story**: reconcile bank-vs-exam
   5–10pp numbers (59.5 vs 79.6 — context pools/budgets differ), decide the
   paper's replacement sentence, present numbers in chat, then edit Overleaf.
   Optional: per-task-intercept refit.
2. **#27 Run the Gemini rules loop** — all blockers cleared; FRESH run id;
   design frozen in `config/params.py`. (User postponed for the paper.)
3. **#36 v11-restart vs v12 decision** — spec ready (`ALGORITHM-RL-V12.md`);
   update its two stale rationales first (§5).
4. **#31 Write seed into rollout parquet** + wire the cell-level score cache
   (key: task, phrase, episode_id, rep, seed).
5. **#29 Verify val8 per-task layout grid sizes** on a pod (INT-ACT
   `custom_scenes/put_on_in_new.py`) — needed for 0–17×2 vs 0–35×1
   comparability claims.
6. Paper long tail: page-1 teaser only via `\twocolumn[]` (needs user opt-in);
   192 sealed traces pending PREREG Amendment 3; user must rotate the leaked
   GitHub token.
