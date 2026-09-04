All code should live in a git environment in the current working directory.

If a git repository does not already exist, create one with the name of the current directory, and push it to github.

All code and artifact changes should be pushed to github. Commit often. Most likely this should be done after any logical code change, and likely after each command you run that finishes writing code. The only thing that should go in the codebase is ML data and large models. 

All models should go into a results/checkpoints directory. In the same directory should be a json tracking all relevant metrics for each experiment and checkpoints (for example, val metric, test metric). Charts can go in results/charts. 

There should also be a results/experiments.json file which tracks the commit we ran the experiment with and a thorough textual overview of the experiment as well as the exact training command used to run the experiment.

When writing training code, always make sure it's resumable from a checkpoint, and always make sure best val and final checkpoints are produced.

When running remotely, those checkpoints should always be pulled locally. Ideally, they should also be synced to some external storage (git is fine, maybe we'll find something better along the way). 

When running on a remote server like runpod, frequently ping to check the status of the experiment in case it failed, and when it fails, try to fix it and take necessary steps to continue.

Running on the remote pod should always happen via github syncs and scripts to make sure we're not relying on scp'ing commands.

If running locally, always use some sort of virtual environment, whether 'uv' or 'venv' or 'conda' or similar. Choose whichever is appropriate.

## Durable project conventions (see HANDOFF.md for full state)

### Build / run
- Local venv is `.venv` (Python 3.11): run everything as `.venv/bin/python scripts/<x>.py`. No scikit-learn installed — small fits are hand-rolled with numpy (see `scripts/verify_reward_fit.py:fit_logistic`).
- The eval design is an executable spec: `python config/params.py` prints it. Any protocol change edits that file (constants carry provenance comments, decisions go in its DECISIONS log) in the same commit as the code change.
- Charts are scripted: `scripts/make_*.py` → `results/charts/*.png`. Edit the script and rerun; never hand-edit a PNG.
- Reward-fit evaluation: `.venv/bin/python scripts/verify_reward_fit.py` → `results/analysis/reward_verify_434.json`.

### Architecture invariants
- Bank join key is always `(task, phrase)`; drop rows with null `z` or `grip`. Resume logic must key on non-null scores, never on row existence.
- Ground-truth rollout composition standard: layouts 0–17 × 2 reps (n=36). Rollout determinism: decode noise seeded by `crc32(task|ep_id|rep)`, phrase excluded.
- Rules-loop runs get a FRESH run id under `results/rules_runs/`; never reuse a stale run-local `bank.parquet`. Train-only banking: held-out scores are never banked.
- The calibrated success logistic `sigmoid(8.123 + 0.4445·z + 11.3193·(−grip))` is frozen; its sigmoid output is a ranking signal, never an absolute success rate. Do not refit on pooled phrases (worsens held-out ranking).
- Sealed-set discipline: no generation touching sealed phrases without a PREREG amendment documented first, `FINAL_EVAL=1`, and a preflight print of the phrases.
- Derived analysis artifacts go in `results/analysis/`, never the session scratchpad.
- Qwen-VL loads via `AutoModelForImageTextToText` (never `AutoModelForCausalLM` — silently wrong module tree).

### Remote pods
- Secrets (`RUNPOD_API_KEY`, `GEMINI_API_KEY`, `OVERLEAF_GIT_TOKEN`, `HF_TOKEN`) are grep/piped from `~/.zshrc` at point of use, never echoed or put in URLs/`ps`-visible commands. Pod git remotes embed a token: never print a pod remote URL.
- Long pod jobs: `setsid nohup`; pod-stop from inside tmux runs foreground. Verify every push with `git ls-tree origin/<branch>` from local — never trust exit codes or drain logs.
- Archive checkpoints to git immediately as split tars (`ckpt_archive.sh` pattern); verify newest-archived vs newest-trained at every status pass and before any pod shutdown.

### Paper
- Canonical tex is the Overleaf clone at `../phrase-rl-paper`: `git pull` before ANY edit, push after, then mirror to `paper/main.tex` and commit here. Compile-check with `pdflatex` twice; visual-check via `pdftoppm`.
- Style: full words, no invented acronyms; "in-distribution"/"out-of-distribution"; every number verified in chat before entering the tex; edit only the section the user asked for.

---

# Project context (reconstructed 2026-09-04)

Written after a machine migration that lost all local session transcripts, `.env`,
and the auto-memory directory. Everything below is grounded in a file, commit, or
config in this repo; claims I could not verify are marked **[inferred]**. Sources
are cited as `file:line`. The authoritative running state is still `HANDOFF.md`
(written 2026-08-10, updated to 08-31) — but note it now lags HEAD by ~4 days of
very heavy activity, so prefer commit messages for anything after 2026-08-31.

## What this project is trying to do

Improve a **frozen** VLA policy purely by **rephrasing the task instruction**.
No policy weights are ever trained (`HANDOFF.md:3-4`; `pyproject.toml:4`). The
research question, stated in `WRITEUP.md:14-17`: can a small model be trained to
produce the rephrasings a *specific* frozen policy prefers, using only that
policy's own action loss as reward? Positioned against **CoVer** (`arXiv:2602.12281`,
`EXPERIMENT.md:7`), which generates with a fixed frontier VLM and selects at test
time; the claimed novelty is using policy loss as a *training* signal
(`EXPERIMENT.md:12`).

Three methods were tried. Per `HANDOFF.md:4-8`:

1. **LLM rules distillation loop** — the best method and the paper's centerpiece.
2. **SFT back-translation** — corrupt-and-invert.
3. **GRPO RL on a rephraser** — a **null result**: reward climbs, ground truth
   stays flat. Documented as such in the paper, not hidden.

The RL arm is dead-ended by its own verdicts. `ALGORITHM-RL-V12.md:257-293`:
v12 paired 768-episode judges went step 0 = 44.40 → step 150 = 39.71 (−4.7pp);
v13 = −4.0pp. Rank-noise amplification, identity collapse and input-mix confounds
were each eliminated as the cause and the outcome did not change. Its own
conclusion: "the exploit surface is the learned verifier itself", and "the rules
loop (#27) remains the productive path meanwhile" (`ALGORITHM-RL-V12.md:293`).

## Two experiment lines are live in parallel

Both run right now; do not conflate them. The loop is abstracted over a
`(policy, dataset)` **environment registry** (`scripts/rules_loop_driver.py:71-110`),
selected with `--env`, so the second benchmark is "a second entry, not a refactor"
(`config/params.py:482-491`).

### A. `bridge_pi0` — π0 + Bridge / SIMPLER (the original line)

Frozen policy `juexzz/INTACT-pi0-finetune-rephrase-bridge`, served in-process from
the INT-ACT fork of SimplerEnv (`scripts/setup_pod.sh:31-38`;
`src/phrase_rl/phase0c_rollout.py:53-73`). WidowX tabletop tasks. 228-task pool,
split live1 = 156 train / 31 held (`config/params.py:98-114`) plus **val8**, the 8
clean sim tasks that are the only split with real rollout ground truth
(`config/params.py:118-134`).

Current work: evaluating new natural phrase sets on a pod fleet, and re-distilling
rulebooks so rule sets can be measured over multiple repetitions for significance.

### B. `pi05_libero` — π0.5 + LIBERO (added 2026-09-02)

Physical Intelligence's released `pi05_libero` checkpoint served by stock OpenPI
(`serve_policy --env LIBERO`, pinned `15a9616`) — `FOURTIER-LIBERO.md:20-22`.
Rollouts go through `bank_eval.py` in a **separate public repo**,
`sttawm/interactive-vlas` (`scripts/rules_loop_jobs.py:159-202`).

**No calibrated proxy exists for π0.5 — the bank is rollout-scored**
(`PI05-BANK-HANDOFF.md:22-24`). Split (seed 20260901): train 65 + goal, val 15,
sealed 20 (`PI05-BANK-HANDOFF.md:70-73`).

The headline finding: the adversarial penalty is a property of the *finetune
pairing*, not the policy — in-finetune 97.3 → 47.2 (−50.1pp) under rephrasing,
while out-of-finetune tiers (51.6 / 50.7 / 47.7) are statistically
indistinguishable (`FOURTIER-LIBERO.md:7-11, 51-72`).

## Key design decisions and why

- **Common random numbers.** Every phrase in a group is scored against the *same*
  K draws of (ε, τ), so loss differences reflect conditioning only
  (`src/phrase_rl/pi0_scoring.py:1-19`). Rollout decode noise is pinned by
  `crc32(task|ep_id|rep)` with the **phrase deliberately excluded**, so arms face
  identical noise; without it identical-phrase cells differed 8–16pp
  (`src/phrase_rl/phase0c_rollout.py:107-116`).
- **The sigmoid is gone; the loop reads the LOGIT.** On training frames `grip≈0.1`
  sits outside the calibration's fitted range, so the sigmoid pins 73% of bank
  phrases above 0.99 and compresses rulebook deltas into the 3rd decimal. The
  frozen weights survive only as a bridge between the two channels' scales
  (`config/params.py:492-499`, decision dated 2026-08-30). This is why the
  invariant above says "ranking signal, never an absolute success rate".
- **Composition, not count, is the standard.** Layouts 12–23 are easier than 0–11
  by 4.3pp, so two n=36 numbers drawn from different layout sets are not
  comparable (`config/params.py:68-74, 251-264`).
- **Train-only banking.** Held-out scores are never banked, because the bank feeds
  the distiller's evidence and banking held-out would destroy the only
  generalization read the loop has (`config/params.py:306-315`).
- **Per-rule evaluation was dropped** for an LLM adherence judge: cost scaled with
  rule count. "The book is measured, not its parts" (`config/params.py:562`).
- **Rulebooks are never merged across rephrasers** — a combined book would
  reintroduce the capacity mismatch that killed v2
  (`ALGORITHM-RULES-SEARCH-AND-DISTILLATION.md:82-83`).
- **The master empirical result**, from 49 instructions × 16 single-change probes:
  phrasing interventions are **conditional on corpus membership**. In-vocab, the
  canonical template is the optimum — touch nothing. Out-of-vocab, visual renames
  are the biggest win. `rename_visual` being *worst* in-corpus and *best* OOV is
  the cleanest statement of the whole project's thesis (`EXPERIMENT.md:2362-2373`).
  Relatedly, true-name toxicity is **corpus absence, not rarity**: "ramekin"
  (corpus-absent) costs −41.7pp while "aubergine" and "rack" (corpus-present) are
  free (`PHRASE-SEARCH.md:48, 574-579`).

## Experiment naming conventions

**`A` vs `B` has three era-dependent meanings.** This is the single most confusing
thing in the archive — check the date before reading an `A_*`/`B_*` name:

1. *v2/v3 era*: plain A/B-test language for the flow-vs-L2 reward comparison; the
   arms are named `flow` and `l2`, not A and B (`WRITEUP.md:189`).
2. *v5/v6 era*: the **update rule**. A = 16-list advantage-weighted, B = GRPO
   sample-16 (`WRITEUP.md:392`). This is what the archives `A_bestval`,
   `B_final_0268`, `A_v6step_0060`, `B_v6_final_s190`, `adapters_A`, `adapters_B`
   refer to. Both v5 arms exited on the KL circuit breaker — A at step 279, B at
   step 268, hence `B_final_0268`.
3. *Sealed-test era (2026-07-30 →)*: **arms A–H are pipeline configurations**
   (A = v7a rewriter, B = v7f, C = rules-v3, D = composed router, E = rules-v4 +
   gemini-pro, F = frozen Qwen + rules-v4, G = Claude Fable + rules-v4, H = bare
   Claude) — `EXPERIMENT.md:2399-2404, 2620-2659, 2739-2756, 2850`.

Other stems:

- **`phase2`** — both the RL phase (`EXPERIMENT.md:91`) and the default checkpoint
  dir, which was **reused across run generations** (`EXPERIMENT.md:390-391`).
  Ambiguous by itself; date the commit.
- **`phase2_flow` / `phase2_l2`** — the two candidate reward functions. `flow` =
  CRN flow-matching residual on the GT chunk; `l2` = per-DoF-normalized distance
  between π0's decoded action and the same chunk (`WRITEUP.md:195-198`). L2 won
  the bake-off as the only cross-task-consistent arm (`EXPERIMENT.md:72`).
- **`contexts_sim_native` / `contexts_sim_val8`** — *inputs*, not outputs: rescued
  sim-grounded context tables (frames from successful rollouts) archived to git
  because they existed only on one pod's volume (`HANDOFF.md:262-264`).
  `native` = the 4 in-vocab val tasks at 5 frames/episode; `val8` = the 8 val8 sim
  tasks + 7 distractor scenes at 4 frames/episode.
- **v7 → v13** — the RL checkpoint ladder, archived per step as 90 MB split tars
  (`results/checkpoints/archive/`, packer `scripts/ckpt_archive.sh`).
- **A29–A35** — PREREG'd experiment IDs in the current (Sept) chapter. Recorded in
  `results/analysis/pi05_bank/PREREG.md` and `results/experiments.json`.
- **Rulebook "books"** — `train_only` (proxy + corpus statistics), `sim_only`
  (rollout evidence), `both`; crossed with **conditions** {Original, Natural,
  Adversarial} and **appliers** {claude, gemini, qwen}.

## Vocabulary

- **phrase** — a candidate instruction string. Bank join key is always
  `(task, phrase)`; for LIBERO the episode key is `(suite, task, phrase, init)`
  (`FOURTIER-LIBERO.md:46-47`).
- **kind** — every phrase carries one: `original` / `natural` / `adversarial` /
  `rephrased` / `search` / `unknown`, never guessed
  (`ALGORITHM-RULES-SEARCH-AND-DISTILLATION.md:184-196`).
- **base** — the unrephrased instruction a rulebook is asked to repair, frozen at
  run start. Distinct from the no-rules anchor, which is the rephraser with an
  *empty* rulebook; both are meaningful and they differ (`…:210-213`).
- **z / grip** — the two proxy channels. **z higher is better, grip lower is
  better** (`prompts/rules_loop/distill.md:28-29`). grip is the channel that
  transfers; z is the one that is exploitable (`ALGORITHM-RL-V12.md:82`).
- **trace** — a scene analysis conditioning the rephraser. Two hard rules: generate
  it **from the base phrase, never the canonical** (a canonical-derived trace
  smuggles the answer key into the input), and a trace is per **(task, phrase)**,
  not per task (`PHRASE-GENERATION-RECIPE.md:115-126`).
- **F × C** — frames per episode × contexts per task; pinned at 64
  (`config/params.py:179`).

## How to run

Local analysis venv is `.venv` (Python 3.11). Pods carry **two** venvs because the
trainer and the reward model cannot share one: `.venv` has the INTACT-era lerobot
fork pinned to `transformers==4.48.3`, `.venv-gen` has transformers≥5 for Qwen+peft
(`src/phrase_rl/phase2_score_server.py:1-8`; `scripts/setup_pod.sh:22-27`). They
talk over a file-based IPC protocol in `--ipc-dir`
(`src/phrase_rl/phase2_train.py:6-24`).

**The rules loop** (current chapter; trains no weights) —
`scripts/rules_loop_driver.py:22-28`:

```bash
# smoke: real agents, synthetic scores, no pods, no GPU
.venv/bin/python scripts/rules_loop_driver.py --run-id smoke --dry-run \
    --rephrasers gemini --patience 2 --max-iters 3
# real
.venv/bin/python scripts/rules_loop_driver.py --run-id r1 \
    --rephrasers qwen,claude,gemini --patience 3
```

Pod side: `RUN_ID=r1 bash scripts/rules_loop_worker.sh`. Workers claim jobs
through git — a `.claim` file, then a `.result.parquet`, both committed. Use a
**fresh run id**; never reuse a stale run-local `bank.parquet`.

**The eval design is an executable spec**: `python config/params.py` prints every
parameter, its provenance, the derived cost, the resolution it buys, plus
`KNOWN_ISSUES` and a dated `DECISIONS` log. Read it before changing any protocol
number.

**RL training** (dead-ended, kept for reproduction) — `scripts/run_arm_v13.sh` is
the last arm; it boots the score server in `.venv` and the trainer in `.venv-gen`.

**Eval is always two steps**: generate phrases from a checkpoint
(`scripts/gen_ckpt_phrases.py`), then roll them
(`src/phrase_rl/phase0c_rollout.py`, invoked by `scripts/v13_eval_loop.sh`,
`run_sealed_leg.sh`, `roll_assigned_leg.sh`). Success is the **environment's**
success flag, never a predicate on the text (`phase0c_rollout.py:142`).

## Where the work stopped (2026-09-04, HEAD `ab71eff`)

### The A31 summary grid is COMPLETE

`results/analysis/v2_status_cells.json` → `results/charts/v2_status_board.png`, via
`scripts/make_v2_status_board.py`. Rows = 3 books + no-rules scaffold; columns =
{Adversarial, Natural, Original} × {claude, gemini, qwen}. 33 measured cells; the
3 scaffold×Original cells are marked "X" (not planned, per user, commit `f2726ff8`).
Un-rephrased baselines: Adversarial 24.5, Natural 28.45, Original 36.1.

The result, per `results/experiments.json` id `a31_v2_threebook_sealed` (~62k
episodes, 797 sharded legs, 16 pods, zero failures): **rollout evidence is the
active ingredient.** The rollout-only book wins 7/9 cells and is the only arm with
a large adversarial gain; the train-only book (proxy + corpus statistics, no
rollouts) is **inert**; combining dilutes the rollout signal except on originals.
One harm case: qwen × rollout-only × original, 28.5 vs 36.1 baseline.

### π0 / Bridge line — in flight

PREREG **A33** (regenerate sealed naturals with three balanced authors and a
semantic meaning judge; the lexical gate was dropped because it rejected legit
synonyms and was 2× harsher on the image variant — commit `58fa18b4`), **A34**
(natural condition re-evaluated on the A33 image-conditioned set, 12 arms, ~55k
episodes — `754415f9`), **A35** (Gemini thinking-budget probe, 1024 vs 16384,
queued behind A34 — `7a381ec1`). Latest set: 186 phrases after a judge dropped 6
goal-changing qwen phrases (`25be64ec`).

### π0.5 / LIBERO line — the live decision

Sealed-20 canonical screening **completed** at HEAD
(`results/rules_runs/p_seal/jobs/seal_screen_canon_w0.result.parquet`). Val-15
screening was **retracted from the job queue** and is being run directly as 3
concurrent shards on pod `ps1` to avoid a per-pod claim collision (`b6292c7e`) —
its results are **not yet committed**.

Why the screen exists (`results/analysis/pi05_bank/PREREG.md:33-48`): roughly half
the sealed tasks may be uncompletable by the policy under *any* phrasing, and
floor tasks **inflate** a pooled rephrasing benefit (+4.8pp natural / +5.6pp
adversarial / +9.0pp oracle in the train bank) purely because canonical sits near
zero and any lucky episode is upside. Completability is a property of policy ×
scene, not of wording. Partial results showed 5 of the first 9 sealed tasks at
0/20 canonical and two more at 5% (`PREREG.md:86-88`).

Protocol, fixed in advance: canonical instruction only; screening window inits
**0–19**; final eval reports on the **disjoint** window **30–49**, both arms
measured fresh; drop rule = 0/20 → FLOOR stratum, excluded from human rephrase
collection and from the primary analysis but **reported separately, never silently
dropped**. (The original amendment said window 50–69; LIBERO exposes exactly 50
init states, so that attempt failed with IndexError before rolling any episode —
nothing was contaminated. Corrected at `PREREG.md:69-82`.)

**The open decision, recorded but not made** (`PREREG.md:95-97`): whether surviving
val tasks JOIN the evaluation, with human rephrasings collected for them. This must
be written into `PREREG.md` *before* any rephrasing of val tasks is generated or
rolled. Then: generate natural + adversarial phrases for the surviving sealed set,
then run the sealed test eval.

Sealed-set discipline is enforced three ways: the driver drops `is_sealed(task)`
rows (`config/params.py:327-330`), generation is gated behind `FINAL_EVAL=1` with a
preflight print of every kept line (`PHRASE-GENERATION-RECIPE.md:144-145`), and a
PREREG amendment must be documented *first*.

## Known-stale and inconsistent (verify before quoting)

- **`EXPERIMENT.md` ends 2026-08-06** and is ~1 month behind HEAD; `HANDOFF.md`
  itself warns against trusting it for latest numbers (`HANDOFF.md:332-333`).
- **`RESUME.md` is from 2026-07-13** and describes a pod fleet that no longer
  exists. Superseded by `HANDOFF.md`.
- **`COSTS.md` was last updated 2026-07-08.** Its "grand total ≈ $112" excludes the
  v11–v13 RL runs, the LIBERO four-tier campaign, the rules loops, and all
  Claude/Gemini spend since. Treat as a floor, not a total.
- **`gt_composition` is half-migrated.** `config/params.py:251` now says
  "layouts 0-17 x 1 rep" (n=18) while the invariant above and
  `config/params.py:528` still pin n=36 / 0–17 × 2. Both appear in live code paths.
- **`config/rules_run.py --quick` is silently defeated for pi05**: `pi05_overrides`
  is applied *after* `quick_overrides` (`config/rules_run.py:138-141`), so
  `--env pi05_libero --quick` launches a real 8-iteration opus-5 run. `--env` also
  has no `choices=`, so a typo reaches the driver silently.
- **`ALGORITHM-RULES-SEARCH-AND-DISTILLATION.md` is mid-refactor**: single-edit
  per-rule evaluation was removed 2026-08-10, but the pseudocode comment (`:54`),
  note (12) (`:168-182`) and the cost model (`:228-234`) still assume it exists.
- **`ALGORITHM-RL.md` (v10/v11 history) was never committed** — cited at
  `HANDOFF.md:180`, absent from every branch. Permanently lost with the old Mac.
- **`.ALGORITHM.md.swp`** is a committed vim swap file (`71e6bc77`, 2026-07-19).
  `.gitignore` has no `*.swp` rule.
- **v10 checkpoints 0040–0220 are NOT lost** despite `HANDOFF.md:194`. They are on
  the unmerged branch `origin/ckpt-l40s` (19 `ckpt-sync` commits).

## Machine-migration notes (new host, 2026-09-04)

This clone is a **partial clone** (`--filter=blob:none`) with sparse-checkout
excluding `results/` — `git config remote.origin.partialclonefilter` = `blob:none`,
patterns `/*` and `!/results/`. The full `results/` tree is ~49 GiB of split tar
archives.

**Never run `git ls-tree -l` against `results/`.** In a blobless clone `-l` must
report blob size, which is not local, so git lazily fetches the blob *content* to
answer — a full-tree `-l` scan will start pulling the archives. `GIT_NO_LAZY_FETCH=1`
does **not** stop it on git 2.37.1 (Apple Git). Safe substitutes:
`git ls-tree -r --name-only` for paths, `git cat-file --batch-check
--batch-all-objects` to enumerate *local* objects, and `git cat-file -p
origin/main:<exact-path>` to fetch one named blob deliberately.
