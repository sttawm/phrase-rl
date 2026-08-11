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
