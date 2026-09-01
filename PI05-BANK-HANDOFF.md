# π0.5/LIBERO phrase bank — handoff for the bank-building session

Goal: a rollout-scored phrase bank for the `pi05_libero` environment, analogous
to the bridge_pi0 score bank the rules loop distills from.

## What already exists (start here, costs nothing)
- `results/analysis/fourtier_live/` — 13,025 LIBERO rollout episodes, 30 tasks,
  four phrase tiers (oracle / original / natural / adversarial), keyed
  (suite, task, phrase, init). Inits 0–9 were screening, 20–49 confirmation.
  Converting these to bank rows (task, phrase, gt_success, n_ctx) seeds a bank
  of several hundred measured phrases immediately.
- `FOURTIER-LIBERO.md` — the full analysis of that data (macro/micro trends,
  finetune-set split, the trained-string inversion, oracle-search ceiling).
- `~/dev/interactive-pi` (`sttawm/interactive-vlas`), `pi05_libero/eval/` —
  the rollout harness (`fourtier_eval.py` etc.) that produced the data; this is
  the scoring engine for any new phrases.
- `prompts/rules_loop/generate.md` + `scripts/search_boards.py` — the exact
  historical phrase-generation prompt and the iterative oracle board search
  (16-board, keep-4, informed regen), reusable for bank growth.

## Substrate decision (made for bridge, applies here)
No calibrated proxy exists for π0.5 — the bank is ROLLOUT-scored (the rules
loop's `phase=sim` pattern: seeded inits, measured success, gt_success 0–100,
n_ctx = episodes). Bank growth therefore costs real GPU rollouts; batch new
phrases and reuse the four-tier fleet bringup scripts (Vulkan render pods —
see memory notes: ICD manifest under /usr/share/vulkan/icd.d, libegl1).

## Integration point
`ENVIRONMENTS` registry in `scripts/rules_loop_driver.py` — add/complete the
`pi05_libero` entry: bank sources, sim task list, rollout recipe (delegating to
the interactive-pi harness), corpus (LIBERO instruction strings; the l90 task
suite vocabulary differs sharply from Bridge's). The registry was designed so
this is an entry, not a refactor (see config/params.py DECISIONS 2026-08-30).

## Conventions that bind here
- Fresh run ids under results/rules_runs/; never reuse a stale bank.
- Rollout determinism: decode noise seeded, phrase excluded from the seed.
- Sealed-set discipline for any generation touching sealed phrases.
- Derived artifacts → results/analysis/; checkpoints → results/checkpoints.
- Pod ops: script-file launches, secrets piped from ~/.zshrc, verify pushes
  with git ls-tree, archive anything valuable before pod termination.

## Coordination
A parallel session ("the r1 operator") is babysitting the bridge_pi0 rules
loop on worker pods rw1–rw3 and pushes to main frequently — pull-rebase before
pushing, avoid touching results/rules_runs/r1/ and scripts/rules_loop_worker.sh
without coordinating.

## Split decision (agreed 2026-09-01, build splits.json from this before any new generation)
- Universe: l90 = 90 tasks (20 touched by four-tier: 18 l90_clean + 2 l90_trained_string);
  in-finetune = 40 tasks (10 goal touched, 30 spatial/object/long untouched).
- TRAIN (70): 10 goal + all 20 touched l90 + 40 fresh l90. All existing four-tier
  scores land here — nothing discarded. l90_trained_string tasks pinned to train.
- VAL (10): untouched l90. Small on purpose: evaluated every iteration by rollout.
- SEALED TEST (20): untouched l90, PREREG discipline (no generation, FINAL_EVAL=1,
  preflight print), first measurement at >=20 virgin inits.
- RESERVE: 30 untouched in-finetune tasks (spatial/object/long) for a later
  in-finetune sealed strand. Not assigned now.
- Assignment: uniform random, committed seed, stratified ONLY by the free
  vocab label (in-vocab = every content word of the task string appears in the
  40 finetune strings; computed from text, no rollouts). No pre-measurement.
- Loop reporting: fixed seed-pinned sample of ~48 train bases, rerolled every
  iteration on the SAME screen-window init indices (episode-level same-draw
  pairing); val = the 10 val tasks per iteration; screens inits 0-9, confirm
  20-29, sealed windows 30+.
