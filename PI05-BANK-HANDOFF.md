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
