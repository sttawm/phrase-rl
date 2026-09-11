# The panel workflows that distilled the nine published rulebooks

Recovered 2026-09-11 from the pre-migration session's persisted workflow
scripts (~/Downloads/.claude.zip -> session a45839fa workflows dir). These are
the EXACT scripts, verbatim (REPO paths still point at the old machine's
/Users/sttawm home).

- `v2_distill_panels.js` — r1 draws (2026-09-03/04): per evidence diet, three
  lens-distillers (register / mechanism / conservative) -> one adversarial
  critic with a diet-specific mandatory check (train_only: leakage; sim_only:
  overfit; both: proxy-vs-rollout consistency) -> one synthesizer (effort
  xhigh). Default (Fable) models. Produced
  results/rules_runs/v2distill/{train_only,sim_only,both}/rules.md.
- `v2_distill_replicates.js` — r2/r3 draws: identical prompts plus an
  "INDEPENDENT draw" instruction, lens order rotated for r3, every agent
  model: 'opus'. Produced the *_r2/*_r3 books.

To produce an exchangeable r4 draw: run v2_distill_replicates.js with
REPS=[4] (and labels r4), REPO fixed to this machine's path, prompts
otherwise verbatim. A cheaper single-prompt alternative (NOT exchangeable
with panel draws) is prompts/distill_r4.md.
