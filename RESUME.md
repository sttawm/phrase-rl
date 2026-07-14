# Resume runbook (pods stopped 2026-07-13, ~$36/day paused)

## State at shutdown
- **RAFT x rollout-reward arm**: step 80 (~epoch 1.7), paired gap trending -7.0 -> -2.5pp
  (~1-sigma, unconfirmed). Checkpoint LOCAL: results/checkpoints/phase2_rollout_raft/latest
  and on pod2 /workspace. Plan: continue to plateau (gap flat 2 consecutive epochs), then
  v5 = GRPO x rollout, from scratch, 10/4/6 task split (EXPERIMENT.md).
- **Proxy arms**: concluded. flow v3 (48.9% deployed), L2 v3 (49.3@280 / 48.4@620) — both
  parity with base greedy 50.2 +/- 1.0-1.5pp bars. Never exceeded it. Chapter closed.
- **Never-finished**: first independent eval of the rollout arm (pod4 tf fix was mid-flight).

## Restart procedure (per pod, after console start)
1. bash /workspace/restore.sh   # bashrc tokens + ssh keys + tmux
2. cd /workspace/phrase-rl && git pull
3. Role scripts:
   - pod2 (RAFT continue): UPDATE_RULE=raft TRAIN_CONTEXTS=data/contexts_sim_rollout.parquet \
       nohup bash scripts/run_pod2_rollout_rl.sh &
   - pod4 (eval loop): finish tf repair if needed (uv sync --reinstall-package tensorflow +
       setup_simpler.sh), then tmux: bash scripts/run_pod4_rollout_eval_loop.sh
   - pod1/pod3: eval stations (pod1 needs SIMPLER build; graphics caps OK)
   - v5 at plateau: build 10-task contexts (sim_train_contexts --tasks per EXPERIMENT.md
       split; needs --tasks flag wired + 6 new Gemini traces), then
       nohup bash scripts/run_pod2_rollout_rl.sh &   # defaults: GRPO + train contexts
4. Monitors: fleet + rollout-arm + eval-loop patterns are in this session's history;
   re-arm equivalents.

## Open decisions for the thinking pause
- v5 task split (10/4/6, EXPERIMENT.md) — confirmed by user; val = 4 held-out tasks nominal.
- Whether RAFT continuation is worth it vs going straight to v5-GRPO.
- Eval budget: per-checkpoint ~2x 1,152 episodes (ERT-ID + val-tasks) across 2 stations.
