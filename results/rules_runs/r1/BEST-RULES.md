# r1 training-phase final: best rulebook per model (marked 2026-09-02)

| pass | best iteration | val_avg (proxy) | book |
|---|---|---|---|
| gemini | 3 | 6.3449 | pass_gemini/best_rules.md (sha-verified == rules_03.md) |
| claude | 0 (no-rules anchor) | 6.3097 | pass_claude/best_rules.md == the 1-rule scaffold |
| qwen | 2 | 6.2045 | pass_qwen/best_rules.md (== rules_02.md) |

All numbers are PROXY-scored (no rollouts in r1). These books seed the sim
phase (r1_sim) via --init-rules-from r1.
