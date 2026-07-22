# Checkpoint archive (split tars, see scripts/ckpt_archive.sh)

Restore: `scripts/ckpt_archive.sh unpack <name> [dest_parent]`

| name | what |
|---|---|
| phase2 | v1 GRPO latest (w/ optimizer state) |
| phase2_v1_degenerated | v1 degenerate run (forensics) |
| phase2_flow / phase2_l2 | first flow/L2 A/B best_val |
| phase2_v2_flow_final / phase2_v2_l2_final | phase-2 v2 A/B best_val (the "v2 on flow/L2" arms) |
| phase2_v3_flow_final / phase2_v3_l2_final | phase-2 v3 A/B best_val |
| phase2_rollout_raft | rollout-RAFT arm |
| phase2_v7 | v7 config/stub (no weights yet) |
| sft17k / sft17k_v2 | SFT-17k v1 / v2 (latest+best_val+final, resumable) |
| adapters_A / adapters_B | pod1 deploy trees (v6 steps incl. staged s80) |
| A_bestval / A_final_0279 / B_bestval / B_final_0268 | v5 arm A/B named checkpoints |
| A_v6step_0060 / B_v6_final_s190 | v6 named checkpoints |

Uncurated intermediates live only in local mirrors (`pod1_mirror/`, `tmp_archive_rescue/`), not in git.
