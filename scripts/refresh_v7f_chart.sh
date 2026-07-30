#!/usr/bin/env bash
# One-command v7f chart refresh: ALWAYS fetch fresh train log + curves first.
# Curves scp into UNTRACKED _scp_* staging files (never over git-tracked curve
# files — a stale scp'd copy once rode a pull autostash into a commit and
# deleted a measured point). make_v7f_progress.py max-n merges git + staging.
set -uo pipefail
cd "$(dirname "$0")/.."
scp -q -o ConnectTimeout=20 -P 29671 -i ~/.ssh/id_ed25519 root@103.196.86.46:/workspace/phrase-rl/results/checkpoints/phase2_v7f/train_log.jsonl results/analysis/v7f_train_log.jsonl 2>/dev/null
scp -q -o ConnectTimeout=20 -P 35163 -i ~/.ssh/id_ed25519 root@38.147.83.23:/workspace/phrase-rl/results/analysis/v7f_adv_curve.jsonl results/analysis/_scp_v7f_adv_curve.jsonl 2>/dev/null || true
scp -q -o ConnectTimeout=20 -P 35163 -i ~/.ssh/id_ed25519 root@38.147.83.23:/workspace/phrase-rl/results/analysis/v7f_pol_curve.jsonl results/analysis/_scp_v7f_pol_curve.jsonl 2>/dev/null || true
scp -q -o ConnectTimeout=20 -P 21673 -i ~/.ssh/id_ed25519 root@38.147.83.30:/workspace/phrase-rl/results/analysis/v7f_pol_curve.jsonl results/analysis/_scp_v7f_pol_curve_p6.jsonl 2>/dev/null || true
.venv/bin/python scripts/make_v7f_progress.py
