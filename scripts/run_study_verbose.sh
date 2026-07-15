#!/usr/bin/env bash
# Study verbose re-score (pod 3, gated behind idmirror): raw v/u at the study
# contexts so flow/both verifiers can be evaluated against rollout success.
set -euxo pipefail
eval "$(grep -E '^export (HF_TOKEN|HF_HOME)' ~/.bashrc || true)"
export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
cd /workspace/phrase-rl
while tmux has-session -t idmirror 2>/dev/null; do sleep 120; done
git pull --no-edit || true
test -f results/overnight/raw/rollouts_id_nominal.parquet  # idmirror must have finished clean

cp -f results/phrase_artifacts/study_phrases_all.parquet data/
.venv/bin/python -m phrase_rl.study_verbose_rescore \
  --contexts data/contexts_0c_match.parquet \
  --phrases data/study_phrases_all.parquet \
  --ckpt juexzz/INTACT-pi0-finetune-rephrase-bridge \
  --k 8 --out data/study_verbose_flow.parquet

cp -f data/study_verbose_flow.parquet results/overnight/raw/
git add results/overnight/raw/ \
  && git -c user.name=pod3 -c user.email=pod@runpod commit -m "study verbose re-score: raw v/u for verifier eval [pod]" \
  && git -c user.name=pod3 -c user.email=pod@runpod pull --rebase --no-edit \
  && git push
echo STUDY-VERBOSE-DONE
