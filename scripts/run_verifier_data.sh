#!/usr/bin/env bash
# Verifier feature extraction (shard 2): waits for the scoring venv build, then
# featurizes train (2000) + val (250) Bridge contexts, publishes, self-stops.
set -euxo pipefail
eval "$(grep -E '^export (HF_TOKEN|HF_HOME)' ~/.bashrc || true)"
export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
cd /workspace/phrase-rl
git pull --no-edit || true

# gate on the venv build kicked off separately (setup tmux session)
until grep -qE "SETUP-OK|SETUP-FAIL" /workspace/setup.log 2>/dev/null; do sleep 30; done
grep -q SETUP-OK /workspace/setup.log  # fail loud if the build failed

.venv/bin/python -m phrase_rl.build_verifier_data \
  --contexts data/contexts_train.parquet --stats-contexts data/contexts_train.parquet \
  --ckpt juexzz/INTACT-pi0-finetune-rephrase-bridge \
  --k 8 --k-decode 4 --out data/verifier_features_train.parquet

.venv/bin/python -m phrase_rl.build_verifier_data \
  --contexts data/contexts_val_0b.parquet --stats-contexts data/contexts_train.parquet \
  --ckpt juexzz/INTACT-pi0-finetune-rephrase-bridge \
  --k 8 --k-decode 4 --out data/verifier_features_val.parquet

mkdir -p results/overnight/raw
cp -f data/verifier_features_train.parquet data/verifier_features_val.parquet \
      data/verifier_features_train_meta.json results/overnight/raw/
git add results/overnight/raw/ \
  && git -c user.name=shard2 -c user.email=pod@runpod commit -m "verifier features: 2250 Bridge contexts x3 conditions, raw v/u/decoded [pod]" \
  && git -c user.name=shard2 -c user.email=pod@runpod pull --rebase --no-edit \
  && git push
echo VERIFIER-DATA-DONE

# self-stop, LOUDLY this time (shard2's previous silent stop failure)
export "$(tr '\0' '\n' < /proc/1/environ | grep '^RUNPOD_POD_ID=')" || true
export "$(tr '\0' '\n' < /proc/1/environ | grep '^RUNPOD_API_KEY=')" || true
runpodctl config --apiKey "$RUNPOD_API_KEY" 2>&1 | tail -1
nohup sh -c "sleep 20; runpodctl stop pod $RUNPOD_POD_ID >> /workspace/selfstop.log 2>&1" &
echo SELF-STOP-QUEUED
