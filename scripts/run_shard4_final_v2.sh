#!/usr/bin/env bash
# Shard-4 final session v2: scoring runs CONCURRENTLY with the slow eggplant
# rollout tail (GPU headroom confirmed); publishes incrementally so the frame
# curve unblocks early. Only the SELF-STOP gates on eggplant finishing.
set -euxo pipefail
eval "$(grep -E '^export (HF_TOKEN|HF_HOME)' ~/.bashrc || true)"
export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
cd /workspace/phrase-rl
git pull --no-edit || true

PUBLISH() {  # $@: files under data/ to publish now
  mkdir -p results/overnight/raw
  for f in "$@"; do cp -f "data/$f" results/overnight/raw/; done
  git add results/overnight/raw/ \
    && git -c user.name=shard4 -c user.email=pod@runpod commit -m "shard4 v2: $* [pod]" \
    && git -c user.name=shard4 -c user.email=pod@runpod pull --rebase --no-edit \
    && git push
}

# 1) container-disk scoring venv (pytest+av are runtime deps)
if [ ! -x /root/venv/bin/python ]; then
  export PATH=$HOME/.local/bin:$PATH UV_CACHE_DIR=/root/.cache/uv UV_LINK_MODE=hardlink
  command -v uv >/dev/null || (curl -LsSf https://astral.sh/uv/install.sh | sh)
  export PATH=$HOME/.local/bin:$PATH
  uv venv --python 3.11 /root/venv && ln -sfn /root/venv .venv
  uv pip install -p /root/venv/bin/python -e ".[gpu]"
  uv pip install -p /root/venv/bin/python \
    "lerobot @ git+https://github.com/IrvingF7/lerobot.git@35f6e02315dcb3fa25f3a740478265c6c793a95e" \
    "transformers==4.48.3" pytest av
  rm -rf /root/.cache/uv
fi
PY=/root/venv/bin/python

# 2) multi-t contexts (hub pulls, no GPU)
[ -f data/contexts_val_multit.parquet ] || $PY -m phrase_rl.multi_t_contexts \
  --contexts data/contexts_val_0b.parquet --out data/contexts_val_multit.parquet
[ -f data/contexts_0c_match_multit.parquet ] || $PY -m phrase_rl.multi_t_contexts \
  --contexts data/contexts_0c_match.parquet --out data/contexts_0c_match_multit.parquet

cp -f results/phrase_artifacts/study_phrases_all.parquet data/
cp -f results/phrase_artifacts/chunk_stats.parquet data/

# 3) FRAME-CURVE UNBLOCK FIRST: verifier features on 4-point Bridge val
$PY -m phrase_rl.build_verifier_data \
  --contexts data/contexts_val_multit.parquet --stats-contexts data/chunk_stats.parquet \
  --ckpt juexzz/INTACT-pi0-finetune-rephrase-bridge \
  --k 8 --k-decode 4 --out data/verifier_features_val_multit.parquet
PUBLISH verifier_features_val_multit.parquet

# 4) study multi-t: verbose flow + decode
$PY -m phrase_rl.study_verbose_rescore \
  --contexts data/contexts_0c_match_multit.parquet --phrases data/study_phrases_all.parquet \
  --ckpt juexzz/INTACT-pi0-finetune-rephrase-bridge --k 8 --out data/study_multit_verbose.parquet
$PY -m phrase_rl.pi0_decode \
  --contexts data/contexts_0c_match_multit.parquet --phrases data/study_phrases_all.parquet \
  --stats-contexts data/chunk_stats.parquet \
  --ckpt juexzz/INTACT-pi0-finetune-rephrase-bridge --k 4 --out data/study_multit_l2.parquet
PUBLISH study_multit_verbose.parquet study_multit_l2.parquet

# 5) eggplant sim-grounding (recordings that exist by now are plenty)
if [ ! -f data/sim_contexts_eggplant.parquet ]; then
  $PY -m phrase_rl.sim_contexts_extract --record-dir /workspace/study_traj \
    --per-episode 1 --out data/sim_contexts_eggplant_all.parquet
  $PY - <<'EOF'
import pandas as pd
d = pd.read_parquet("data/sim_contexts_eggplant_all.parquet")
d = d.sample(n=min(60, len(d)), random_state=0)
d.to_parquet("data/sim_contexts_eggplant.parquet", index=False)
print(len(d), "sim contexts kept")
EOF
fi
$PY -m phrase_rl.study_verbose_rescore \
  --contexts data/sim_contexts_eggplant.parquet --phrases data/study_phrases_all.parquet \
  --ckpt juexzz/INTACT-pi0-finetune-rephrase-bridge --k 8 --out data/eggplant_sim_verbose.parquet
$PY -m phrase_rl.pi0_decode \
  --contexts data/sim_contexts_eggplant.parquet --phrases data/study_phrases_all.parquet \
  --stats-contexts data/chunk_stats.parquet \
  --ckpt juexzz/INTACT-pi0-finetune-rephrase-bridge --k 4 --out data/eggplant_sim_l2.parquet
PUBLISH eggplant_sim_verbose.parquet eggplant_sim_l2.parquet sim_contexts_eggplant.parquet
echo SHARD4-V2-SCORING-DONE

# 6) wait for the eggplant rollout to finish (its own session publishes it), then stop
until grep -q "SHARD-DONE widowx_put_eggplant_in_basket" /workspace/shard.log 2>/dev/null; do sleep 180; done
echo SHARD4-FINAL-DONE
export "$(tr '\0' '\n' < /proc/1/environ | grep '^RUNPOD_POD_ID=')" || true
export "$(tr '\0' '\n' < /proc/1/environ | grep '^RUNPOD_API_KEY=')" || true
runpodctl config --apiKey "$RUNPOD_API_KEY" 2>&1 | tail -1 || true
sleep 30
runpodctl stop pod "$RUNPOD_POD_ID" 2>&1 | tee -a /workspace/selfstop.log
