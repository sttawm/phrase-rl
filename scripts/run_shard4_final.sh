#!/usr/bin/env bash
# Shard-4 final session (gated on eggplant rollouts finishing): everything that
# remains GPU-bound for the verifier lock, then STOP THE POD (foreground).
#  1) scoring venv on container disk (MFS-safe)
#  2) multi-t contexts: study (0c_match x4 quartiles) + Bridge val (x4)
#  3) eggplant sim-grounded contexts from /workspace/study_traj recordings
#  4) verbose-flow + decode scoring: study multit + eggplant sim (35 phrases each)
#  5) verifier features on val multit (own/hard/easy, resume-safe per (ep,t))
#  6) publish everything, self-stop
set -euxo pipefail
eval "$(grep -E '^export (HF_TOKEN|HF_HOME)' ~/.bashrc || true)"
export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
cd /workspace/phrase-rl

# gate on the eggplant shard finishing (old-script run: prints SHARD-DONE, no self-stop)
until grep -q "SHARD-DONE widowx_put_eggplant_in_basket" /workspace/shard.log 2>/dev/null; do sleep 120; done
git pull --no-edit || true

# 1) container-disk venv (same fallback as shard2; pytest is a RUNTIME dep of the fork)
if [ ! -x /root/venv/bin/python ]; then
  export PATH=$HOME/.local/bin:$PATH UV_CACHE_DIR=/root/.cache/uv UV_LINK_MODE=hardlink
  command -v uv >/dev/null || (curl -LsSf https://astral.sh/uv/install.sh | sh)
  export PATH=$HOME/.local/bin:$PATH
  uv venv --python 3.11 /root/venv
  ln -sfn /root/venv .venv
  uv pip install -p /root/venv/bin/python -e ".[gpu]"
  uv pip install -p /root/venv/bin/python \
    "lerobot @ git+https://github.com/IrvingF7/lerobot.git@35f6e02315dcb3fa25f3a740478265c6c793a95e" \
    "transformers==4.48.3" pytest av
fi
PY=/root/venv/bin/python

# 2) multi-t contexts (selective per-episode hub pulls; no full dataset needed)
[ -f data/contexts_0c_match_multit.parquet ] || $PY -m phrase_rl.multi_t_contexts \
  --contexts data/contexts_0c_match.parquet --out data/contexts_0c_match_multit.parquet
[ -f data/contexts_val_multit.parquet ] || $PY -m phrase_rl.multi_t_contexts \
  --contexts data/contexts_val_0b.parquet --out data/contexts_val_multit.parquet

# 3) eggplant sim-grounded contexts (subsample recordings to ~60 contexts)
if [ ! -f data/sim_contexts_eggplant.parquet ]; then
  $PY -m phrase_rl.sim_contexts_extract --record-dir /workspace/study_traj \
    --per-episode 1 --out data/sim_contexts_eggplant_all.parquet
  $PY - <<'EOF'
import pandas as pd
d = pd.read_parquet("data/sim_contexts_eggplant_all.parquet").sample(n=60, random_state=0)
d.to_parquet("data/sim_contexts_eggplant.parquet", index=False)
print(len(d), "sim contexts kept")
EOF
fi

cp -f results/phrase_artifacts/study_phrases_all.parquet data/
cp -f results/phrase_artifacts/chunk_stats.parquet data/

# 4a) study multit: verbose flow + decode
$PY -m phrase_rl.study_verbose_rescore \
  --contexts data/contexts_0c_match_multit.parquet --phrases data/study_phrases_all.parquet \
  --ckpt juexzz/INTACT-pi0-finetune-rephrase-bridge --k 8 --out data/study_multit_verbose.parquet
$PY -m phrase_rl.pi0_decode \
  --contexts data/contexts_0c_match_multit.parquet --phrases data/study_phrases_all.parquet \
  --stats-contexts data/chunk_stats.parquet \
  --ckpt juexzz/INTACT-pi0-finetune-rephrase-bridge --k 4 --out data/study_multit_l2.parquet

# 4b) eggplant sim-grounded: verbose flow + decode (its phrases only ride along via task match)
$PY -m phrase_rl.study_verbose_rescore \
  --contexts data/sim_contexts_eggplant.parquet --phrases data/study_phrases_all.parquet \
  --ckpt juexzz/INTACT-pi0-finetune-rephrase-bridge --k 8 --out data/eggplant_sim_verbose.parquet
$PY -m phrase_rl.pi0_decode \
  --contexts data/sim_contexts_eggplant.parquet --phrases data/study_phrases_all.parquet \
  --stats-contexts data/chunk_stats.parquet \
  --ckpt juexzz/INTACT-pi0-finetune-rephrase-bridge --k 4 --out data/eggplant_sim_l2.parquet

# 5) verifier features on Bridge val multit (for the 4-point val-set test)
$PY -m phrase_rl.build_verifier_data \
  --contexts data/contexts_val_multit.parquet --stats-contexts data/chunk_stats.parquet \
  --ckpt juexzz/INTACT-pi0-finetune-rephrase-bridge \
  --k 8 --k-decode 4 --out data/verifier_features_val_multit.parquet

# 6) publish
mkdir -p results/overnight/raw
cp -f data/study_multit_verbose.parquet data/study_multit_l2.parquet \
      data/eggplant_sim_verbose.parquet data/eggplant_sim_l2.parquet \
      data/sim_contexts_eggplant.parquet data/verifier_features_val_multit.parquet \
      results/overnight/raw/
git add results/overnight/raw/ \
  && git -c user.name=shard4 -c user.email=pod@runpod commit -m "multi-t study+val scoring, eggplant sim-grounded scores [pod]" \
  && git -c user.name=shard4 -c user.email=pod@runpod pull --rebase --no-edit \
  && git push
echo SHARD4-FINAL-DONE

export "$(tr '\0' '\n' < /proc/1/environ | grep '^RUNPOD_POD_ID=')" || true
export "$(tr '\0' '\n' < /proc/1/environ | grep '^RUNPOD_API_KEY=')" || true
runpodctl config --apiKey "$RUNPOD_API_KEY" 2>&1 | tail -1 || true
# FOREGROUND stop (nohup+& dies with tmux server teardown)
sleep 20
runpodctl stop pod "$RUNPOD_POD_ID" 2>&1 | tee -a /workspace/selfstop.log
