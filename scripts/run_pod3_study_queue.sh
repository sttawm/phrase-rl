#!/usr/bin/env bash
# Pod-3 study queue: scoring venv rebuild -> K=64 proxy scoring (flow + decoded-L2)
# -> spoon rollout shard -> ERT2 top-up for the other tasks.
set -euxo pipefail
eval "$(grep -E '^export (HF_TOKEN|HF_HOME|GEMINI_API_KEY)' ~/.bashrc || true)"
export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
export VLA_DATA_DIR=/workspace/vla_data VLA_LOG_DIR=/workspace/vla_log WANDB_MODE=offline
cd /workspace/phrase-rl
git pull --no-edit || true

# 1) scoring venv (.venv, lerobot fork) — lost in the volume incident
if [ ! -x .venv/bin/python ]; then
  export UV_CACHE_DIR=/root/.cache/uv UV_LINK_MODE=copy   # container disk is 20G, workspace tight
  bash scripts/setup_pod.sh
  rm -rf /root/.cache/uv
fi

# 2) proxy scoring at K=64, per-draw saved (subsampling gives all smaller K free)
.venv/bin/python -m phrase_rl.phase0c_score \
  --contexts data/contexts_0c_match.parquet \
  --phrases data/study_phrases_all.parquet \
  --ckpt juexzz/INTACT-pi0-finetune-rephrase-bridge \
  --k 64 --out data/study_scores_flow_k64.parquet

.venv/bin/python -m phrase_rl.pi0_decode \
  --contexts data/contexts_0c_match.parquet \
  --phrases data/study_phrases_all.parquet \
  --stats-contexts data/contexts_0c_match.parquet \
  --ckpt juexzz/INTACT-pi0-finetune-rephrase-bridge \
  --k 8 --out data/study_scores_l2_k8.parquet

mkdir -p results/overnight/raw
cp -f data/study_scores_flow_k64.parquet data/study_scores_l2_k8.parquet results/overnight/raw/
git add results/overnight/raw/ && git -c user.name=pod3 -c user.email=pod@runpod commit -m "study proxy scores: flow K=64 + decoded-L2 K=8 [pod]" || true
git -c user.name=pod3 -c user.email=pod@runpod pull --rebase --no-edit || true
git push || true

# 3) spoon rollout shard (no self-stop here — pod 3 has more queue after)
cp -f results/phrase_artifacts/study_phrases_rollable.parquet data/
/workspace/INT-ACT/.venv/bin/python - <<'PY'
import pandas as pd
d = pd.read_parquet("data/study_phrases_rollable.parquet")
d[d.task == "widowx_spoon_on_towel"].to_parquet("data/shard_phrases.parquet", index=False)
print(len(d[d.task == "widowx_spoon_on_towel"]), "spoon phrases")
PY
cd /workspace/INT-ACT
/workspace/INT-ACT/.venv/bin/python /workspace/phrase-rl/src/phrase_rl/phase0c_rollout.py \
  --int-act-root /workspace/INT-ACT \
  --config config/experiment/simpler/pi0_finetune_bridge_ev.yaml \
  --ckpt juexzz/INTACT-pi0-finetune-rephrase-bridge \
  --phrases /workspace/phrase-rl/data/shard_phrases.parquet \
  --episode-ids 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 \
  --repeats 12 \
  --out /workspace/phrase-rl/data/rollouts_study_widowx_spoon_on_towel.parquet

# 4) corrected-ERT2 top-up for the other three tasks (v2 phrases; shards rolled v1 junk)
cd /workspace/phrase-rl
/workspace/INT-ACT/.venv/bin/python - <<'PY'
import pandas as pd
d = pd.read_parquet("data/study_phrases_rollable.parquet")
d = d[(d.arm == "redteam2") & (d.task != "widowx_spoon_on_towel")]
d.to_parquet("data/ert2_topup.parquet", index=False)
print(d[["task", "phrase"]].to_string())
PY
cd /workspace/INT-ACT
/workspace/INT-ACT/.venv/bin/python /workspace/phrase-rl/src/phrase_rl/phase0c_rollout.py \
  --int-act-root /workspace/INT-ACT \
  --config config/experiment/simpler/pi0_finetune_bridge_ev.yaml \
  --ckpt juexzz/INTACT-pi0-finetune-rephrase-bridge \
  --phrases /workspace/phrase-rl/data/ert2_topup.parquet \
  --episode-ids 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 \
  --repeats 12 \
  --out /workspace/phrase-rl/data/rollouts_study_ert2_topup.parquet

cd /workspace/phrase-rl
cp -f data/rollouts_study_widowx_spoon_on_towel.parquet data/rollouts_study_ert2_topup.parquet results/overnight/raw/
git add results/overnight/raw/ && git -c user.name=pod3 -c user.email=pod@runpod commit -m "study: spoon shard + ert2 topup x12 [pod]" || true
git -c user.name=pod3 -c user.email=pod@runpod pull --rebase --no-edit || true
git push || true
echo STUDY-QUEUE-DONE
