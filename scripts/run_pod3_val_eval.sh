#!/usr/bin/env bash
# Pod-3 validation rollout eval (episode_ids 0-9 = VAL split; test = 10-33, untouched).
# Arms: original / base / tuned-flow (+ tuned-l2 when its best_val exists).
# Self-contained on pod 3: pulls adapters from the training pods, generates
# deployment phrases locally (.venv-gen, greedy p_single + self-owned traces),
# then rolls out in SIMPLER (INT-ACT venv). Idempotent: rollouts resume by key,
# rerunning after the l2 adapter appears only adds the new arm's episodes.
set -euxo pipefail
eval "$(grep -E '^export (HF_TOKEN|HF_HOME|GEMINI_API_KEY)' ~/.bashrc || true)"
export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
POD1="ssh -o StrictHostKeyChecking=no -p 39484 root@38.147.83.25"
POD2="ssh -i $HOME/.ssh/id_ed25519 -o StrictHostKeyChecking=no -p 22159 root@69.30.85.249"
cd /workspace/phrase-rl
git pull --no-edit
mkdir -p data results/checkpoints/eval_adapters

# 1) pull freshest best_val adapters (flow required, l2 optional)
scp -q -r -o StrictHostKeyChecking=no -P 39484 \
  root@38.147.83.25:/workspace/phrase-rl/results/checkpoints/phase2/best_val \
  results/checkpoints/eval_adapters/flow
scp -q -r -o StrictHostKeyChecking=no -P 22159 -i "$HOME/.ssh/id_ed25519" \
  root@69.30.85.249:/workspace/phrase-rl/results/checkpoints/phase2_l2/best_val \
  results/checkpoints/eval_adapters/l2 2>/dev/null || echo "l2 best_val not there yet — flow-only round"
cat results/checkpoints/eval_adapters/flow/info.json || true

# 2) deployment phrases per arm (greedy p_single; nominal instructions + cached traces)
cp -f results/phrase_artifacts/contexts_0c_tasks.parquet data/contexts_0c_tasks.parquet
.venv-gen/bin/python -m phrase_rl.phase4_generate_eval \
  --tasks results/phrase_artifacts/cover_gemini_tasks.parquet \
  --adapter results/checkpoints/eval_adapters/flow \
  --out data/phrases_val_eval_flow.parquet
if [ -d results/checkpoints/eval_adapters/l2 ]; then
  .venv-gen/bin/python -m phrase_rl.phase4_generate_eval \
    --tasks results/phrase_artifacts/cover_gemini_tasks.parquet \
    --adapter results/checkpoints/eval_adapters/l2 \
    --out data/phrases_val_eval_l2.parquet
fi

# 3) merge arms: original + base from the flow file; tuned arms renamed per reward
.venv/bin/python - <<'PY'
import os
import pandas as pd
fl = pd.read_parquet("data/phrases_val_eval_flow.parquet")
fl["arm"] = fl["arm"].map({"base": "base", "original": "original", "tuned": "tuned_flow"})
frames = [fl]
if os.path.exists("data/phrases_val_eval_l2.parquet"):
    l2 = pd.read_parquet("data/phrases_val_eval_l2.parquet")
    l2 = l2[l2.arm == "tuned"].assign(arm="tuned_l2")
    frames.append(l2)
out = pd.concat(frames, ignore_index=True).drop_duplicates(["task", "arm"])
out.to_parquet("data/phrases_val_eval.parquet", index=False)
print(out.to_string())
PY

# 4) VAL rollouts: episode_ids 0-9 (never use 10+ here — that's the TEST split)
cd /workspace/INT-ACT
/workspace/INT-ACT/.venv/bin/python /workspace/phrase-rl/src/phrase_rl/phase0c_rollout.py \
  --int-act-root /workspace/INT-ACT \
  --config config/experiment/simpler/pi0_finetune_bridge_ev.yaml \
  --ckpt juexzz/INTACT-pi0-finetune-rephrase-bridge \
  --phrases /workspace/phrase-rl/data/phrases_val_eval.parquet \
  --episode-ids 0 1 2 3 4 5 6 7 8 9 \
  --out /workspace/phrase-rl/data/rollouts_val_eval.parquet

# 5) publish
cd /workspace/phrase-rl
mkdir -p results/overnight/raw
cp -f data/phrases_val_eval.parquet data/rollouts_val_eval.parquet results/overnight/raw/
git add results/overnight/raw/
git -c user.name=pod3 -c user.email=pod@runpod commit -m "val rollout eval: original/base/tuned arms, ep 0-9 [pod]" || true
git -c user.name=pod3 -c user.email=pod@runpod pull --rebase --no-edit || true
git push || true
.venv/bin/python - <<'PY'
import pandas as pd
d = pd.read_parquet("data/rollouts_val_eval.parquet")
print(d.groupby(["task", "arm"])["success"].agg(["mean", "size"]).to_string())
print("\n=== overall by arm ===")
print(d.groupby("arm")["success"].mean().to_string())
PY
echo VAL-EVAL-DONE
