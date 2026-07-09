#!/usr/bin/env bash
# Pod-3 RED-TEAM validation eval (user 2026-07-10: next eval matches CoVer's setup).
# Inputs = CoVer's 4 ERT instructions. Arms:
#   redteam_direct : ERT phrase straight to pi0 (their pi0-baseline replication)
#   gemini_zeroshot: Gemini's single simple phrase (the ablation CoVer never ran)
#   base           : Gemini trace -> base Qwen single greedy phrase
#   tuned_flow/l2  : same with the freshest best_val adapters
# Protocol: OUR val states (ep 0-24, stock horizon) — CoVer's exact reset-seed/150-step
# protocol stays SEALED for the one final run (avoids checkpoint-selection contamination).
set -euxo pipefail
eval "$(grep -E '^export (HF_TOKEN|HF_HOME|GEMINI_API_KEY)' ~/.bashrc || true)"
export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
export VLA_DATA_DIR=/workspace/vla_data VLA_LOG_DIR=/workspace/vla_log WANDB_MODE=offline
mkdir -p "$VLA_DATA_DIR" "$VLA_LOG_DIR"
cd /workspace/phrase-rl
git pull --no-edit
mkdir -p data results/checkpoints/eval_adapters results/overnight/raw

# 1) Gemini assets (traces on ERT phrases + zeroshot arm) — 8 calls, cached in git afterwards
if [ ! -f results/phrase_artifacts/redteam_eval_assets.parquet ]; then
  .venv-gen/bin/python -m phrase_rl.gemini_redteam_assets \
    --out results/phrase_artifacts/redteam_eval_assets.parquet
  git add results/phrase_artifacts/redteam_eval_assets.parquet
  git -c user.name=pod3 -c user.email=pod@runpod commit -m "redteam eval assets: Gemini traces + zeroshot phrases [pod]" || true
  git -c user.name=pod3 -c user.email=pod@runpod pull --rebase --no-edit || true; git push || true
fi

# 2) fresh adapters
rm -rf results/checkpoints/eval_adapters/flow results/checkpoints/eval_adapters/l2
scp -q -r -o StrictHostKeyChecking=no -P 39484 \
  root@38.147.83.25:/workspace/phrase-rl/results/checkpoints/phase2/best_val \
  results/checkpoints/eval_adapters/flow
scp -q -r -o StrictHostKeyChecking=no -P 22159 -i "$HOME/.ssh/id_ed25519" \
  root@69.30.85.249:/workspace/phrase-rl/results/checkpoints/phase2_l2/best_val \
  results/checkpoints/eval_adapters/l2
cat results/checkpoints/eval_adapters/flow/info.json results/checkpoints/eval_adapters/l2/info.json || true

# 3) tuned + base phrases (Gemini traces via assets)
cp -f results/phrase_artifacts/contexts_0c_tasks.parquet data/contexts_0c_tasks.parquet
.venv-gen/bin/python -m phrase_rl.phase4_generate_eval \
  --tasks results/phrase_artifacts/cover_gemini_tasks.parquet \
  --assets results/phrase_artifacts/redteam_eval_assets.parquet \
  --adapter results/checkpoints/eval_adapters/flow \
  --out data/phrases_rt_flow.parquet
.venv-gen/bin/python -m phrase_rl.phase4_generate_eval \
  --tasks results/phrase_artifacts/cover_gemini_tasks.parquet \
  --assets results/phrase_artifacts/redteam_eval_assets.parquet \
  --adapter results/checkpoints/eval_adapters/l2 \
  --out data/phrases_rt_l2.parquet

# 4) merge the 5 arms
.venv-gen/bin/python - <<'PY'
import pandas as pd
fl = pd.read_parquet("data/phrases_rt_flow.parquet")
fl["arm"] = fl["arm"].map({"base": "base", "original": "redteam_direct", "tuned": "tuned_flow"})
l2 = pd.read_parquet("data/phrases_rt_l2.parquet")
l2 = l2[l2.arm == "tuned"].assign(arm="tuned_l2")
a = pd.read_parquet("results/phrase_artifacts/redteam_eval_assets.parquet")
zs = pd.DataFrame({"task": a.task, "arm": "gemini_zeroshot",
                   "phrase": a.gemini_zeroshot, "instruction": a.ert_instruction})
out = pd.concat([fl, l2, zs], ignore_index=True).drop_duplicates(["task", "arm"])
out.to_parquet("data/phrases_redteam_eval.parquet", index=False)
print(out[["task", "arm", "phrase"]].to_string())
PY

# 5) rollouts on the VAL states (0-24)
cd /workspace/INT-ACT
/workspace/INT-ACT/.venv/bin/python /workspace/phrase-rl/src/phrase_rl/phase0c_rollout.py \
  --int-act-root /workspace/INT-ACT \
  --config config/experiment/simpler/pi0_finetune_bridge_ev.yaml \
  --ckpt juexzz/INTACT-pi0-finetune-rephrase-bridge \
  --phrases /workspace/phrase-rl/data/phrases_redteam_eval.parquet \
  --episode-ids 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 \
  --out /workspace/phrase-rl/data/rollouts_redteam_eval.parquet

# 6) publish + table
cd /workspace/phrase-rl
cp -f data/phrases_redteam_eval.parquet data/rollouts_redteam_eval.parquet results/overnight/raw/
git add results/overnight/raw/
git -c user.name=pod3 -c user.email=pod@runpod commit -m "red-team val eval: 5 arms, ep 0-24 [pod]" || true
git -c user.name=pod3 -c user.email=pod@runpod pull --rebase --no-edit || true
git push || true
.venv-gen/bin/python - <<'PY'
import pandas as pd
d = pd.read_parquet("data/rollouts_redteam_eval.parquet")
print((d.pivot_table(index="task", columns="arm", values="success", aggfunc="mean") * 100).round(0).to_string())
print("\noverall (%):"); print((d.groupby("arm")["success"].mean() * 100).round(1).to_string())
PY
echo REDTEAM-EVAL-DONE
