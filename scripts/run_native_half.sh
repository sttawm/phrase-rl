#!/usr/bin/env bash
# Native half of the fine-discrimination exam, whole chain on one render+score pod:
# record successful trajectories for the 4 IN-VOCAB val tasks under their
# oracle-best phrases (ground-truth-quality contexts, mirrors the OOV half's
# sim-grounded a* recordings) -> extract 5 frames/episode -> score the exam panel
# via the local score server -> push features. Then make_fine_grid native/all runs
# locally (driver side).
set -uo pipefail
eval "$(grep -E '^export (HF_TOKEN|HF_HOME)' ~/.bashrc || true)"
export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
export VLA_DATA_DIR=/workspace/vla_data VLA_LOG_DIR=/workspace/vla_log WANDB_MODE=offline
mkdir -p "$VLA_DATA_DIR" "$VLA_LOG_DIR"
cd /workspace/phrase-rl
mark() { echo "[native $(date -u +%H:%M)] $*" | tee -a /workspace/native.log; }

mark "build phrase list (val8_oracle, ID tasks)"
.venv-gen/bin/python - <<'PY'
import pandas as pd
IV = ["widowx_carrot_on_plate", "widowx_spoon_on_towel", "widowx_stack_cube",
      "widowx_put_eggplant_in_basket"]
d = pd.read_parquet("results/phrase_artifacts/ph_val8_reference.parquet")
d = d[(d.arm == "val8_oracle") & (d.task.isin(IV))].copy()
d["arm"] = "native_best"
d["instruction"] = d["phrase"]
d[["task", "arm", "phrase", "instruction"]].to_parquet("data/phrases_native_best.parquet", index=False)
print(d[["task", "phrase"]].to_string())
PY

mark "record trajectories (4 tasks x 24 layouts x 2 reps, success-only)"
cd /workspace/INT-ACT
.venv/bin/python /workspace/phrase-rl/src/phrase_rl/phase0c_rollout.py \
  --int-act-root /workspace/INT-ACT \
  --config config/experiment/simpler/pi0_finetune_bridge_ev.yaml \
  --ckpt juexzz/INTACT-pi0-finetune-rephrase-bridge \
  --phrases /workspace/phrase-rl/data/phrases_native_best.parquet \
  --episode-ids $(seq 0 23) --repeats 2 \
  --out /workspace/phrase-rl/data/rollouts_native_record.parquet \
  --record-dir /workspace/phrase-rl/data/sim_traj_native --record-success-only \
  > /workspace/native_record.log 2>&1 || { mark "RECORD FAIL"; exit 1; }
cd /workspace/phrase-rl
mark "recorded: $(ls data/sim_traj_native | wc -l) episodes"

mark "extract 5 frames/episode"
PYTHONPATH=src .venv-gen/bin/python -m phrase_rl.sim_contexts_extract \
  --record-dir data/sim_traj_native --out data/contexts_sim_native.parquet --per-episode 5 \
  >> /workspace/native.log 2>&1 || { mark "EXTRACT FAIL"; exit 1; }

mark "score exam panel on local score server"
PYTHONPATH=/workspace/phrase-rl/src .venv-gen/bin/python scripts/fine_exam_score.py \
  --contexts data/contexts_sim_native.parquet \
  --phrases results/analysis/fine_exam_phrases.parquet \
  --out results/analysis/fine_exam_features_native.parquet \
  --ipc-dir /workspace/ipc6 >> /workspace/native.log 2>&1 || { mark "SCORE FAIL"; exit 1; }

timeout 300 bash -c "git add results/analysis/fine_exam_features_native.parquet && git commit -q -m 'fine exam: native-half features [pod6]' && git pull -q --rebase && git push -q" \
  && mark "PUSHED native features" || mark "PUSH-DEFERRED native features"
mark "NATIVE-HALF-COMPLETE"
