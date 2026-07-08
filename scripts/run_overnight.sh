#!/usr/bin/env bash
# Overnight redo (2026-07-07): step-0 A/B (inline vs Gemini-trace conditioning),
# 0b-redo on CoVer-template distributions, 0c-redo rollouts + scoring.
# Ordered so RL-blocking answers land first. Publishes to git per milestone.
# Does NOT self-stop — RL starts on this pod in the morning.
set -euxo pipefail
export UV_CACHE_DIR=/workspace/uv_cache UV_LINK_MODE=copy
eval "$(grep -E '^export (HF_TOKEN|HF_HOME|GEMINI_API_KEY|VLA_DATA_DIR|VLA_LOG_DIR|WANDB_MODE)' ~/.bashrc || true)"
export HF_HOME="${HF_HOME:-/workspace/hf_cache}" WANDB_MODE=offline
export VLA_DATA_DIR="${VLA_DATA_DIR:-/workspace/vla_data}" VLA_LOG_DIR="${VLA_LOG_DIR:-/workspace/vla_log}"
cd /workspace/phrase-rl
git pull --no-edit || true

publish () {  # publish <milestone> <files...>
  local msg="$1"; shift
  mkdir -p results/overnight/raw
  cp -f "$@" results/overnight/raw/ 2>/dev/null || true
  git add results/overnight && git -c user.name=pod -c user.email=pod@runpod commit -m "overnight: $msg [pod]" || true
  git push || true
}

# ---- Stage 1: step-0 A/B generation (RL-blocking) ----
.venv-gen/bin/python -m phrase_rl.qwen_cover_generate --contexts data/contexts_val_0b.parquet \
  --arm inline --out data/cover_qwen_inline_val.parquet --n 32 --limit 120
.venv-gen/bin/python -m phrase_rl.qwen_cover_generate --contexts data/contexts_val_0b.parquet \
  --arm trace --traces results/phrase_artifacts/rephrases_val_0b.parquet \
  --out data/cover_qwen_trace_val.parquet --n 32 --limit 120
publish "step0 A/B generation done" data/cover_qwen_inline_val.parquet data/cover_qwen_trace_val.parquet

# ---- Stage 2: 0b-redo scoring on the CoVer-template arms (RL-blocking) ----
source .venv/bin/activate
python -m phrase_rl.phase0b_score --contexts data/contexts_val_0b.parquet \
  --arm cover_gemini=results/phrase_artifacts/cover_gemini_val.parquet \
  --arm cover_qwen_inline=data/cover_qwen_inline_val.parquet \
  --arm cover_qwen_trace=data/cover_qwen_trace_val.parquet \
  --ckpt juexzz/INTACT-pi0-finetune-rephrase-bridge --out data/scores_0b_redo.parquet
deactivate
publish "0b-redo scoring done" data/scores_0b_redo.parquet

# ---- Stage 3: 0c-redo phrases for the 4 SIMPLER tasks ----
.venv-gen/bin/python -m phrase_rl.qwen_cover_generate --contexts data/contexts_0c_tasks.parquet \
  --arm inline --out data/cover_qwen_tasks.parquet --n 16
.venv-gen/bin/python - <<'PY'
import pandas as pd
ctx = pd.read_parquet("data/contexts_0c_tasks.parquet")
qw = pd.read_parquet("data/cover_qwen_tasks.parquet")
gm = pd.read_parquet("results/phrase_artifacts/cover_gemini_tasks.parquet")
rows = []
for c in ctx.itertuples():
    rows.append({"task": c.task, "arm": "original", "phrase": c.instruction})
    seen = {c.instruction.strip().lower()}
    for arm, frame in [("cover_gemini", gm), ("cover_qwen_inline", qw)]:
        m = frame[(frame.episode_index == c.episode_index) & (frame.t == c.t)]
        if len(m) == 0: continue
        for p in list(m.iloc[0]["rephrases"])[:16]:
            k = str(p).strip().lower()
            if k not in seen:
                seen.add(k); rows.append({"task": c.task, "arm": arm, "phrase": str(p).strip()})
df = pd.DataFrame(rows); df.to_parquet("data/phrases_0c_redo.parquet", index=False)
print(df.groupby(["task"]).size().to_string())
PY
publish "0c-redo phrases built" data/phrases_0c_redo.parquet data/cover_qwen_tasks.parquet

# ---- Stage 4: 0c-redo rollouts (long; resumable) ----
cd /workspace/INT-ACT
.venv/bin/python /workspace/phrase-rl/src/phrase_rl/phase0c_rollout.py \
  --int-act-root /workspace/INT-ACT \
  --config config/experiment/simpler/pi0_finetune_bridge_ev.yaml \
  --ckpt juexzz/INTACT-pi0-finetune-rephrase-bridge \
  --phrases /workspace/phrase-rl/data/phrases_0c_redo.parquet \
  --episode-ids 0 1 2 3 4 5 6 7 8 9 \
  --out /workspace/phrase-rl/data/rollouts_0c_redo.parquet
cd /workspace/phrase-rl
publish "0c-redo rollouts done" data/rollouts_0c_redo.parquet

# ---- Stage 5: 0c-redo flow-loss scoring on matched real contexts ----
source .venv/bin/activate
python -m phrase_rl.phase0c_score --contexts data/contexts_0c_match.parquet \
  --phrases data/phrases_0c_redo.parquet \
  --ckpt juexzz/INTACT-pi0-finetune-rephrase-bridge --out data/scores_0c_redo.parquet
deactivate
publish "0c-redo scoring done — OVERNIGHT COMPLETE" data/scores_0c_redo.parquet

echo "OVERNIGHT ALL DONE"
