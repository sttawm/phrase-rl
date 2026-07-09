#!/usr/bin/env bash
# One-off (pod 1): base-vs-tuned rephrase rank data for the 4 SIMPLER tasks.
# Samples 16 trace-conditioned rephrases from BASE Qwen and from the TUNED
# checkpoint (phase2/best_val) with paired seeds, then CRN-scores everything
# through the LIVE phase-2 score server via IPC (no second pi0 load; the job
# id sorts after trainer jobs so training keeps priority). The trainer is
# paused ONLY for generation (GPU memory) and relaunched (resume-safe).
set -euxo pipefail
eval "$(grep -E '^export (HF_TOKEN|HF_HOME|GEMINI_API_KEY)' ~/.bashrc || true)"
export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
cd /workspace/phrase-rl
git pull --no-edit

# 0) let val@200 finish first so best_val is as fresh as possible (max 35 min)
for i in $(seq 70); do
  grep -q '"type": "val", "step": 200' results/checkpoints/phase2/train_log.jsonl && break
  sleep 30
done

# 1) traces for the 4 task contexts (from the cached Gemini CoVer responses)
.venv-gen/bin/python - <<'PY'
import pandas as pd
from phrase_rl.cover_prompt import extract_trace
g = pd.read_parquet("results/phrase_artifacts/cover_gemini_tasks.parquet")
rows = [{"episode_index": int(r.episode_index), "t": int(r.t),
         "trace": extract_trace(str(r.raw_response))} for r in g.itertuples()]
assert all(r["trace"] for r in rows), "trace extraction failed"
pd.DataFrame(rows).to_parquet("data/traces_0c_tasks.parquet", index=False)
print(f"{len(rows)} traces")
PY

# 2) pause trainer (latest checkpoint is <=25 steps old; fully resume-safe)
tmux kill-session -t train 2>/dev/null || true
pkill -f phrase_rl.phase2_train || true
sleep 15

# 3) generate — base then tuned, same contexts, paired sampling seeds
CUDA_VISIBLE_DEVICES=0 .venv-gen/bin/python -m phrase_rl.qwen_cover_generate \
  --contexts data/contexts_0c_tasks.parquet --arm trace --traces data/traces_0c_tasks.parquet \
  --out data/cover_qwen_trace_tasks_base.parquet --n 16 --gen-seed 7
CUDA_VISIBLE_DEVICES=0 .venv-gen/bin/python -m phrase_rl.qwen_cover_generate \
  --contexts data/contexts_0c_tasks.parquet --arm trace --traces data/traces_0c_tasks.parquet \
  --out data/cover_qwen_trace_tasks_tuned.parquet --n 16 --gen-seed 7 \
  --adapter results/checkpoints/phase2/best_val

# 4) relaunch trainer (verbatim run_phase2.sh line; resumes from latest)
tmux new-session -d -s train \
  "bash -lc 'set -o pipefail; eval \"\$(grep -E \"^export (HF_TOKEN|HF_HOME|GEMINI_API_KEY|UV_CACHE_DIR|UV_LINK_MODE)\" ~/.bashrc || true)\"; cd /workspace/phrase-rl && \
   CUDA_VISIBLE_DEVICES=0 .venv-gen/bin/python -m phrase_rl.phase2_train \
     --ipc-dir /workspace/ipc --resume \
     --beta 0.15 --lr 5e-6 --kl-abort 1.2 \
     --traces results/phrase_artifacts/cover35_teacher_train.parquet \
     --val-traces results/phrase_artifacts/rephrases_val_0b.parquet \
   2>&1 | tee -a results/checkpoints/phase2/train.log; \
   echo \"trainer exited rc=\$?\"; sleep infinity'"

# 5) score all phrases via the live score server (context x phrase cross-join)
.venv/bin/python - <<'PY'
import json, os, time
import numpy as np, pandas as pd
ctx = pd.read_parquet("data/contexts_0c_match.parquet")
tasks4 = pd.read_parquet("data/contexts_0c_tasks.parquet")
meta = {(int(r.episode_index), int(r.t)): (r.task, r.instruction) for r in tasks4.itertuples()}
phr = []
for arm, path in [("base", "data/cover_qwen_trace_tasks_base.parquet"),
                  ("tuned", "data/cover_qwen_trace_tasks_tuned.parquet")]:
    for r in pd.read_parquet(path).itertuples():
        task, _ = meta[(int(r.episode_index), int(r.t))]
        for p in r.rephrases:
            phr.append({"task": task, "arm": arm, "phrase": str(p).strip()})
for (ep, t), (task, orig) in meta.items():
    phr.append({"task": task, "arm": "original", "phrase": str(orig)})
ph = pd.DataFrame(phr).drop_duplicates(["task", "phrase"])  # keep first arm on collision
ph.to_parquet("data/phrases_rankchart.parquet", index=False)
print(ph.groupby(["task", "arm"]).size().to_string())

rows = []
for r in ctx.itertuples():
    for q in ph[ph.task == r.task].itertuples():
        rows.append({"context_id": f"{r.task}|{r.episode_index}|{r.t}",
                     "image_png": r.image_png,
                     "state": np.asarray(r.state, dtype=np.float32),
                     "action_chunk": np.asarray(r.action_chunk, dtype=np.float32),
                     "phrase": q.phrase})
ipc = "/workspace/ipc"
pd.DataFrame(rows).to_parquet(f"{ipc}/rankchart.in.parquet", index=False)
spec = {"in_parquet": f"{ipc}/rankchart.in.parquet",
        "out_parquet": "/workspace/phrase-rl/data/scores_rankchart_raw.parquet",
        "k": 16, "seed": 0, "tau_min": 0.0}
tmp = f"{ipc}/zz_rankchart.req.json.tmp"  # 'zz' sorts after trainer's s000NNN jobs
open(tmp, "w").write(json.dumps(spec))
os.replace(tmp, f"{ipc}/zz_rankchart.req.json")
for i in range(360):
    if os.path.exists(f"{ipc}/zz_rankchart.done.json"):
        break
    if os.path.exists(f"{ipc}/zz_rankchart.failed.json"):
        raise SystemExit(open(f"{ipc}/zz_rankchart.err.txt").read())
    time.sleep(10)
else:
    raise SystemExit("score server did not answer within 60 min")
sc = pd.read_parquet("data/scores_rankchart_raw.parquet")
sc["task"] = sc.context_id.str.split("|").str[0]
sc.merge(ph, on=["task", "phrase"], how="left").to_parquet("data/scores_rankchart.parquet", index=False)
print("scored rows:", len(sc))
PY

# 6) publish artifacts
mkdir -p results/overnight/raw
cp -f data/phrases_rankchart.parquet data/scores_rankchart.parquet \
      data/cover_qwen_trace_tasks_base.parquet data/cover_qwen_trace_tasks_tuned.parquet \
      results/overnight/raw/
git add results/overnight/raw/
git -c user.name=pod1 -c user.email=pod@runpod commit -m "rankchart: base vs tuned(best_val) trace-arm 16-lists + CRN flow scores [pod]" || true
git -c user.name=pod1 -c user.email=pod@runpod pull --rebase --no-edit || true
git push || true
echo RANKCHART-DONE
