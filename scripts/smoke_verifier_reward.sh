#!/usr/bin/env bash
# Smoke the verifier reward server end-to-end on 2 contexts x 4 phrases.
set -euxo pipefail
cd /workspace/phrase-rl
eval "$(grep -E '^export (HF_TOKEN|HF_HOME)' ~/.bashrc || true)"
export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
mkdir -p /workspace/ipc_smoke
.venv/bin/python - <<'PY'
import pandas as pd, json
ctx = pd.read_parquet("data/contexts_train.parquet").head(2)
rows = []
for i, r in enumerate(ctx.itertuples()):
    for p in [str(r.instruction), "move the thing to the other thing",
              "pick up the object and place it down", str(r.instruction).upper()]:
        rows.append({"context_id": f"smoke{i}", "image_png": r.image_png, "state": r.state,
                     "action_chunk": r.action_chunk, "phrase": p})
pd.DataFrame(rows).to_parquet("/workspace/ipc_smoke/smoke_in.parquet", index=False)
json.dump({"in_parquet": "/workspace/ipc_smoke/smoke_in.parquet",
           "out_parquet": "/workspace/ipc_smoke/smoke_out.parquet",
           "reward_mode": "verifier"}, open("/workspace/ipc_smoke/job000.req.json", "w"))
PY
.venv/bin/python -m phrase_rl.phase2_score_server --ipc-dir /workspace/ipc_smoke \
  --stats-contexts results/phrase_artifacts/chunk_stats.parquet \
  --verifier-ensemble results/checkpoints/verifier_reward_ensemble.json --once
test -f /workspace/ipc_smoke/job000.done.json
.venv/bin/python - <<'PY'
import pandas as pd
d = pd.read_parquet("/workspace/ipc_smoke/smoke_out.parquet")
print(d[["context_id", "phrase", "loss"]].to_string())
own = d.groupby("context_id").first().loss
assert len(d) == 8 and d.loss.notna().all()
print("\nSMOKE-OK: own-instruction losses", own.round(3).tolist(),
      "(should be lowest within each group)")
PY
