#!/usr/bin/env bash
# SFT-17k rollout eval (dead-simple spec step 4): roll the SFT and frozen_bare
# reconstructions on the touched-8 val suite. Per-arm sequential so partial
# results land early; skip-guards make reruns idempotent.
#   bash scripts/run_sft_eval.sh greedy    # x12 primary verdict (2 arms x 2,304 eps)
#   bash scripts/run_sft_eval.sh sampled   # k=8 x1 distribution view (2 arms x 1,536 eps)
set -uo pipefail
eval "$(grep -E '^export (HF_TOKEN|HF_HOME)' ~/.bashrc || true)"
export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
export VLA_DATA_DIR=/workspace/vla_data VLA_LOG_DIR=/workspace/vla_log WANDB_MODE=offline
cd /workspace/phrase-rl
mark() { echo "[sft-eval $(date +%H:%M:%S)] $*" | tee -a /workspace/sfteval.log; }

LEG=${1:-greedy}
NW=${VAL_WORKERS:-3}
ARMS="sft frozen_bare"
if [ "$LEG" = greedy ]; then
  PH=data/phrases_sft17k_ert_greedy.parquet; REPS=12; TAG=sfteval12
elif [ "$LEG" = trace ]; then
  # trace-conditioned arms, greedy: frozen_trace = the missing same-suite
  # anchor; sft_trace = the SFT adapter under the CoVer trace prompt — a
  # zero-training v2 preview (train/deploy-mismatch caveat applies)
  PH=data/phrases_trace_greedy.parquet; REPS=12; TAG=sfteval12; ARMS="frozen_trace sft_trace"
  if [ ! -f "$PH" ]; then
    if [ ! -f data/ph4_trace_tmp.parquet ]; then
      mark "generating trace-conditioned greedy phrases (phase4, both arms)"
      PYTHONPATH=src .venv-gen/bin/python -m phrase_rl.phase4_generate_eval \
        --tasks data/val_screen_frames.parquet --ctx data/val_screen_frames.parquet \
        --assets data/val_screen_assets.parquet \
        --adapter results/checkpoints/sft17k/final \
        --out data/ph4_trace_tmp.parquet >> /workspace/sfteval.log 2>&1 \
        || { mark "TRACE GEN FAILED"; exit 1; }
    fi
    .venv-gen/bin/python -c "
import pandas as pd
d = pd.read_parquet('data/ph4_trace_tmp.parquet')
d = d[d.arm.isin(['base', 'tuned'])].copy()
d['arm'] = d.arm.map({'base': 'frozen_trace', 'tuned': 'sft_trace'})
d.to_parquet('$PH', index=False)
print(d.arm.value_counts().to_dict())" >> /workspace/sfteval.log 2>&1 || { mark "TRACE FILTER FAILED"; exit 1; }
  fi
elif [ "$LEG" = selftrace ]; then
  # fully self-contained: Qwen writes its own trace (minimal 2-section
  # prompt), then the same phase4 plumbing as the Gemini-trace leg
  PH=data/phrases_selftrace_greedy.parquet; REPS=12; TAG=sfteval12; ARMS="frozen_selftrace"
  if [ ! -f "$PH" ]; then
    if [ ! -f data/val_screen_assets_selftrace.parquet ]; then
      mark "generating Qwen self-traces (minimal prompt)"
      PYTHONPATH=src .venv-gen/bin/python -m phrase_rl.sft17k_selftrace_gen \
        >> /workspace/sfteval.log 2>&1 || { mark "SELFTRACE GEN FAILED"; exit 1; }
    fi
    mark "generating self-trace-conditioned phrases (phase4, both arms)"
    PYTHONPATH=src .venv-gen/bin/python -m phrase_rl.phase4_generate_eval \
      --tasks data/val_screen_frames.parquet --ctx data/val_screen_frames.parquet \
      --assets data/val_screen_assets_selftrace.parquet \
      --adapter results/checkpoints/sft17k/final \
      --out data/ph4_selftrace_tmp.parquet >> /workspace/sfteval.log 2>&1 \
      || { mark "SELFTRACE PHRASE GEN FAILED"; exit 1; }
    .venv-gen/bin/python -c "
import pandas as pd
d = pd.read_parquet('data/ph4_selftrace_tmp.parquet')
d = d[d.arm.isin(['base', 'tuned'])].copy()
d['arm'] = d.arm.map({'base': 'frozen_selftrace', 'tuned': 'sft_selftrace'})
d.to_parquet('$PH', index=False)
print(d.arm.value_counts().to_dict())" >> /workspace/sfteval.log 2>&1 || { mark "SELFTRACE FILTER FAILED"; exit 1; }
  fi
elif [ "$LEG" = selftrace2 ]; then
  # the deferred diagnostic arm (v1 adapter + self-trace, phase4 conditioning)
  PH=data/phrases_selftrace_greedy.parquet; REPS=12; TAG=sfteval12; ARMS="sft_selftrace"
elif [ "$LEG" = v2 ]; then
  # v2 under its NATIVE conditioning: own wrapper + Qwen self-trace prepended
  # (pre-registered primary read: sft_v2 vs frozen_selftrace, trace source held)
  PH=data/phrases_v2_selftrace.parquet; REPS=12; TAG=sfteval12; ARMS="sft_v2"
  if [ ! -f "$PH" ]; then
    mark "generating v2 phrases (native trace format)"
    PYTHONPATH=src .venv-gen/bin/python -m phrase_rl.sft17k_generate_eval \
      --adapter results/checkpoints/sft17k_v2/final \
      --assets data/val_screen_assets.parquet \
      --trace-assets data/val_screen_assets_selftrace.parquet \
      --arm sft_v2 --out "$PH" >> /workspace/sfteval.log 2>&1 \
      || { mark "V2 GEN FAILED"; exit 1; }
  fi
else
  PH=data/phrases_sft17k_ert_sam8.parquet; REPS=1; TAG=sftsam8x1
  if [ ! -f "$PH" ]; then
    mark "generating sampled phrases (k=8, both arms)"
    PYTHONPATH=src .venv-gen/bin/python -m phrase_rl.sft17k_generate_eval \
      --adapter results/checkpoints/sft17k/final --include-frozen \
      --assets data/val_screen_assets.parquet --sample-k 8 \
      --out "$PH" >> /workspace/sfteval.log 2>&1 || { mark "SAMPLED GEN FAILED"; exit 1; }
  fi
fi
[ -f "$PH" ] || { mark "missing $PH — run the chain / generation first"; exit 1; }

for ARM in $ARMS; do
  OUT="results/val_screens/${TAG}_${ARM}.parquet"
  [ -f "results/val_screens/.done_${TAG}_${ARM}" ] && { mark "skip $ARM (done)"; continue; }
  ARM=$ARM PH=$PH NW=$NW .venv-gen/bin/python - <<'PY'
import os
import pandas as pd
d = pd.read_parquet(os.environ["PH"])
d = d[d.arm == os.environ["ARM"]]
assert len(d), f"no rows for arm {os.environ['ARM']}"
tasks = sorted(d.task.unique())
nw = int(os.environ["NW"])
for w in range(nw):
    wt = [t for i, t in enumerate(tasks) if i % nw == w]
    d[d.task.isin(wt)].to_parquet(f"data/sfte_w{w}.parquet", index=False)
print(f"{os.environ['ARM']}: {len(d)} phrase-rows over {len(tasks)} tasks")
PY
  rm -f data/sfte_out_w*.parquet
  cd /workspace/INT-ACT
  pids=""
  for w in $(seq 0 $((NW-1))); do
    sleep 45   # stagger pi0 loads (simultaneous loads OOM-race)
    /workspace/INT-ACT/.venv/bin/python /workspace/phrase-rl/src/phrase_rl/phase0c_rollout.py \
      --int-act-root /workspace/INT-ACT \
      --config config/experiment/simpler/pi0_finetune_bridge_ev.yaml \
      --ckpt juexzz/INTACT-pi0-finetune-rephrase-bridge \
      --phrases /workspace/phrase-rl/data/sfte_w$w.parquet \
      --episode-ids 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 \
      --repeats $REPS \
      --out /workspace/phrase-rl/data/sfte_out_w$w.parquet > /workspace/sfte_w$w.log 2>&1 &
    pids="$pids $!"
  done
  wfail=0
  for p in $pids; do wait $p || wfail=1; done
  cd /workspace/phrase-rl
  [ $wfail = 1 ] && { mark "WORKER FAILED $ARM — inspect /workspace/sfte_w*.log"; exit 1; }
  OUT=$OUT PH=$PH ARM=$ARM REPS=$REPS .venv-gen/bin/python - <<'PY' || { mark "MERGE FAILED $ARM"; exit 1; }
import glob
import os
import pandas as pd
d = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob("data/sfte_out_w*.parquet"))], ignore_index=True)
ph = pd.read_parquet(os.environ["PH"])
expect = len(ph[ph.arm == os.environ["ARM"]]) * 24 * int(os.environ["REPS"])
assert len(d) == expect, f"expected {expect} eps, got {len(d)}"
d.to_parquet(os.environ["OUT"], index=False)
s = d.groupby("task").success.mean() * 100
print(f"[sft-eval] RESULT {os.environ['OUT']}: pooled {d.success.mean()*100:.1f}% |",
      s.round(1).to_dict(), flush=True)
PY
  touch "results/val_screens/.done_${TAG}_${ARM}"
  git add results/val_screens/ && git -c user.name=pod4 -c user.email=pod@runpod commit -q -m "sft eval ${TAG} ${ARM} [pod]" && git -c user.name=pod4 -c user.email=pod@runpod pull -q --rebase --no-edit && git push -q
  mark "$ARM DONE -> $OUT"
done
mark "SFT-EVAL-COMPLETE ($LEG)"
