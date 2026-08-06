#!/bin/bash
# Pod6: after the 230-250 gens finish, extract the step-60 adapter and
# generate the sealed step-60 arms (A26), then push the phrase files.
cd /workspace/phrase-rl
mark() { echo "[step60 $(date +%H:%M)] $*" >> /workspace/step60.log; }

# wait for the gen worker to go idle (230-250 backfill in flight)
while pgrep -f "[g]en_ckpt_phrases" >/dev/null; do sleep 120; done
sleep 30

d=/workspace/v10_adapters/step_0060
if [ ! -f "$d/adapter_model.safetensors" ]; then
  mkdir -p /tmp/v10ck && rm -rf /tmp/v10ck/*
  cat results/checkpoints/archive/v10_step_0060.tgz.part-* | tar xzf - -C /tmp/v10ck 2>/dev/null
  src=$(find /tmp/v10ck -name adapter_model.safetensors | head -1)
  [ -n "$src" ] && { mkdir -p "$d"; cp "$(dirname $src)"/* "$d"/; } || { mark "NO ARCHIVE 0060"; exit 1; }
fi
mark "gen start"
FINAL_EVAL=1 .venv-gen/bin/python scripts/gen_v10_step60_anchors.py "$d" > /workspace/step60_gen.log 2>&1 || { mark "GEN FAIL"; exit 1; }
git add results/sealed/ph_sealed_v10step60_polish.parquet results/sealed/ph_sealed_v10step60_repair.parquet
git commit -q -m "A26 assets: v10 step-60 sealed phrases both conds (preflighted in step60_gen.log) [pod6]"
for i in 1 2 3 4; do
  timeout 300 bash -c "git -c rebase.autoStash=true pull -q --rebase && git push -q" && { mark "pushed"; break; }
  git rebase --abort 2>/dev/null; sleep $((30 * i))
done
mark "STEP60-DONE"
