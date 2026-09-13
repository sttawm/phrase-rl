#!/bin/bash
# Pod-side supervisor for one LIBERO minimal-pair round on a single-GPU pod.
#   scp scripts/libero_round_runner.sh root@pod:/workspace/run_r8.sh
#   ssh pod 'ROUND=8 TARGETS="450 450 450 450 400 400 400 400" setsid nohup /workspace/run_r8.sh >/dev/null 2>&1 < /dev/null &'
# Expects /workspace/r<ROUND>shard<i>.queue.json (one per target) and a policy server
# on :8000 started by /workspace/start_server.sh. Launches one bank_eval.py worker per
# shard (staggered 7 s so they do not race a warming server), restarts the server if
# :8000 closes, relaunches any worker that died short of its target (bank_eval.py
# resumes per episode on its --out file; never delete those), and touches
# /workspace/R<ROUND>_DONE when every shard reaches its target.
# Copy the file verbatim (scp) — writing it through an ssh heredoc expands every $
# on the remote side and silently produces a runner that launches nothing.
: "${ROUND:?set ROUND}" "${TARGETS:?set TARGETS (space-separated per-shard episode counts)}"
. /workspace/interactive-vlas/pi05_libero/.openpi_env
cd /workspace/interactive-vlas/pi05_libero/eval
TARGET=($TARGETS); NS=${#TARGET[@]}
P=/workspace/r${ROUND}
srv_up(){ (ss -ltn 2>/dev/null||netstat -ltn 2>/dev/null) | grep -q ':8000'; }
launch(){ MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONPATH="$LIBERO_PYTHONPATH" "$LIBERO_VENV/bin/python" bank_eval.py --queue ${P}shard$1.queue.json --out ${P}roll$1.jsonl --port 8000 --seed 7 >> ${P}roll$1.log 2>&1 & sleep 7; }
rm -f /workspace/R${ROUND}_DONE
for ((i=0; i<NS; i++)); do launch $i; done
while true; do
  srv_up || { echo "$(date -u +%H:%M) r$ROUND SERVER DOWN -> restart" >> /workspace/chain.log; setsid nohup /workspace/start_server.sh >/dev/null 2>&1 < /dev/null & sleep 90; }
  all=1
  for ((i=0; i<NS; i++)); do
    have=$(wc -l < ${P}roll$i.jsonl 2>/dev/null || echo 0); [ "$have" -ge "${TARGET[$i]}" ] && continue; all=0
    pgrep -f "r${ROUND}shard$i[.]queue[.]json" >/dev/null || launch $i
  done
  [ "$all" = 1 ] && { echo "$(date -u +%H:%M) R$ROUND COMPLETE" >> /workspace/chain.log; touch /workspace/R${ROUND}_DONE; exit 0; }
  sleep 120
done
