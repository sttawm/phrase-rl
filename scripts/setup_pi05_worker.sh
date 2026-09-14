#!/bin/bash
# Turn a freshly built pi05 LIBERO pod (scripts/setup_pi05_libero_pod.sh) into a
# git-job-queue worker for one rules-loop run. Idempotent: rerunning repairs
# whatever is missing (clone, venv, env file, server, worker, watchdog) and
# leaves what is already healthy alone.
#
#   POD_GIT_TOKEN=... RUN_ID=p_v2 bash scripts/setup_pi05_worker.sh
#
# What it sets up:
#   /workspace/phrase-rl          blobless partial clone, cone sparse-checkout of
#                                 scripts/ prompts/ config/ src/ results/rules_runs/$RUN_ID/
#                                 (the jobs dir MUST be in the checkout: a bare
#                                 "!/results" pattern blanks the queue)
#   /workspace/phrase-rl/.venv-gen  pandas+numpy+pyarrow -- rules_loop_worker.sh
#                                 hard-codes .venv-gen/bin/python for jobs.py
#   /workspace/pi05_worker.env    LIBERO env for the libero_bank_eval job kind
#   /workspace/start_server.sh    policy server on :8000 (started if not listening)
#   /workspace/pi05_watchdog.sh   5-min loop: restart server / relaunch worker
#                                 (NOT after a "COMMIT FAILED" halt: that stall
#                                 stays visible; fix git, rerun this script)
#
# Only libero_bank_eval score jobs run here; apply/generate jobs need the LLM
# stack, so /workspace/.can_apply is deliberately NOT created. Drain the worker
# with `touch /workspace/.worker_stop` (the watchdog honours it too).
#
# No set -e: every step reports its own failure and the final STATUS line says
# what is broken. The token is never echoed; git output is masked before print.
set -uo pipefail

RUN_ID="${RUN_ID:-p_v2}"
POD_GIT_TOKEN="${POD_GIT_TOKEN:-}"
if [ -z "$POD_GIT_TOKEN" ]; then
  echo "setup_pi05_worker: POD_GIT_TOKEN is not set" >&2
  echo "STATUS clone=FAIL(no token) venv=skip server=skip worker=skip watchdog=skip"
  exit 2
fi
REPO_URL="https://x-access-token:${POD_GIT_TOKEN}@github.com/sttawm/phrase-rl.git"
CLONE=/workspace/phrase-rl
ENV_FILE=/workspace/pi05_worker.env
WATCHDOG=/workspace/pi05_watchdog.sh
OPENPI_ENV=/workspace/interactive-vlas/pi05_libero/.openpi_env
SPARSE_DIRS=(scripts prompts config src "results/rules_runs/$RUN_ID")
export PATH="/root/.local/bin:/root/.cargo/bin:$PATH"
# uv's managed pythons and cache default to /root (container disk): a pod stop
# would leave .venv-gen/bin/python a dangling symlink. Keep them on the volume.
export UV_PYTHON_INSTALL_DIR=/workspace/uv_python UV_CACHE_DIR=/workspace/uv_cache UV_LINK_MODE=copy

say() { echo "[setup_pi05_worker $(date -u +%H:%M:%S)] $*"; }
# git prints the remote URL on some failures; scrub the token before it reaches
# stdout. Pure-bash substitution so the token never sits on a sed/awk cmdline.
mask() { local l; while IFS= read -r l || [ -n "$l" ]; do printf '%s\n' "${l//"$POD_GIT_TOKEN"/***}"; done; }
port_up() { (exec 3<>/dev/tcp/127.0.0.1/8000) 2>/dev/null; }

# pod identity for the git author (same lookup the worker uses)
export "$(tr '\0' '\n' < /proc/1/environ 2>/dev/null | grep '^RUNPOD_POD_ID=')" 2>/dev/null || true
POD="${RUNPOD_POD_ID:-$(hostname)}"

st_clone=FAIL st_venv=skip st_server=skip st_worker=skip st_watchdog=skip

# --- 1. clone / refresh -------------------------------------------------------
if [ ! -d "$CLONE/.git" ]; then
  say "cloning phrase-rl (blobless, sparse) into $CLONE"
  rm -rf "$CLONE"
  if git clone --filter=blob:none --no-checkout "$REPO_URL" "$CLONE" 2>&1 | mask; then
    if git -C "$CLONE" sparse-checkout init --cone 2>&1 | mask \
       && git -C "$CLONE" sparse-checkout set "${SPARSE_DIRS[@]}" 2>&1 | mask \
       && git -C "$CLONE" checkout -q main 2>&1 | mask; then
      st_clone=ok
    else
      say "sparse-checkout/checkout FAILED"
    fi
  else
    say "git clone FAILED"
  fi
else
  say "clone exists -- refreshing"
  git -C "$CLONE" remote set-url origin "$REPO_URL"
  # re-apply the sparse set every run: a new RUN_ID needs its jobs dir added
  git -C "$CLONE" sparse-checkout init --cone 2>&1 | mask
  git -C "$CLONE" sparse-checkout set "${SPARSE_DIRS[@]}" 2>&1 | mask
  if timeout 600 git -C "$CLONE" fetch -q origin 2>&1 | mask \
     && timeout 600 git -C "$CLONE" -c rebase.autoStash=true pull -q --rebase origin main 2>&1 | mask; then
    st_clone=ok
  else
    git -C "$CLONE" rebase --abort 2>/dev/null
    say "fetch/pull FAILED (rebase aborted; local commits kept)"
  fi
  # self-heal an empty worktree: if the first run's `checkout main` died (blob
  # fetch timeout) the index is empty and pull --rebase is a no-op forever
  if [ "$st_clone" = ok ] && [ ! -f "$CLONE/scripts/rules_loop_worker.sh" ]; then
    say "worktree empty (first checkout never completed) -- checking out main"
    if timeout 600 git -C "$CLONE" checkout -q main 2>&1 | mask \
       && [ -f "$CLONE/scripts/rules_loop_worker.sh" ]; then
      git -C "$CLONE" sparse-checkout reapply 2>&1 | mask
    else
      st_clone=FAIL
      say "checkout main FAILED -- worktree still empty"
    fi
  fi
fi
if [ "$st_clone" = ok ] && [ ! -d "$CLONE/results/rules_runs/$RUN_ID" ]; then
  say "WARNING: results/rules_runs/$RUN_ID not on origin yet -- worker will idle until the driver pushes it"
fi
if [ "$st_clone" != ok ]; then
  echo "STATUS clone=$st_clone venv=$st_venv server=$st_server worker=$st_worker watchdog=$st_watchdog"
  exit 1
fi

# --- 2. git identity (repo-local: a relaunched worker must not depend on the env)
git -C "$CLONE" config user.email "worker@phrase-rl.local"
git -C "$CLONE" config user.name  "phrase-rl worker $POD"

# --- 3. .venv-gen ---------------------------------------------------------------
if ! command -v uv >/dev/null 2>&1; then
  say "installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1 || say "uv install FAILED"
fi
st_venv=FAIL
if command -v uv >/dev/null 2>&1; then
  [ -x "$CLONE/.venv-gen/bin/python" ] || (cd "$CLONE" && uv venv --python 3.11 .venv-gen)
  if ! "$CLONE/.venv-gen/bin/python" -c "import pandas, numpy, pyarrow" >/dev/null 2>&1; then
    (cd "$CLONE" && uv pip install --python .venv-gen/bin/python pandas numpy pyarrow)
  fi
  "$CLONE/.venv-gen/bin/python" -c "import pandas, numpy, pyarrow" >/dev/null 2>&1 && st_venv=ok
fi
[ "$st_venv" = ok ] || say ".venv-gen is not usable (pandas/numpy/pyarrow import failed)"

# --- 4. env file for the libero_bank_eval job kind ------------------------------
st_env=FAIL
if [ -f "$OPENPI_ENV" ]; then
  # shellcheck disable=SC1090
  . "$OPENPI_ENV"
  if [ -x "${LIBERO_VENV:-}/bin/python" ]; then
    cat > "$ENV_FILE" <<ENV
export RUN_ID=$RUN_ID
export INTERACTIVE_PI_ROOT=/workspace/interactive-vlas
export LIBERO_PY=$LIBERO_VENV/bin/python
export LIBERO_PYTHONPATH=$LIBERO_PYTHONPATH
export MUJOCO_GL=egl
export LIBERO_PARALLEL=${LIBERO_PARALLEL:-6}   # bank_eval sub-shards per job (rules_loop_jobs.py)
export PYOPENGL_PLATFORM=egl
ENV
    st_env=ok
  else
    say "LIBERO_VENV python missing: ${LIBERO_VENV:-unset}/bin/python"
  fi
else
  say "$OPENPI_ENV missing -- run scripts/setup_pi05_libero_pod.sh first"
fi

# --- 5. policy server on :8000 --------------------------------------------------
start_server() {
  [ -x /workspace/start_server.sh ] || { say "/workspace/start_server.sh missing"; return 1; }
  date +%s > /workspace/.pi05_server_started
  setsid nohup /workspace/start_server.sh >/dev/null 2>&1 < /dev/null &
  say "started policy server (pid $!)"
}
if port_up; then
  st_server=up
else
  st_server=FAIL
  # a server still loading the checkpoint is not listening yet; do not double-start
  sp=$(pgrep -f '[s]erve_policy.py' | head -n1 || true)
  if [ -n "$sp" ]; then
    # a server this script did not launch (setup_pi05_libero_pod.sh starts one)
    # still needs the watchdog's 15-min startup grace: stamp its start time,
    # else the first tick after a >10-min load computes age=now and kills it
    { stat -c %Y "/proc/$sp" 2>/dev/null || date +%s; } > /workspace/.pi05_server_started
    say "serve_policy process exists (pid $sp) but :8000 not listening -- waiting"
  else
    start_server
  fi
  for _ in $(seq 1 60); do
    sleep 10
    port_up && { st_server=up; break; }
  done
  [ "$st_server" = up ] || say "policy server not listening after 10 min (see /workspace/server.log)"
fi

# --- 6. worker ------------------------------------------------------------------
# liveness by process, never by tmux session. Refuse to run two workers on one
# pod: both would claim under the same POD id.
worker_pid() { pgrep -f '[r]ules_loop_worker.sh' | head -n1; }
start_worker() {
  # shellcheck disable=SC1090
  . "$ENV_FILE"
  cd "$CLONE" || return 1
  RUN_ID="$RUN_ID" setsid nohup bash scripts/rules_loop_worker.sh > /workspace/rules_worker.out 2>&1 < /dev/null &
  echo $! > /workspace/rules_worker.pid
}
if [ "$st_env" = ok ] && [ "$st_venv" = ok ]; then
  wp=$(worker_pid || true)
  if [ -n "$wp" ]; then
    wrun=$(tr '\0' '\n' < "/proc/$wp/environ" 2>/dev/null | grep '^RUN_ID=' | cut -d= -f2-)
    if [ "$wrun" = "$RUN_ID" ]; then
      # running this script asks for a live worker: a leftover stop file would
      # make it exit on its next loop pass and the watchdog would not relaunch
      [ -f /workspace/.worker_stop ] && { rm -f /workspace/.worker_stop; say "cleared stale /workspace/.worker_stop (worker pid $wp would have drained)"; }
      st_worker="pid:$wp"
    else
      say "a worker for run '${wrun:-?}' is already running (pid $wp); touch /workspace/.worker_stop, let it exit, rerun"
      st_worker="FAIL(other run $wrun pid $wp)"
    fi
  else
    [ -f /workspace/.worker_stop ] && { rm -f /workspace/.worker_stop; say "cleared stale /workspace/.worker_stop"; }
    start_worker
    sleep 3
    wp=$(worker_pid || true)
    if [ -n "$wp" ]; then st_worker="pid:$wp"; else st_worker="FAIL(see /workspace/rules_worker.out)"; fi
  fi
else
  st_worker="skip(env=$st_env venv=$st_venv)"
fi

# --- 7. watchdog ----------------------------------------------------------------
# Written to a file and run as `bash $WATCHDOG` so its cmdline never contains the
# pgrep patterns it uses (a `bash -c '...'` launch would match itself).
cat > "$WATCHDOG" <<'WD'
#!/bin/bash
# pi05 worker watchdog (written by scripts/setup_pi05_worker.sh). Every 5 min:
# restart the policy server if :8000 stops listening, relaunch the rules-loop
# worker if its process died. Never uses pkill -f; kills only pids from pgrep.
set -uo pipefail
ENV_FILE=/workspace/pi05_worker.env
CLONE=/workspace/phrase-rl
relaunches=0
log() { echo "[watchdog $(date -u '+%m-%d %H:%M')] $*" >> /workspace/pi05_watchdog.log; }
port_up() { (exec 3<>/dev/tcp/127.0.0.1/8000) 2>/dev/null; }
while true; do
  sleep 300
  # shellcheck disable=SC1090
  . "$ENV_FILE" 2>/dev/null || { log "env file missing"; continue; }
  if ! port_up; then
    started=$(cat /workspace/.pi05_server_started 2>/dev/null || echo 0)
    age=$(( $(date +%s) - started ))
    pids=$(pgrep -f '[s]erve_policy.py' || true)
    if [ -n "$pids" ] && [ "$age" -lt 900 ]; then
      log "server still starting (${age}s), :8000 not up yet"
    else
      # a live-but-deaf server holds the GPU; the replacement would OOM beside it
      if [ -n "$pids" ]; then
        log "server pid(s) $pids alive but :8000 down for ${age}s -- killing"
        for p in $pids; do kill "$p" 2>/dev/null; done
        sleep 10
        for p in $pids; do kill -9 "$p" 2>/dev/null; done
      fi
      date +%s > /workspace/.pi05_server_started
      setsid nohup /workspace/start_server.sh >/dev/null 2>&1 < /dev/null &
      log "restarted policy server (pid $!)"
    fi
  fi
  if ! pgrep -f '[r]ules_loop_worker.sh' >/dev/null 2>&1; then
    if [ -f /workspace/.worker_stop ]; then
      log "worker down and /workspace/.worker_stop present -- not relaunching"
    elif tail -n1 /workspace/rules_worker.log 2>/dev/null | grep -q 'COMMIT FAILED'; then
      # the worker's hard halt is deliberate (wedged git state must stay
      # visible); relaunching every 5 min would just repeat the failure.
      # Operator: fix git in $CLONE, then rerun setup_pi05_worker.sh (or
      # append any line to rules_worker.log) to lift this.
      log "worker halted on COMMIT FAILED (last rules_worker.log line) -- NOT relaunching; fix git state and rerun setup"
    else
      cd "$CLONE" || { log "clone missing"; continue; }
      RUN_ID="$RUN_ID" setsid nohup bash scripts/rules_loop_worker.sh >> /workspace/rules_worker.out 2>&1 < /dev/null &
      echo $! > /workspace/rules_worker.pid
      relaunches=$((relaunches + 1))
      log "relaunched worker for $RUN_ID (pid $!, relaunch #$relaunches since watchdog start)"
    fi
  fi
done
WD
chmod +x "$WATCHDOG"
wdp=$(pgrep -f '[p]i05_watchdog.sh' | head -n1 || true)
if [ -z "$wdp" ]; then
  setsid nohup bash "$WATCHDOG" >/dev/null 2>&1 < /dev/null &
  sleep 1
  wdp=$(pgrep -f '[p]i05_watchdog.sh' | head -n1 || true)
fi
if [ -n "$wdp" ]; then st_watchdog="pid:$wdp"; else st_watchdog=FAIL; fi

# --- 8. status --------------------------------------------------------------------
echo "STATUS run=$RUN_ID pod=$POD clone=$st_clone venv=$st_venv env=$st_env server=$st_server worker=$st_worker watchdog=$st_watchdog"
case "$st_venv$st_env$st_server$st_worker$st_watchdog" in
  *FAIL*) exit 1 ;;
esac
exit 0
