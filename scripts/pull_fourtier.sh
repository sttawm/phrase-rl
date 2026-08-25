#!/bin/bash
# Pull the four pi0.5/LIBERO four-tier shard logs from the lb fleet into
# results/analysis/fourtier_live/ (read-only ssh cat; pods hold no push token).
# Pod->shard map: lb5 runs shard lb4. SSH endpoints come from the session
# scratchpad lb_ssh.json via the sshp helper (SSHP env var).
set -uo pipefail
SSHP="${SSHP:?set SSHP=/path/to/sshp helper}"
DEST="$(cd "$(dirname "$0")/.." && pwd)/results/analysis/fourtier_live"
mkdir -p "$DEST"
ok=0
for pair in lb1:lb1 lb2:lb2 lb3:lb3 lb5:lb4; do
  pod=${pair%%:*}; sh=${pair##*:}
  if timeout 45 "$SSHP" "$pod" "cat /workspace/fourtier_${sh}.jsonl 2>/dev/null" > "$DEST/fourtier_${sh}.jsonl.tmp" 2>/dev/null \
     && [ -s "$DEST/fourtier_${sh}.jsonl.tmp" ]; then
    mv "$DEST/fourtier_${sh}.jsonl.tmp" "$DEST/fourtier_${sh}.jsonl"
    ok=$((ok+1))
  else
    rm -f "$DEST/fourtier_${sh}.jsonl.tmp"
  fi
  timeout 45 "$SSHP" "$pod" "cat /workspace/fourtier_${sh}.jsonl.boards.json 2>/dev/null" \
    > "$DEST/fourtier_${sh}.boards.json.tmp" 2>/dev/null \
    && [ -s "$DEST/fourtier_${sh}.boards.json.tmp" ] \
    && mv "$DEST/fourtier_${sh}.boards.json.tmp" "$DEST/fourtier_${sh}.boards.json" \
    || rm -f "$DEST/fourtier_${sh}.boards.json.tmp"
done
echo "pulled $ok/4 shard logs -> $DEST"
