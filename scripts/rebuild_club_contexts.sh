#!/bin/bash
# Regenerate data/contexts_club.parquet (1.4G, gitignored) from the committed
# manifest results/analysis/contexts_club_manifest.parquet.
#
# The manifest is the reproducible part: 16,956 rows naming the exact
# (instruction, episode_index, t) frames the phrase search scored across its 213
# instructions. This script re-pulls those frames' images/state/action chunks
# from the Bridge dataset, so the heavy payload never enters git.
#
#   bash scripts/rebuild_club_contexts.sh          # on a pod with HF_HOME set
#
# Verify afterwards: row count 16956, 213 instructions, 4 frames/episode.
set -euo pipefail
cd /workspace/phrase-rl
export HF_HOME="${HF_HOME:-/workspace/hf_cache}"

META="${META:-$HF_HOME/datasets--IPEC-COMMUNITY--bridge_orig_lerobot/snapshots}"
META_FILE=$(find "$META" -name episodes.jsonl 2>/dev/null | head -1)
if [ -z "$META_FILE" ]; then
  echo "episodes.jsonl not found under $META -- set META=<dir> or pre-fetch the dataset" >&2
  exit 1
fi

# Same parameters the original build used (build_club_contexts.py defaults):
#   --min-eps 16  --per-instruction 20  --points 4
# The manifest is the source of truth; these must reproduce it.
.venv-gen/bin/python scripts/build_club_contexts.py \
  --meta "$META_FILE" \
  --out data/contexts_club.parquet \
  --min-eps 16 --per-instruction 20 --points 4

.venv-gen/bin/python - <<'PY'
import pandas as pd
got = pd.read_parquet("data/contexts_club.parquet",
                      columns=["instruction", "episode_index", "t"])
want = pd.read_parquet("results/analysis/contexts_club_manifest.parquet",
                       columns=["instruction", "episode_index", "t"])
g = got.sort_values(list(got.columns)).reset_index(drop=True)
w = want.sort_values(list(want.columns)).reset_index(drop=True)
print(f"rebuilt {len(g)} rows / {g.instruction.nunique()} instructions "
      f"(manifest: {len(w)} / {w.instruction.nunique()})")
print("EXACT MATCH" if g.equals(w) else "DIFFERS from manifest -- check dataset revision")
PY
