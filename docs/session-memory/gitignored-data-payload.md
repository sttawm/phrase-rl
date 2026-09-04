---
name: gitignored-data-payload
description: data/ contexts parquets live ONLY on pods — archive before any pod retirement; regen recipe is seeded+verifiable
metadata: 
  node_type: memory
  type: project
  originSessionId: a45839fa-48ee-4aea-bdee-72eb4fc9dccf
---

The trainer's gitignored `data/` payload (contexts_train.parquet, contexts_val_0b.parquet,
contexts_train_multit.parquet, ~2-3GB) exists ONLY on whichever pod runs training — it is
NOT in git and NOT covered by the checkpoint relay. 2026-07-19: cleared a pod for shutdown
as "nothing unique" and lost the whole payload; v7's first launch died on the missing file.

**Why:** pod /workspace is the only home for large non-git artifacts; "results are pushed"
does not cover training INPUTS.

**How to apply:** before retiring any trainer pod, pull `data/*.parquet` to the Mac archive
(or verify another live pod has byte-identical copies). Loss is recoverable — regeneration
is deterministic: `extract_contexts --split train --n 2000 --seed 0` / `--split val --n 250
--seed 0` from IPEC-COMMUNITY/bridge_orig_lerobot, then `multi_t_contexts --points 4
--seed 0`; VERIFY the (episode_index, t) set against git's
results/phrase_artifacts/chunk_stats.parquet before trusting it (~2-3h, network-dominated).
Related: [[runpod-podstate-backup]].

**2026-08-14 quota-crisis addendum — check LOCAL FIRST:** the Mac's `data/`
holds MASTER copies of every context table (contexts_train, contexts_val_0b,
contexts_train_multit{,16}, contexts_val_multit{,16}) — they were built
locally and shipped to pods. When a pod loses its payload, upload from local
(~1GB, minutes) before scavenging pod volumes or regenerating.
Second resort: `extract_contexts.py` regenerates deterministically from
IPEC-COMMUNITY/bridge_orig_lerobot (hash-split, seeded).
Also two operational footguns from the same incident: `mv dir target/` onto an
existing `target/dir` FAILS (or nests) — verify the move landed before any
cleanup; and never launch `rm -rf old_repo` in the background until every
salvage `mv` is verified (`ls` the destination, not the exit code).
