---
name: score-server-batches-over-phrases
description: phase2_score_server bills ~2.7s per CONTEXT regardless of phrase count — batch phrases per context or pay a factor of P
metadata: 
  node_type: memory
  type: project
  originSessionId: a45839fa-48ee-4aea-bdee-72eb4fc9dccf
  modified: 2026-08-08T07:16:03.916Z
---

`phase2_score_server` batches over the PHRASE LIST inside a context, not over
rows. Its own log line is the tell: `64 contexts / 64 rows in 182s` (one phrase)
versus `64 contexts / 512 rows in 275s` (eight phrases) — the per-context cost is
roughly fixed at ~2.7s, and extra phrases riding along cost ~12.8s each in total.

Cost model that fits both: **`time ≈ 173s + 12.8s × P`** per task at F=4 C=16.

With one phrase per call the GPU sits at **9–17%**; with eight it runs **85–100%**.
Measured 177s/phrase versus 34.4s/phrase — a **5.1× difference**, or 118 hours
versus 15 for a 1,648-phrase bank.

**Why:** this inverts the intuition that per-(frame, phrase) work dominates. It
does not; the context setup does. Any scoring job structured one-phrase-at-a-time
(for resumability, for depth-first ordering, for per-phrase flushing) silently
pays a factor of P.

**How to apply:** always send a chunk of phrases per context. When per-phrase
completion granularity is wanted, chunk (8 for tasks averaging ~8 phrases, 16 for
the sim tasks carrying ~30) rather than going to 1 — it keeps interruption cost
bounded while recovering essentially all the batching. Check the server log's
`N contexts / M rows in Ts` line to confirm M >> N. Related:
[[phase0c-out-parquet-resume]].
