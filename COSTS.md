# Costs

Running ledger of external spend. **Authoritative sources:** Google AI Studio
billing console (Gemini) and the RunPod billing dashboard (compute) — the numbers
below are my estimates, good to ~±10% for Gemini (measured token counts × list
pricing) and ~±30% for RunPod (wall-clock estimates; check the dashboard).

Last updated: 2026-07-07.

## Gemini API  —  ≈ $11 total

Pricing (2026, interactive/non-batch): 3.1 Pro $2.00/$12.00 per M in/out;
3.5 Flash $1.50/$9.00. Input ≈ 458 tok/call (256×256 image ≈ 258 + prompt ≈ 200).

| Run | Model | Calls | Out tok/call | Est. cost |
|---|---|---|---|---|
| 0b sensitivity, 32-list | 3.1 Pro | 250 | 466 | $1.63 |
| 0b sensitivity arm, 32-list | 3.5 Flash | 250 | 486 | $1.26 |
| Phase 1 teacher, 16-list + trace | 3.1 Pro | 1873 | 274 | $7.88 |
| **Gemini subtotal** | | | | **≈ $10.8** |

Free / $0 (didn't bill): the `gemini-3.1-pro` 404 and free-tier 429 attempts; the
free-tier 3.5-flash slow crawl (~35 calls before switching); the 0c rephrase
attempt (credits depleted, 0 succeeded — 0c phrases came from Qwen, local/free);
the teacher resume for the last 127 (depleted, 0 succeeded).

**Notes / lessons**
- The teacher batch is 72% of Gemini spend. It's now *ablation* data (SFT
  warm-start), not the primary path — good to know it wasn't on the critical path.
- One earlier run lost ~$5 of paid results to an in-memory-only save + a
  depletion crash. Fixed (incremental saves); counted as sunk, not re-billed.
- Primary Phase 2 (RL-from-base Qwen) needs **no** Gemini rephrases — only the
  already-generated traces as conditioning. Near-zero further Gemini spend expected
  unless we run the teacher-SFT ablation or top up 0c/teacher gaps.

## RunPod compute  —  ≈ $10 est. (check dashboard)

| Pod | GPU | ~Hours | ~$/hr | Est. cost |
|---|---|---|---|---|
| US-KS-2 (Phase 0a/0b) | RTX A6000 48GB | ~11 | 0.50 | ~$5.5 |
| CA-MTL-1 (Phase 0c, ongoing) | A40 48GB | ~8 | 0.40 | ~$3.2 |
| CPU pod (data pull) + network volume 200GB | — | — | — | ~$1.5 |
| **RunPod subtotal** | | | | **≈ $10** |

Self-stop on job completion keeps idle time near zero; the volume (~$0.47/day) is
the only always-on cost — delete it when the project pauses.

## Grand total so far  ≈ $21

## How to update
Append a row when a run finishes. For Gemini, measure output tokens from the
output parquet (`len(trace)+len(json(rephrases))` ÷ 4) × list price; input ≈ 458
tok/call. For RunPod, read the dashboard.

## Rate-limit / 429 hygiene (added 2026-07-07)
429-rejected requests do not bill — the money lost earlier was successful calls
held in memory when a run died (fixed: incremental saves). Guards now in
`gemini_rephrase.py`: (1) preflight single call before any batch — dead
billing/quota fails in seconds, not after thousands of doomed tasks; (2) hard
abort of all pending calls on unretryable 429s (depleted credits / quota 0) —
no retry storms; (3) retries honor the server's suggested retryDelay; (4)
default concurrency lowered to 6.
