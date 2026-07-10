# Costs

**Authoritative source: the billing consoles.** Estimates below are reconciled to
actual billing where known.

Last updated: 2026-07-08.

## ⚠️ Estimate correction (2026-07-08)

Earlier versions of this file estimated Gemini at ~$11. **Actual billed ≈ $100.**
Two compounding errors: (1) output-token counts were measured from *stored* parquets
(~274 tok/call), but CoVer-style inline reasoning is far longer at generation time
(often 1–2k tok/call); (2) the faithfulness-judge scripts were re-run many times
during debugging, each re-billing before the cache filled. Lesson: the *judge*, not
generation, is the silent money sink — and it recurs every RL step.

## Gemini — ≈ $100 spent (sunk), going-forward slashed

Rough attribution of the ~$100 (not re-billable):
- Teacher batches incl. the crashed-then-resumed run: ~$40 (long traces)
- Faithfulness judging across 0c + re-judges + A/B, with debug re-runs: ~$40
- 0b/0c/CoVer generation arms: ~$20

**Going-forward changes (2026-07-08):**
- **Judge model → `gemini-3.1-flash-lite` ($0.25/$1.50 per M) — 6× cheaper than the
  3.5-flash judge ($1.50/$9).** This is the recurring in-RL cost, so it dominates.
- **In-loop gate votes 3 → 1.** Majority-of-3 reserved for offline analysis only;
  in-loop the gate is a soft filter over 16 candidates, single-vote is fine.
- Disk cache (keyed on original+candidate) already skips repeats.
- **Traces cost $0 going forward** — inline conditioning won the A/B, so no Gemini
  call at train or deploy time. Cached teacher traces feed the ablation only.
- Combined effect: in-loop gate ≈ **18× cheaper per step** (6× model × 3× votes).

**Forward estimate for a full Phase 2 RL run + analysis:** ~$2–5 total
(≈1000 steps × 2 flash-lite judge calls/step ≈ $1.50, minus cache hits; plus a few
one-time analysis passes). Phase 3 eval baseline (frontier rephrases on the test
set) is a one-time ~$3–5 if kept on 3.1-pro for quality.

## RunPod compute — ≈ $12 est. (check dashboard)

A6000/A40 pods self-stop on completion; the network volume (~$0.47/day) is the only
always-on cost.

## Grand total so far ≈ $112

## Cost discipline going forward
- Judge = flash-lite, votes=1 in the loop.
- Never re-run a judging script over the same phrases without confirming the cache
  is warm (cache path `data/gate_cache.json`).
- Enable auto-recharge to avoid depletion stalls (3 so far), OR keep a buffer;
  forward burn is now small enough that a modest buffer lasts.

## 2026-07-10 incident: judge thinking-tokens drain
- Switched gate to gemini-3.5-flash without disabling THINKING -> ~1-2k billed output
  tokens per ~30-token verdict (~500k out / 15 min, ~$5/hr). Est. damage: $20-30 over ~5h.
- Fixed: thinking_budget=0 + include_reasons=False in-loop (verdict-only ~30 tokens);
  judge now env-selectable, DEFAULT = local frozen-base Qwen ($0). Both trainers reverted.
- Rule: any new Gemini call site MUST set thinking_config(thinking_budget=0) unless
  reasoning text is the deliverable — and even then it goes in the body, not thinking.
