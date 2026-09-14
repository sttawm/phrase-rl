# single_edit_pairs — every single-edit phrase pair at n=50 (π0.5 / LIBERO)

Built by `scripts/build_libero_single_edit_pairs.py` from the per-episode records in
`roll50/`, `round2/`..`round8/`, `pilot_raw/{pilotA,pilotC,minpair_partial}.jsonl`
(18,379 episodes). One row per unordered pair of phrases on the same task whose
whitespace tokens differ in exactly one contiguous span. Every phrase has exactly one
episode on each of inits 0–49 (seed 7), so the two sides of a pair were measured on
the same 50 initial states. Success = the environment's goal check within the suite's
step budget (spatial/object/goal 300, libero_90 400, libero_10 520).

Files: `single_edit_pairs.parquet` / `.csv` (426 pairs, 71 tasks), `n50_phrases.parquet`
(367 phrases at n=50, with a `succ_replicate` column for the 21 pilotA phrases that
pilotB re-measured).

| column | meaning |
|---|---|
| task, suite, task_id, canonical | LIBERO task and its finetune/canonical string |
| split_20260901 | train / val / sealed_test from `splits.json`; `unassigned` = the reserve in-finetune suites (spatial, object, 10) and libero_90 tasks outside that split |
| eval_set_20260904 | stratum in the 22-task FINAL_EVAL set (`eval_set.json`), else blank |
| phrase_a, phrase_b | the pair; phrase_a is the canonical when one side is (`a_is_canonical`), else alphabetical |
| edit_type, span_a, span_b, span_pos | difflib opcode (replace/insert/delete), the differing tokens on each side, token index of the span |
| edit_class | lexicon heuristic on the span: case, punctuation, determiner, color, preposition/particle, verb/frame, words added/removed, noun/other |
| succ_a, succ_b, delta | success % of each side (n=50) and delta = succ_b − succ_a in pp |
| p_z, p_fisher | two-proportion z test; Fisher exact (two-sided) |
| both, a_only, b_only, neither, p_mcnemar | paired 2×2 over the 50 shared inits and exact McNemar p on the discordant inits |
| designed, design_note | the pair was a (canonical, edit) row in a round manifest, with that round's group/label/rationale |
| source_a, source_b | which file each side's episodes came from |
| significant_18pp | \|delta\| ≥ 18 pp and p_z < 0.05 (18 pp is the resolution floor at n=50: repeat-measurement spread 1.6 pp mean / 8 pp max) |

Counts (2026-09-14): 195 canonical-anchored pairs, 23 significant / 172 within noise;
231 further cross pairs between edits of the same canonical (ladder rungs), 92 significant.
Two sealed_test tasks (libero_90/30, libero_90/38) are present — filter on
`split_20260901` / `eval_set_20260904` once the test set is chosen.
