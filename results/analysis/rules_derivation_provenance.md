# Rules derivation: the exact algorithm and adversarial involvement (2026-08-07 forensics)

Multi-agent reconstruction (12 agents, 7/8 verifications CONFIRMED; the one
REFUTED verdict corrected a channel-attribution clause and confirmed the
substantive claim with exact episode-level recounts). Run wf_ceba86b0-e0e.

## The algorithm (simulation iteration -> rules v1-v3; training overlay -> v4)

- **Round 0 (Jul 21): evidence assembly.** b4_rules_inputs/ files 01-07:
  corpus stats (17,297 instructions), OXE paraphrase sample, 139 CI-cleared
  contrast pairs, 73 greedy phrasing cells (8 sim tasks, n>=288 — including
  the val-quartet ERT-repair arms), 435 sampled-pool draws, case studies,
  self-traces. README = the distiller prompt.
- **Round 1 (Jul 21): v1.** Fresh-context Claude agent ("B4 runner") authors
  10 rules from files 01-07.
- **Round 2 (Jul 22): v2.** Triggers: v1 produced 7 verbatim failure modes on
  12 unseen instructions (text-only; roll intercepted), and the val-8 phrase
  search completed (Claude-proposed 16-phrase boards/scene, 36 eps/phrase,
  leaders confirmed on held-out layouts x72). Fresh-context rerun with the
  search scoreboards (declared highest authority) + the failure list.
- **Round 3 (Jul 23): v3 / v3.1.** Triggers: the executor-bound finding
  (Qwen-9B could not execute v2; rules re-targeted as a frontier-ICL
  protocol) and evidence purification. New evidence: the **xcert
  certification suite** — per task a PLAIN base phrase + ~6 single-edit
  variants, pre-registered, rolled 24 layouts x 6 reps (n=144/phrase) = the
  "cert" numbers. Authoring: 7-agent workflow (3 drafts -> red-team against
  the 7 historical failure ERTs -> synthesis); v3.1 adds the exact
  corpus-vocabulary appendix.
- **Round 4 (Jul 30): v4.** Training-corpus overlay (213 boards, 664-delta
  16-transform probe, 4-round hypothesize-then-holdout: 14/14 refuted) via a
  second 7-agent workflow (3 miners -> draft -> 2 adversarial verifiers ->
  revision -> leakage sweep).

## Adversarial involvement — the four channels

1. **As repair-arm INPUTS (largest channel).** The 4 CoVer-release ERT
   instructions + 4 Gemini "keeper-quartet" ERTs were the standing hostile
   inputs of every val-quartet repair arm; rephraser outputs rolled at
   n=288-1,152/cell, and the ERT instructions THEMSELVES were rolled at
   pooled n=360-648/task (battery passthrough 288 + dedicated screen 72 +
   tuned-checkpoint echoes 288 on carrot/stack). These cells pervade
   evidence files 03/04/05 read by every distillation round.
2. **As red-team GRADERS in v3 authoring** (text-only, no rollouts): drafts
   were stress-tested against the 7 historical failure ERTs.
3. **As hostile-STYLE single edits in the cert suite.** The cert bases are
   PLAIN training-register phrases; the variants probe adversarial-style
   edits (basket->bin, category words, "upright"...). The v4 legend's
   "measured on ornate/adversarial inputs" is loose: cert measured
   hostile-style EDITS on plain bases.
4. **The +12..+69 "rebuild" gains are NOT adversarial-input results**: their
   baselines are the tasks' own bad NOMINALS ("put coke can on ramekin"
   9.0% -> "white bowl" rebuild 78.1% = +69.1). None of the four rescue
   baselines is a CoVer ERT string.

## Sealed-suite contact: one text-only qualification

No sealed rollout number and no sealed ERT instruction entered derivation.
Qualification: on Jul 22, v1 was applied to the 12 sealed ERTs (roll
intercepted); TWO failing rewrite outputs were quoted verbatim in evidence
file 09 read by later rounds ("put the purple object on the black keyboard",
"put the block on the dish") — two sealed task compositions were thereby
inferable. Disclose as a footnote; no measurements leaked.

## Paper-impact corrections

- Say "repair rules were distilled from evidence including ERT-repair arms
  and hostile-style single-edit certification" — not "measured on
  adversarial inputs" (loose) and not "rebuild gains on adversarial inputs"
  (false; gains were over bad nominals).
- The held-out-attacker + held-out-tasks claim for sealed ERT stands, with
  the file-09 footnote.
