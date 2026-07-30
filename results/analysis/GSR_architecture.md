# GSR — Gated Snapback-Surgery Router (architecture panel synthesis, 2026-07-30)

Combination of the project's four assets: val-8 rules, corpus-mined rules,
vocab-membership tests, RL rewriter. Produced by a 3-architect design panel +
synthesis chair over the experimental ledger.

# PANEL VERDICT — Recommended architecture: **GSR — Gated Snapback-Surgery Router** (deploy stack now, v8-gsr consolidation after)

Skeleton is Architect 3's CGS router (cleanest routing logic, binary census, minimalist reward discipline). Grafted in: Architect 1's snap-back retrieval and exact-census GATE 0 (the single best new idea — the optimal repair of a hostile paraphrase of a known instruction is a lookup, not a generation) plus its provenance-scoped rename table and day-0 desk audits; Architect 2's computed-2x2-tag move for v8 (the only proposal that closes the train/deploy tag gap) and its SFT dataset design. Verified against EXPERIMENT.md (through 2026-07-30) and PREREG_closing_leg.md; all cited artifacts exist in-repo.

## 1. Deployment pipeline (src/phrase_rl/deploy_gate.py + rename_lexicon.json; CPU except R2b)

```
gate(instr):                                   # ONE shared function: dataset builder, trainer, deploy
  if census_exact_or_normalized_hit(instr):    return R0    # GATE 0, bridge_train_uniques.parquet
  nouns = extract_content_nouns(instr)         # sealed_vocab_audit.json code path, lemmatized n-gram
  oov   = [n for n in nouns if census_count(n) == 0]        # BINARY, nonzero=in-vocab, no exceptions
  form  = template_ok(instr)                   # b4_phrasing_rules_v3.md Step-0 as regex; calibrated
                                               # so >=95% of 20k census originals PASS (row-17 lesson)
  if form and not oov:  return R0
  if form and oov:      return R1(oov)
  return R2

route(instr):
  R0: emit instr byte-identical.               # no model, no trace, no API
  R1: for each oov noun with a lexicon entry in evidence-scope: swap noun only
      (ramekin->"white bowl", pepsi brand-carry, keyboard->keep+color, wheel, juice...
       ~15-20 entries, provenance+scope fields, oracle_confirmed > certified > trace-majority;
       DO-NOT-RENAME: orange_juice; color license ONLY on kept-alien nouns).
      No entry -> keep name. Every other byte preserved. Emit.
  R2a SNAP-BACK: normalize nouns via lexicon; retrieve canonical instruction by
      (object, destination, relation) from census/club bank. Fire ONLY if retrieval is
      UNIQUE and role assignment is confident (parse order + relation agree); else R2b.
  R2b MODEL: v7f best ckpt (later v8-gsr), tag "[input: adversarially reworded]",
      self-trace, greedy, gen_sealed_arm.py path, strip emitted tag.
  OUTPUT LINT (all branches except R0): re-run gate on output; unforced zero-count noun
      -> lexicon swap; brand-token diff -> restore dropped brand; hard template failure
      (fragment/no verb) -> rules-v3 pipeline (executor-flat, cheapest available) -> else passthrough.
```

Killed at deploy (per measurement): unconditional rewriting of clean input (row 17 -1.3 pooled, -14..-19 in-vocab cells), any model on R0/R1, executor-tier engineering (0.6pp spread), sampling/best-of-n (greedy peaks, twins gated), trace calls outside R2b.

## 2. v8-gsr training recipe (consolidation, not prerequisite — the stack ships without it)

**Tags:** replace provenance tags with the computed 2x2 from `gate()` ([in-corpus|oov:<nouns>] x [canonical|non-canonical]) — deterministic, identical train/deploy, unit-tested against sealed_vocab_audit.json's 12 verdicts. SFT from BASE Qwen (provenance-tag lineage conflicts; v7a cold-start precedent is fine for SFT).

**SFT (~20k rows, loss on output only, overnight L40S):**
- COPY ~5k, loss-weight 0.5: 213 club instructions (results/analysis/club_phrase_ranking.parquet) + census sample of >=2-ep canonical forms -> target = input. Kill switch: drop entirely if keyboard-repair tripwire fires.
- SNAP ~8k: SFT-17k ERT bank (gen_traces_17k.py lineage) where underlying task in-corpus -> target = canonical original; plus inverse-corruption rows from scripts/probe_transforms.py operators (rename_visual, elaborate, destination_first, add_spatial, verb_synonym, color_adj) applied to club originals -> target = undo. The transform grid becomes supervision.
- SYNTH-OOV ~5k: club instructions with one noun swapped to a census-verified zero-count noun (~100-entry lexicon, LLM-generated once, census-filtered) -> target = original; include keep+color keyboard-class, brand-restore, closed-family rows.
- JARGON-WINNER ~1k: club instructions where the high-fidelity ranking winner beats original at significant z -> target = winner. Everything else in-corpus: target = original.
- Keep ~25% [withheld] dropout rows.

**RL (scripts/run_arm_v8.sh = sed-clone of run_arm_v7f.sh):** --init-adapter results/checkpoints/sft_v8/final; KL reference = the SFT init (v7b: beta/KL-target not a lever, re-pointing is free); reward = v7f's instrument UNCHANGED — grip-pure rank01, F=4 x C=10 same-instruction club contexts, GRPO, beta 0.05, lr 7e-6. **NO shaped reward terms** (CGS wins this conflict: every shaping term is an unmeasured component; the external gate + output lint covers identity and emissions mechanically). Source mix hostile-heavy 40% real ERT / 30% transform-corrupted / 20% synthetic-OOV / 10% withheld — no clean-canonical RL regime (deploy never routes it to the model). Probes from birth: probe8 + probe8_adv tagged + NEW gate-probe (byte-identity rate, monitoring only). Checkpoint of record: 2-rep tagged adv val-8 (n=384), keyboard cell as capability tripwire, one polish sanity point. Sync watcher from birth. Ship rule: v8 replaces v7f on R2b only if 2-rep adv beats v7f's best by >=1.5pp with keyboard cell >= v7f's 14.6.

## 3. Staged measurement plan

**Stage 0 — desk, today/tomorrow, CPU only, no sealed exposure:**
1. Gate coverage audit (Arm F): run `gate()` over ~20k census + 213 club. Targets: >=95% originals route R0; every canonical ERT routes R2. Calibration dial for the form regex.
2. Snap-back precision/coverage audit: the ERT bank pairs (source instruction, ERT) are free ground truth — measure offline what fraction of banked ERTs R2a retrieves, and how often it retrieves the RIGHT canonical. This one number conditions every prediction below; freeze predictions after it, before any roll.
3. rename_lexicon.json curated with provenance/scope fields; tag-function unit tests.
4. Composite predictions from banked parquets (arm-C methodology), filed before rolling.

**Stage 1 — val-8 arms, existing components, ~1 pod-day total (2-rep, n=384/condition; references Orig 40.6, Adv 34.5, Oracle 54.7, frozen-Qwen repair 35.9, v7f adv 2-rep 38.3-40.4):**
- Arm G-NOM (stack, nominal input): **predict 43.5 (band 41.5-46)** — in-vocab pinned at Original by R0, ramekin-class lexicon rescue on OOV (ramekin Orig 11.1, rename-family datum 66.7). Test: >= Orig; entire delta in OOV cells.
- Arm G-ADV (stack, adversarial input — headline): **predict 41.0 (band 38.5-43)** at >=70% snap coverage; degrade toward v7f-alone if Stage-0 shows lower coverage. Tests: >= v7f band top; >= frozen 35.9 + 3; in-vocab adv cells recover toward 53.7.
- v7f steps 80/100 2-rep (already queued) gate the v8 fork checkpoint and whether RL is worth running at all.

**Stage 2 — sealed, ONLY by documented amendment filed before the affected closing-leg arms roll (else val-8-only; prereg discipline is absolute):** gated-surgical nominal **predict 39.5 (band 38.5-40.5)** vs originals 36.1, arm C predicted 36.8 — composition: 5 in-vocab tasks at 48.3 byte-identical + OOV ~34.0 (rules rescues +13.2/+16.0/+17.4 kept, taxes structurally deleted). Adversarial stack **predict 33-35** vs rules 31.6 / passthrough 26.6. Only 2-3 genuinely new phrase cells need rolling (~1h micro-leg); the rest is banked-parquet arithmetic.

**Stage 3 — after v8 trains (~4-5 pod-days, next week):** D0 (SFT-only) repair predict 38.0 (36.5-39.5), gate-probe byte-identity >=95%; failure criterion D0 < 38 blocks RL. v8-gsr repair predict 42.5 (41-44), gate-probe >=90%, keyboard >=14.6. Belted-vs-pure delta <=0.5pp (internalization diagnostic, free re-score). Tag-ablation probe: untagged drop >=3pp expected (tags load-bearing but now computable).

## 4. Two biggest risks

**Risk 1 — wrong-canonical snap / rename misfire (the worst measured outcome class: wrong-object rename ~3% absolute; role inversion is expensive — destination_first is significantly harmful).** Mitigation: R2a fires only on unique retrieval + confident roles, else v7f; renames only through scoped lexicon entries with keep-name default and the DO-NOT-RENAME list; decisive because it is measurable offline at Stage 0 against the ERT bank's known source instructions — if snap precision <95% there, tighten or demote R2a to v7f-first before anything rolls.

**Risk 2 — v8 regime interference / distillation ceiling: COPY split bleeds copying into the repair tag (v7a's per-task zero-sum churn), or RL never beats its mechanical teacher, leaving v8 = arm C with extra steps.** Mitigation: the stack never depends on v8 (v7f holds R2b; belted deploy makes polish drift moot); COPY split down-weighted with a kill switch; keyboard-cell + rewrite-rate-on-corrupted tripwires at every eval; hard gates at D0 (>=38 repair) and RL step 100 (> D0+1 or ship D0-belted); v7f's 80/100 points decide whether the RL phase launches at all.

**Sequencing:** gate+lexicon+Stage-0 audits immediately (no pod contention with the rolling closing leg); Stage-1 val-8 arms overnight; sealed amendment filed today if the closing leg's C arm has not yet rolled; SFT data gen in parallel; v8 RL takes the L40S after v7f 100. Total: ~2-3 eng-days, ~1 pod-day this week (Stages 0-1), ~4-5 pod-days for v8, <$20 API.