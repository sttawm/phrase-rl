# V7 DECISION MEMO — phrase-rl
**To:** Project lead | **Date:** 2026-07-18 | **Status:** Final, panel-adjudicated (6 analysts + red-team attack pass)

---

## 1. DIAGNOSIS: why the rollout ladder is flat

You asked whether the reward is a failed proxy. Answer: **partly yes — on exactly one axis — and the benchmark is simultaneously saturated on the other.** Three mechanisms, each now measured, jointly explain the verifier-vs-rollout divergence. Panel probability split: greedy fixed point ~30%, fine-axis reward blindness ~30%, headroom exhaustion ~25%, GRPO collapse ~5%, eval noise ~2%.

**(a) The greedy fixed point (byte-proven).** Tuned greedy phrases at A60/B60/B80/B120 are byte-identical to frozen repair's canonical phrase on 6/8 tasks (keyboard: all four checkpoints emit exactly frozen's string). Under CRN pinning, identical phrase = bit-identical episodes — the ladder has been partly **re-counting the same rollouts** (eggplant 81.9, ramekin 64.2, keyboard 22.9 frozen across rungs and arms). The 38.0/39.2/37.2 "flatness" is largely the argmax being pinned at frozen's attractor, while the verifier val series measures the whole 16-candidate distribution. Two different objects; they can diverge without contradiction.

**(b) Fine-axis proxy failure (your hypothesis — confirmed, with a named mechanism).** The residual differences ARE real and reproducible: adding "center" costs **−12.8pp on spoon** (0.500→0.378→0.250) and **kills wheel** (nominal "on the black tire" 29.2%; frozen "center of the black tire" 9.7%; tuned "center of the tire" 0.0%); "red can"→"red cola can" is **+9.7pp** on coke_on_plate. Mechanism: frozen repair wins by *dropping* unexecutable hostile qualifiers; the tuned model *re-imports* ERT's "exactly in the middle" as "center" — a semantic cousin of v5's echo-Goodhart that survives the byte-level echo-strip. The reward's fine ranking (0.517 pairwise, saturated) is blind to precisely these 1–3-word deltas that move rollouts 10–30pp. The coarse relevance instrument remains excellent (AUC 0.94–0.996, validated in vivo both directions) — your hypothesis is wrong about that half.

**(c) Headroom exhaustion (my hypothesis — confirmed pooled, refuted per-task).** Pooled headroom is 1.6pp (clean 41.8 vs repair 40.2) under a ±3–4pp task-clustered CI floor: pooled superiority on these 8 tasks is **arithmetically unreachable**. But pooling cancels a large inversion: the per-task realizable ceiling from *already-measured phrases* is **47.7% pooled, ~+7.5pp above the bar**, concentrated in carrot_on_wheel (+17.7–19.4pp), eggplant (+13.9–14.9pp), stack (+11.8pp) — offset by coke_on_ramekin where repair 63.9 beats clean 31.9 (likely a real phrasing effect, "vertically inside"; audit pending, see §3). Headroom is **mislocated, not absent** — and it lives exactly on the fine axis the reward can't see.

**GRPO dynamics: exonerated.** Duplication series (computed from tlog6_B.jsonl, 172 steps): dup_rate 8.1% → ~25–26% (single-step max 46%), effective uniques 14.7 → ~12 of 16, n_pos pinned at 128, parse-fail 0.0, adv_max stable ~2.5, grad_norm 0.51→0.46. Rising, not collapsing; linear projection hits 50% dup near step ~350–450, past the stop rule. KL fits **exponentially** (doubling ~every 70 steps): projected ~0.61 at step 300 — under the 1.2 abort, but 2x the naive linear estimate.

**Correction to the briefing:** measured step time is **398s (6.6 min)**, not 2.7 min — gen 56s (14%), score 119s (30%), update 223s (56%). Budget everything at 6.6 min/step.

---

## 2. YOUR AGENDA, ITEM BY ITEM

**(1) Two new SIMPLER tasks with a\* — ADOPT, modified.** Correct instinct; wrong a\* premise. No Bridge demos exist for any new combo (or any CoVer OOD task) — a\* means **pseudo-demos from recorded successful frozen-policy clean-instruction rollouts** (proven eggplant pipeline: 60 trajectories → 2,100 feature rows; ~1–4 A6000-h/task on idle pod4). Two corrections from the attack pass: (i) the clean-minus-repair "improvability" gap is winner's-curse-prone — the repair leg is effectively ONE byte-frozen phrase at n=72 (±9pp SE); **re-measure shortlisted tasks with multiple distinct ERT instructions x12 and halve measured gaps in the power calc** before adopting; (ii) binding caveat: sim frames invert learned fine readouts (0.172–0.310 pairwise) — sim a\* rows feed only rel-gate+grip scoring, never learned fine heads. Also note: a\* is needed only for the offline-reward path; evaluation and rollout-reward training need none (analytic success).

**(2) Chunk-diff-only small MLP — REJECT. It was built and it lost.** Executed per your spec (32x16, 5-seed, hard negatives kept; results/analysis/chunk_reward_bakeoff.json): coarse AUC 0.829/0.777 vs the ensemble's 0.976; fine 0.736/0.632 vs 0.889; full-feature variant **inverts on sim (0.310)**. Dominated everywhere, including as the fine term inside the gate (0.793/0.778/0.604 vs hybrid 0.862/0.910/0.806). Salvage only as sim-augmented/hard-negative experiments scored against Gate Zero labels.

**(3) Gripper-only — MODIFY.** Standalone is dead: coarse AUC **0.191 on sim, actively inverted**. But as the fine term behind the relevance gate it is the leading v7 reward (eggplant-sim 0.862 pairwise, multit 0.910, oldctx 0.806, best own-spearman 0.672). Honesty note the red team forced: the hybrid **currently fails its own adoption gates** — top1_regret 8.0 (carrot-multit) and 19.4 (stack-oldctx) vs the ≤6pp cap, and its multit number includes 5 train-overlapping episodes. It must be repaired (rank-blend / winsorized grip / per-task gate calibration) and re-cleared with clustered-bootstrap CI lower bounds, with **top-1/top-3 regret under argmax as the primary gate metric** (that is what GRPO exploits — v5 precedent). No silent waivers.

**(4) 8f/16f scoring — REJECT.** Measured from tlog: 8f = **1.30x** step time (your 1.25x guess: close), 16f = **1.90x** (your 1.5x: optimistic). The 1/2/4f AUC curve (0.945/0.969/0.976) projects +0.002–0.005 for 8f — on the coarse axis, which is already saturated — while study-side **fine ranking declined 2f→4f (0.917→0.868)**: more frames are contraindicated for the failing axis. Plus 8f needs a new feature-generation campaign (~2x the 4f build, hours of pod GPU, 5-seed retrain, reference re-pin = val-series level break). Minority salvage (analyst 5): 8f as checkpoint-selection-only scorer, out of the training path.

**(5) Keep v6-B to the stop rule — ADOPT, conditionally, at honest cost.** True marginal cost is **$20–27** (398s/step + the x12 at ≥300 on dual pods), not $5. Amendment: run a **best-of-16 (verifier-pick + oracle-pick) rollout column at s160 today** (~$2–3). If oracle-minus-greedy < 3pp (phrase space saturated), amend the stop to fire at the next eval12 (~step 200–220) — an exogenous cost-model correction, saving $10–15; if ≥3pp, the model is improving under greedy's shadow and running to 300 is justified *and* best-of-N deployment becomes a live lever. Tripwires: dup alert 0.40 / halt 0.50; KL kill at >0.5 before 250 with ladder <38 (note: the exponential fit crosses ~0.5 near step ~285 anyway). No mid-run knob changes. Harvest: s160/s200 verdicts, KL/dup tail, spoon s80-vs-s120 byte-audit.

**(6) 10 val / 13 test — MODIFY.** Registry v3 already assigns **VAL-11 (8 touched + 3 CoVer OOD) / TEST-12**, and the CoVer OOD assets are already ported (reference/cover_ood_port; ~40-line env wiring each). That gets val 8→11 with **zero unsealing** and keeps test larger (12>11) — strictly better than your 10/13, which spends sealed capital unnecessarily. **Open decision you must make (dissent d-below):** analyst 3 argues the CoVer trio (redbull/zucchini/tennis) is the only direct head-to-head vs CoVer's published OOD ERT rows and should stay sealed in TEST (VAL-10/TEST-14 from natives instead). Until you rule, the trio stays untouched — screening burns it permanently. Either way: registry entry (date+commit) before any v7 asset touches a new task; CI floor improves ±4 → ~±3.1–3.6pp at 11–14 test tasks.

**(7) Duplication monitoring — ADOPT; question answered above.** Series computed; rising (8%→26%), not collapsing; effective group ~12/16. Add to v7 code (not mid-run): per-group n_unique/n_parsed logging + degenerate-group fraction (zero advantage spread), since pooled dup_rate is only a proxy. Tripwires as in (5).

**(8) Bigger lr / n=32 — REJECT; the premise is measured false.** Generation is **14%** of step time; the **update dominates (56%)**. Naive n=32 costs 1.85–1.9x/step for √2 noise reduction, and at 25% dup the marginal 16 candidates add <13 uniques. lr 2x projects KL 0.65–1.3 at step 300 (abort territory, v5 precedent); minority position (analyst 1): ≤1.5x could be entertained with the KL breaker — not recommended. **Designated cheap response if dup sustains >0.4:** the compute-neutral swap **4 ctx × n=32** (same 128 candidates/step, ~$0 marginal, stratification 2/2/4 → 1/1/2).

---

## 3. THE V7 PLAN

**Day 0 (today; parallel with v6-B tail) — ~$6, mostly pod4 (idle A6000) + Mac.**
1. **Ramekin cross-pipeline audit FIRST** (pod4, minutes–$2): score repair's "vertically inside the white bowl" under the eval12nom v2-asset pipeline and the clean phrase under the screen pipeline, same 24 episodes. Adjudicates real-phrasing-effect vs artifact; corrects the headroom denominator every other number uses. (Honest prior: real.)
2. **Gate Zero — phrase-variant reward-ranking audit** (pod4, <$0.50, ~30 min): pre-register ~8–10 powered pairs (pooled n≥288/phrase, CI excludes zero: spoon±center, wheel±center, coke±cola, stack, eggplant); score live ensemble, repaired hybrid, grip, modifier-penalty variant; require correct sign on ALL pairs + report Spearman margin; case/punctuation-normalized control (a candidate whose ordering flips under normalization is keying on tokenization). This is a **necessary-condition gate**: FAIL kills the offline-reward path outright; PASS only licenses proceeding and must be followed by a within-group discrimination check on actual tlog GRPO groups.
3. **Best-of-16 column at s160** (ladder A6000, $2–3) → decides the v6-B early-stop amendment per §2(5).
4. v6-B continues under tripwires (owner: supervisor, $12–27 depending on amendment).
5. Mac: writeup sections (§4).

**Day 0–1 — bake-off repair + task screening, ~$5–10.**
6. **Bake-off, gates enforced** (pod4): fix hybrid top-1 regret, clustered CIs with lower-bound clearance, recompute multit minus the 5 overlapping episodes. Additionally require **out-of-sample scoring on labeled rows from ≥1 new task** (pseudo-demo a\* rows) — the only non-circular transfer evidence, since all current tables cover the 4 old tasks.
7. **Improvability re-screen** of 2–3 ex-TRAIN natives (nut_on_wheel, eggplant_on_sponge, carrot_on_ramekin): multiple ERT instructions, x12, headroom vs frozen best-of-16 (not clean greedy), gaps halved for power. Adopt the 2 largest. (~$3–5; screened-but-unadopted = permanently val-burned — screen only what you'd keep.)
8. **Your call on the CoVer trio** (val vs test) → wiring if val (0.5 day labor, $0 compute).

**Day 1–2 — attribution baseline + assets, ~$15–25.**
9. **Gate Minus-One (new, from the attack pass): screen archived v6-B checkpoints (s60/80/120) x12 on the newly adopted improvable tasks** ($5–10, pod4, eval-only). This is the decisive adjudicator the fork was missing: (a) if v6-B beats frozen repair there → saturation confirmed, the v6 *reward* is vindicated, and v7 primary becomes v6-recipe + new-task contexts with the reward swap demoted to a secondary arm; (b) if v6-B fails despite demonstrated headroom → proxy failure confirmed, proceed to the fork below. Without this, a Design-A win confounds "new reward" with "new tasks."
10. Pseudo-demo harvest (gated on the offline path surviving; log layout distribution, cap per-layout share — success harvesting biases toward easy layouts), ERT assets for new tasks (~$1–5 API), **anchor pinning x12 before v7's first ladder read**, registry entry with the two-tier endpoint: **PRIMARY = paired-CRN contrast on the improvable subset** (10–15pp/task effects vs ±6–8pp CI — detectable at 30–50% capture); SECONDARY = pooled 11-task (reported, not claimed on).

**Day 2–4 — v7 primary run, ~$55–110.**
11. **The fork, gated not argued:**
   - **Design A** (offline reward swap): from-scratch GRPO, v6 recipe otherwise frozen (n=16, lr 7e-6, beta 0.15, tags, dropout), reward = repaired hybrid, +~25% new-task pseudo-demo context draws. Launch **only if** a candidate clears the enforced gates AND Gate Zero AND the out-of-sample test. ~$55–70, 2 days.
   - **Design B** (rollout-reward × hostile inputs — the untried quadrant): zero proxy gap by construction (reward = deployment success; echoing hostile input is punished by the reward itself, passthrough 20.4 vs repair 40.2); v5 infra proven at 5–6 min/step. **Precondition: improvable tasks in its training/eval cells** — on the old 8 it's pre-falsified by the CI floor. ~$80–145, 3–4 days, kill gate: no separation by step 200.
   - Current evidence (hybrid failing its own gates; chunk-MLP dead) says **Design B is the likely outcome — resource it as the default, not the contingency.**
12. Shared regardless: modifier-echo countermeasures (per-task spatial-qualifier-import counter; offline-tested echo penalty — three analysts independently converged on this as the cheapest +1–2pp lever), per-group logging, KL abort 1.2, dup tripwires.

**Total: ~$85–130 over 3.5–4 days, inside the ~$35/day envelope** (front-loaded days are cheap; the run dominates).

---

## 4. WRITE UP MEANWHILE ($0 compute)

WRITEUP.md exists (389 lines) — extend it. **Five sections are evidence-complete today:** CoVer protocol audit; instrument calibration (0b/0c); the two-instrument reward decomposition (coarse AUC 0.94–0.996 vs fine 0.517-saturated / sim-inverted, with the frame curve); the v5 echo-Goodhart forensic; and the new **v6 constraint-retention mechanism** (frozen repair wins by dropping adversarial qualifiers; fidelity-optimizing rewards re-insert them) — the two negative-results case studies are publishable regardless of v7. **The cost/headline section waits ~1 hour for the ramekin audit**; until then phrase it audit-robustly: "boot-time repair yields **1.8–2.0x** hostile success at **1/40** CoVer test-time cost, recovering **87–96%** of clean" — the headroom conclusion is branch-invariant (~5.6pp above the bar either way, because ramekin enters bar and ceiling symmetrically). Pre-register both endpoints before v7's first ladder read; any hostile-hard-subset claim must compare against a **freshly re-measured, CRN-paired** frozen-repair bar, never the stale selection-biased cells. TEST task names stay unenumerated in drafts.

---

## 5. DISSENTS (unresolved, on the record)

- **(a) v7 primary design:** offline-reward swap (analysts 2, 5) vs rollout-reward × hostile (analyst 6). Resolved procedurally by Gate Minus-One + Gate Zero + enforced bake-off gates; current data leans B.
- **(b) CoVer OOD trio:** val capital (registered VAL-11 position; majority) vs sealed test for head-to-head comparability (analyst 3). **Needs your ruling before anything touches them.**
- **(c) Harvest-task choice:** measured-improvable-but-expensive (wheel, clean ~10–29% success) vs cheap-moderate (stack 19 / keyboard 22 / coke_plate 25). Compromise adopted: pilot on eggplant (demos exist, 95%), screen the natives.
- **(d) Ramekin inversion:** real phrasing effect (analyst 3; attack pass agrees — likely real) vs pipeline artifact (analyst 5). $2 audit decides Day 0.
- **(e) v6-B tail value:** "protocol integrity worth it" (majority) vs "near-zero info under the greedy fixed point" (analyst 6). Resolved via the conditional early-stop + changed-phrase pre-filtering of future screens.
- **(f) Selection disclosure:** new val tasks chosen partly for improvability must be disclosed in the writeup (analyst 2) — **adopted as a condition.**
- Minority technical notes: lr ≤1.5x permissible with KL breaker (analyst 1, not recommended); 8f as checkpoint-selection-only scorer (analyst 5, recorded).

**Bottom line:** the optimizer works, the coarse reward works, and the benchmark can't pay out. Stop tuning knobs; spend the next $100 on tasks with measured headroom, a fine-axis reward that passes a labeled audit it currently fails, and — most likely — the one quadrant (rollout-reward × hostile) where the proxy gap is zero by construction. The noninferiority result is already banked; write it down before v7 rolls.
---
## ADDENDUM (post-memo, same day): training-vocabulary coverage of eval-task nouns
2,000 train contexts = 1,355 unique instructions, 542 content words, 329 object /
480 receptacle phrases — style diversity is adequate. But the four OOD eval tasks'
nouns appear ~ZERO times (coke 0, keyboard 0, wheel 0, ramekin 0, tire 1) vs
19-106 mentions for the four native-task nouns. The reward therefore never
carried any signal about pi0's preferred vocabulary for the OOD objects — the
"tire" 0.0% failure is an out-of-coverage fallback to the LM prior, unfixable by
more Bridge contexts (Bridge contains none of these objects). Strengthens:
pseudo-demo harvest (puts OOD-object contexts INTO the reward loop) and Design B
(rollout reward = direct gradient on those objects). Script:
results/analysis/train_vocab_coverage.py
