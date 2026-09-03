===RULES===
1. Reuse unchanged the head noun of every physical thing the instruction names — the thing to be moved and the place it goes ("bowl", "book", "caddy", "cabinet", "microwave", "butter", "alphabet soup") — and never replace one with a category word, an appearance description or a function description ("vessel", "basin", "the dark concave thing", "the fat used for baking", "the radiation oven").
2. Write the rewrite in the same plain, spoken register and at roughly the same length as the instruction you were given, and never turn it into formal, ceremonial or padded prose.
3. Say only what the instruction already said: do not narrate the scene, do not explain what an object is for, do not introduce an object or a location the instruction never mentioned, and do not add a position word ("on the left", "at the back") the instruction did not contain.
4. Use the scene description only to confirm that the instruction's own nouns name real objects, never as a licence to swap one of those nouns for whatever the scene calls it instead.
5. Change word order, voice, politeness, question form, punctuation, clause count and contractions as freely as you like — every one of these is measured to have no effect — and in particular do not force the sentence to begin with its verb.
6. Apply rules 1-5 identically to every instruction, because nothing in the wording tells you whether the task is easy, hard or impossible; and if you cannot produce a rewrite that satisfies rules 1-3, return the instruction unchanged rather than reaching for a bolder one.

===RATIONALE===

## Scope and method

Evidence: `results/analysis/pi05_bank/distill_evidence_ood/phrases.csv` — 55 out-of-finetune
`libero_90` tasks, 2,119 measured phrasings, 36,490 real rollout episodes, mean canonical success
40.3%. Bands: floor(<20%) 27 tasks, mid(20-70%) 12 tasks, high(>70%) 16 tasks. No in-finetune
(`libero_goal`) task is present and none was used.

Every number below is my own recomputation with `.venv/bin/python` + pandas, built from an
independently written feature extractor (head nouns = content tokens of the canonical minus a
function-word stoplist, minus spatial words, minus colour words, minus a verb lexicon; `-s`
lemmatisation so bowl/bowls match).

Estimator, applied uniformly:
- Contrasts are **paired inside each (task x kind) cell**, episode-weighted, then averaged over the
  task's cells, then over tasks. Tier and style are confounded by construction, so nothing is ever
  compared across prompt families except the one contrast explicitly labelled as a tier contrast.
- Uncertainty is a **cluster bootstrap over TASKS** (4,000 draws, 95% percentile), never over rows.
  Bootstrapping over rows or cells at these n manufactures rules; that is the single most important
  methodological fact about this evidence.
- Every promoted effect additionally carries a **within-cell label-permutation null** (1,500-2,000
  shuffles of `pct` inside each (task x kind) cell). Bootstrap-only "findings" that die under
  permutation were not promoted.
- **`oracle_board` (764 rows) is excluded from all rule evidence.** Those rows were searched for high
  scores against these same rollouts. Four independent "goodness" features (verb-initial, opening-verb
  reuse, brevity, noun retention) invert sign inside that family — the signature of conditioning on
  the outcome. The rule evidence is the 1,300 unsearched rows (`natural` + `adversarial`), 16,250
  episodes: high band 370 rows / 4,550 eps, mid 360 / 5,400, floor 570 / 6,300.
- The decision-relevant summary throughout is a **sealed-20 simulation**: draw 20 of the 55 tasks
  without replacement, emit one qualifying rewrite per task, score its measured `delta_vs_canon`
  (tasks with no qualifying row fall back to the canonical, delta 0), 20,000 draws.

---

## Rule 1 — keep the object and place nouns

Feature: does the rewrite contain every object/place head noun of the canonical.

| band | effect of keeping all object nouns | tasks | permutation p |
|---|---|---|---|
| pooled | **+3.19pp [+0.65, +6.10]** | 55 | **0.0005** (null sd 0.81) |
| floor | +0.36 [-0.09, +0.88] | 27 | 0.37 |
| mid | +0.01 [-4.44, +6.33] | 12 | 1.00 |
| high | **+10.34 [+3.51, +18.52]**, 11+/3- | 16 | **0.0005** (null sd 2.16) |

This is the largest and cleanest single measurement in the file, and the only one that is not a
cross-family comparison. It replicates **inside one prompt family**: adversarial-only high band
+11.19 [+4.13, +19.78] (16 tasks), natural-only +3.43 [-0.15, +7.24] (12 tasks) — same sign, and the
adversarial arm alone clears on its own. Leave-one-task-out on the high band never falls below
**+7.50** (drop `libero_90:55`) and never exceeds +11.40, so it is not one task. In a within-cell
demeaned, episode-weighted regression on the high band with `n_words` and `oov_rate` as controls,
`keep_obj` = +4.12 while `n_words` = +0.017 and `oov_rate` = +4.85 (wrong sign) — it is a specific
noun effect, not generic lexical overlap or brevity.

**The payload is tail risk, not average lift.** High band, catastrophic rows (delta <= -40pp):
14.3% of the 105 noun-dropping rows collapse versus 6.4% of the 265 noun-keeping ones; task-clustered
difference **+15.77pp [+3.91, +28.96]**, 6 tasks positive / 1 negative, and +17.74 [+5.24, +32.62]
within the ornate family alone. The losing rows are all periphrasis: *"the black-pigmented basin"*
(0% against a 95% canonical), *"the dark concave receptacle from which one might eat cereal"* (0%),
*"the canned item that helps children learn their letters"* (10% against 100%), *"the radiation oven"*
(20% against 80%).

**Two honest limits.** (a) It is insurance, not protection: 17 of the 32 high-band collapse rows
*keep* the object noun. (b) It buys nothing on 39 of 55 tasks — floor +0.36 and mid +0.01 are
indistinguishable from zero — so its population value is a fraction of its high-band value. A plain
rephraser breaks it on 25.2% of rewrites; an ornate one on 47.7%.

## Rule 2 — plain register, same length

This is the one **tier contrast** in the rulebook and it is labelled as such: `natural` and
`adversarial` are two prompt families, not two treatments. They differ by construction — 10.5 vs
20.8 words, out-of-vocabulary rate 0.31 vs 0.60, verb-initial 51.1% vs 3.4%.

Paired within task on identical composition (10 phrasings x ~10 episodes per arm, all 55 tasks,
16,250 episodes), plain minus ornate = **+2.68pp [-0.10, +5.82]**; within-task label permutation
(2,000 draws) gives null +0.02 ± 0.61, **z = +4.45, p = 0.0005**. By band: floor -0.37 [-1.29, +0.38],
mid +2.46 [-4.43, +9.37], high **+8.00 [+0.91, +16.92]**. Against the canonical rather than against
each other: ornate -1.92 [-7.33, +3.69] pooled and **-10.87 [-23.10, -1.14]** on the high band, versus
plain +0.76 [-2.78, +5.42] and -2.87 [-6.76, +0.32]. Heavy paraphrase wins in no band. In the
sealed-20 draw, plain beats ornate in **91.3%** of draws.

**Why it ranks below rule 1 despite a larger population number.** Conditioning both arms on rule-1
compliance shrinks the gap to +1.39 [-1.54, +4.71] pooled and +4.69 [-1.36, +13.27] on the high band
— no longer significant. The two rules are not independent, and the register contrast is the
confounded half. It is kept because (i) it survives partially even after conditioning, (ii) it is the
lever with the largest measured population value taken alone (see the ladder below), and (iii) the
mechanism the ornate family supplies — periphrasis — is exactly what rule 1 forbids, so the two are
mutually reinforcing rather than double-counted.

**Counterexample stated honestly:** ornate rewrites that keep the object nouns still reach ceiling on
several high-band tasks — *"You are hereby instructed to execute a transfer of the cream cheese..."*
100%, *"If it is not too much trouble, I would greatly appreciate it if you could relocate the
chocolate..."* 90% (+15pp). Register is a tendency, not a constraint; the nouns are the constraint.

## Rule 3 — add nothing the instruction did not say

Feature: fraction of the rewrite's content words absent from the canonical's, median-split within each
(task x kind) cell.

| band | below-median added content | tasks | permutation p |
|---|---|---|---|
| pooled | **+1.49pp [+0.00, +3.02]** | 55 | **0.0073** (null sd 0.58) |
| floor | +0.01 [-0.84, +0.85] | 27 | — |
| mid | +0.44 [-3.61, +4.30] | 12 | 0.79 |
| high | **+4.78 [+1.32, +8.66]**, 9+/5- | 16 | **0.0020** (null sd 1.35) |

Not a tier artifact — it replicates inside each family separately on the high band: natural +3.28
[-0.08, +7.43], adversarial +6.78 [+1.98, +12.21]. Not rule 1 in disguise either: restricted to rows
that already keep every object noun, the high band still shows **+2.39 [+0.06, +4.92]**. In the joint
high-band regression the added-content coefficient (-11.99 per unit) is the larger of the two axes,
with `n_words` dead at +0.017 — length itself does nothing; *new content* does.

The explicit "no invented position word" clause is the one specific sub-form that clears a nominal
bar: among rows that keep the object nouns, adding a spatial word the canonical lacked is -2.17
[-4.03, -0.52] pooled and -6.05 [-12.16, -0.48] on the high band. Unconditionally it is -0.61
(p=0.50), and three prior operationalisations of it were null, so it is stated as part of rule 3
rather than as its own rule.

**This rule is what closes the high-band loss.** Operating points versus the canonical, high band,
16 tasks:

| policy | high-band delta vs canonical |
|---|---|
| any unsearched rewrite | **-6.87 [-15.03, -0.81]** |
| + keep object nouns | -4.39 [-11.09, +0.65] |
| + plain register | -2.87 [-6.52, +0.37] |
| + plain and keep nouns | -2.49 [-6.04, +1.03] |
| + plain, keep nouns, add little | **+0.03 [-4.45, +4.57]** |

## Rule 4 — scene confirms, never substitutes

There is no positive evidence for scene-conditional rewriting anywhere in this evidence, and the
failure mode this rule blocks is the most expensive one measured. Adding a colour or material
descriptor looks costly unconditionally (-3.42 [-8.20, +0.74], permutation p=0.025) but the sign is
entirely rows that *also swapped the object noun*: restricted to noun-keeping rows it is +1.39
[-0.83, +5.00] and on the high band +2.22 [0.00, +6.67]. So attaching an accurate descriptor to a
retained noun ("the yellow stick of butter", "the red and white can of alphabet soup" — both 100%) is
harmless; *replacing* the noun with what the scene shows is rule 1's -10 to -17pp. Independently,
adding a spatial qualifier the instruction lacked is -0.61 pooled, permutation p=0.50 — grounding buys
nothing. Two prior lenses tested task-family-specific styles (permutation p >= 0.29 for every feature)
and selector preservation with synonym matching (+2.25 [-3.06, +7.69]); both null.

## Rule 5 — the licence

Stated as a permission because a rephraser given only prohibitions invents its own, and the ones it
would invent measure negative. Every style axis, within-cell, unsearched, 55 tasks, 16,250 episodes,
task-clustered bootstrap plus permutation:

| feature (pooled) | effect | perm p |
|---|---|---|
| shorter than cell median | +0.87 [-0.11, +1.92] | 0.14 |
| lower rare-word rate | +0.79 [-0.42, +2.13] | 0.16 |
| keeps canonical's spatial words | +1.35 [-0.69, +4.17] | 0.10 |
| keeps canonical's colour words | +2.05 [-0.17, +5.12] | 0.11 |
| polite / hedged wrapper | +0.07 [-0.87, +1.03] | 0.93 |
| verb-initial | **-1.01** [-2.34, +0.16] | 0.17 |
| contains a comma | +0.46 [-1.10, +2.18] | 0.59 |

All seven cross zero. Verb-initial is called out in the rule text because its point estimate is
negative in every band (-1.62 on high) and it is the single most likely rule for a naive reader to
invent from "robot commands should be imperative". Two single-band cells look publishable and are
not — high-band politeness +2.92 [+0.85, +5.15] and high-band commas +3.90 [+1.38, +7.44] — against
14 tests, 2 nominal hits is chance, and both die pooled (p=0.93, p=0.59).

## Rule 6 — uniform policy, and unchanged is allowed

**Difficulty is invisible from the text.** Three tasks in this evidence share the *identical* canonical
string "put the black bowl on top of the cabinet": `libero_90:10` at 95.0% (high), `libero_90:25` at
0.0% (floor), `libero_90:31` at 9.3% (floor). "pick up the book and place it in the left compartment
of the caddy" appears at 10.0%, 39.1% and 25.0%. No band-conditional rule is executable, and none is
written.

**Rewriting is a coin flip.** Unsearched pool minus canonical, per task, episode-weighted: -0.58pp
[-4.76, +4.28] over 55 tasks, median exactly 0.00, 10 tasks better / 23 worse / 22 exactly tied.
Sealed-20 draw: P(>0) = 0.40. Plain-only: +0.76 [-2.78, +5.42], P(>0) = 0.53.

**Yet a hard "return unchanged" forfeits the one real prize.** `libero_90:44`, canonical "turn on the
stove", scores 1.65% over 60 episodes and 3.3% over 90 more when re-run — the canonical string itself
is pathological. Its 15 plain rewrites average 96.8% and its 15 ornate rewrites 99.6%. That is one
task in 55 worth roughly +95pp, about +1.7pp of population average by itself, and *any* rewrite catches
it (rule-1-compliant ones included: rewrites keeping "stove" average 96.0%). Only 6 other floor tasks
move at all, and 20 of the 27 floor tasks are hard zeros — 420 phrasings, 4,200 episodes, every single
one exactly 0.0%.

The rule therefore permits but does not require the identity, and forbids reaching further when
stuck — the shape that keeps the `libero_90:44` upside without buying the high-band downside.

---

## What we tested and rejected

- **Spatial-word retention as its own rule.** Pooled +1.35 [-0.69, +4.17], permutation p=0.10; the
  high band reads +7.82 but on only 5 tasks, and prior work showed it collapses to +0.14 without
  `libero_90:10`. Rejected. (It is a semantic-correctness matter, unmeasured here.)
- **Colour-word retention.** +2.05 [-0.17, +5.12], p=0.11; *negative* on the high band (-1.23, 0 of 4
  tasks). Rejected.
- **Reuse the canonical's verb.** Null in this population (prior recomputes: +1.26 [-0.29, +2.66],
  permutation p=0.18; its apparent high-band support rests on 2 adversarial cells). Rejected.
- **Length / brevity as an independent lever.** Below-median length +0.87, p=0.14; `n_words` = +0.017
  in the joint high-band model. Length is a symptom of rule 3, not a lever. Rejected.
- **Rare or out-of-finetune vocabulary as such.** +0.79, p=0.16; +4.85 (wrong sign) in the joint
  model. Rejected — what matters is whether the instruction's nouns were *replaced*, not whether
  unusual words are present.
- **Verb-initial / imperative word order.** -1.01 [-2.34, +0.16], p=0.17. Rejected, and explicitly
  inverted into rule 5 so nobody re-derives it.
- **Politeness, question form, commas, clause count, pronouns.** All null (table above).
- **"Preserve the goal prepositional phrase verbatim."** The largest catastrophe cluster in the file
  is `libero_90:10`, where keeping the literal string "on top of the cabinet" gives -8.75pp and 0 of 7
  collapses while mutating it gives -68.33pp and 78% collapses (18 of the 32 high-band collapses in
  the whole population). It does not generalise: the other two tasks with that identical canonical
  give +0.00 and +5.32. One task. Rejected.
- **Scene-family-specific styles, and selector preservation.** Permutation p >= 0.29 across features;
  selector preservation +2.25 [-3.06, +7.69]. Rejected.
- **"Always change something."** Rejected — see rule 6; a near-identity feature reads -15pp on floor
  entirely from `libero_90:44` and +5pp on high.
- **Chasing `task_summary.csv` headroom.** Against an explicit binomial null simulated at the real
  per-row n on the depth-matched unsearched probe (every task got the same ~24 phrasings), best-of-k
  excess over pure selection noise is floor **+0.01**, mid **+2.94**, high **+2.57**, overall +1.39
  [+0.06, +2.91]. The mid band's 28.3pp of apparent headroom is essentially all winner's curse, and
  only 3 of 55 tasks exceed 10pp of genuine excess. Rejected as a source of rules.

## How this differs from the in+out rulebook

`in_plus_ood_v1.md` was built on a corpus where 10 in-finetune `libero_goal` tasks sit at 96.7%
canonical with a 60pp cliff below them. That geometry — "the canonical string is a memorised training
string, so protect it and never echo it" — does not exist here, and it was carrying five of its seven
rules.

- **Its rule 1 (head nouns) survives and is promoted, on new evidence.** That rulebook measured the
  noun effect at -32.16pp in-finetune and reported it as *free* out-of-finetune: "-1.58pp, 55 tasks,
  33 negative / 34 positive, median exactly 0.0, [-4.0, +0.8]". That reading came from pooling all 55
  OOD tasks across bands and bootstrapping over 109 *cells*. Stratified by band and clustered over
  tasks, the same feature is +3.19 [+0.65, +6.10] pooled (p=0.0005) and +10.34 [+3.51, +18.52] on the
  16 high-band tasks (p=0.0005). It was never free out here; it was hidden by the 27 floor tasks where
  nothing moves.
- **Its rule 2 (copy every spatial word verbatim) is DROPPED.** It cited -2.89pp OOD, [-6.0, -0.3],
  over 56 cells. Clustered over tasks with a permutation null the same feature is +1.35 [-0.69, +4.17],
  p=0.10, and its high-band appearance is one task. Cell-level bootstrapping is what made it look real.
- **Its rule 3 (reuse the action verb) is DROPPED.** That rulebook's own evidence for it was
  in-finetune (-2.31pp among noun-keepers) and it conceded the OOD number was "a wash" (+1.66pp
  [-1.5, +5.0]). Confirmed a wash; removed.
- **Its rule 4 ("never emit the instruction unchanged") is REVERSED into rule 6's permission.** It
  rested on exactly two tasks whose canonical is verbatim a finetune string, `libero_90:44` and
  `libero_90:77`. In this population `libero_90:44` is 1 task in 55, it *is* the entire apparent
  floor-band gain (removing it takes plain rewriting from +0.76 to -0.99 and the pool from -0.58 to
  -2.38), and mandating a change is a coin flip that costs -6.87pp on the high band when the change is
  liberal. This rulebook permits a rewrite, keeps the `libero_90:44` upside, and permits the identity.
- **Its rule 5 ("do not compress or de-formalise; keep politeness and extra length; never shorten") is
  REVERSED into rule 2 and neutralised in rule 5.** It cited non-verb-initial phrasing at +2.87pp OOD
  [+0.6, +5.8] over 68 cells and politeness at +1.07pp. Clustered over tasks: verb-initial -1.01
  [-2.34, +0.16] (p=0.17) and politeness +0.07 (p=0.93) — both dead — while the "keep extra length"
  half is contradicted outright, since the 2x-length ornate family is -2.68pp against plain
  (permutation z=+4.45) and -10.87pp against the canonical on the high band. That rulebook instructs
  the rephraser to do the single most expensive thing measurable in this population.
- **Its rule 6 (attach a scene descriptor) is WEAKENED from an instrument to a permission.** There it
  was the recommended way to satisfy "always change something". Here, adding material is worth
  -1.49pp pooled and -4.78pp on the high band (rule 3); an accurate descriptor around a *retained*
  noun is merely harmless (+1.39 [-0.83, +5.00]), not useful.
- **Its rule 7 ("in" not "inside") is DROPPED** as noise at this n.
- **Net:** 7 rules become 6, but only one of the seven (head nouns) survives with its direction intact,
  and two of them (never echo; never shorten) are inverted.

## Expected effect at test

Sealed-20 simulation, 20,000 draws of 20 tasks without replacement, one emitted rewrite per task:

| policy | mean vs canonical | P(> canonical) | 5-95% |
|---|---|---|---|
| unconstrained rewriting | **-0.44pp** | 0.46 | [-7.12, +6.17] |
| ornate + drops a noun (worst case) | **-4.48pp** | 0.19 | [-12.67, +3.75] |
| this rulebook (plain + nouns + little added) | **+1.13pp** | 0.64 | [-3.55, +6.37] |
| this rulebook, `libero_90:44` removed | **-0.10pp** | 0.50 | [-4.13, +3.58] |

**My honest prediction: no measurable gain over the canonical instruction on the sealed 20.** The
expected value of this rulebook against simply passing the instruction through is about +1pp, its
interval spans roughly -3.5 to +6.4, and essentially all of the point estimate is the chance that the
sealed 20 happens to contain a `libero_90:44`-style pathological canonical (about 1 task in 55, worth
~+95pp when it appears). Strip that one task and the expected gain is exactly zero. If the sealed 20
matches this band composition, ~7 of them will be hard zeros that no wording can move, ~4 will be mid
tasks whose apparent 28pp of headroom is winner's curse, and ~6 will be high-band tasks where the only
thing on the table is not losing.

The value of this rulebook is the **+1.6pp gap to an unconstrained rephraser** (paired sealed-20
comparison: +1.59 [-5.50, +9.00], beats it in 66% of draws) and the **+5.6pp gap to a bad one**. It is
damage control, and it should be sold as such.

**What would show it working.** Be warned that the sealed 20 is underpowered for every comparison
on offer, and this should be settled before the run rather than after. Per-task deltas in this
population have a standard deviation of 17.8pp (11.9pp excluding `libero_90:44`), so a paired
per-task test at k=20 has an 80%-power minimum detectable effect of **11.2pp** (7.5pp excluding that
task) against a predicted rulebook-vs-canonical effect of +1.1pp. Even the far larger
rulebook-versus-unconstrained-rephraser contrast, paired per task, has per-task difference sd 10.7pp
and an MDE of **6.7pp** at k=20 against a predicted +1.6pp; restricted to the ~6 high-band tasks the
MDE is 12.9pp against a predicted +6.6pp. Every headline comparison is three to ten times smaller
than what 20 tasks at one rewrite each can resolve.

So: do not read a null sealed result as a refutation, and do not read a positive one as
confirmation — at these n a single `libero_90:44`-style task landing in the draw swings the mean by
~5pp on its own. Three things would give the run actual power, in order of cost:

1. **Emit several rewrites per sealed task, not one**, and score the rulebook's arm against an
   unconstrained arm with matched composition. That is the design this evidence itself uses (10
   phrasings per arm per task), and it is what makes the +1.6pp population gap and the +6.9pp
   high-band gap (-6.87 -> +0.03) measurable rather than notional.
2. **Raise episodes per phrasing well above 10.** At n=10 a single episode is 10pp and roughly 80% of
   within-cell variance is binomial; the true per-phrase effect standard deviation in this population
   is only 4-5pp, i.e. exactly at the detection threshold.
3. **Report the catastrophic-collapse rate** (a rewrite scoring >=40pp below its own canonical), not
   just the mean. It is a proportion over rewrites rather than a mean over tasks, it is the effect
   this rulebook actually buys, and the prediction is specific: on high-band tasks it should fall from
   ~14% of rewrites to ~6%.

Always stratify by measured canonical band. Pooled across bands the effect is diluted by the ~half of
sealed tasks that cannot move in either direction, and a pooled rulebook-vs-canonical mean will come
back indistinguishable from zero — a result fully consistent with this rulebook being correct.

---

## Independent audit of rule 1

Three auditors re-derived the head-noun feature independently (purely lexical head-final NP parse;
task-definition-derived closed lexicon; determiner-span parse with matched pairs), and this section
adjudicates them against a fourth, independent re-derivation of my own. Scripts:
`results/analysis/pi05_bank/audit_rule1_adjudication/` (`feat.py` = determiner-span head-noun
extractor built from the 48 canonical strings alone, `est.py` = estimator, `r1`-`r9` = the contrasts).
Nothing in this audit changes any number above it; the arithmetic in the rule-1 section reproduces.

**Estimator, held fixed everywhere below.** Every contrast is computed inside a (`kind` x `canon_band`)
cell; inside a task the two arms are episode-weighted by `n`; the task is the unit and tasks are
weighted equally; CIs are a 20,000-draw cluster bootstrap over tasks; the noise floor is a
3,000-draw permutation of the feature label **within task** (weights stay glued to rows).
`oracle_board` is reported for corroboration only and never in a headline, because those rows were
searched against these same rollouts.

### 1. What reproduces exactly

Running the rulebook's own feature (all canonical content tokens minus verbs, determiners,
prepositions, spatial words and colour words) through my estimator:

| quantity | rulebook | this audit |
|---|---|---|
| pooled | +3.19 [+0.65, +6.10] | +2.97 [+0.55, +5.97] |
| high band | +10.34 [+3.51, +18.52] | +10.86 [+4.00, +19.04] |
| adversarial-only, high band | +11.19 [+4.13, +19.78], 16 tasks | **+11.19** [+3.97, +19.58], 16 tasks |
| natural-only, high band | +3.43 [-0.15, +7.24], 12 tasks | **+3.43** [-0.12, +7.19], 12 tasks |
| high-band collapse rate (delta <= -40) | 14.3% dropping vs 6.4% keeping | **14.3% vs 6.4%** |

The rule-1 section is arithmetically correct. Everything in dispute is interpretation.

### 2. What is established (four independent feature implementations agree)

**Prevalence — the confound the reader flagged is real and large.** Violation rate is
**45.4% in `adversarial` vs 21.1% in `natural`**: renaming is 2.1x more frequent in the tier that was
generated to be indirect. Rows that actually vary on the feature, per cell (head-noun feature):

| kind | band | rows | violations | tasks varying / tasks | episodes keep vs viol |
|---|---|---|---|---|---|
| adversarial | floor | 285 | 130 (45.6%) | 26/27 | 1690 / 1460 |
| adversarial | mid | 180 | 103 (57.2%) | 12/12 | 1085 / 1615 |
| adversarial | high | 185 | 62 (33.5%) | 16/16 | 1465 / 810 |
| natural | floor | 285 | 59 (20.7%) | 19/27 | 2480 / 670 |
| natural | mid | 180 | 55 (30.6%) | 10/12 | 1875 / 825 |
| natural | **high** | 185 | **23 (12.4%)** | **8/16** | 1980 / **295** |
| oracle_board | mid | 536 | 262 (48.9%) | 11/12 | 6835 / 5945 |
| oracle_board | floor / high | 137 / 91 | 15 / 20 | 3/3, 4/5 | — (searched, ignore) |

`natural` x high is the cell the claimed replication rests on and it is **underpowered by
construction**: 8 of 16 tasks vary, 23 violation rows, 295 episodes, permutation noise floor
SD 3.17pp. It cannot resolve anything below roughly 6.4pp.

**Keep-all minus violation, within kind x band** (episode-weighted, task-clustered, head-noun feature):

| kind | band | effect | median task | sign | tasks vary | null SD | perm p |
|---|---|---|---|---|---|---|---|
| natural | floor | -0.22 [-1.56, +1.05] | +0.00 | 1+/2- | 19/27 | 0.55 | 0.71 |
| natural | **mid** | **-7.38 [-13.72, -1.96]** | -7.12 | 2+/8- | 10/12 | 3.15 | **0.021** |
| natural | high | +4.25 [+0.20, +8.70] | +1.83 | 5+/2- | 8/16 | 3.17 | 0.184 |
| natural | all | -1.19 [-3.71, +1.04] | +0.00 | 8+/12- | 37/55 | 1.15 | 0.30 |
| adversarial | floor | +0.65 [-0.18, +1.73] | +0.00 | 4+/1- | 26/27 | 0.44 | 0.14 |
| adversarial | mid | +1.23 [-5.49, +9.82] | -3.74 | 4+/8- | 12/12 | 2.66 | 0.66 |
| adversarial | **high** | **+13.58 [+4.63, +24.42]** | +3.72 | 10+/3- | 16/16 | 2.53 | **0.0003** |
| adversarial | all | +4.61 [+1.17, +8.75] | +0.00 | 18+/12- | 54/55 | 0.97 | 0.0003 |
| oracle_board | high | +10.02 [+1.29, +24.38] | +4.40 | 3+/0- | 4/5 | 4.09 | 0.019 (searched) |

Established: **the effect is high-band only.** The floor band is flat in both unsearched tiers and the
`natural` x mid cell runs the **wrong way** at -7.38pp, p=0.021, on 10 of 12 tasks — a significant
counterexample to a rule written as "never", and one the pooled mid row (+0.01) above conceals.

### 3. Settling the auditors' disagreements

**(a) "natural-only pooled is -1.19, which contradicts the claimed +3.43" — mis-mapped, withdrawn.**
The rule-1 section's +3.43 is the natural-only **high-band** figure on 12 tasks, which I reproduce to
two decimals with the same feature and the same task count. The -1.19 is natural-only **pooled over
bands**, which I also reproduce. Both are true; they are different quantities and there is no
contradiction. What is fair to say instead is that "it replicates inside one prompt family" is too
strong: the adversarial arm clears on its own, the natural arm is a same-sign point estimate with
permutation p=0.18 against a 3.17pp noise floor and a bootstrap CI that a 8-cluster bootstrap cannot
be trusted to place.

**(b) "The tail-risk payload does not survive matching" — rests on the one threshold where it vanishes.**
That auditor tested P(score == 0). Unmatched, `adversarial` x high gives 4.9% (keep) vs 6.5% (viol) on
that outcome — a near-null. On the outcome the rulebook actually claims (`delta_vs_canon` below a
threshold) the gap is large at every threshold tested:

| threshold | keep | violation |
|---|---|---|
| delta <= -20 | 15.4% | 30.6% |
| delta <= -40 | 9.8% | **24.2%** |
| delta <= -50 | 7.3% | 17.7% |
| delta <= -60 | 6.5% | 12.9% |

Task-clustered, `adversarial` x high, delta <= -40: **+0.23 [+0.07, +0.42]**, 6 tasks positive /
1 negative, null SD 0.05, p=0.0005. The payload stands — **within the adversarial family only**. In
`natural` x high it is -0.04 (keep 3.1%, renamed 0.0% over 23 rows — no information), and
`oracle_board` x high has zero collapses at any level. The quoted 14.3% -> 6.4% is a cross-family
mixture: 65% of the violation rows are adversarial against 44% of the keep rows.

**(c) Is it a tier artifact? No — the noun effect is real inside the adversarial family.**
Four controls, all inside `adversarial` x high, none of which can be a tier effect because the tier
is held constant:

- 16 of 16 tasks vary on the feature; the contrast is never computed across families.
- Task x kind fixed effects plus `n_words` and `n_oov` as distortion controls: `beta_viol` =
  **-10.36 [-18.89, -4.26]** (natural x high -5.45 [-9.86, -1.55]; nat+adv high -8.42 [-15.42, -4.18]).
- 1:1 length-caliper matching within task (|delta n_words| <= 3, 46 pairs, 16 tasks) moves the estimate
  **up**, to **+16.34 [+5.89, +28.60]**.
- Splitting adversarial-high at each task's median `n_oov`, the effect is as large in the **less**
  distorted half (+15.66 [+5.69, +26.85], 15 tasks) as in the more distorted half
  (+13.37 [+3.02, +27.97], 11 tasks). Placebo: among noun-**keeping** adversarial-high rows the
  demeaned slope of success on `n_oov` is -0.27pp/word and on `n_words` -0.07pp/word — renaming is
  not simply an index of how mangled the sentence is.

**(d) `libero_90:10` does not drive rule 1.** Dropping it leaves adversarial-high at +13.67 (from
+13.58) and the collapse difference at +0.24 (from +0.23). It *does* inflate the raw rates: without
it the keep-side collapse rate falls from 6.0% to 1.5% while the violation side stays at 12.8%
(adversarial: 2.7% vs 17.5%). The task-clustered difference is unaffected; the raw proportions in the
headline are not a safe way to quote it.

**(e) The mean is heavy-tailed; quote the median.** adversarial-high mean +13.58 but median task
effect **+3.72**, 10+/3- (exact sign test p=0.092), 10%-trimmed +10.79, drop-three-most-influential
+5.36, leave-one-out range [+9.74, +14.82]. Four tasks supply most of the mass (`libero_90:55` +71.3
on 2 rows, `:50` +38.9 on 1 row, `:56` +37.5, `:33` +31.7). A permutation test on the **median**
statistic still rejects (+3.72, null SD 0.92, p=0.012; nat+adv high +3.79, null SD 0.84, p=0.001),
so a real ~4pp median effect survives the robustness pass even though the ~14pp mean does not.

### 4. Deleting vs replacing, and mild vs aggressive renaming

**Deletion is untested, not null.** Classifying each missing head noun by anchoring on the canonical
tokens that survived and inspecting the phrase span between the flanking anchors: of 1,300 unsearched
rows, **11 are pure deletions** (a looser classifier used by one auditor finds 62) and **zero of them
are in the high band; zero are in `adversarial` x high**. The "reuse unchanged / never drop" half of
rule 1 has no evidence behind it in either direction.

**In the tier where the effect lives, there is no such thing as a mild rename.** Counting the novel
content words that occupy the replaced noun's slot:

| tier, high band | one-word swap | >= 3 words | median novel words |
|---|---|---|---|
| adversarial | **0%** | 89% | **8** |
| natural | 43% | 26% | 2 |

Every one of the 62 adversarial-high violations is a multi-word definite description
("the dark concave receptacle from which one might eat cereal", "the canned item that helps children
learn their letters", "the appliance we use to quickly heat up leftovers"). Every natural-high
violation is a near-synonym in register (bowl -> dish, mug -> cup, stove -> burner/hob/stovetop,
moka pot -> espresso maker, cabinet -> cupboard, chocolate pudding -> chocolate dessert). The
severity axis is therefore perfectly confounded with the tier axis and the corpus cannot separate them
cleanly. Every gradient that is visible points the same way:

- natural-high renames (all mild): mean 90.4% vs 91.5% for keepers, contrast +4.50 [-1.05, +10.59]
  for one-word swaps (p=0.49, 10 rows, 6 tasks), and **0 of 23 collapse**. Corpus-wide, `natural`
  renames collapse on 0.0% of rows vs 1.9% for keepers; `adversarial` renames on 7.5% vs 3.4%.
- inside adversarial-high, collapse rate by substitution size: keepers 9.8%, 2 novel words 14.3%,
  3-4 words 10.0%, **5+ words 28.9%**.

So the answer is "aggressive renaming is costly", with the caveat that this evidence never observes a
mild rename inside an ornate sentence or an aggressive one inside a plain sentence.

**Only the manipulated object's noun is tested.** Restricting violations to the target noun versus to
landmark/destination nouns only:

| contrast | effect | tasks vary | rows |
|---|---|---|---|
| adversarial high, TARGET replaced | **+13.47 [+4.50, +24.09]** | 16/16 | 57 |
| adversarial high, LANDMARK only | +2.32 [+0.00, +3.64], p=0.60 | 3/16 | 5 |
| natural high, TARGET replaced | +4.99 [+0.16, +10.29], p=0.26 | 6/16 | 11 |
| high band, LANDMARK only (nat+adv) | +0.91 [-1.11, +3.69], p=0.57 | 4/16 | 17 |
| **all bands, LANDMARK only (nat+adv)** | **+1.18 [-1.77, +4.17], p=0.42** | 23/55 | 117 |

The last row has real power (23 tasks, 117 rows, 985 rows in the comparison) and is null. "Every
physical thing the instruction names — the thing to be moved **and the place it goes**" is not
supported for the second half.

### 5. Verdict

Rule 1 is **a real noun effect, not a tier artifact — inside a scope roughly one quarter the size of
the one it claims.** What survives: replacing the head noun of the *object being manipulated* with a
*multi-word description* costs roughly +4pp (median task effect, the robust number) to +14pp (mean,
tail-driven) on tasks whose canonical already exceeds 70%, and roughly doubles-to-triples the rate of
catastrophic collapse there. What does not survive as written: the pooled framing (a mixture of one
band that moves and 39 tasks that do not, with one cell running the wrong way), the destination-noun
half, the never-drop half, and "replicates within one prompt family".

One further structural caveat, since rule 1 and rule 2 are sold as mutually reinforcing: the rule-1
section notes that conditioning rule 2 on rule 1 shrinks rule 2's gap. The reverse conditioning is
the more damaging one. **Conditional on the rewrite already being plain (`natural`), rule 1 buys
+4.25 [+0.20, +8.70] (p=0.18) on the high band and -1.19 [-3.71, +1.04] pooled.** A rephraser that
already obeys rule 2 is being asked to obey a rule whose measured payload lives almost entirely inside
the register rule 2 has already forbidden.

### 6. PROPOSED narrowing (rule 1 above is left as written; this is for the user to decide)

Band-conditional wording is not executable — rule 6 above is right that difficulty is invisible from
the text — so the narrowing is on the linguistic axis instead. Tested as a feature, the narrowed rule
captures the entire measured payload while binding on **one third as many plain rewrites**:

| feature | high band effect | high band collapse diff | violated by plain rewrites | by ornate |
|---|---|---|---|---|
| rule 1 as written (any head noun missing) | +15.22 [+5.58, +26.64] | +0.23 [+0.07, +0.42] | 21.1% | 45.4% |
| **narrowed (target noun expanded into >= 2 novel words)** | **+15.66 [+5.62, +27.06]** | **+0.24 [+0.08, +0.43]** | **7.7%** | 38.3% |

> **PROPOSED replacement for rule 1:**
>
> Carry the instruction's own head noun for the object being moved or acted on straight into the
> rewrite ("bowl", "book", "butter", "pot", "soup", "drawer"), and never expand that one noun into a
> multi-word description of what it looks like, what it is made of, what it is for or what category
> it belongs to ("the dark concave receptacle from which one might eat cereal", "the canned item that
> helps children learn their letters", "the appliance we use to quickly heat up leftovers"); a
> single-word near-synonym in the same plain register ("bowl" -> "dish", "mug" -> "cup", "stove" ->
> "burner") carries no measured penalty, and the noun naming *where* the object goes is not measured
> at all.

Executable from instruction plus scene alone: take the head noun of the instruction's first noun
phrase, keep it or swap a single word for it, and never replace it with a phrase. If this narrowing
is adopted, the two "honest limits" stated under rule 1 should be joined by a third: on mid-band
tasks under plain phrasing the measured sign is negative (-7.38 [-13.72, -1.96], p=0.021), so
"never" is contradicted somewhere in this population.
