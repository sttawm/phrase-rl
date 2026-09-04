===RULES===
1. Reproduce every content noun of the incoming instruction in your rewrite as the same word ("bowl", "drawer", "cabinet", "stove", "plate", "rack", "wine bottle", "cream cheese"). Never replace one with a synonym, a category word, an appearance or function description, or a definite description ("dish", "burner", "cooking range", "cupboard", "receptacle", "the flat dining vessel", "that specific cavity"), and never drop one. This applies to the noun naming the thing being moved AND to the noun naming the place it goes — both, with no ordering between them. A plural/singular change is fine; so is a compound that literally contains the noun ("stovetop" for "stove", "cabinetry" for "cabinet"). Copy the instruction's spatial words ("middle", "top", "front", "inside") through unchanged as well: this costs nothing and the case where it matters is untested.

2. Default to returning the instruction exactly as you received it. Rewrite only when the instruction cannot be executed against the scene as written. There is no rewrite in this evidence that is worth anything: every measured improvement over the incoming instruction is indistinguishable from rollout noise, while the losses reach 100 percentage points. Identity is not merely safe here — it strictly dominates.

3. If you must rewrite, return roughly the register and the length you were handed. Do not turn a plain spoken instruction into formal, ceremonial, hedged, narrated or padded prose. This is the only cost beyond the nouns that is measurable at all, and it does not decompose into any single named feature — so the instruction is about the overall move away from plain register, not about any one word you might add. It is NOT a licence to compress: shortening below the instruction's own length has no measured penalty and no measured benefit.

4. Once every content noun is verbatim and the register is unchanged, the rest of the sentence is yours. Change the verb, the word order, the voice, the mood; make it a question; add "please"; add a comma or a subordinate clause; attach a colour, material or size word to a noun you have kept; use vocabulary the policy never saw. Every one of these is measured as a null on this population. Do not invent a prohibition against any of them, and in particular do not force the sentence to start with its verb.

===RATIONALE===

## Scope, estimator, and what the geometry allows

Evidence: `results/analysis/pi05_bank/distill_evidence_indist/phrases.csv` — the 10 `libero_goal`
tasks whose canonical instruction *is* one of the 40 strings the pi0.5 policy was finetuned on.
468 measured phrasings, 11,260 real rollout episodes, mean canonical success 96.8% (per-task
88.35–100.0). Excluding the 10 `original` rows (the canonical string itself, delta-zero by
construction, an anchor that only shrinks contrasts toward zero) leaves **458 phrasings / 10,660
episodes**; the unsearched subset (`natural` + `adversarial`) is **299 phrasings / 4,990 episodes**.

Every number below is my own recomputation with `.venv/bin/python` + pandas from an independently
written feature extractor (content nouns = canonical tokens minus a function-word stoplist, minus a
verb lexicon, minus spatial words; `-s`/`-es`/`-ies` stemming so bowl/bowls match). It rebuilds the
same noun sets as the shipped lens work — `libero_goal:2` → {wine, bottle, cabinet}, `:7` has one
noun, `:0/:1/:4/:5/:8` have two — and reproduces the prevalence (68.6%, 321/468) and the dose table
(93.2 / 60.8 / 17.5 / 1.4%) to the digit.

**Estimator, held fixed everywhere.** Contrasts are computed inside each `(task x kind)` cell,
episode-weighted within the cell; cell deltas are averaged within a task weighted by cell episodes;
the statistic is the unweighted mean over tasks, with the **task as the clustering unit**. Cells
with only one arm are dropped and counted. Intervals are a 20,000-draw cluster bootstrap over tasks.
p-values are an **exact sign-flip permutation over the k task deltas** (2^k enumeration) — at k=10
the attainable floor is p=0.0020, at k=9 it is 0.0039, at k=7 it is 0.0156, at k=6 it is 0.031, at
k=5 it is 0.0625. Each headline also carries a **within-`(task x kind)` label-permutation floor**
(95th percentile of |null statistic|) and an approximate 80%-power minimum detectable effect
(2.9 x task SD / sqrt(k)). `oracle_board` rows were searched against these same rollouts and are
never headlined; every rule reports its unsearched replication.

**Why the output is prohibitions.** Canonical mean 96.8%. Every one of the ten tasks has an observed
best phrasing of exactly 100.0%, and a binomial null — each phrasing assigned exactly its own
canonical's success probability at its real episode count, 20,000 draws per task — puts the
*expected* best-of-m at 100.0% on all ten tasks, with P(null best >= observed best) >= 0.999
everywhere. The entire apparent headroom in `task_summary.csv` (0.0 to 11.7pp, 3 tasks at exactly
0.0) is max-over-40-draws noise. **No gain-shaped rule is establishable from this population, and
none is written.**

---

## Rule 1 — every content noun, verbatim

| specification | effect | k | cells | episodes | p (sign-flip) | perm floor | MDE |
|---|---|---|---|---|---|---|---|
| all kinds | **+26.3pp** [+18.1, +35.3] | 10 | 25 | 8,565 | **0.0020** (floor) | 6.6 | 13.5 |
| unsearched (nat+adv) | **+39.4pp** [+30.3, +48.1] | 10 | 16 | 3,990 | **0.0020** | 9.3 | 13.9 |
| adversarial only | +47.2pp [+39.8, +54.8] | 10 | 10 | 2,500 | 0.0020 | 12.3 | 11.8 |
| natural only | +18.8pp [+4.2, +34.5] | 6 | 6 | 1,490 | 0.094 | 11.4 | 25.0 |
| oracle_board (SEARCHED) | +8.3pp | 8 | 8 | 4,260 | 0.109 | — | 11.5 |
| **realistic length only (ratio < 2x)** | **+10.9pp** [+5.1, +17.7] | 8 | 15 | 5,810 | **0.0156** | 5.5 | 10.1 |
| ratio < 2x, single-noun task 7 dropped | +8.1pp [+3.8, +12.5] | 7 | 12 | 4,990 | 0.031 (floor) | 5.2 | 7.1 |

10 of 10 tasks positive; leave-one-task-out spans +23.4 to +28.3 with p=0.0039 in all ten refits.
Levels: **92.8%** on 311 noun-preserving rows / 7,830 episodes versus **50.2%** on 147 noun-changing
rows / 2,830 episodes; unsearched, 88.4% versus 36.3%. The catastrophe rate (a rewrite landing 40pp
or more below its own canonical) is **8.3% when every noun survives and 70.1% when one does not**;
task-clustered that difference is **+56.7pp** [+47.2, +66.8], k=10, p=0.0020.

**The honest magnitude is 8–11pp, not 26pp.** The 26pp figure is what a noun substitution costs when
it arrives inside a twenty-word circumlocution. Conditioned on the lengths a rephraser would actually
emit (< 2x the canonical), it is +10.9pp — barely above its own MDE of 10.1 — and +8.1pp once the
single-noun task is removed. Quote 8–11pp as the expected cost; quote 26–39pp only as what happens
when the substitution rides inside a rewritten sentence.

**Both slots, no ordering.** Splitting the noun set into the manipulated object (all nouns but the
last) and the destination/landmark (the last), on the 9 tasks with two or more nouns:

| contrast | effect | k | p | note |
|---|---|---|---|---|
| object slot kept | **+32.9pp** [+23.1, +43.5] | 9 | 0.0039 (floor) | 9/9 tasks, LOTO +29.0 to +35.4 |
| landmark slot kept | +28.6pp [+9.8, +48.9] | 9 | 0.0195 | 8/9, est **below** its MDE of 31.1 |
| object, given landmark kept | +32.0pp [+17.0, +49.1] | 9 | 0.0039 | 9/9 |
| landmark, given object kept | **+22.4pp** [+9.3, +41.2] | 7 | 0.0156 (floor) | 7/7 |
| *which slot is worse to lose* | +0.9pp [-31.4, +31.5] | 7 | **1.00** | MDE 51.5 — no test |

Joint cells, episode-weighted: both kept 92.3%, landmark lost 74.2%, object lost 55.5%, both lost
11.7%. The object slot is solid; the landmark slot is thinner (its mean is carried by tasks 6, 8, 9
at +57.5/+71.5/+78.5 against a median task delta of 15.3) but its *conditional* form — losing the
landmark while keeping the object — is 7/7 tasks positive at the k=7 exact floor. **Which slot is
more damaging cannot be established at k=10**: the direct paired contrast is +0.9pp with p=1.00
against an MDE of 51.5pp. The rule therefore protects both and asserts no ranking.

**Dose is monotone; the step sizes are not separable.** Episode-weighted success by number of
canonical nouns missing: 93.2% (321 rows, 8,430 eps, 10 tasks) / 60.8% (103, 2,160, 10) / 17.5%
(41, 600, 9) / 1.4% (3, 70, 2). Both adjacent steps resolve — 0-vs-1 = +16.1pp, p=0.0059, k=10,
floor 6.2; 1-vs-2+ = +38.6pp, p=0.0078, k=9, floor 16.8 — but the *comparison of the two steps* does
not (+25.6pp, p=0.109, 5 of 9 tasks positive, MDE 36.9). Report the monotone trend and stop.
Note also that 38 of the 41 two-or-more-missing rows are adversarial, so the second step's magnitude
is entangled with circumlocution style.

**The two carve-outs are earned, not decorative.** A token matcher condemns "stovetop" and
"cabinetry"; a substring matcher rescues them. Those 15 rescued rows average **89.1%** against
**43.9%** for true substitutions — within-cell +31.9pp, positive in 4 of 5 tasks and 9 of 10 cells,
but k=5 and p=0.125, which is *suggestive, not established*. The substring matcher also scores the
rule slightly better overall (+29.3pp vs +26.3pp), which is why compounds are carved out rather than
banned. Inflection is free by construction of the matcher.

**Where the rule is weakest.** Outside the adversarial tier it does not independently clear:
natural + cross_scene gives +20.3pp at p=0.094 (k=6), natural alone +18.8pp at p=0.094 (k=6).
The non-adversarial pool including the searched oracle rows does clear (+11.8pp, p=0.0156, k=8), but
that is partly a searched population. So four of the five tiers' worth of evidence rests on
adversarial circumlocutions, where "noun replaced" and "whole sentence distorted" arrive together.
The direction survives every cut; the tier-independence does not.

**Substitutes cannot be ranked, and no whitelist should be attempted.** The same substitute inverts
inside a single task and scene: on `libero_goal:5`, "Please slide the plate so it rests directly in
front of the stovetop" scores 90.0% while "Nudge the flat dinner plate to the area just before the
stovetop" scores 0.0% — same word, same scene, both preserving "plate". On `libero_goal:3` the four
adversarial "dish" rows score 100/20/0/0. The formal alternative (short one-word swaps are safer
than long circumlocutions) does not test at all: k=3 usable tasks, +25.6pp, p=0.50, against a 33.6pp
permutation floor. That is not a failed test; it is no test.

---

## Rule 2 — default to identity

This rule is decision-theoretic, and it is the one place where the ceiling geometry, not a contrast,
does the work.

| rewrite policy | mean delta vs canonical | k=10 | tasks better |
|---|---|---|---|
| any unsearched rewrite | **-27.8pp** [-40.3, -16.9] | 10 | **0 / 10** |
| noun-preserving rewrite | -9.5pp [-17.7, -3.0] | 10 | 1 / 10 |
| noun-preserving + plain register | -4.9pp [-12.6, +0.4] | 10 | 3 / 10 |
| noun-preserving + plain + within 1.5x length | -4.5pp [-12.1, +0.8] | 10 | 3 / 10 |

Even the best-behaved rewrite arm this rulebook can define is a wash at best and -12pp at worst, and
the one task that shows a clean "gain" (`libero_goal:8`, all 15 plain noun-preserving rows at 100.0%
against a 95.0% canonical, +5.0pp) is exactly what a 95%-canonical binomial produces at these n.
Against that, the downside is unbounded: 111 of 150 adversarial phrasings lose ground, 31 unsearched
phrasings score exactly 0.0% against canonicals of 88.4–98.4%, and 70.1% of noun-changing rewrites
lose 40pp or more.

The rule is stated as "rewrite only when the instruction cannot be executed as written" rather than
"never rewrite" because the rephraser cannot tell which population it is in, and out of finetune
there is one known pathological canonical worth about +95pp (see the in-vs-out section). Inside this
population that trigger never fires and identity always wins.

---

## Rule 3 — do not inflate the register

This is the only residual beyond the nouns that is measurable, and it is stated with its confound
attached: `natural` and `adversarial` are two prompt families, not two treatments, so the contrast
is a **tier contrast** and is labelled as one.

| contrast | effect | k | p | tasks positive | LOTO |
|---|---|---|---|---|---|
| plain minus ornate, raw | +36.7pp [+26.2, +48.8] | 10 | 0.0020 | 10/10 | — |
| plain minus ornate, **noun-kept only** | **+12.6pp** [+2.1, +26.4] | 10 | **0.037** | 8/10 | [+6.9, +15.3] |
| same, substring noun matcher | +14.2pp [+2.7, +28.2] | 10 | 0.037 | 8/10 | [+8.7, +17.0] |
| same, both arms <= 20 words | +9.6pp [+1.0, +18.7] | 9 | 0.094 | 6/9 | [+7.1, +12.0] |
| same, noun **and** spatial word kept | +7.4pp [+1.4, +13.8] | 7 | 0.125 | 5/7 | [+5.4, +8.9] |
| plain minus ornate, among noun-**dropping** rows | +45.6pp | 6 | 0.031 | 6/6 | — |

So roughly two thirds of the raw 36.7pp tier gap is noun mangling and roughly one third — 7 to 14pp
depending on how strictly "noun kept" is defined — is a genuine residual cost of the ornate register.
No single task carries it. It is not decomposable: on the 311 noun-preserving rows every named
feature is null (see rule 4), so the instruction has to be about the register as a bundle.

**Two things this rule deliberately does not say.**
*It does not say "be brief."* Among noun-preserving rows, length at or below 1.34x the canonical is
-1.4pp (p=0.35, k=10, MDE 3.9) — no penalty for being short, and no benefit either. Within the
ornate tier, splitting noun-preserving adversarial rows at each task's median length gives +4.1pp
(p=0.47, k=10, MDE 16.1, floor 8.7) under my operationalisation; a sibling analysis using a looser
noun-keeping rule reports +17.0pp at p=0.005 for the same comparison. A contrast whose size and
significance move that much with the definition of its control variable is not shippable, so
"do not compress" is **not** a rule here.
*It does not license expansion either.* Every noun-preserving adversarial phrase in this population
is already 2.0–4.5x the canonical length and every noun-preserving natural phrase is <= 2.0x, so
length and tier are collinear by construction and no cell mixes them. The rule is "return what you
were handed", which is executable from text alone and does not require the untestable claim.

---

## Rule 4 — the permission

Stated as a permission because a rephraser given only prohibitions invents its own, and the ones it
would invent measure as nothing. All contrasts below are on the **311 noun-preserving rows**
(7,830 episodes, 10 tasks, episode-weighted mean 92.8%, -3.5pp against canonical):

| feature | effect | k | cells | p | perm floor | MDE |
|---|---|---|---|---|---|---|
| canonical verb reused | +2.2pp [-1.6, +7.3] | 10 | 22 | 0.48 | 3.0 | 7.1 |
| canonical spatial word kept | +1.8pp [+0.6, +3.0] | 4 | 9 | 0.25 | 2.9 | 2.1 |
| politeness / hedging wrapper | +2.4pp [-2.3, +6.3] | 9 | 19 | 0.33 | 4.3 | 6.8 |
| question form | -1.7pp [-4.4, +0.4] | 7 | 12 | 0.34 | 7.2 | 4.0 |
| added colour/material/size descriptor | -0.4pp [-2.0, +0.9] | 10 | 21 | 0.58 | 2.7 | **2.3** |
| verb-initial word order | +0.2pp [-2.3, +3.2] | 10 | 19 | 0.91 | 2.8 | 4.3 |
| above-median novel-word load | -2.1pp [-5.1, +0.4] | 10 | 31 | 0.24 | 2.6 | 4.4 |
| subordinate clause | +1.7pp [-4.1, +8.4] | 9 | 21 | 0.65 | 5.2 | 9.8 |
| contains a comma | -7.1pp [-22.9, +4.2] | 9 | 13 | 0.48 | 9.2 | **22.0** |
| >= 20 words | +0.2pp [-14.6, +11.8] | 8 | 8 | 0.98 | 10.0 | **21.1** |

Nine of ten point estimates sit inside their own permutation floor. The bounds differ enormously,
though, and the rule text should be read with them: descriptors, spatial words, verb-initial order
and novel vocabulary are bounded tightly (roughly +/- 2 to 4pp); verb choice, politeness and
question form at about +/- 4 to 7pp; **commas, subordination and length are not bounded at all**
(MDE 10 to 22pp) because their per-task deltas are wildly dispersed and, for length, every usable
cell is adversarial — 52% of adversarial noun-preserving rows reach 20 words against 0% of natural,
oracle and cross-scene ones. Those three are *untested outside one generator's style*, not shown
inert; rule 3 is what covers them.

**Why the permission is load-bearing.** Every one of these features looks damaging if you compute it
without holding the nouns fixed, because terse rewrites in this corpus are disproportionately the
ones that keep the noun. Verb-initial word order reads -6.2pp (p=0.039, 9/10 tasks negative)
marginally and +0.2pp (p=0.91) once nouns are held fixed — a naive analysis would prohibit the
canonical's own sentence form. "Added content" reads +22.2pp (p=0.0020, 10/10) marginally and
+3.2pp (p=0.26, MDE 7.4) once nouns are held fixed. A rephraser that derived its own rules from
uncontrolled marginals here would end up forbidding exactly the phrasing the policy was trained on.

**Residual structure is real and unexplained.** 23 noun-preserving unsearched rows across 6 tasks
land 20pp or more below their canonical (`libero_goal:5` supplies 8, `:0` 6, `:6` 5), and the
within-task standard deviation among noun-preserving rows is 26.4pp on `libero_goal:0`, 24.3pp on
`:5` and 20.9pp on `:6`. Rule 4 is a bound on what these ten features explain, not a claim that
wording beyond the nouns is inert.

---

## What we tested and rejected

- **Ranking or whitelisting noun substitutes.** Rejected — the same substitute inverts within a
  single task and scene (rule 1). Any "safe synonym" list built from this population would be
  fitting rollout noise.
- **Ranking the object noun above the destination noun.** Rejected as untestable: +0.9pp, p=1.00,
  MDE 51.5pp. Folded into rule 1 as "both, with no ordering".
- **"The second substitution is the worse step."** Rejected: +25.6pp, p=0.109, 5 of 9 tasks
  positive, MDE 36.9pp. The monotone trend survives; the comparison of steps does not.
- **"Compounds are safe."** Not encoded as a positive claim. The 15 substring-rescued rows do
  average 89.1% vs 43.9%, but k=5, p=0.125, and the class contains two hard collapses
  ("Nudge the flat dinner plate ... stovetop" 0.0%; a "cabinetry" row at 10.0%). It is carved out of
  the prohibition, not blessed.
- **"Do not compress" / "brevity is not protective."** Rejected — definition-unstable (+4.1pp
  p=0.47 under my noun matcher, +17.0pp p=0.005 under a looser one), and every usable cell is inside
  the ornate tier where all rows are already >= 2x canonical length. Rule 3 says "keep the register
  you were handed" instead.
- **A hedging-preamble prohibition.** Rejected. Under my detector, hedged noun-preserving rows are
  5 phrases / 70 episodes / 4 tasks scoring 100, 100, 100, 100, 30 percent; the contrast is -11.6pp
  at p=0.31 with 2 of 4 tasks positive, and dropping `libero_goal:7` flips the sign to +1.2. On all
  adversarial rows it is -5.5pp, p=0.52, 4 of 9 tasks positive. This is a one-task effect.
- **A colour rule, in either direction.** Untestable, not disproven. No canonical here contains a
  colour word, so only *adding* a colour is ever tested — and of the 69 colour-bearing rows, 59 are
  searched `oracle_board` and 6 more are the single `cross_scene` task, leaving **4 rows** in the
  unsearched pool. The tight null on "added descriptor" (-0.4pp, MDE 2.3) means adding one is free;
  nothing here speaks to contradicting one.
- **A spatial-modifier prohibition as its own rule.** Rejected as a *finding*: within noun-preserving
  rows the contrast is +1.8pp on 4 tasks (p=0.25, MDE 2.1); a sibling recompute gets -0.7pp on the
  same 4 tasks (p=0.875). Only one of those four canonicals ("the middle drawer" among three
  drawers) has a modifier that actually discriminates between objects in the scene, so the case a
  rephraser would care about is untested. Carried inside rule 1 as a **prior at zero measured
  cost**, explicitly not as evidence.
- **Chasing `task_summary.csv` headroom.** Rejected — P(binomial null best >= observed best) >= 0.999
  on all ten tasks. There is no measurable gain to chase.
- **Any band- or difficulty-conditional rule.** Not executable: the rephraser sees text only, and the
  next section shows that identical text scores 100.0% and 2.0% in two different scenes.

---

## THE IN-VS-OUT CONTRAST

Side by side with `ood_only_v1.md` (55 out-of-finetune `libero_90` tasks, 2,119 phrasings, 36,490
episodes, mean canonical 40.3%, bands: 27 floor <20%, 12 mid 20–70%, 16 high >70%). **Everything in
this section is recomputed by me with the same featuriser and the same estimator on both files**, so
the columns are comparable line by line. Unsearched rows only (`natural` + `adversarial`) on both
sides; `original` excluded on both sides.

The right comparison for the in-finetune column is the **OOD high band**, because those 16 tasks
match this population on the only axis that matters for dynamic range — a canonical already above
70%. The pooled OOD column is shown too, because that is what a deployed rephraser actually meets.

### The headline table

| feature (identical code, both populations) | IN-FINETUNE k=10, canon 96.8% | OOD HIGH k=16, canon >70% | OOD POOLED k=55, canon 40.3% |
|---|---|---|---|
| keep every content noun | **+39.4** [+30.3,+48.1] p=0.0020, **10/10** | **+10.0** [+3.1,+18.5] p=0.0051, 11/16 | +2.8 [+0.4,+5.7] p=0.039, **19/55** |
| — same, ornate tier only | +47.2 p=0.0020, 10/10 | +11.3 p=0.0068, 10/16 | +3.8 p=0.018, 18/54 |
| — same, **plain tier only** | **+18.8** p=0.094, 5/6 | **+1.0** p=0.56, 4/12 | **-0.7** p=0.32, 8/43 |
| object noun, given landmark kept | **+32.0** p=0.0039, 9/9 | +7.7 p=0.022, 8/15 | +2.5 p=0.082, 15/47 |
| **landmark noun, given object kept** | **+22.4** p=0.0156, **7/7** | +2.4 p=0.375, k=4 (no power) | **-0.2** p=0.88, 6/19 |
| extra catastrophe rate if a noun is dropped | **+56.7pp** (8.3% → 70.1%) | +14.9pp (5.8% → 11.9%) | +4.9pp (2.5% → 3.9%) |
| dose 0 / 1 / 2+ nouns missing | **88.4 / 47.6 / 16.3%** | 89.8 / 82.3 / 84.7% | 42.7 / 41.5 / 42.0% |
| plain minus ornate, raw | **+36.7** p=0.0020, 10/10 | +8.0 p=0.054, 9/16 | +2.7 p=0.079, 20/55 |
| plain minus ornate, **noun-kept** | **+12.6** p=0.037, 8/10 | +4.3 p=0.30, 8/16 | +1.2 p=0.45, 17/54 |
| below-median added content, all rows | **+22.2** p=0.0020, 10/10 | +5.8 p=0.011, 11/16 | +1.5 p=0.058, 22/55 |
| below-median added content, noun-kept | +3.2 p=0.26, 6/9 | +4.3 p=0.12, 7/15 | +0.6 p=0.57, 13/49 |
| reuse the canonical verb (noun-kept) | +2.2 p=0.48 | +5.9 p=0.044, 8/15 | +0.5 p=0.83 |
| verb-initial order (noun-kept) | +0.2 p=0.91 | -1.7 p=0.42 | -1.2 p=0.21 |
| politeness (noun-kept) | +2.4 p=0.33 | +2.5 p=0.31 | +0.9 p=0.44 |
| any unsearched rewrite vs canonical | **-27.8** [-40.3,-16.9], **0/10 better** | -6.9 [-14.8,-0.8], 2/16 better | -0.6, median **0.0**, 10 better / 23 worse / 22 tied |
| best constrained rewrite arm vs canonical | -4.9 [-12.6,+0.4] | -2.8 [-6.8,+0.8] | +0.5 [-3.1,+5.1] |

### Rule by rule

**Rule 1 (content nouns) — SHARED IN DIRECTION, IN-SPECIFIC IN MAGNITUDE, AND ITS DESTINATION HALF
IS IN-SPECIFIC OUTRIGHT.** This is the one prohibition both rulebooks reach independently, and it is
the only one that transfers. But it is a different size and a different shape in the two
populations:
- *Magnitude:* +39.4pp in-finetune against +10.0pp on the matched OOD high band and +2.8pp pooled —
  roughly **4x** even after matching on canonical band, and **14x** against the population a
  deployed rephraser meets. The honest realistic-length in-finetune figure (+8–11pp) is still at or
  above the *entire* OOD high-band effect.
- *Universality:* 10 of 10 tasks positive in-finetune; 11 of 16 out-of-finetune high band; 19 of 55
  pooled, with 45 of 97 cells degenerate and the **mid band running the wrong way** (-1.3pp,
  p=0.52), which the `ood_only_v1` audit also found (-7.38pp, p=0.021, on the plain tier).
- *Register-independence:* the sharpest single difference. In the plain register — the only register
  a real rephraser writes in — the noun rule is worth **+18.8pp in-finetune** and **+1.0pp
  (p=0.56) out-of-finetune on high-band tasks**, **-0.7pp (p=0.32) pooled**. Out of finetune the
  rule's entire measured payload lives inside the ornate family; in finetune it survives, at
  two-thirds strength, in plain English too. `ood_only_v1`'s own audit reaches the same verdict from
  the other side ("aggressive renaming is costly ... this evidence never observes a mild rename
  inside an ornate sentence").
- *Shape:* the dose response is the tell. In-finetune it is a cliff — 88.4 / 47.6 / 16.3% for zero,
  one, two-or-more nouns missing. On the OOD high band it is 89.8 / 82.3 / 84.7% — a ~7pp step then
  flat, i.e. *not monotone at all*. Pooled OOD it is flat (42.7 / 41.5 / 42.0), and on the OOD floor
  band dropping a noun is **better** (nmiss=1 delta +20.7pp vs +4.1pp) because that band contains
  the pathological canonical. Rephrasing a memorised string is a cliff; rephrasing an unmemorised one
  is a gentle slope on the tasks that work and noise everywhere else.
- *Tail:* the payload is catastrophe insurance in both, but at wildly different premiums —
  dropping a noun raises the catastrophe rate by **+56.7pp** in-finetune, +14.9pp on the OOD high
  band, +4.9pp pooled, and **+0.0pp** on the 27 OOD floor tasks.
- *The destination noun is the piece that is genuinely in-finetune-only.* Holding the object noun
  fixed, keeping the landmark noun is worth **+22.4pp, 7 of 7 tasks, p=0.0156** in-finetune and
  **-0.2pp, p=0.88, k=19** out-of-finetune pooled (+2.4pp, k=4, no power, on the high band). The
  `ood_only_v1` audit independently concluded that "the place it goes ... is not supported for the
  second half" of its own rule 1. Both files now agree: **protecting the destination noun is a
  statement about training language, not about language.** `in_plus_ood_v1.md`'s rule 1 asserted it
  for both populations and was wrong for one of them.

**Rule 2 (default to identity) — DIRECTIONALLY SHARED, BUT FOR OPPOSITE REASONS, AND ONLY HERE IS IT
STRICT.** In-finetune, 0 of 10 tasks improve under any rewrite and the pooled cost is -27.8pp;
identity strictly dominates because the gain side is provably empty (P(null best >= observed) >=
0.999 on all ten tasks). Out of finetune the same policy is a coin flip — -0.6pp, median exactly
0.0, 10 tasks better / 23 worse / 22 exactly tied — and `ood_only_v1` correctly *permits* rather
than requires the identity, because one task in 55 (`libero_90:44`) has a pathological canonical
worth about +95pp. That task is the load-bearing exception and it is invisible from text: the
identical string "turn on the stove" scores **100.0% over 165 episodes in `libero_goal:7`** and
**2.0% over 150 episodes in `libero_90:44`**, and across the 41 phrasings measured in both scenes,
`libero_90:44` is better on 22, worse on 3 and tied on 16 (66.6% vs 95.4% mean). The rewrite effect
does not merely shrink out of finetune — **it inverts**. Rule 2 is therefore written as "default to
identity, rewrite only when the instruction cannot be executed as written": strict in this
population, and leaving the one door open that the other population needs.
The rulebook this most directly refutes is `in_plus_ood_v1.md` rule 4, "never emit the instruction
back unchanged". On this population that instruction is worth -27.8pp.

**Rule 3 (do not inflate the register) — LARGELY IN-SPECIFIC.** The raw plain-minus-ornate gap is
+36.7pp in-finetune, +8.0pp on the OOD high band, +2.7pp pooled. Conditioning on noun preservation
leaves a residual of **+12.6pp (p=0.037, 8/10 tasks)** in-finetune, **+4.3pp (p=0.30)** on the OOD
high band and **+1.2pp (p=0.45)** pooled — i.e. the residual register cost is about **3x** larger
in-finetune and it is the only place it clears. `ood_only_v1` keeps its register rule anyway (its
rule 2), on a permutation null and a sealed-20 win rate rather than a clustered p-value, and openly
notes it dies after conditioning on nouns. **Both books are right about their own population:** out
of finetune the register rule is a weak population-average tendency; in finetune it is a real,
LOTO-robust residual on 8 of 10 tasks. Neither book should be read as showing that ornate prose is
intrinsically harmful — in both it is mostly the noun mangling that ornate prose brings with it.

**`ood_only_v1` rule 3 ("say only what the instruction already said") — DOES NOT TRANSFER INWARD AS
A SEPARATE RULE.** Uncontrolled, "added content" looks enormous in-finetune (+22.2pp, p=0.0020,
10/10). Conditioned on noun preservation it is +3.2pp, p=0.26, MDE 7.4 — inside the noise. On the
OOD high band the same conditioning leaves +4.3pp, p=0.12 (the OOD book reports +2.39 with a
cell-level interval that excludes zero; under a task-clustered sign-flip it does not clear for me
either). So in-finetune, "added content" was **noun replacement wearing a descriptor costume**, and
it is folded into rule 1 and rule 3 rather than shipped separately. Concretely, in-finetune the
descriptor half of that OOD rule is contradicted: attaching a colour, material or size word to a
*retained* noun is -0.4pp with an MDE of 2.3pp — one of the tightest nulls in the file, and the
same direction the OOD book reports for retained nouns (+1.39 there).

**`ood_only_v1` rules 4 and 5 (scene never substitutes; the style licence) — TRANSFER UNCHANGED.**
The style licence is the cleanest cross-population agreement in the exercise: verb-initial order,
politeness, question form, commas, clause count and novel vocabulary are null in *both* populations
once nouns are held fixed, with no sign agreement problems and no feature clearing its floor
anywhere (in-finetune verb-initial +0.2 p=0.91 vs OOD -1.2 p=0.21; politeness +2.4 p=0.33 vs +0.9
p=0.44). Both books also independently warn against re-deriving a verb-initial prohibition from
uncontrolled marginals, which reads -6.2pp (p=0.039) here and -1.0pp there. The scene rule survives
inward by construction: rule 1's prohibition on category words and definite descriptions is exactly
"do not swap the instruction's noun for whatever the scene calls it instead".
One genuine cross-population disagreement worth recording rather than resolving: **reusing the
canonical verb** is +2.2pp (p=0.48, MDE 7.1) in-finetune and +5.9pp (p=0.044, 8/15 tasks) on the OOD
high band. Both books reject a verb rule; against roughly 14 features screened per population, one
nominal hit is chance, and I do not promote it.

**Net.** Of the four rules here, **one transfers with its direction intact and a quarter of its
magnitude** (content nouns for the manipulated object), **one transfers as a permission**
(the style licence), **one is in-finetune-specific in strength** (register), and **two specific
claims are in-finetune-only outright**: protecting the *destination* noun, and treating identity as
strictly dominant rather than merely permitted. Going the other way, the single most valuable move
in the OOD population — rewriting `libero_90:44`'s pathological canonical for +95pp — has no
analogue here at all, and its trigger is not visible in text.

**The one-sentence version.** When the instruction is a string the policy memorised, the nouns are a
retrieval key and changing any of them — including the one naming the destination — falls off a
cliff; when it is not, the nouns are just a description, and only genuinely worse descriptions on
tasks the policy can already do cost anything. A rephraser that cannot tell the two apart should
obey the in-finetune rule, because it costs nothing in the population where it does not apply.

---

## Limits

- **k=10 tasks, all `libero_goal`, one suite, one embodiment, one policy.** Every interval here is
  a 10-cluster interval. Nothing in this file has been shown to transfer off `libero_goal`, and the
  in-vs-out section is the only evidence offered that any of it does.
- **The noise floor is high and should be quoted with every claim.** Random binary features drawn at
  a fixed prevalence on the real cell/episode structure (1,500 simulations each) give, on the
  all-kinds grid: 95th-percentile |null| of 13.1 / 8.4 / 6.0 / 4.9 / 3.9pp at 5 / 10 / 20 / 30 / 50%
  prevalence, with 80%-power MDEs of 18.9 / 12.6 / 8.7 / 6.9 / 5.8pp. On the unsearched grid the
  floors are about 1.5x larger (8.9pp floor, 13.2pp MDE at 20% prevalence). **A feature violated by fewer than about one row in ten needs a >18pp per-task
  effect before it is distinguishable from a coin flip.** Nothing subtle is findable here, and the
  correct reading of every null in rule 4 is "not resolvable", not "absent".
- **Ceiling compression makes half the population uninformative.** Three tasks sit at a 100.0%
  canonical, and in the noun-preserving pool `libero_goal:1`, `:2`, `:7` and `:8` have standard
  deviations of 2.1–3.5pp — those cells can return nothing but zero, and the task-mean estimator
  gives them equal weight, which mechanically shrinks every permission contrast toward zero.
  Rules 3 and 4 are really estimated on 5–6 tasks.
- **Tier and style are confounded by construction and cannot be fully unpicked at this n.** Length,
  commas and subordination exist only inside the adversarial tier among noun-preserving rows
  (prevalence of >= 20 words: adversarial 52%, natural 0%, oracle 0%, cross-scene 0%). Rule 3 is a
  tier contrast and is labelled as one; rule 4's bounds on those three features are +/- 10 to 22pp,
  not the +/- 2 to 4pp that its tighter features carry.
- **`cross_scene` is one task.** All 21 rows are `libero_goal:7`. The tier mean of -33.3pp quoted in
  the evidence brief is one task's number and must never be cited as a tier effect; it is used here
  only for the paired two-scene comparison in the in-vs-out section, which is k=1 canonical string.
- **`oracle_board` rows were searched against these same rollouts.** They are 138 of 468 rows and
  5,355 of 11,260 episodes — the largest single tier by episode count — and 38 of them are
  re-measurements of strings already measured under another tier label. No rule here is headlined on
  them; every rule reports an unsearched replication, and rule 1 is stronger, not weaker, when they
  are removed. Relatedly, only **419 of the 468 rows are distinct strings** (45 `(task, phrase)`
  pairs are measured twice under two kind labels), so arm *means* are mildly pseudo-replicated even
  though the within-cell contrasts are not.
- **Colour is untestable, not disproven.** No canonical here contains a colour word.
- **Deletion of a noun is not separately tested.** The population's noun-changing rows are almost all
  substitutions; pure deletions are too few to contrast, so rule 1's "never drop one" clause is a
  prior carried along with the substitution evidence.
- **What cannot be established at k=10, stated as results:** which of the two noun slots is more
  damaging to lose (p=1.00, MDE 51.5pp); whether the second substitution is worse than the first
  (p=0.109, 5/9); whether short one-word swaps are safer than long circumlocutions (k=3, p=0.50,
  floor 33.6pp); whether a spatial modifier that actually discriminates between objects in the scene
  matters (one task has one); whether length, commas or subordination cost anything outside the
  ornate tier (no non-adversarial cell exists); whether hedging costs anything (one task); and any
  gain-shaped rule whatsoever (P(null best >= observed) >= 0.999 on all ten tasks). Each of those is
  a finding about this design's resolution, and each is a reason not to write the rule.
- **What would actually settle the open questions:** more tasks, not more phrasings — every limit
  above is a k=10 limit, and per-phrasing n (10–105 episodes, binomial SE 5–16pp) is not the binding
  constraint on any headline in this file except the thin-cell contrasts.

---

*Reproducibility: analysis scripts (feature extractor, estimator, five contrast runs) were written
for this distillation and every number above was recomputed from the two `phrases.csv` files and,
for the two-scene comparison only, `results/analysis/pi05_bank/bank.parquet`. Nothing new was
written under `results/` except this file.*
