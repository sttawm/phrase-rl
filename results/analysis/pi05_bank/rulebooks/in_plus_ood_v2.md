===RULES===
1. Carry every noun that names a physical object or a container through into your rewrite verbatim — bowl, plate, drawer, cabinet, stove, mug, bottle, book, caddy, basket, tray, rack, pot, pan, box, shelf, compartment, microwave, and named foods such as ketchup or cream cheese — and never replace one with a synonym, a category word, or a description ("hot plate", "vessel", "the thing you drink from").
2. Whenever the instruction says top, middle or bottom, carry that word through verbatim and do not paraphrase it, because it names which drawer or which shelf.
3. Build the rewrite out of the instruction's own words: introduce at most six content words that were not in the incoming instruction, and prefer none to two.
4. Never shorten, tidy or smooth a sentence at the cost of a word that rules 1-3 protect — length is free, so let the rewrite run as long as keeping those words requires.
5. Write the sentence in whatever form reads best — polite or blunt, question or command, one clause or two, verb-first or not — because sentence form carries no measured cost once rules 1-3 hold.
6. Keep the instruction's own action verb (put, pick up, place, open, close, push, turn on) unless the rewrite genuinely reads better without it.
7. You may add one colour, material or position attribute that the scene description actually asserts of the object ("the black bowl", "the wooden cabinet"), but only in front of a noun you have already kept and never in place of it.

===RATIONALE===

All numbers are my own recomputation from `results/analysis/pi05_bank/distill_evidence/phrases.csv`
(2,587 measured phrasings, 65 tasks, 47,750 real rollout episodes) with
`/Users/sttawm/dev/robotics/phrase-rl/.venv/bin/python` + pandas. The analysis base drops the 65
`kind=original` rows, the 21 `cross_scene` rows and every phrase byte-identical to its canonical:
**2,471 rewrites, 65 tasks, 42,990 episodes**.

Two contrast forms are quoted throughout:

- **task-paired** — episode-weighted mean success among rule-obeying phrases minus rule-violating
  phrases *inside one task*, tasks then averaged with equal weight. Cells need >=10 episodes per arm.
- **tier-controlled** — the same contrast computed inside each (task x generation tier) cell before
  averaging to the task. This is the honest test: the README warns that natural / adversarial /
  oracle_board came from one prompt family, so length, novel vocabulary, clause structure and
  noun-dropping co-occur by construction, and any pooled contrast partly measures "which prompt
  wrote this row".

Brackets are 2.5/97.5 percentiles of a 4,000-draw bootstrap over tasks. Noun matching is
inflection-tolerant and uses a corpus-derived head-noun extractor
(`results/analysis/pi05_bank/audit_noun_lexical/lexicon.py`); a hand-written 30-noun list reproduces
every headline within 0.5pp. Sign tests count only tasks whose two arms actually differ — a third of
tasks sit at a hard 0% or 100% and contribute exact zeros.

---

**Rule 1 — carry the object and container nouns through verbatim.**
This is the rulebook. It is the only rule that survives every control I could apply, and it accounts
for most of what the whole book is worth.

- Task-paired, all 65 tasks: **+11.15pp**, 33 tasks positive / 10 negative / 22 tied,
  CI [+6.46, +16.29], sign p = 0.0006, 27,575 vs 15,415 episodes.
- On the 10 tasks whose instruction is in the policy's finetune set: **+43.76pp, 10 of 10 tasks
  positive**, CI [+35.27, +51.78], 9,445 episodes.
- On the 55 out-of-finetune tasks: **+5.22pp**, 23+/10-, CI [+1.64, +9.37], sign p = 0.035.
- Tier-controlled: all tasks **+8.07pp** CI [+4.28, +12.16]; out-of-finetune **+3.93pp**
  CI [+0.85, +7.69]. Leave-one-task-out on the out-of-finetune arm moves it only between
  +2.68 and +4.26 — no single task carries it.
- It is not oracle-search selection: the oracle_board rows were searched *upward* for winners and
  their noun-swapping rows still lose (+1.23pp), and the adversarial tier alone gives +4.61pp
  CI [+1.23, +8.51] out-of-finetune.

What it protects against is a bounded-downside / unbounded-upside asymmetry, which is the real
argument for applying it unconditionally. Dropping or renaming a noun raises the rate of
catastrophic collapse (>=30pp below the canonical) by **+15.10pp** overall and **+51.21pp** on
finetuned instructions (worse on 10 of 10 such tasks, better on none); on in-finetune tasks a
noun swap collapses the rollout 49.7% of the time and gains >=20pp exactly 0% of the time across
146 rewrites. Meanwhile obeying it costs essentially nothing: in the natural tier out-of-finetune
the extra-collapse figure is **-1.28pp**, i.e. a wash.

Concrete cases, each n>=15 episodes: `put the bowl on the stove` (100%) -> "place the grey bowl on
the **hot plate**" = 0%. `turn on the stove` (100%) -> "power on the **hot plate**" = 0% and
"switch on the **hot plate**" = 0%. `push the plate to the front of the stove` (98%) -> "propelled
the flat dining **vessel** toward the anterior region of the cooking **appliance**" = 0%.

Honest caveat: out-of-finetune the *average* gain is small. An episode-weighted WLS with task x kind
fixed effects and task-clustered standard errors gives keep-nouns = +5.04pp (t = 2.74) overall,
+18.60pp (t = 4.62) in-finetune, but only +0.97pp (t = 0.86) out-of-finetune. Treat this as cheap
insurance against a ~50pp tail, not as an expected gain on every instruction. Containers carry it
more than food words (keep-all-containers +12.39pp, 62 tasks), so if two protected words ever
conflict, protect the container.

**Rule 2 — carry top / middle / bottom through verbatim.**
20 tasks have an ordinal in the canonical (823 rewrites, 14,940 episodes).

- Task-paired: **+9.54pp**, 10+/3-/7 tied, CI [+3.18, +18.00].
- Conditional on rule 1 already being obeyed (so this is not rule 1 in disguise): **+6.53pp** all
  tasks (10+/3-), CI [+1.20, +13.78]; **+6.25pp** out-of-finetune (7+/2-), CI [+0.61, +14.57].
- Tier-controlled and noun-conditioned it weakens to +3.20pp CI [-0.09, +7.60] (out-of-finetune
  +3.96pp CI [-0.10, +9.14]) — positive but at the significance boundary.
- Leave-one-task-out never falls below **+3.64pp** (the largest single contributor, libero_90:10,
  is worth +61.5pp on its own but removing it leaves the rule intact).
- Per word: `top` +10.00pp (14 tasks, 7+/2-), `middle` +10.47pp (5 tasks, 3+/0-), `bottom` +0.41pp
  (3 tasks — too thin to judge).

This is scoped deliberately narrowly. The lateral words are a different story and are **not** in the
rule: keeping left/right/front/back is +2.64pp unconditioned (21 tasks, 4+/7-) and **-1.22pp** once
nouns are preserved (19 tasks, 3+/6-), with `right` at -1.90pp and `back` at -3.14pp. Sign test on
rule 2 is p = 0.09, so it is directional rather than decisive; it earns its slot because it is
never negative in any slice, it is free to obey, and the physical failure it prevents (top drawer
vs bottom drawer) is a wrong-target error rather than a wording error.

**Rule 3 — build the rewrite from the instruction's own words.**
Measured on the noun-preserving subset only, so this is an independent channel from rule 1.

- Task-paired, `<=6 new content words`: **+5.48pp** all tasks (27+/14-, CI [+1.97, +9.77],
  22,605 vs 4,490 episodes); out-of-finetune +3.19pp CI [+0.09, +7.17]; in-finetune +18.96pp
  CI [+7.56, +32.81].
- Tier-controlled: +3.59pp CI [+0.84, +6.54] all tasks; +2.88pp CI [+0.28, +5.95] out-of-finetune.
- As a policy the threshold is a plateau, not a cliff. Adding it on top of rules 1-2 and scoring the
  resulting task-mean delta against passing the instruction through unchanged: `<=2` +1.10,
  `<=3` +1.29, `<=4` +1.37, `<=5` +1.36, **`<=6` +1.40**, `<=7` +1.24, `<=8` +1.10, `<=9` +1.03,
  no cap +0.92. Anything from 2 to 6 buys about the same +0.5pp over rules 1-2 alone; the value
  decays smoothly above 7.

This is the weakest of the three constraint rules and I flag it as such: the paired contrast is a
clean spike at exactly 6 with thresholds 5 and 7 individually null, and the continuous version in a
task x kind FE regression on noun-preserving rows is not significant (new content words -0.42pp/word,
t = -1.06 overall; -1.47, t = -2.05 in-finetune; -0.23, t = -0.52 out-of-finetune). It stays in the
book because the policy sweep is a genuine plateau, because it is the only defence against the
25-new-word rewrite (the single worst-performing form in the bank), and because it costs almost
nothing — the best-found phrasing survives the cap on 21 of the 45 live tasks versus 22 under rule 1
alone.

**Rule 4 — length is free; never trim a protected word.**
This exists to stop the applier from mis-serving rules 1-3 by being terse, which is the natural
failure mode of a rulebook whose first three entries are preservation constraints.

- A length cap of the kind the bank's earlier analyses favoured (`words <= max(canonical+3, 13)`)
  looks worth +4.19pp task-paired on noun-preserving rows, but **tier-controlled it is -0.83pp,
  CI [-3.58, +1.81]** — a null with the sign pointing the wrong way. "At most twice the canonical's
  length" behaves the same: +5.18pp task-paired, +0.55pp CI [-2.20, +3.02] tier-controlled.
- In the FE regression, raw word count is *positively* signed everywhere: +0.44pp/word (t = 2.15) on
  noun-preserving rows, +0.86pp/word (t = 3.57) in the full model, +0.48pp/word out-of-finetune.
- Adding the length cap on top of rules 1-3 in the policy simulation changes the task-mean from
  +1.40 to +1.39 (no gain) while dropping the tasks that retain a >=10pp winner from 21 to 20 and
  the best-achievable task-mean delta from +13.31 to +12.42.
- The best-found phrasing is **longer** than the canonical on 31 of the 45 live tasks and shorter on
  5 (mean 13.3 words vs 9.2). Compliant long winners exist and are ordinary-looking, e.g.
  libero_90:64 48% -> 80% from a 20-word rewrite, libero_goal:6 98% -> 100% from a 19-word one.

**Rule 5 — sentence form is free.**
Five independent form features, each measured on noun-preserving rows, tier-controlled, all null:

| feature | tier-controlled effect | tasks | rows / episodes |
|---|---|---|---|
| politeness frame absent | +0.64pp CI [-0.73, +2.13] | 63 | 237 polite rows, 4,220 eps |
| not a question | +1.43pp CI [-1.19, +3.97] | 38 | 68 rows, 1,500 eps |
| no subordinate clause | -0.22pp CI [-2.46, +2.03] | 59 | 126 rows, 1,665 eps |
| verb-initial | -0.46pp CI [-1.62, +0.68] | 64 | 783 rows, 15,345 eps |
| both steps kept as separate verbs | +0.07pp CI [-2.68, +2.50] | 21 | 25 two-step tasks |

Verb-initial deserves a note because it looked like a real rule before the control: task-paired it is
+2.28pp overall and +7.66pp in-finetune (8+/2-), but tier-controlled it goes slightly negative
(17+/21-). It was a proxy for "resembles the trained string", which is rule 1. The clause-count
result is the surprising one: on the 25 two-step canonicals, **79% of rewrites collapse the two
imperatives into one clause** (92.5% in the adversarial tier), and it costs nothing measurable — so
the applier does not need to preserve clause structure, only the words.

**Rule 6 — prefer the instruction's own action verb.**
Task-paired, all tasks: **+2.95pp**, 29+/11-/16 tied, CI [+0.87, +5.34], sign p = 0.0064, 6,395 vs
20,185 episodes. Out-of-finetune +2.06pp CI [-0.10, +4.36]; in-finetune +7.07pp (9+/1-).
Tier-controlled it softens to +1.31pp CI [-0.78, +3.41], which is why the rule is worded as a
preference rather than a prohibition. Its sign test is actually the strongest of any secondary rule
(40 non-tied tasks, 29 positive), and it is free to obey. Stated as the mirror contrast, substituting
the canonical verb costs -3.81pp over 60 tasks (14+/28-, CI [-6.65, -1.13]) and -2.95pp on
noun-preserving rows (11+/29-, CI [-5.34, -0.87]).

One honest correction to an argument made elsewhere in this project: I do **not** reproduce verb
substitution as a clean placebo for rule 1. In my recompute it is a real cost, just a smaller one —
about a quarter the size of the noun effect (+2.95pp versus +11.15pp task-paired, and +7.07pp versus
+43.76pp on finetuned instructions). So the correct reading is not "only nouns matter" but "nouns
matter roughly four times as much as verbs, and everything else measures null." Rules 1 and 6 are
ordered accordingly.

**Rule 7 — you may add a visible scene attribute in front of a kept noun.**
Task-paired on noun-preserving rows: **+3.17pp**, 20+/10-/3 tied, CI [+0.66, +5.85], 5,845 vs
16,215 episodes across 33 tasks; out-of-finetune +3.31pp CI [+0.58, +6.75]. Tier-controlled it is
+0.91pp CI [-1.40, +3.37] overall and +2.29pp CI [-0.54, +5.66] out-of-finetune — positive in every
slice I ran and negative in none, but never significant once tier is held fixed. On the 39 tasks
whose canonical contains a spatial disambiguator (the "several like objects" case) it is +4.64pp
task-paired, out-of-finetune +5.42pp, but only 7+/5- on sign — suggestive, not established. It is
therefore written as a permission, and the guard clause is doing the real work: attribute-adding
rewrites drop a noun 45% of the time in this bank (232 of 517), and that is rule 1's cost, not
rule 7's benefit.

---

**What the whole book is worth**

Compliance with rules 1-3 as a single flag: 1,109 of 2,471 rewrites (45%) comply — 71% of natural
rewrites, 54% of oracle_board, only 10% of adversarial ones.

- Task-paired: **+7.49pp**, 29+/14-/22 tied, CI [+4.08, +11.38], 20,100 vs 22,890 episodes.
- In-finetune **+29.31pp, 10 of 10 tasks**; out-of-finetune **+3.52pp** CI [+0.86, +6.84].
- Tier-controlled: **+5.05pp** CI [+2.56, +7.97] overall, **+3.03pp** CI [+0.91, +5.52]
  out-of-finetune. It replicates inside all three tiers separately: natural +4.29pp (49 tasks),
  adversarial +8.81pp (42 tasks, 24+/2-), oracle_board +5.13pp (28 tasks).

Against passing the instruction through unchanged (delta = 0), scored per generation tier:

| rephraser style | unfiltered rewriting | rules 1-3 applied |
|---|---|---|
| natural | -0.81pp (in-finetune -9.42) | **+0.23pp** (in-finetune -4.49) |
| adversarial / verbose | -5.12pp (in-finetune -36.08) | **+2.70pp** (in-finetune -1.67) |
| oracle_board (searched) | +4.04pp | +4.42pp |

Read that honestly: **blind rephrasing is net-negative, and this rulebook's job is mostly damage
control.** Against a well-behaved rephraser it buys roughly +1pp; against a verbose one it buys
~8pp and turns a losing policy into a winning one. Almost all of the value is in not wrecking the
instructions the policy was trained on. The headroom cost is small — 21 of 45 live tasks still
retain a >=10pp winner under rules 1-3, versus 23 with no rules at all, and the best-achievable
task-mean delta falls only from +16.07 to +13.31.

---

**What we tested and rejected**

- *Gate on whether to rewrite at all* (leave short/terse/finetune-looking instructions alone). No
  text feature finds the regime. `turn on the stove` is the canonical of two different tasks, at
  1.65% and 100%; `put the black bowl on top of the cabinet` is the canonical of three, at 9.3%,
  95.0% and 0.0%. Only the scene differs. A *perfect* oracle gate that is told the canonical's
  success rate beats blanket conservative rewriting by -0.08pp, i.e. not at all.
- *"Add at most four new content words."* Threshold-fished; the marginal cell is near-empty and the
  effect is null at every neighbouring cutoff once tier is controlled.
- *Length caps of any form* — absolute, canonical-relative, or 2x-ratio. Confounded with the
  adversarial prompt family; tier-controlled every version is null or negatively signed, and raw
  word count is positively signed in the FE regression.
- *"Stay verb-initial."* Tier-controlled -0.46pp; it was a proxy for resembling the trained string.
- *"Never use a question form."* The high-band effect rests on two tasks (libero_goal:0 and :7) and
  reverses on the unbiased tiers.
- *"Strip politeness / hedging / subordinate clauses."* Well-powered nulls across five slices.
- *"Keep every spatial word."* Only the ordinals survive. Lateral words (left / right / front /
  back) are null-to-negative and are deliberately excluded from rule 2.
- *"Keep the colour/type modifier the instruction gave you."* Conditional on the head noun surviving,
  keeping the colour word is -0.1pp to -0.77pp — inert in both directions. The policy keys on the
  head noun, not the noun phrase.
- *"Never rename an object with a second descriptive noun"* and *"never add an off-list adjective."*
  Both are rule 1 wearing a mask: 86% of off-list-noun adders and 52% of off-list-adjective adders
  also drop a canonical noun. Stratified on noun preservation both go to zero (-0.51pp and +0.11pp).
- *"Don't drop a content word."* Once nouns are protected, dropping the rest is free: `drop <=1
  content word` is +1.10pp CI [-0.36, +2.75] tier-controlled. Attributes and spatial phrases can be
  reworded or dropped freely.
- *"Preserve both steps of a two-step instruction as two verbs."* Measured null (+0.07pp), and 79%
  of the bank's rewrites already collapse them. Kept out of the rules on evidence grounds; a
  rephraser should still preserve both steps for plain correctness reasons the bank cannot score.
- *"Suspend the rules for instructions longer than 12 words."* Word count is nearly collinear with
  canonical difficulty; the effect is carried by one task and reverses on 5 of 6 testable ones.

**Open questions**

1. *The rules are insurance whose expected value is concentrated where the applier cannot look.*
   Rule 1 is +43.76pp on the 10 finetuned instructions and roughly +1 to +4pp elsewhere, and no
   feature of the instruction text separates the two groups — 38 of the 55 out-of-finetune
   canonicals contain zero out-of-vocabulary content words, exactly like all 10 finetuned ones.
2. *Most tasks cannot be moved at all.* 20 of 65 tasks score 0% under every one of their 21+
   phrasings; 22 sit at canon >=90% where rewriting can only lose (across 712 such rewrites and 12,550
   episodes, none gained >=20pp, only 8 gained >=10pp, and the largest single gain is +10.0pp). Essentially all the upside lives in ~18 mid-band tasks, and nothing in the
   instruction or the scene predicts which band a task is in.
3. *Three tasks carry two-thirds of the measured upside* (libero_90:44, :77, :12). libero_90:44 in
   particular is a canonical pathology: the trained string `turn on the stove` scores 1.65% while 41
   of its 44 rephrasings score exactly 100% (bank mean 99.1%). Any headline "rewriting helps" number should be
   discounted accordingly.
4. *The in-vocabulary direction is barely sampled.* Rule 1 says keep the noun the instruction gave
   you; whether *upgrading* a noun the policy never saw in finetuning toward the finetune vocabulary
   would beat that is essentially unmeasured. `tray` is the test case — 7 tasks, 305 rewrites, 239
   keep the word; of the 66 that do not, 52 simply delete the destination and only 14 substitute
   anything at all (receptacle 8, container 2, box 2, platter 2, ...). No arm has the power to
   answer it.
5. *Two forms were never generated, so neither can be forbidden or endorsed:* narration ("the arm
   reaches...") appears in 1 phrase of 2,587, and multi-sentence output effectively never.
6. *Rule 7's benefit is not established* — positive in every slice, significant in none once tier is
   controlled. What is established is that it never hurts.
7. *Per-row episode counts are 5-50 (median 10)*, so a single 20pp task difference is ~1.3 sigma.
   Every number above is a contrast averaged over tasks with both the task count and both arms'
   episode totals stated; treat any rule resting on fewer than ~15 non-tied tasks as directional.
