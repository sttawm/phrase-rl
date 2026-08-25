# Four-tier phrasing evaluation: π0.5 on LIBERO

**One-line result:** π0.5's sensitivity to instruction wording is a property of
its *finetune pairing*, not of the policy: where the canonical string was
finetuned, it is unbeatable and ornate rephrasing halves success; where it was
not, all phrasings perform alike — except that a string finetuned in one scene
actively *fails* in another (0/20) while every rephrasing of it succeeds
(140/140), and searched phrasings recover large fractions of lost success on
about half of out-of-finetune tasks.

Everything here is computed by `scripts/analyze_fourtier.py` from
`results/analysis/fourtier_live/*.jsonl` →
`results/analysis/fourtier_summary.json`. 10,700 rollout episodes, 30 tasks,
2026-08-25, one day on four RunPod RTX-4090-class pods (all terminated after
verified salvage). Experiment record: `results/experiments.json` id
`pi05_libero_fourtier`.

## 1. What was run

- **Policy**: π0.5, Physical Intelligence's released `pi05_libero` checkpoint,
  served by stock OpenPI (`serve_policy.py --env LIBERO`, pinned `15a9616`).
  The finetune dataset (`physical-intelligence/libero`) contains exactly 40
  task strings — the four eval suites × 10 tasks, ~50 teleop demos each — with
  `prompt_from_task=True`: the canonical string is the *only* wording each task
  was ever paired with.
- **Tasks (30)**: all 10 `libero_goal` (in-finetune); 20 mid-band `libero_90`
  tasks (out-of-finetune scenes; prior canonical 1–4 of 5). Two of them share
  their exact instruction string with finetuned tasks (44 "turn on the stove",
  77 caddy/book) → stratum `l90_trained_string`; the other 18 are `l90_clean`.
- **Tiers per task**: original (canonical, 10→30 trials after deepening);
  5 natural + 5 adversarial rephrasings at 5→10 trials each, generated with
  the *same prompt and model* as the SIMPLER bank tiers
  (`prompts/rules_loop/generate.md` verbatim, text-only, gemini-pro-latest).
- **Oracle**: rollout board search, port of the SIMPLER `search_boards.py`
  spec — board of 16 seeded with canonical + naturals, 5-trial screens on init
  states 0–4, keep-4, Gemini regenerates 12 informed by the ranked board
  (image-conditioned flash), converge ≤4 rounds — then **confirm top-4 +
  canonical at 10 trials on virgin init states 20–29**; the confirm winner is
  the reported oracle. An `oracleplus` pass re-searched every task whose
  winner was <90% (8 rounds, adversarials join the seed, rotating
  plain/visual/motion lenses). Deepening re-measured all confirm finalists on
  a *second* virgin window (inits 30–49, `confirm2`).
- **Protocol amendments** (all recorded in `experiments.json`): screens 3→5
  trials before any libero_90 board ran (goal-6 grandfathered); the episode
  measurement key is `(suite, task, phrase, init)` — arm labels are metadata,
  reuse is by exact pairing.

## 2. Macro trends

Pooled success, Wilson 95% intervals (per-episode n in parentheses):

| stratum | original | natural | adversarial | oracle (confirmed) |
|---|---|---|---|---|
| goal, in-finetune (10) | **97.3** [94.8–98.6] (300) | 86.2 [82.9–88.9] (500) | **47.2** [42.9–51.6] (500) | 99.0 [94.5–99.8] (100) |
| l90 clean (18) | 51.6 [47.2–56.0] (490) | 50.7 [47.2–54.2] (775) | 47.7 [44.2–51.3] (775) | **65.6** [58.4–72.1] (180) |
| l90 trained-string (2) | **11.7** [5.8–22.2] (60) | 75.0 [65.7–82.5] (100) | 78.0 [68.9–85.0] (100) | **95.0** [76.4–99.1] (20) |

1. **The adversarial penalty lives in the finetune, not the policy.**
   In-finetune, ornate/indirect phrasing costs −50.1pp (97.3→47.2).
   Out-of-finetune the three tiers are statistically indistinguishable
   (51.6 / 50.7 / 47.7). The same generator, the same phrase styles — the
   collapse only happens where a canonical string was trained on top of them.
2. **The canonical string is unbeatable at home, ordinary abroad.** Oracle
   search returns the canonical itself on 7/10 goal tasks (99.0 pooled ≈ the
   97.3 baseline); on `l90_clean` it finds +14.0pp (51.6→65.6), and on the
   trained-string stratum +83.3pp (11.7→95.0).
3. **Selection honesty.** Screen-vs-confirm gap: +2.0pp (30 tasks) — 5-trial
   screens barely inflate. Confirm-vs-confirm2 (second virgin window, 18
   deepened tasks): +3.9pp mean, but with large per-task swings — see §3.4.
   Robust per-task estimates below always quote confirm2 when it exists.

## 3. Micro trends

### 3.1 The trained-string inversion (the headline mechanism)

"turn on the stove" is the trained string of `libero_goal/7`. The full 2×2,
same phrase strings in both kitchens:

| same phrases | trained kitchen (goal/7) | novel kitchen (l90/44) |
|---|---|---|
| canonical | **100%** (30/30) | **0%** (0/30) |
| 5 naturals | 72% | 100% |
| 5 adversarials | 14% | 100% |

Every one of the 20 distinct rephrasings tried in the novel kitchen succeeded
in it (140/140 episodes) while the trained string went 0/20 there. Replicated
on the other trained-string task (77: canonical 23% pooled vs oracle 90%,
winner "Put the standing black notebook into the rear compartment…"). Chart:
`results/charts/l90_44_inversion.png`.

The cross-scene cells split the binding into two separable symptoms:
- *Canonical-fails-abroad* (both pairs): the trained wording triggers
  behavior bound to the training scene and misfires elsewhere.
- *Everything-else-fails-at-home* (stove pair only): in goal/7's kitchen the
  l90/44 board pooled 70/105 — mechanism-grounded phrases ("twist the black
  knob", "rotate the black dial") work in BOTH kitchens (5/5 each side), but
  the ornate phrases that were perfect abroad collapse to 0–2/5 at home. The
  book pair shows the opposite: its trained scene accepts all 56 phrasings
  (278/280). Tightening is task-dependent; the inversion is not.

### 3.2 A binding gradient, not a binary

`l90/10` is out-of-finetune, but its string is one word from trained goal/4
("put the **black** bowl on top of the cabinet"). It behaves exactly like an
in-finetune task: canonical 95%, naturals 64%, adversarials **0%**. String
proximity alone reproduces the trained-task signature in a novel scene.

### 3.3 Phrasing failures vs capability failures

The same string, two novel scenes: "put the black bowl on top of the cabinet"
scores **95–100% on l90/10 and 10% on l90/31 — and on 31 no wording helps**
(two independent searches; best found 20% on the confirm window). Likewise
l90/35 (microwave): everything ≤10%. Rollout search cleanly separates tasks
that phrasing can rescue (44, 77, 12) from tasks the policy simply cannot do
in that scene (31, 35, 14) — a distinction a static benchmark number hides.

### 3.4 Which oracle recoveries are real (confirm → confirm2)

| task | orig | confirm | confirm2 | verdict |
|---|---|---|---|---|
| l90/44 "Could you start the burners?" | 0 | 100 | **100** | robust (+100pp) |
| l90/12 "…back grey bowl on the red and white plate" | 53 | 100 | **80** | robust (+27pp) |
| goal tasks (canonical winners) | 97 | 100 | 95–100 | at ceiling |
| l90/28 "close the open drawer on the left" | 40 | 70 | **40** | evaporates |
| l90/79 "Into the left slot of the caddy…" | 40 | 60 | **30** | evaporates |

The second virgin window is what makes the oracle column credible: reported
recoveries that were selection luck (28, 79) are caught; the trained-string
and visual-rename recoveries hold.

### 3.5 What winning phrases look like

Across the 21 tasks where a non-canonical phrase won or tied, winners are
mundane, echoing SIMPLER: **visual renames** — "grey bowl" beats the
canonical "black bowl" twice (l90/12, l90/60; the bowls render greyish),
"silver coffee pot" for moka pot (l90/19), "green and orange can" for tomato
sauce (l90/59); **spatial/state references** — "close the open drawer on the
left", "Into the left slot of the caddy"; **mechanism references** — "twist
the black knob"; and plain **politeness reformulations** ("Can you place the
cream cheese onto the tray?"). Ornate register only ever wins abroad (l90/70's
"Would you be so kind as to relocate the cocoa-flavored dessert…", 90%).

## 4. Relation to the π0/SIMPLER program

The SIMPLER four-tier structure transfers: original ≥ natural ≫ adversarial
in-distribution, oracle recovering a large fraction of the gap. What LIBERO
adds is the *clean separation of the cause*: because PI's finetune paired each
task with exactly one string, the in-finetune stratum isolates
string-binding, and the trained-string/novel-scene cells expose it as an
inversion — the strongest evidence yet that "instruction-following fragility"
in these VLAs is memorized wording→behavior shortcuts, not language
understanding. Rephrasing is a *repair* exactly where a shortcut exists
(in-finetune ornate inputs; trained strings abroad) and a lottery where none
does.

## 5. Reproduction

- Harness (in `sttawm/interactive-vlas`): `pi05_libero/eval/fourtier_eval.py`
  (`--phase main|deepen|oracleplus|cross`), launchers `run_fourtier.sh` /
  `run_fourtier_ext.sh`, tier generator `gen_fourtier_phrases.py`, task lists
  `fourtier_tasks{,_ext}.json`, phrases `fourtier_phrases.json`.
- Data: `results/analysis/fourtier_live/fourtier_lb{1..4}.jsonl` (one row per
  episode) + `.boards.json` sidecars + `.runlog`s, all pulled and
  count-verified before pod termination.
- Charts: `scripts/make_fourtier_progress.py`,
  `scripts/make_l90_44_inversion.py` → `results/charts/`.
- Caveats: per-cell n is 10–30 (tier CIs are honest but individual per-task
  cells swing); the 5-trial screen amendment landed mid-run (before any
  libero_90 board; goal-6 grandfathered under 3-trial screens); rollouts
  reproduce exactly only within a pod (SIMPLER cross-pod drift doctrine).
