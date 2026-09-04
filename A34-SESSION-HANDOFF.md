# A34 session handoff — new-Mac session, 2026-09-04

Written for a session on the OLD laptop with none of this context. Scope: the
π0/Bridge/SIMPLER line only (A34, the "more-natural" naturals re-run). Nothing
was committed or pushed except this file; no pods were started or stopped.

> **CAVEAT — this was written with limited context.** It comes from a fresh
> clone on a replacement Mac with **no session transcripts and no auto-memory
> directory** (`~/.claude/projects/-Users-sttawm-dev-robotics-phrase-rl/memory/`
> did not survive). Everything below was reconstructed from committed code,
> configs, and commit messages. Additional limits:
>
> - The clone is `--filter=blob:none` with `results/` sparse-excluded, so most of
>   `results/` was never read — only ~10 specific files were fetched by path.
> - `EXPERIMENT.md` ends 2026-08-06 and `HANDOFF.md` was last updated 2026-08-31,
>   so both lag HEAD; commit messages were the primary source for September work.
> - Some prior decisions were recovered only because the user supplied them in
>   chat (e.g. that gemini and claude had already been judged adherent) — there
>   may be more such decisions this document silently contradicts.
>
> The old laptop has strictly more context than the session that wrote this.
> **Where this document and your own memory/transcripts disagree, trust yours.**
> Treat §1 (the phrase-set decision) and §5 (the missing baseline arm) as the
> load-bearing parts; they were made explicitly with the user in this session.

> **KNOWN DIVERGENCE at time of writing.** You (the old laptop) pushed
> `91c41ee9` "A34 natural legs: 4 books x 3 appliers on the judged 186-phrase
> image set (144 legs)" and pods began claiming them immediately. Those legs are
> staged on the **186**-phrase set; §1 below cuts a further 8 phrases to **178**.
> The user is aware and unconcerned — the extra phrases are easy to drop at
> analysis time, since results key on `(task, phrase)`. No action needed unless
> you want the rolled set to match §1 exactly. (One arm, `d34bcln`, stages 180
> rows across its 12 legs — presumably post-dedupe.)

---

## 1. Decision: the natural set is now 178 phrases ("more-natural")

`results/sealed/ph_sealed_natural_v2_img.parquet` (192) → **178**, in three cuts.
All 14 dropped phrases were **qwen-authored**.

**−6 goal-changing** (pre-session, already in the repo at commit `25be64ec`):
- `Give me the wheel, then the Coke can.` — coke_can_on_wheel
- `Rest the eggplant beside the keys` — eggplant_on_keyboard
- `Slide the purple eggplant to the keyboard` — eggplant_on_keyboard
- `Drop the eggplant next to the sponge.` — eggplant_on_sponge
- `Slide the juice box over to the flat circle.` — orange_juice_on_plate
- `Move the beverage container onto the yellow surface.` — orange_juice_on_plate

**−5 adversarial-register descriptive renames** (this session, user-approved):
- `Place the liquid breakfast on the dish.` — orange_juice_on_plate
- `Slide the can onto the rectangular black device` — coke_can_on_keyboard
- `Put the drink can on the input device` — coke_can_on_keyboard
- `Set the red beverage tin in the middle of the typing board` — coke_can_on_keyboard
- `Set the drink container on top of the plate.` — orange_juice_on_plate

**−3 borderline** (this session, user-approved):
- `Placed the round nut on the tire.` — past tense, not an imperative
- `Attach the nut to the tire.` — `attach` changes the action
- `balance the green-yellow dish on the cube` — `balance` changes the action

**Explicitly KEPT** by user decision: `Move the carrot and put it into the
ceramic dish.` (carrot_on_ramekin) — flagged by my filter, judged fine.

Rationale for the register cut: by the project's own `kind` definitions, `natural`
is "a fluent rewording, the kind of thing a person would actually say" and
`adversarial` is "deliberately awkward, ornate or indirect". Descriptive renames
in the natural tier are exactly what a rulebook repairs, so they would depress the
un-rephrased natural baseline and inflate apparent rulebook gain — and it is
author-confounded, since all of them are qwen's.

Result: **178 phrases — gemini 72 / claude 60 / qwen 46.**

**Per-task counts are now uneven** (was a uniform 16): orange_juice_on_plate 12,
coke_can_on_keyboard 13, eggplant_on_keyboard 14, nut_on_wheel 14,
coke_can_on_wheel 15, eggplant_on_sponge 15, small_plate_on_green_cube 15,
remaining five tasks 16. Matters if per-task cells are reported rather than pooled.

Regenerate with: filter `ph_sealed_natural_v2_img.parquet` on the 8 phrase strings
above (the other 6 are already absent from that file).

---

## 2. Verifications — re-usable, do not redo

- **Trace pre-flight PASSED.** All bases resolve to a **per-base** trace in
  `results/phrase_artifacts/traces_rules_v1.parquet` — 186/186 at the time of
  check, zero legacy-instruction fallback, zero placeholder. This is the
  "ERT-derived-trace defect" A31 fixed; it is intact. Matched traces were merged
  into that file by commit `452615e2`.
- **Prompt has not drifted.** `sha1(prompts/rules_loop/apply.md)[:12]` =
  `0bc4fc615551`, matching `results/rules_runs/r1_sim/config.json`
  `prompt_hashes["apply.md"]`. So the A34 protocol is byte-identical to A31's.
- **Applies are stateless per phrase** (`call_llm(..., session=None)`, by design so
  they don't cross-contaminate). Therefore **shrinking the base set never requires
  re-running applies — just filter rows.** This is how qwen's four books survived
  the 192→178 cut for free. Do not re-run them.

---

## 3. Findings

**Qwen fails the copy gate.** On identical inputs with the `sim_only` book,
substantive rewrite rate: **qwen 92% vs gemini 53%**. Concrete rule violations in
qwen's output:

| base | qwen | violated |
|---|---|---|
| `...set it in the white ramekin.` | `...put it in the white bowl` | Rule 7 — `set` is a safe verb to keep |
| `...set it on top of the sponge.` | `...set it on the sponge` | Rule 11 — never collapse "on top of" |
| `Set the carrot down inside the ramekin.` | `set the carrot into the ramekin` | Rule 11 — never delete "down"; also skipped the ramekin swap it applied elsewhere |

`apply.md:17` tells every applier "If no rule applies, return the instruction
unchanged. Returning it unchanged is a valid answer, not a failure." User confirmed
gemini and claude were previously judged **adherent**; qwen is the known outlier.
Consistent with A31, where qwen produced the campaign's only harm case
(sim_only × Original, 28.5 vs 36.1 baseline).

**Substantive change rate by book** (normalization-only edits excluded):
train_only 52–53%, sim_only 92% (qwen) / 53% (gemini), both 98%, scaffold 98%.
The train-only book is the conservative one.

**Schema mismatch — will silently produce a null result if missed.** Qwen's pod
job results are `(task, phrase=BASE, rewrite=OUTPUT)`. The local `ph_a34_*` files
are `(task, k, base, phrase=OUTPUT)`. Normalize before staging legs, or the qwen
arms will roll **un-rephrased bases**.

---

## 4. Arm status — 5 of 12 done

| arm | state |
|---|---|
| qwen × train_only / sim_only / both / scaffold | ✅ done, free (filtered from existing pod results) |
| gemini × sim_only | ✅ done — **the only real API spend**, 186 calls, 4 min 08 s |
| gemini × train_only / both / scaffold | ❌ remaining |
| claude × train_only / sim_only / both / scaffold | ❌ remaining |

Command: `FINAL_EVAL=1 RULES_APPLY_WORKERS=8 .venv/bin/python
scripts/gen_a34_applies.py results/rules_runs/v2distill/<arm>/rules.md <applier> <tag>`
with tags `t`/`s`/`b`/`sc`. NOTE: that script hardcodes
`ph_sealed_natural_v2_img.parquet` — point it at the 178-phrase set.

**Claude applier is `claude-opus-5`, not `claude-fable-5`** (`r1_sim/config.json`;
switched by commit `422637f2` for fable credits, caveat recorded in PREREG). A34's
claude column is therefore **not directly comparable to A31's**. Gemini and qwen
columns are.

---

## 5. Missing arm: there is no un-rephrased baseline

PREREG A34's "12 arms" = 4 books × 3 appliers. The scaffold row is *no rules but
still rephrased by an applier* — not the same thing. The 28.45 on the A31 board is
a **v1 preliminary number on a different prompt and a different phrase set** and
cannot serve as this set's baseline.

Add a **13th arm**: the raw 178 naturals rolled with no applier at all.
178 × 24 = **4,272 episodes** (~8% on top of 51,264). User agreed this is needed.

---

## 6. Leg staging — NOT written yet

A31's stager was never committed. Everything needed to write one:

- **Convention:** one leg per (arm, task) → 12 arms × 12 tasks = 144 legs, plus 12
  for the baseline arm. A31 ran 797 legs total across 16 pods.
- **Job id prefix:** A31 used `b31<book><applier><cond>` (e.g. `b31tqwo` =
  train-only × qwen × orig); re-runs used `c32*`.
- **Files per leg:** `<jid>.payload.parquet` (columns `task`, `phrase`) +
  `<jid>.spec.json`, dropped into `results/rules_runs/r1_sim/jobs/`, then committed
  and pushed. Workers claim by committing `<jid>.claim`; the push is the atomic
  arbiter.
- **Exact spec schema** (verbatim from `b31bcla_138ddf3803.spec.json`):

```json
{"job_id": "...", "kind": "score", "method": "rollout", "draw": 0, "seed": 7,
 "proxy": {"C": 8.123, "bz": 0.4445, "bg": 11.3193},
 "rollout": {"config": "config/experiment/simpler/pi0_finetune_bridge_ev.yaml",
             "ckpt": "juexzz/INTACT-pi0-finetune-rephrase-bridge",
             "seed": 42, "episode_ids": [0,1,...,23], "repeats": 1}}
```

- **Dedupe rewrites before rolling** — `rules_loop_jobs.py` aggregates by
  `(task, phrase)` anyway, so duplicates burn GPU for nothing. A31 `drop_duplicates()`'d.
- Natural condition is **24 layouts × 1 rep** (adversarial and original were ×2).

---

## 7. Artifacts on the new Mac worth transferring

| file | note |
|---|---|
| `results/sealed/ph_a34_s_gemini_nat.parquet` | **the only paid artifact** — transfer this |
| `results/rules_runs/r1_sim/apply_cache_gemini_a34sge_nat_977d9faf3f49.parquet` | its cache; `llm_log/` has all 186 prompt/response pairs |
| `results/sealed/ph_sealed_natural_v3.parquet` | the 178 set — trivially regenerable from §1 |
| `results/sealed/ph_a34_{t,s,b,sc}_qwen_nat.parquet` | qwen, normalized + filtered — regenerable for free |
| `scripts/make_a34_status_board.py`, `results/analysis/a34_status_cells.json` | greyed status grid; fill cells as legs land and re-run |
| `CLAUDE.md` | +307 lines appended, original 45 lines untouched |

---

## 8. Gotchas

- **NEVER run `git ls-tree -l` against `results/` in the blobless clone.** `-l`
  needs blob size, which isn't local, so git lazily fetches blob *content*. A
  full-tree `-l` scan starts pulling the 49 GiB of archives, and
  `GIT_NO_LAZY_FETCH=1` does **not** stop it on git 2.37.1 (Apple Git). I pulled
  ~428 MB this way before it was killed. Safe: `git ls-tree -r --name-only` for
  paths, `git cat-file --batch-check --batch-all-objects` for local objects,
  `git cat-file -p origin/main:<path>` to fetch one named blob deliberately.
- `claude` CLI needs a one-time interactive sign-in for headless `-p` use.
- The Gemini key must belong to a **paid** project; a fresh AI Studio key 429s.
- `av` (PyAV) won't build without `pkg-config`/ffmpeg — skip it; it's only for video.
- Pods created from the old laptop hold **only its SSH key**. Don't fix by
  restarting: `POD-SETUP.md:78,125` — `/root` doesn't survive stop/restart and the
  venvs are symlinked there.
- 18 pods were RUNNING at **$12.52/hr** with a fully drained queue (804 job ids,
  804 results) as of 2026-09-04 ~11:00 UTC.

---

## 9. Loose ends left on the new Mac

**Only this file was pushed.** Everything in §7 is local-only on the replacement
Mac. In particular `results/sealed/ph_a34_s_gemini_nat.parquet` (+ its
`apply_cache_gemini_a34sge_nat_977d9faf3f49.parquet` and the 186 prompt/response
pairs in `results/rules_runs/r1_sim/llm_log/`) represents **186 paid Gemini
calls**. Re-running gemini × sim_only will re-bill it. Ask for the file if you
want it; otherwise just re-run.

**Two independent CLAUDE.md reconstructions now exist.** The new Mac has +307
uncommitted lines appended to `CLAUDE.md` (original 45 lines untouched: project
framing, the two experiment lines, design decisions, experiment naming
conventions, glossary, how to run, where work stopped, stale-doc warnings).
Separately you pushed `b8eacf56` "pi0.5/LIBERO: screen findings + eval-design
implications; CLAUDE.md project context". **These were written independently and
should be reconciled deliberately, not blind-merged.**

**~428 MB of `results/` blobs are sitting in the new Mac's `.git`** (1.4 GB
total) from the `ls-tree -l` accident described in §8. Harmless but wasteful;
`git repack -a -d --filter=blob:none` drops them. Not run.

**Git identity on the new Mac** was set repo-locally (not `--global`) to
`sttawm <sttawm@users.noreply.github.com>` so this commit could be made. The
global config is still unset there.

**Pod fleet:** 18 running at $12.52/hr. They sat idle on a fully drained queue
for ~3 hours before your `91c41ee9` leg push gave them work; they are busy now.

**Task state at handoff:** applies 5/12 done (qwen ×4 free, gemini × sim_only
paid); 7 remain (gemini × t/b/sc, claude × t/s/b/sc). Leg staging: you have
already done it for the 186 set. The un-rephrased passthrough baseline arm (§5)
was still missing as of this writing.

---

## 10. Corrections to the repo's own record, found while reconstructing

- **`HANDOFF.md:194` says v10 checkpoints 0040–0220 are lost. They are not** —
  they're on the unmerged branch `origin/ckpt-l40s`, 19 `ckpt-sync` commits.
- **`ALGORITHM-RL.md`** (cited at `HANDOFF.md:180` as the v10/v11 RL history) was
  **never committed on any branch** — old-laptop-only.
- The "gripper is 25× the verifier" framing is a raw-coefficient artifact already
  retracted from the paper (`HANDOFF.md:174`); standardized within-task, z carries
  ~2× the gripper. The A31 evidence diets exposed **both** z and grip (99.3%
  populated on both) — no book was gripper-only.
