# π0.5 / LIBERO — screen findings and eval-design implications

> **CAVEAT — written with limited context.** This was produced 2026-09-04 from the
> replacement machine during the migration, by a session that had **none** of the
> project's accumulated context: no Claude Code transcripts, no auto-memory
> directory (`~/.claude/projects/.../memory/`, which holds the pod-repair recipes),
> no local `.env`/`~/.zshrc`, no Overleaf clone, and no SSH access to any pod. It
> could not read `ALGORITHM-RL.md` (never committed to any branch) or the ultracode
> distillation script (lived in the old machine's session workflows dir).
>
> Everything here was reconstructed from **committed artifacts and commit messages
> only**. `results/` was not checked out — individual blobs were fetched one at a
> time. In particular, the state of the val-15 re-screen on pod `ps1` is
> **unknown**: it was launched outside the job queue and could not be inspected
> without SSH.
>
> The old laptop has strictly more context than this session did. **Where anything
> here conflicts with what that machine knows, trust the old laptop.** Treat this
> as a starting point to verify, not as an authority.
>
> One unexplained observation, recorded because it may matter: between 04:45 and
> 04:55 PDT on 2026-09-04, eleven files appeared in this working tree that this
> session did not create — A34 apply outputs (`ph_a34_*_qwen_nat.parquet`,
> `ph_a34_s_gemini_nat.parquet`), an `llm_log/` directory and a gemini apply cache
> (so real API calls ran), a status board plus `scripts/make_a34_status_board.py`,
> and a new `results/sealed/ph_sealed_natural_v3.parquet`. They were left untouched
> and uncommitted. If a second session or the recovered laptop produced them, the
> sealed-set artifact in particular should be checked against PREREG discipline
> before anything downstream trusts it.

Written 2026-09-04 from the new machine (`tararaffi`) during the migration, while
`results/` was NOT checked out (partial clone, sparse-checkout excludes it). Every
artifact below was read by fetching the single blob with
`git cat-file -p origin/main:<path>`. Numbers are quoted from files, not recalled.
Where I am inferring, it says so.

Companion docs: `PI05-BANK-HANDOFF.md` (split design), `FOURTIER-LIBERO.md`
(four-tier campaign), `results/analysis/pi05_bank/PREREG.md` (sealed contact log),
`PHRASE-GENERATION-RECIPE.md` (prompts/models/temps for every generation stage).

---

## 1. Sealed-20 canonical screen — COMPLETE, and it is brutal

Source: `results/rules_runs/p_seal/jobs/seal_screen_canon_w0.result.parquet`
(job `seal_screen_canon_w0`, claimed by pod `d090psulx896lv`/ps1 at 07:33 UTC,
result committed 09:27 UTC as `ab71eff`). Protocol per the spec: canonical
instruction only, `method=libero_bank_eval`, inits 0–19 (n=20), seed 7, port 8000.

**15 of the 20 sealed tasks scored exactly 0/20.** The PREREG expected "roughly
half"; the realised figure is 75%.

| task | screen % | in_vocab | gen_tier | canonical |
|---|---|---|---|---|
| 51 | **100** | True | novel_in_vocab | pick up the butter and put it in the basket |
| 30 | **85** | True | string_in_train | put the black bowl on the plate |
| 7 | 5 | True | novel_in_vocab | open the top drawer of the cabinet |
| 43 | 5 | True | novel_in_vocab | put the white bowl on top of the cabinet |
| 53 | 5 | True | novel_in_vocab | pick up the orange juice and put it in the basket |

Floor stratum (0/20): 8, 11, 18, 26, 45, 58, 62, 65, 69, 73, 80, 83, 85, 86, 88.
Of those, **in-vocab**: 8, 11, 26, 69, 73, 80, 83, 85. **Out-of-vocab**: 18, 45,
58, 62, 65, 86, 88.

Three of the five survivors are 1/20, so the sealed set really contributes
**two tasks with dynamic range**.

## 2. Val-15 — an n=10 prior already exists; the n=20 re-screen does not

`results/analysis/pi05_bank/val_canonicals.parquet` already carries a canonical
screen: `window=screen`, `n=10` (inits 0–9), `source=bank_v1`. I cross-checked all
15 canonicals against `splits.json` `lang`: **0 mismatches**.

| task | n=10 % | in_vocab | gen_tier | canonical |
|---|---|---|---|---|
| 47 | **100** | True | novel_in_vocab | pick up the cream cheese box and put it in the basket |
| 67 | **100** | True | novel_in_vocab | put the white mug on the left plate |
| 20 | **90** | True | string_in_train | turn on the stove |
| 0 | 10 | True | string_in_train | close the top drawer of the cabinet |
| 13 | 10 | False | novel_out_of_vocab | put the black bowl at the front on the plate |

Zero at n=10: 3, 4, 5, 6, 27, 42, 49, 63, 75, 87.

**5/15 nonzero**, versus 5/20 for sealed. Caveat: a 0/10 has a ~26% upper 95%
bound, so some of those ten may not be true floor — which is the case for the
n=20 re-screen.

**Status of the n=20 re-screen: NOT DONE and NOT RECOVERABLE from here.** It was
staged, then retracted from the queue (`b6292c7e`) and launched *directly* as 3
concurrent shards on `ps1`. Direct runs write outside `$JOBS/`, so the worker's
orphan-salvage never commits them, and nothing has landed in 2+ hours. Timeline
suggests it may have died: `ps1` was created 06:50 UTC, was running
`seal_screen_canon_w0` from 07:33–09:27, and the 3-shard val launch went in at
08:31 — i.e. four concurrent model loads on one 4090, with no stagger. Commit
`04cd91c6` added a 45s stagger precisely because "simultaneous model loads OOM'd
chunk 2 on rv4."

**Only the old laptop can check this** — its SSH key is the one in every pod's
`env.PUBLIC_KEY` (comment `sttawm`). The RunPod REST API has no exec/terminal
endpoint (23 endpoints; pods support only get/patch/delete/reset/restart/start/
stop/update), so the API key gives fleet control, not shell.

### Re-staged val screen job (unpushed, reproducible)

I staged `results/rules_runs/p_seal/jobs/val_screen_canon_w0.{spec.json,payload.parquet}`
but deliberately did **not** commit them — pushing makes a worker claim the job and
start a ~2h run, which conflicts with checking ps1 first. Schema was verified
identical to the sealed job (payload `[task, phrase]`, both `str`). Spec:

```json
{"job_id": "val_screen_canon_w0", "kind": "score", "method": "libero_bank_eval",
 "prereg": "2026-09-04 amendment: canonical-only screen, window 0-19, report on 30-49, drop at 0/20",
 "rollout": {"inits": [0..19], "seed": 7, "port": 8000}}
```

Payload = the 15 val tasks as `libero_90:<task_id>` with their `splits.json` `lang`.

## 3. The three rulebooks — located

All in `results/analysis/pi05_bank/rulebooks/`:

| book | rules | note |
|---|---|---|
| `in_only_v1.md` | 4 | in-finetune evidence only (`a3731fbb`) |
| `ood_only_v1.md` | 6 | libero_90 only; rule 1 narrowed by audit (`cc0fd77e`, `6c783a15`) |
| `in_plus_ood_v2.md` | 7 | both; **byte-identical to `results/analysis/pi05_bank/rules_bank_v1.md`** |
| `in_plus_ood_v1.md` | — | SUPERSEDED by v2 (full 13-agent re-run) |

Verified by sha256 that `in_plus_ood_v2.md` and `rules_bank_v1.md` are the same
file, and that v1 differs from v2.

## 4. What the books' own analysis says about the eval design

From the `===RATIONALE===` section of `in_plus_ood_v2.md`, recomputed over
**2,471 rewrites / 65 tasks / 42,990 episodes**:

- **Rule 1 (carry object and container nouns verbatim) *is* the rulebook.**
  Task-paired all 65 tasks: **+11.15pp**, 33+/10−/22 tied, CI [+6.46, +16.29],
  sign p = 0.0006.
- Split by finetune membership:
  - **10 in-finetune tasks: +43.76pp, 10 of 10 positive**, CI [+35.27, +51.78].
  - 55 out-of-finetune tasks: **+5.22pp**, 23+/10−, CI [+1.64, +9.37], p = 0.035.
- **"Most tasks cannot be moved at all."** 20 of 65 score 0% under *every one* of
  their 21+ phrasings. 22 sit at canon ≥90% where rewriting can only lose: across
  712 such rewrites and 12,550 episodes, **none gained ≥20pp**, only 8 gained
  ≥10pp, largest single gain +10.0pp. Essentially all upside lives in **~18
  mid-band tasks**, and *nothing in the instruction or the scene predicts which
  band a task is in*.
- **Three tasks carry two-thirds of the measured upside** (`libero_90:44`, `:77`,
  `:12`). Task 44 is a pathology: trained string "turn on the stove" scores 1.65%
  while 41 of its 44 rephrasings score exactly 100%.
- Per-row episode counts are 5–50 (median 10), so a single 20pp task difference is
  ~1.3σ.

From the closing recommendations of `ood_only_v1.md`:

1. **Emit several rewrites per sealed task, not one** (its own evidence uses 10
   phrasings per arm per task), scored against an unconstrained arm with matched
   composition.
2. **Raise episodes per phrasing well above 10.** At n=10 a single episode is
   10pp and ~80% of within-cell variance is binomial, while the true per-phrase
   effect SD in this population is only 4–5pp — exactly at the detection threshold.
3. **Report the catastrophic-collapse rate** (a rewrite scoring ≥40pp below its
   own canonical), not just the mean. Specific prediction: on high-band tasks it
   should fall from ~14% of rewrites to ~6%.
4. **Always stratify by measured canonical band.** Pooled across bands the effect
   is diluted by the ~half of tasks that cannot move, and "a pooled
   rulebook-vs-canonical mean will come back indistinguishable from zero — a
   result fully consistent with this rulebook being correct."

## 5. Salvage: real, but rare

Measured on the 24 zero-canonical **train** libero_90 tasks in
`results/analysis/pi05_bank/bank.parquet` (2,797 rows), each with ~21 searched
phrasings:

- **4 of 24 rescued above 0%.**
- Only one is dramatic — task 44, 0% → 100% — and that is the known trained-string
  collision (`FOURTIER-LIBERO.md:80-84`). Excluding it: **3 of 23, capped at
  10–20%** (tasks 15 → 20%, 36 → 20%, 52 → 10%).
- **All four salvaged tasks are `in_vocab=True`.** No out-of-vocab task salvaged.

Implication for the floor stratum: the plausible salvage pool among the 15 sealed
floor tasks is the 8 in-vocab ones. Task **73** is the single best bet — it is
`string_in_train` *and* in-vocab, the same profile as task 44.

## 6. Base rates for calibration

From the bank, canonical (`kind=original`) rows:

| population | n tasks | mean canonical | at exactly 0% |
|---|---|---|---|
| libero_goal (in-finetune, train) | 10 | **98.0** | 0 |
| libero_90 (out-of-finetune, train) | 55 | 39.4 | 24 (44%) |

In-finetune per-condition means over the 10 goal tasks: canonical **98.0**,
natural **88.2**, adversarial **52.4**. Natural barely moves (−1 to −17pp, except
task 5 at −53); adversarial drops 16–93pp. So in-finetune contributes
**adversarial-repair** headroom, not canonical headroom.

The sealed-20's 75%-zero rate is a **bad draw**, not the population — train
libero_90 is 44% zero. (Inferred: the random split at seed 20260901 happened to
concentrate hard tasks in sealed.)

## 7. Design consequence — the productive band is nearly empty

Combining the sealed n=20 and val n=10 screens, the out-of-finetune pool is
**bimodal**:

```
100, 100, 100, 90, 85   can only lose (books: none of 712 rewrites gained >=20pp)
 10,  10,   5,  5,  5   near floor
        [ nothing ]     <- the ~18-mid-band where rules demonstrably work
      25 tasks at 0     immovable
```

This argues for **inverting** the "handful of in-finetune" plan:

- **In-finetune should be a large stratum**, drawn from the ~30 **untouched
  reserve** (`libero_spatial`, `libero_object`, `libero_10` minus one exclusion —
  `splits.json: reserve_in_finetune_suites`, `reserve_exclusions`). Do **not** use
  the 10 `libero_goal` tasks: they are in the train split and would be
  contaminated. In-finetune is where Rule 1 is +43.76pp with 10/10 consistency.
- **Keep the floor stratum token-sized** and report it separately, per the PREREG.
- **Keep the 85–100% tasks**, but as a *collapse-rate* stratum testing the
  ~14% → ~6% prediction, not as an improvement stratum.
- Consider **re-screening the survivors at higher n** to resolve true band — a
  100% at n=10 may be 80% at n=50, which would move it into the productive band.

## 8. Open PREREG items — amendments needed BEFORE generation

`results/analysis/pi05_bank/PREREG.md` states deviations must be logged as a
further amendment. Three are outstanding:

1. **Do val tasks join the evaluation?** `PREREG.md:95-97` records this as an
   explicit undecided: "Whether surviving val tasks JOIN the evaluation (with human
   rephrasings collected for them) is a separate decision to be recorded here
   before any rephrasing of val tasks is generated or rolled."
2. **Deliberately including 0/20 floor tasks** contradicts the fixed drop rule
   (`PREREG.md:56-58`: floor tasks are "excluded from human rephrase collection and
   from the primary analysis").
3. **Generated vs human-written sealed naturals.** `PREREG.md:11-15` specifies the
   sealed natural tier as **human-written** via `human_eval/rephrase_sheets.html`.
   Switching to generated naturals is a change of instrument.

Also note an asymmetry to record: val tasks have **already been measured** on
inits 0–9 (`bank_v1`), which the sealed 20 have not. The screening/reporting
window separation (screen 0–19, report 30–49) still holds, but prior exposure
differs between the two sources.

## 9. Fleet facts

- **Only two pods are LIBERO-capable**: `ps1` (`d090psulx896lv`, both seal screens)
  and `pw1` (`cdwnjbvv64eb27`, ran the `p0smoke` `libero_bank_eval` job). The 16
  `rv*` pods are bridge/SIMPLER stack and cannot run LIBERO jobs. LIBERO work is
  therefore bottlenecked on two GPUs.
- 18 pods were RUNNING at $12.52/hr ($300/day) with a fully drained queue
  (804 job ids, 804 results) as of 2026-09-04 ~11:00 UTC.
- Pods hold no git push token for this purpose beyond the worker's own; results
  reach `main` only via `rules_loop_worker.sh` committing `$JOBS/<jid>.result.parquet`.
