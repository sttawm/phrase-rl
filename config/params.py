#!/usr/bin/env python3
"""Running log of iteration parameters for the rules loop and its evaluation.

Readable AND executable:  python config/params.py   prints every parameter with
its meaning, the derived costs, and the resolution each choice buys.

Rules for this file
-------------------
* Every MEASURED constant carries its provenance. If a number has no source it
  is a guess and must be labelled one.
* Every DECISION carries the date and the reason. Superseded decisions stay in
  the log, marked, so the record shows what changed and why.
* Nothing here is imported by training code yet; it is the single place the
  design is written down, so the calculator, the chart and the driver cannot
  quietly disagree about what N means.

DESIGN IN ONE LINE (2026-08-09)
    Every phrase the loop evaluates is measured at BANK GRADE, so evaluation and
    bank-building are the same act rather than two budgets. Sim phrases get
    n=36 rollouts, train phrases get F*C=64 proxy contexts -- both matching the
    grade of what is already banked, so old and new rows stay comparable.
"""
from dataclasses import dataclass

# ===========================================================================
# MEASURED CONSTANTS -- each with the artefact it came from
# ===========================================================================


@dataclass(frozen=True)
class Measured:
    sec_per_context_row: float = 2.7
    """Proxy scoring: seconds the score server bills per context-row for the
    FIRST phrase. Measured on e1/e6, 64 contexts x 1 phrase = 177s."""

    sec_per_extra_phrase_row: float = 0.2
    """Each ADDITIONAL phrase riding the same context row. Measured: 64 contexts
    x 8 phrases = 275s, so (275-177)/7/64 ~= 0.2s. This is why depth-first
    scoring one phrase at a time cost a factor of P -- the server batches over
    the phrase list within a context, and the context is the expensive part."""

    min_per_episode: float = 0.42
    """Sim rollout wall-clock per episode, one pod. Measured on the e6 leg."""

    sd_phrase_success: float = 0.193
    """SD of rollout success ACROSS phrases WITHIN a task -- the dominant noise
    term in every sim resolution below. Binomial-corrected, measured on
    sim_rollouts_natadv_0of2.parquet: 96 phrases (64 natural + 32 adversarial),
    8 val8 tasks, 36 rollouts each.

    USE THIS ONE, NOT fine_exam's 0.096. fine_exam is a set of CLOSE
    PARAPHRASES built to test fine reward discrimination, so its spread is small
    BY CONSTRUCTION -- 0.096 binomial-corrected against 0.193 here, a factor of
    2.0. The loop's bases are a nat/adv mix, which is the high-spread case.
    Quoting fine_exam's sigma made every sim resolution look ~1.6x better than
    it is; corrected 2026-08-09."""

    sd_proxy_logit: float = 1.376
    """SD of the calibrated proxy logit across phrases within a task, over the
    4,204 measured training phrases."""

    base_success: float = 0.42
    """val8 success rate for natural phrasing, the operating point every
    resolution number below is computed at. nat24 step-0 baseline = 42.19."""

    layout_half_gap_pp: float = 4.3
    """Layouts 12-23 are EASIER than 0-11 by this much, for natural rephrases.
    EXPERIMENT.md:3286, 'A24 exchangeability REFUTED'. This is why a measurement
    is defined by its layout COMPOSITION and not by its episode count: two n=36
    numbers built from different layout sets are not comparable. A half-layout
    read already produced one false conclusion here ('99% repair', corrected to
    72% on uniform-24)."""

    replay_fraction_of_repeat_cells: float = 0.76
    """Of 46,720 rollout cells rolled more than once across the 293k-episode
    corpus, 9.4% disagree on success. Independent draws at p=0.42 would disagree
    48.7% of the time, so ~76% of repeats are EXACT REPLAYS and ~24% are
    independent redraws at a different seed. Rollouts are deterministic given
    the seed; the seed is a CLI arg (phase0c_rollout.py:38, default 42) that is
    never written to the output parquet. See CachePolicy."""

    reference_effect_pp: tuple = (1.0, 3.2, 5.2)
    """A28 measured v3->v4 rulebook gains: qwen +1.0, gemini +3.2, claude +5.2.
    A design whose resolution exceeds these cannot see what it looks for."""


M = Measured()

# ===========================================================================
# EVAL SETS
# ===========================================================================


@dataclass(frozen=True)
class EvalSets:
    t_train: int = 25
    """Training tasks. The distiller SEES these -- its evidence file is built
    from these tasks only (rules_loop_driver.py:1296 passes train_tasks)."""

    t_train_held: int = 31
    """Held-out real-robot tasks. Proxy-scored every iteration as the
    generalization read, and NEVER banked -- see BankPolicy."""

    t_sim_held: int = 8
    """val8 sim tasks, CLEAN only, frozen as VAL8_TASKS in the driver and
    identical across all 36 splits.json files.

    DELIBERATELY EXCLUDES the 7 val-side distractor variants:
      1. They INHERIT their phrases from the clean base (jaccard 0.51-0.65), so
         they re-measure the same phrases in a cluttered scene rather than
         adding phrasing evidence.
      2. Base rates 7.9 / 20.8 / 24.8 pp against val8's ~42 -- far less
         headroom, so a rulebook effect is diluted, not measured.
      3. Zero distractor episodes exist in the 293k-episode rollout corpus; the
         only ground truth anywhere is 1,296 episodes in one file.
      4. Their proxy scores are broken-thin: 5 of 7 scored below the script's
         own context floor, one at a SINGLE successful episode.
      5. Statistically a wash anyway: at equal episodes, 8 tasks with more bases
         reads 2.83pp against 15 tasks at 2.98pp.
    Use them ONCE on the promoted rulebook as a generalization read."""


SETS = EvalSets()

# ===========================================================================
# PER-ITERATION SIZE
# ===========================================================================


@dataclass(frozen=True)
class Sizes:
    n_per_task: int = 16
    """Bases per task, the SAME on all three eval sets.

    Uniform on purpose: the train-vs-train_held gap is the loop's overfitting
    signal, and it reads cleanest when both sides carry the same phrase count
    and the same measurement grade.

    -> 25 x 16 = 400 train, 31 x 16 = 496 train_held, 8 x 16 = 128 sim.

    NOT gated by the phrase bank -- that was an error corrected 2026-08-09. The
    bank's sim rows are proxy-scored EVIDENCE for distillation; eval bases are
    generated. Different pools with different jobs."""

    fxc: int = 64
    """Context-rows per proxy phrase: F=4 frames x C=16 contexts.

    64, NOT 40, because 64 is what the bank already holds --
    bank_scores_train_0of2.parquet carries n_ctx=64 on all 2,105 rows. Scoring
    the eval at 40 would bank it at a grade that cannot be compared with those
    rows. The extra costs ~1.5 GPU-h per iteration; a split-grade bank costs the
    comparison.

    Both channels are means, so the linear predictor's EXPECTATION is
    budget-independent and ranking is valid at any C -- but the calibrated
    PROBABILITY is not, and must never be read as a success rate."""

    orig_nat_adv: tuple = (0.20, 0.50, 0.30)
    """Base mix. Costs the same whatever it is -- every base costs one rephrase
    and one measurement. It sets what the number MEANS, not what it costs.
    OUTSTANDING: the ERT adversarial reference exists for 0 of 8 val8 tasks, so
    the 0.30 share must be generated before the first real iteration. Note also
    that 254 of the ~350 existing val8 phrases are fine_exam CLOSE PARAPHRASES,
    which fill none of these three buckets cleanly."""

    iterations: int = 6
    rephrasers: int = 1
    """Gemini first. Claude/Qwen legs are separate runs and multiply the total
    by their count -- they do not share rollouts."""


SIZE = Sizes()

# ===========================================================================
# ROLLOUT / LAYOUT POLICY
# ===========================================================================


@dataclass(frozen=True)
class Rollouts:
    gt_n: int = 36
    """Rollouts per sim phrase. EVERY evaluated sim phrase gets the full 36, so
    the eval output is bank-grade by construction and there is no separate
    gt-subset pass to schedule or reconcile.

    36 is the repo's established SEARCH grade, and it is a structural number,
    not a power calculation: 18 training layouts x 2 reps. Its companions in
    that protocol are estimation at n=72 on virgin layouts 18-23, and
    certification at n=144 (all 24 layouts x 6, paired SE 5.9pp).

    Consequence to keep in view: a banked n=36 row is SCREENING grade. That is
    the right grade for evidence the distiller ranks over. If a single phrase's
    number later becomes a CLAIM inside a rule ('phrase X gives +N pp'), it
    needs certification at 144 and the n=36 row cannot carry it."""

    gt_composition: str = "layouts 0-17 x 2 reps"
    """HOW the 36 is composed -- and this, not the count, is the standard.

    The bank currently holds two forms:
      fine_exam    254 phrases, layouts 0-17 x 2 reps  (EXPERIMENT.md:1719)
      sim nat/adv  180 phrases, episode_ids 0-35 x 1   (roll_sim_arms.sh:60)
    Because episode_id wraps modulo the per-task grid, ids 0-35 on a 24-layout
    task resolve to layouts 0-23 once plus 0-11 again -- the same 2/3 hard,
    1/3 easy balance as 0-17 x 2, so the two agree. On an 18-layout task they
    agree exactly. On any OTHER grid size they diverge.

    That is a coincidence, not a design. Pin this composition explicitly on
    every new measurement and record it per row. See KNOWN_ISSUES: the val8
    per-task grid sizes are still unverified."""

    reps_per_layout: int = 2
    """Follows from gt_composition: 18 layouts x 2 = 36.

    Note this REVERSES the K=1 rule that governs a rulebook-mean design. Both
    are right for their own job:
      * estimating a rulebook MEAN -- spend on distinct bases, K=1, because
        Var = 2p(1-p)/E + 2*sd^2/n: the binomial term depends only on total
        episodes while the phrase term depends only on distinct bases.
      * estimating ONE PHRASE for the bank -- spend on that phrase, because
        there is no averaging across bases to help it.
    Measuring every eval phrase at bank grade means the second objective sets
    the budget and the first inherits whatever resolution falls out. That trade
    is why sim resolution below is 5.1pp rather than 3.0pp."""

    layout_pool: tuple = tuple(range(24))
    """Layouts available. All 24.

    The 0-17 / 18-23 reservation in EXPERIMENT.md:398-407 governs ROLLOUT RL --
    its header is literally 'FAIR-EVAL PROTOCOL (rollout RL trains in the same
    sim it is evaluated in)'. That RL gradient-updates on layout 0-17 outcomes
    and can memorise them, so 18-23 is its generalization control.

    The rules loop trains no weights. No mechanism carries a layout outcome into
    a rulebook except the choice of which rulebook to promote -- ~6 bits over a
    whole run, which is reported rather than fenced off. (An earlier draft of
    this file reserved 18-23 here. Wrong: it imported a rollout-RL control into
    a loop with no gradient.)

    gt_composition uses 0-17 anyway, to match the banked rows."""


ROLL = Rollouts()

# ===========================================================================
# BANK POLICY
# ===========================================================================


@dataclass(frozen=True)
class BankPolicy:
    bank_train: bool = True
    """train (25 tasks x 16) is banked at F*C=64 after every iteration. The
    distiller already sees these tasks, so banking them leaks nothing."""

    bank_train_held: bool = False
    """train_held (31 tasks x 16) is NEVER banked. The bank feeds the
    distiller's evidence, so banking the held-out set would destroy the only
    real-robot generalization read the loop has. Today the driver's train_tasks
    filter (rules_loop_driver.py:1296) would exclude them anyway -- this flag
    says do not rely on that filter as the sole guard."""

    bank_sim: bool = True
    """val8 (8 tasks x 16) is banked at gt n=36, into the SIM bank."""

    sim_bank_visible_to_proxy_loop: bool = False
    """Sim rows are for the sim loop only. Verify this holds for any NEW
    evidence writer, not just the existing one."""

    sealed_excluded: bool = True
    """rules_loop_driver.py:507 drops is_sealed(task) rows. The 4 sealed
    distractor variants inherit sealed status and go with their bases."""

    KNOWN_ISSUES: tuple = (
        "results/rules_runs/live1/bank.parquet is STALE (Aug 7 19:08, before "
        "bank_generated.parquet on Aug 8 10:25) and holds ZERO natural and ZERO "
        "adversarial phrases. seed_bank() returns the cached file "
        "unconditionally when it exists, so re-running live1 silently distills "
        "from the old bank. Delete or rebuild before the next run.",
        "scripts/finish_sim_bank.sh invokes phase0c_rollout without exporting "
        "VLA_DATA_DIR / VLA_LOG_DIR / WANDB_MODE -- the only script in the repo "
        "that does not. The rollout dies, and with no rc check under "
        "'set -uo pipefail' (no -e) scoring proceeds over thin trajectories and "
        "the run self-reports success. This is work item #26.",
        "episode_id wraps MODULO the per-task grid "
        "(len(xy_configs) * len(quat_configs)), and grid sizes differ per task "
        "(4 to 360 in the vendored envs). Verify each val8 task's grid before "
        "trusting that 0-17x2 and 0-35x1 measurements are comparable. The env "
        "classes live in the external INT-ACT fork, so this needs a pod read.",
        "The rollout parquet does not record the seed, so a cached cell cannot "
        "be told apart from one that would be an independent redraw. One "
        "column fixes it. See CachePolicy.",
    )


BANK = BankPolicy()

# ===========================================================================
# SCORE-BANK-AS-CACHE
# ===========================================================================


@dataclass(frozen=True)
class CachePolicy:
    enabled: bool = True
    """The bank doubles as a scoring cache: if an iteration emits a rewrite that
    has already been measured, reuse the measurement instead of re-rolling. The
    project already does this by hand -- the row-14 confirm round pulled its
    layout-18-23 cells out of the sealed-leg parquets rather than re-rolling."""

    key: tuple = ("task", "phrase", "episode_id", "rep", "seed")
    """Cache at CELL level, not at aggregate level.

    Aggregate caching (reusing a phrase's gt_success at n=36) breaks the paired
    comparison: the eval pairs rulebooks on the same (base, layout) cells, and
    layouts differ by 4.3pp, so a cached arm measured on different layouts from
    its fresh counterpart leaks that gap into the difference. Cell-level keying
    lets the eval take exactly the cells it needs.

    BLOCKER: `seed` is not in the schema. phase0c_rollout.py:38 accepts --seed
    (default 42) and never writes it out. Until it does, ~24% of apparent cache
    hits are independent redraws (Measured.replay_fraction_of_repeat_cells)."""

    expected_hit_rate: float = 0.197
    """A LOWER BOUND, from pass-through alone: 13 of 66 rows in
    results/rules_runs/_rule_cache.parquet have rewrite == base, and a
    pass-through whose base is already banked at the loop's cells is free.
    Cross-iteration hits come on top and grow as rulebooks converge."""


CACHE = CachePolicy()

# ===========================================================================
# DERIVED
# ===========================================================================


def n_train():
    return SETS.t_train * SIZE.n_per_task


def n_train_held():
    return SETS.t_train_held * SIZE.n_per_task


def n_proxy_phrases():
    return n_train() + n_train_held()


def n_sim_bases():
    return SETS.t_sim_held * SIZE.n_per_task


def episodes():
    return n_sim_bases() * ROLL.gt_n


def _proxy_task_sec():
    return SIZE.fxc * (M.sec_per_context_row
                       + M.sec_per_extra_phrase_row * (SIZE.n_per_task - 1))


def proxy_hours():
    return (SETS.t_train + SETS.t_train_held) * _proxy_task_sec() / 3600


def rollout_hours():
    return episodes() * M.min_per_episode / 60


def rephraser_hours(sec_per_call=2.0):
    return (n_proxy_phrases() + n_sim_bases()) * sec_per_call / 3600


def hours_per_iteration():
    return proxy_hours() + rollout_hours() + rephraser_hours()


def sim_resolution_pp(n=None, per=None, sd=None):
    """Detectable paired difference between two rulebooks, 95%."""
    import math
    n = n_sim_bases() if n is None else n
    per = ROLL.gt_n if per is None else per
    sd = M.sd_phrase_success if sd is None else sd
    p = M.base_success
    return 1.96 * math.sqrt(2 * p * (1 - p) / (n * per) + 2 * sd ** 2 / n) * 100


def sim_resolution_floor_pp(n=None, sd=None):
    """Best achievable at that base count, infinite rollouts per phrase."""
    import math
    n = n_sim_bases() if n is None else n
    sd = M.sd_phrase_success if sd is None else sd
    return 1.96 * math.sqrt(2 * sd ** 2 / n) * 100


def proxy_resolution_pp(n=None):
    import math
    n = n_proxy_phrases() if n is None else n
    p = M.base_success
    lo = math.log(p / (1 - p))
    d = 1.96 * M.sd_proxy_logit * math.sqrt(2) / math.sqrt(n)
    return (1 / (1 + math.exp(-(lo + d))) - p) * 100


def per_phrase_ci_pp(n_rollouts):
    """95% CI on ONE phrase's success -- why gt_n is not a per-iteration knob."""
    import math
    p = M.base_success
    return 1.96 * math.sqrt(p * (1 - p) / n_rollouts) * 100


def bases_for(target_pp, per=None):
    """Smallest base count reaching target_pp, in multiples of the 8 val8 tasks."""
    per = ROLL.gt_n if per is None else per
    n = SETS.t_sim_held
    while n < 8000 and sim_resolution_pp(n=n, per=per) > target_pp:
        n += SETS.t_sim_held
    return n


# ===========================================================================
# DECISION LOG
# ===========================================================================

DECISIONS = [
    ("2026-08-09", "Every evaluated phrase is measured at BANK GRADE",
     "Sim at n=36, train at F*C=64. Evaluation and bank-building become one act "
     "instead of two budgets, and no separate gt-subset pass has to be "
     "scheduled or reconciled. Costs sim resolution -- see the 5.1pp entry."),
    ("2026-08-09", "N = 16 per task, uniform across all three eval sets",
     "The train vs train_held gap is the overfitting signal and reads cleanest "
     "at equal phrase counts and equal grade."),
    ("2026-08-09", "F*C = 64, not 40",
     "bank_scores_train_0of2.parquet holds n_ctx=64 on all 2,105 rows. Scoring "
     "at 40 would bank a grade that cannot be compared with them, to save "
     "~1.5 GPU-h per iteration."),
    ("2026-08-09", "train is banked; train_held is NOT",
     "The bank feeds the distiller's evidence. Banking the held-out set would "
     "destroy the only real-robot generalization read the loop has."),
    ("2026-08-09", "sigma_phrase = 0.193, not 0.119",
     "CORRECTS every earlier sim resolution number. 0.119 came from fine_exam, "
     "a set of close paraphrases built to test fine discrimination, so its "
     "spread is small by construction. The loop's bases are a nat/adv mix."),
    ("2026-08-09", "gt COMPOSITION (0-17 x 2), not just gt_n=36, is the standard",
     "Two n=36 measurements from different layout sets are not comparable -- "
     "the halves differ by 4.3pp. The bank already holds two forms that agree "
     "only because the per-task grid happens to be 24."),
    ("2026-08-09", "Layouts drawn from all 24; the 0-17/18-23 split does not apply",
     "That reservation governs rollout RL, which gradient-updates on layout "
     "outcomes. A text-distillation loop has no such channel."),
    ("2026-08-09", "8 clean val8 tasks; no distractor variants in the loop",
     "They inherit their bases' phrases, have 8-25pp base rates against val8's "
     "42, have essentially no ground truth, and their proxy scores are "
     "broken-thin. Use once on the promoted rulebook."),
    ("2026-08-09", "The bank doubles as a cell-level scoring cache",
     "Keyed on (task, phrase, episode_id, rep, seed). Aggregate-level caching "
     "would break the paired comparison. Blocked on writing seed to the "
     "rollout parquet."),
    ("2026-08-09", "SUPERSEDED: K=1 with many bases + a 32-phrase gt subset",
     "Right for a rulebook-mean design, and it resolved to 3.9pp under the "
     "wrong sigma. Dropped in favour of measuring everything at bank grade, "
     "which is simpler to operate and yields 128 banked sim phrases per "
     "iteration instead of 32."),
    ("2026-08-09", "Per-rule evaluation dropped in favour of an adherence judge",
     "Per-rule cost scaled with the rule count and the loop could not afford "
     "it. The book is measured, not its parts."),
]


def main():
    W = 84

    def rule(c="="):
        print(c * W)

    def row(k, v, note=""):
        print(f"  {k:<24} {str(v):<16} {note}")

    rule()
    print("  ITERATION PARAMETERS -- rules loop")
    print("  every evaluated phrase is measured at bank grade")
    rule()

    print("\nEVAL SETS")
    row("T_train", SETS.t_train, "distiller sees these; BANKED")
    row("T_train_held", SETS.t_train_held, "generalization read; NEVER banked")
    row("T_sim_held", SETS.t_sim_held, "val8 CLEAN only; BANKED")

    print("\nSIZE")
    row("N per task", SIZE.n_per_task, "uniform across all three sets")
    row("  -> train", n_train(), "phrases")
    row("  -> train_held", n_train_held(), "phrases")
    row("  -> sim", n_sim_bases(), "bases")
    row("F x C", SIZE.fxc, "matches the 2,105 banked rows at n_ctx=64")
    row("orig:nat:adv", "%.0f:%.0f:%.0f" % tuple(x * 100 for x in SIZE.orig_nat_adv),
        "free -- sets meaning, not cost")

    print("\nROLLOUTS")
    row("gt_n", ROLL.gt_n, ROLL.gt_composition)
    row("episodes", episodes(), f"= {n_sim_bases()} bases x {ROLL.gt_n}")
    row("layout pool", "0-23", "RL's 0-17/18-23 reservation does not apply here")

    print("\nCOST PER ITERATION")
    row("proxy scoring", f"{proxy_hours():.2f} GPU-h",
        f"{n_proxy_phrases()} phrases at F*C={SIZE.fxc}")
    row("sim rollouts", f"{rollout_hours():.2f} GPU-h", f"{episodes()} episodes")
    row("rephraser API", f"{rephraser_hours():.2f} h", "not GPU")
    row("TOTAL", f"{hours_per_iteration():.2f} GPU-h")
    rh = rollout_hours()
    print(f"    sim wall-clock:  1 pod {rh:.1f}h | 2 pods {rh/2:.1f}h | "
          f"3 pods {rh/3:.1f}h | 4 pods {rh/4:.1f}h")
    tot = hours_per_iteration() * SIZE.iterations * SIZE.rephrasers
    row(f"x {SIZE.iterations} iterations", f"{tot:.0f} GPU-h",
        f"1 pod {tot/24:.1f}d | 2 pods {tot/2/24:.1f}d | 3 pods {tot/3/24:.1f}d")

    print("\nBANK YIELD PER ITERATION")
    row("train", n_train(), f"proxy rows at F*C={SIZE.fxc}")
    row("sim", n_sim_bases(), f"gt rows at n={ROLL.gt_n}")
    row("train_held", 0, f"({n_train_held()} measured, deliberately not banked)")
    print(f"    over {SIZE.iterations} iterations: {n_train()*SIZE.iterations} train "
          f"+ {n_sim_bases()*SIZE.iterations} sim rows")

    print("\nRESOLUTION (detectable rulebook difference, 95%, paired)")
    row("proxy", f"{proxy_resolution_pp():.1f} pp", f"on {n_proxy_phrases()} phrases")
    row("sim", f"{sim_resolution_pp():.1f} pp",
        f"on {n_sim_bases()} bases x {ROLL.gt_n}")
    row("sim floor", f"{sim_resolution_floor_pp():.1f} pp",
        "at infinite rollouts per phrase")
    q, g, c = M.reference_effect_pp
    print(f"    A28 reference: qwen +{q} / gemini +{g} / claude +{c} pp")
    if sim_resolution_pp() > g:
        nb = bases_for(g)
        print(f"    ** sim {sim_resolution_pp():.1f}pp EXCEEDS the gemini effect "
              f"(+{g}pp): underpowered for a gain that size, adequate for a")
        print(f"       claude-sized one (+{c}pp). Reaching +{g}pp needs {nb} bases "
              f"({nb // SETS.t_sim_held}/task) = {nb*ROLL.gt_n*M.min_per_episode/60:.0f} GPU-h.")

    print("\nPER-PHRASE PRECISION")
    for n in (4, 12, ROLL.gt_n, 72, 144):
        tag = {36: "  <- bank / search grade", 72: "  <- estimation grade",
               144: "  <- certification grade"}.get(n, "")
        row(f"n={n} rollouts", f"+/-{per_phrase_ci_pp(n):.1f} pp", tag)

    print("\nKNOWN ISSUES")
    for i, s in enumerate(BANK.KNOWN_ISSUES, 1):
        print(f"  {i}. {s}")

    print("\nDECISION LOG")
    for date, what, why in DECISIONS:
        print(f"  {date}  {what}\n              {why}")
    print()


if __name__ == "__main__":
    main()
