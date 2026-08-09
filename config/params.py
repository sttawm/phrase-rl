#!/usr/bin/env python3
"""Running log of iteration parameters for the rules loop and its evaluation.

Readable AND executable:  python config/params.py   prints every parameter with
its meaning, the derived costs, and the resolution each choice buys.

Rules for this file
-------------------
* Every MEASURED constant carries its provenance. If a number has no source, it
  is a guess and must be labelled one.
* Every DECISION carries the date and the reason. Superseded decisions stay,
  struck through in the text, so the record shows what changed and why.
* Nothing here is imported by training code yet; it is the single place the
  design is written down, so the calculator, the chart and the driver cannot
  quietly disagree about what N_sim means.
"""
from dataclasses import dataclass, field, asdict

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

    sd_phrase_success: float = 0.119
    """SD of rollout success ACROSS phrases WITHIN a task. From fine_exam:
    254 phrases x 36 rollouts each, 8 val8 tasks.
    CAVEAT: sim nat/adv phrases measured 0.194 instead. If the truth is nearer
    0.194 every sim resolution below degrades ~1.6x. The e6 shard-1 rollouts
    settle this; until then treat sim resolution as a lower bound on the gap."""

    sd_proxy_logit: float = 1.376
    """SD of the calibrated proxy logit across phrases within a task, over the
    4,204 measured training phrases."""

    base_success: float = 0.42
    """val8 success rate for natural phrasing, the operating point every
    resolution number below is computed at. nat24 step-0 baseline = 42.19."""

    layout_half_gap_pp: float = 4.3
    """Layouts 12-23 are EASIER than 0-11 by this much, for natural rephrases.
    EXPERIMENT.md:3286, 'A24 exchangeability REFUTED'. Lives here because it is
    the reason LAYOUT_POOL is balanced rather than 'the first L ids'. A
    half-layout read already produced one false conclusion in this project
    ('99% repair', corrected to 72% on uniform-24)."""

    proxy_vs_gripper_rank_corr: float = 0.645
    """Rank correlation between the calibrated proxy and gripper-only ranking.
    The driver aborts a run if the first 3 jobs average below RVG_FLOOR."""


M = Measured()

# ===========================================================================
# EVAL SETS -- what each iteration measures, and on what
# ===========================================================================


@dataclass(frozen=True)
class EvalSets:
    t_train: int = 25
    """Training tasks. The distiller SEES these -- its evidence file is built
    from these tasks only (rules_loop_driver.py:1296 passes train_tasks)."""

    t_train_held: int = 31
    """Held-out real-robot tasks. Never distilled from; proxy-scored only."""

    t_sim_held: int = 8
    """val8 sim tasks, CLEAN only. Frozen as VAL8_TASKS in the driver and
    identical across all 36 splits.json files.

    DELIBERATELY EXCLUDES the 7 val-side distractor variants. Reasons, in order
    of weight:
      1. They INHERIT their phrases from the clean base (jaccard 0.51-0.65), so
         they re-measure the same phrases in a cluttered scene rather than
         adding phrasing evidence.
      2. Their base rates are 7.9 / 20.8 / 24.8 pp against val8's ~42 -- far
         less headroom, so a rulebook effect gets diluted, not measured.
      3. Zero distractor episodes exist in the 295k-episode rollout corpus;
         the only ground truth anywhere is 1,296 episodes in one file.
      4. Their proxy scores are broken-thin: 5 of 7 scored below the script's
         own context floor, one at a SINGLE successful episode.
      5. Statistically it is a wash anyway -- at equal episodes, 8 tasks with
         more bases reads 2.83pp vs 15 tasks at 2.98pp.
    Use them ONCE on the promoted rulebook as a generalization read."""


SETS = EvalSets()

# ===========================================================================
# PER-ITERATION SIZE
# ===========================================================================


@dataclass(frozen=True)
class Sizes:
    n_task: int = 8
    """Bases per task on the two PROXY sets -> 56 x 8 = 448 phrases scored."""

    n_sim: int = 48
    """Bases per task on val8 -> n = 8 x 48 = 384 bases.

    DECOUPLED from n_task on purpose. The proxy pools 56 tasks, val8 pools 8,
    so at a shared N the sim leg is phrase-starved by ~7x. Raising a shared N
    would drag proxy scoring nobody needs along with it.

    NOT gated by the phrase bank -- that was an error corrected 2026-08-09. The
    bank's sim rows are proxy-scored EVIDENCE; val8 eval bases are generated.
    325 of 384 already exist on disk; 59 to generate (~$3)."""

    fxc: int = 40
    """Context-rows per proxy phrase: F frames x C contexts. The proxy was
    calibrated at F=4, C=10-24 (F*C ~= 72), so 40 sits below calibration --
    ranking stays valid (both channels are means, so the linear predictor's
    expectation is budget-independent) but the PROBABILITY must never be read
    as a success rate at this budget."""

    orig_nat_adv: tuple = (0.20, 0.50, 0.30)
    """Base mix. Costs the same whatever it is -- every base costs one rephrase
    and one score. It sets what the number MEANS, not what it costs.
    OUTSTANDING: the ERT adversarial reference exists for 0 of 8 val8 tasks, so
    the 0.30 share has to be generated before the first real iteration."""

    iterations: int = 6
    rephrasers: int = 1
    """Gemini first. Claude/Qwen legs are separate runs, and multiply the total
    by their count -- they do not share rollouts."""


SIZE = Sizes()

# ===========================================================================
# ROLLOUT / LAYOUT POLICY
# ===========================================================================


@dataclass(frozen=True)
class Rollouts:
    layouts_per_base: int = 4
    """L. Distinct layouts each base is rolled on."""

    reps_per_layout: int = 1
    """K. Episodes per layout.

    K=1 is OPTIMAL, not a compromise. Paired variance is
        Var = 2p(1-p)/E + 2*sd_phrase^2/n        with E = n * L * K
    The binomial term depends only on TOTAL episodes; the phrase term only on
    the number of distinct bases. So at a fixed episode budget, spend on bases,
    never on repeats. This is already how nat24 is built: 24 distinct phrases
    per task, one layout each."""

    layout_pool: tuple = tuple(range(24))
    """Layouts the loop may draw from. ALL 24.

    The 0-17 / 18-23 reservation in EXPERIMENT.md:398-407 governs ROLLOUT RL --
    its header is literally 'FAIR-EVAL PROTOCOL (rollout RL trains in the same
    sim it is evaluated in)'. That RL gradient-updates on layout 0-17 outcomes
    and can memorise them, so 18-23 is its generalization control.

    The rules loop trains no weights; it emits text. No mechanism carries a
    layout outcome into a rulebook except the choice of which rulebook to
    promote -- ~6 bits over a whole run. That is reported, not fenced off.
    (An earlier draft of this file reserved 18-23 here. Wrong: it imported a
    rollout-RL control into a loop with no gradient.)"""

    layout_balance: str = "2 from 0-11, 2 from 12-23"
    """Because layouts 12-23 run +4.3pp easier than 0-11 (see
    Measured.layout_half_gap_pp), an unbalanced fixed set biases every
    delta-vs-original the loop reports."""

    layouts_fixed_across_iterations: bool = True
    """Fixing the set is what makes the paired variance formula true -- there is
    no layout term in it. It also pairs iteration-to-iteration comparisons, so
    the 6-point trajectory shows the rulebook moving, not layout noise.
    Cost: absolute numbers become specific to this layout set. Acceptable, since
    the loop measures deltas; the confirmation read on 18-23 recovers the
    absolute claim."""

    gt_n: int = 36
    """Rollouts per phrase required for a phrase to enter the SCORE BANK with
    ground truth. The established standard: fine_exam_phrases.parquet (254
    phrases) and sim_rollouts_natadv_0of2.parquet (96 phrases) both carry
    gt_n=36 exactly.

    THIS IS NOT L, AND THE EVAL CANNOT SUPPLY IT.
    At L=4 a single phrase's success carries a 95% CI of +/-48.4pp; at gt_n=36
    it is +/-16.1pp. The eval is a rulebook-MEAN design -- it buys coverage by
    giving up per-phrase precision, which is the right trade for comparing
    rulebooks and the wrong one for scoring a phrase.
    Rolling all 384 bases at 36 would cost 13,824 episodes = 96.8 GPU-h per
    iteration, ~9x the L=4 design. Rejected.
    Resolution: see gt_subset_per_iter."""

    gt_subset_per_iter: int = 32
    """Rewrites promoted to a full gt_n=36 measurement each iteration, 4 per
    val8 task, and added to the SIM bank. 32 x 36 = 1,152 episodes = 8.1 GPU-h.
    Yields 192 gt-scored sim phrases over 6 iterations at +59% on the rollout
    leg. Pick them where they inform most -- widest spread, thinnest coverage.

    The L=4 episodes are still recorded, tagged n=4, and are honest in AGGREGATE
    (that is the whole eval) but must never be read as a per-phrase score."""


ROLL = Rollouts()

# ===========================================================================
# BANK / CONTAMINATION POLICY
# ===========================================================================


@dataclass(frozen=True)
class BankPolicy:
    sim_bank_visible_to_proxy_loop: bool = False
    """Sim phrases scored by the sim loop go into the sim bank and must NOT
    reach the train/proxy loop's evidence. Enforced today by the driver passing
    train_tasks (not val8) to write_new_measurements -- rules_loop_driver.py:1296.
    Verify this holds for any new evidence writer."""

    sealed_excluded: bool = True
    """rules_loop_driver.py:507 drops is_sealed(task) rows from the bank. The 4
    sealed distractor variants inherit sealed status and are excluded with
    their bases."""

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
        "(4 to 360 in the vendored envs). Verify each val8 task's grid is >= 18 "
        "before freezing LAYOUT_POOL, or some 'distinct' layouts are the same "
        "scene.",
    )


BANK = BankPolicy()

# ===========================================================================
# DERIVED -- cost and resolution
# ===========================================================================


def n_proxy_phrases():
    return (SETS.t_train + SETS.t_train_held) * SIZE.n_task


def n_sim_bases():
    return SETS.t_sim_held * SIZE.n_sim


def episodes_eval():
    return n_sim_bases() * ROLL.layouts_per_base * ROLL.reps_per_layout


def episodes_gt():
    return ROLL.gt_subset_per_iter * ROLL.gt_n


def proxy_hours():
    per_task = SIZE.fxc * (M.sec_per_context_row
                           + M.sec_per_extra_phrase_row * (SIZE.n_task - 1))
    return (SETS.t_train + SETS.t_train_held) * per_task / 3600


def rollout_hours():
    return (episodes_eval() + episodes_gt()) * M.min_per_episode / 60


def rephraser_hours(sec_per_call=2.0):
    calls = n_proxy_phrases() + n_sim_bases()
    return calls * sec_per_call / 3600


def hours_per_iteration():
    return proxy_hours() + rollout_hours() + rephraser_hours()


def sim_resolution_pp(L=None, n=None):
    """Detectable paired difference between two rulebooks, 95%."""
    import math
    L = ROLL.layouts_per_base if L is None else L
    n = n_sim_bases() if n is None else n
    E = n * L * ROLL.reps_per_layout
    p = M.base_success
    return 1.96 * math.sqrt(2 * p * (1 - p) / E
                            + 2 * M.sd_phrase_success ** 2 / n) * 100


def sim_resolution_floor_pp(n=None):
    """Resolution at infinite layouts -- bases alone set this."""
    import math
    n = n_sim_bases() if n is None else n
    return 1.96 * math.sqrt(2 * M.sd_phrase_success ** 2 / n) * 100


def proxy_resolution_pp():
    import math
    p = M.base_success
    lo = math.log(p / (1 - p))
    d = 1.96 * M.sd_proxy_logit * math.sqrt(2) / math.sqrt(n_proxy_phrases())
    return (1 / (1 + math.exp(-(lo + d))) - p) * 100


def per_phrase_ci_pp(n_rollouts):
    """95% CI on ONE phrase's success at n rollouts -- why gt_n != L."""
    import math
    p = M.base_success
    return 1.96 * math.sqrt(p * (1 - p) / n_rollouts) * 100


# ===========================================================================
# DECISION LOG
# ===========================================================================

DECISIONS = [
    ("2026-08-09", "K=1, not K>1",
     "Binomial variance depends on total episodes, phrase variance on distinct "
     "bases. At fixed budget more bases always beats more repeats."),
    ("2026-08-09", "N_sim decoupled from N_task",
     "val8 pools 8 tasks vs the proxy's 56; a shared N starves the sim leg or "
     "overpays the proxy leg."),
    ("2026-08-09", "N_sim is NOT gated by the phrase bank",
     "CORRECTS an earlier claim. Bank sim rows are proxy-scored evidence for "
     "distillation; val8 eval bases are generated. Different pools."),
    ("2026-08-09", "8 clean val8 tasks, no distractor variants in the loop",
     "Distractors inherit their bases' phrases, have 8-25pp base rates against "
     "val8's 42, have essentially no ground truth, and their proxy scores are "
     "broken-thin. Statistically a wash. Use once on the promoted rulebook."),
    ("2026-08-09", "Layouts FIXED across iterations, drawn from all 24",
     "The 0-17/18-23 reservation governs ROLLOUT RL, which gradient-updates on "
     "layout outcomes. The rules loop trains no weights, so it does not apply. "
     "Balance 2 from 0-11 + 2 from 12-23 against the +4.3pp half gap."),
    ("2026-08-09", "gt_n=36 stays the bank standard; the eval does not feed it",
     "L=4 gives +/-48pp on a single phrase. A gt_subset of 32 rewrites per "
     "iteration is promoted to 36 rollouts and banked."),
    ("2026-08-09", "Per-rule evaluation dropped in favour of an adherence judge",
     "Per-rule cost scaled with R and the loop could not afford it. The book is "
     "measured, not its parts."),
]


def main():
    W = 78
    def rule(c="="): print(c * W)
    def row(k, v, note=""):
        print(f"  {k:<26} {str(v):<18} {note}")

    rule(); print("  ITERATION PARAMETERS -- rules loop"); rule()
    print("\nEVAL SETS")
    row("T_train", SETS.t_train, "distiller sees these")
    row("T_train_held", SETS.t_train_held, "proxy-scored, never distilled from")
    row("T_sim_held", SETS.t_sim_held, "val8 CLEAN only (no distractors)")
    print("\nPER-ITERATION SIZE")
    row("N_task", SIZE.n_task, f"-> {n_proxy_phrases()} proxy phrases")
    row("N_sim", SIZE.n_sim, f"-> {n_sim_bases()} val8 bases")
    row("F x C", SIZE.fxc, "context-rows per proxy phrase")
    row("orig:nat:adv", "%.0f:%.0f:%.0f" % tuple(x * 100 for x in SIZE.orig_nat_adv),
        "free -- sets meaning, not cost")
    print("\nROLLOUTS")
    row("L (layouts/base)", ROLL.layouts_per_base, ROLL.layout_balance)
    row("K (reps/layout)", ROLL.reps_per_layout, "optimal at 1")
    row("layout pool", "0-23", "all 24 -- the RL reservation does not apply")
    row("eval episodes", episodes_eval(), "= n x L x K")
    row("gt_n (bank standard)", ROLL.gt_n, f"+/-{per_phrase_ci_pp(ROLL.gt_n):.0f}pp per phrase")
    row("gt subset/iter", ROLL.gt_subset_per_iter, f"= {episodes_gt()} episodes, banked")
    print("\nCOST PER ITERATION")
    row("proxy scoring", f"{proxy_hours():.2f} GPU-h")
    row("rollouts", f"{rollout_hours():.2f} GPU-h",
        f"{episodes_eval()} eval + {episodes_gt()} gt")
    row("rephraser API", f"{rephraser_hours():.2f} h", "not GPU")
    row("TOTAL", f"{hours_per_iteration():.2f} GPU-h")
    tot = hours_per_iteration() * SIZE.iterations * SIZE.rephrasers
    row("x %d iters" % SIZE.iterations, f"{tot:.0f} GPU-h",
        f"1 pod {tot/24:.1f}d | 3 pods {tot/3/24:.1f}d")
    print("\nRESOLUTION (detectable rulebook difference, 95%, paired)")
    row("proxy", f"{proxy_resolution_pp():.1f} pp", f"on {n_proxy_phrases()} phrases")
    row("sim", f"{sim_resolution_pp():.1f} pp", f"on {n_sim_bases()} bases x {ROLL.layouts_per_base}")
    row("sim floor", f"{sim_resolution_floor_pp():.1f} pp", "at infinite layouts")
    print("     reference: A28 measured v3->v4 at +1.0 qwen / +3.2 gemini / +5.2 claude pp")
    print("\nPER-PHRASE PRECISION (why gt_n is not L)")
    for n in (ROLL.layouts_per_base, 12, 24, ROLL.gt_n, 64):
        row(f"n={n} rollouts", f"+/-{per_phrase_ci_pp(n):.1f} pp")
    print("\nKNOWN ISSUES")
    for i, s in enumerate(BANK.KNOWN_ISSUES, 1):
        print(f"  {i}. {s}")
    print("\nDECISION LOG")
    for date, what, why in DECISIONS:
        print(f"  {date}  {what}\n              {why}")
    print()


if __name__ == "__main__":
    main()
