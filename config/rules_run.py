#!/usr/bin/env python3
"""Launch configuration for the rules loop -- the single place run parameters
live. One applier per invocation (run them separately for qwen, gemini,
claude); all invocations share run_id "r1", so they share one bank and later
appliers inherit earlier passes' measurements (ALGORITHM note 2: run the
strongest applier last).

    .venv/bin/python config/rules_run.py claude             # print real-run params
    .venv/bin/python config/rules_run.py gemini --quick     # quick-run overrides
    .venv/bin/python config/rules_run.py qwen --exec        # print, then launch
"""
import argparse
import shlex
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# --- real-run parameters (applier-independent) -------------------------------
P = dict(
    run_id="r1",

    # reasoning stack (distiller / judge / planner / corpus): always Claude
    distiller="claude",
    claude_model="claude-fable-5",
    claude_effort="max",        # reasoning roles

    # evaluation sample: 96 bases per eval set (train / val_held / val8);
    # targets ~1 h per iteration
    sample_n=96,

    # loop control
    patience=3,
    max_iters=12,
    max_probes=20,

    # scoring operating point -- matches the calibration's aggregation
    score_budget=64,            # target F*C forward passes per phrase
    contexts_per_task=16,       # C
    frames_per_episode=4,       # F floor (saturates at 4-5, measured)
    min_eps_per_task=8,         # task admission floor

    seed=7,
)

# --- per-applier knobs -------------------------------------------------------
APPLIERS = {
    # rephraser=claude applies with claude_model; effort is its lever
    "claude": dict(apply_effort="max"),
    # rephraser=gemini applies via the API; thinking budget is its lever
    "gemini": dict(gemini_model="gemini-pro-latest", gemini_thinking_budget=0),
    # rephraser=qwen runs pod-side: Qwen3.5-9B greedy (rules_loop_jobs.py); no knob here
    "qwen": dict(),
}

ENV = dict(
    RULES_APPLY_WORKERS="8",    # parallel claude/gemini apply calls
)

# --- phase: train (proxy on the training corpus) vs sim (rollouts) -----------
# The plan (user 2026-08-30): converge on training data first, then re-run on
# the sim tasks with rules seeded from the training run. The sim phase has no
# validation data -- overfitting is bounded by max_iters (2-3), not a val split.
def sim_overrides(p):
    p = dict(p)
    p.update(
        run_id=p["run_id"] + "_sim",
        phase="sim",
        init_rules_from=p["run_id"],   # seed from the training-phase rulebook
        max_iters=3,                   # "run the loop twice or three times"
        patience=99,                   # unused: no validation to stop on
        sample_n=16,                   # 16 bases x 18 episodes = 288 rollouts/eval
        max_probes=6,                  # probes cost rollouts here
    )
    return p


# --- quick-run: end-to-end shakeout with real LLMs + pod, minutes not hours --
def quick_overrides(p, applier):
    p = dict(p)
    p.update(
        run_id="quick",
        claude_model="claude-sonnet-5",
        claude_effort="medium",
        sample_n=12,
        patience=1,
        max_iters=2,
        max_probes=6,
        max_train_tasks=20,     # cap the pool so evidence files stay small
    )
    if applier == "claude":
        p["apply_effort"] = "medium"
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("applier", choices=sorted(APPLIERS))
    ap.add_argument("--phase", default="train", choices=["train", "sim"],
                    help="train: proxy-scored on the training corpus. "
                         "sim: rollout-scored on the sim tasks, rules seeded "
                         "from the training run, no validation split")
    ap.add_argument("--env", default="bridge_pi0",
                    help="which (policy, dataset) pair (see driver ENVIRONMENTS)")
    ap.add_argument("--quick", action="store_true", help="quick-run overrides")
    ap.add_argument("--exec", dest="run", action="store_true",
                    help="launch the driver after printing")
    ap.add_argument("--dry-run", action="store_true",
                    help="append --dry-run (plumbing only, no LLMs/pod)")
    args = ap.parse_args()

    p = {**P, **APPLIERS[args.applier], "rephrasers": args.applier,
         "env": args.env, "phase": "train"}
    if args.quick:
        p = quick_overrides(p, args.applier)
    if args.phase == "sim":
        p = sim_overrides(p)

    cmd = [".venv/bin/python", "scripts/rules_loop_driver.py"]
    for k, v in p.items():
        cmd += [f"--{k.replace('_', '-')}", str(v)]
    if args.dry_run:
        cmd.append("--dry-run")

    mode = "QUICK RUN" if args.quick else "REAL RUN"
    print(f"# rules loop launch config -- {mode} -- applier: {args.applier} "
          f"-- phase: {args.phase} -- env: {args.env}\n")
    for k, v in p.items():
        print(f"  {k:22s} {v}")
    for k, v in ENV.items():
        print(f"  {k:22s} {v}   (env)")
    print("\n  " + " ".join(shlex.quote(c) for c in cmd) + "\n")
    if not args.run:
        print("(--exec to launch; a pod worker must be consuming jobs unless --dry-run)")
        return
    import os
    sys.stdout.flush()
    env = {**os.environ, **ENV}
    sys.exit(subprocess.call(cmd, cwd=REPO, env=env))


if __name__ == "__main__":
    main()
