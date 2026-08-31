#!/usr/bin/env python3
"""Launch configuration for the rules loop -- the single place run parameters
live. Prints the resolved parameter table and the exact driver command;
--exec runs it.

    .venv/bin/python config/rules_run.py            # real-run params + command
    .venv/bin/python config/rules_run.py --quick    # quick-run overrides
    .venv/bin/python config/rules_run.py --exec     # print, then launch

Real-run defaults target ~1 h per iteration (user 2026-08-30): sample_n=96
sits near the noise/cost knee -- SE halves twice vs 24, while the scoring
fixed cost (per distinct task touched) stays at ~half the task pool.
"""
import argparse
import shlex
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# --- real-run parameters -----------------------------------------------------
P = dict(
    run_id="r1",

    # passes: one rulebook per applier, weakest applier first so the strongest
    # distills against the fullest bank (ALGORITHM note 2)
    rephrasers="qwen,gemini,claude",

    # reasoning stack (distiller / judge / planner / corpus): always Claude
    distiller="claude",
    claude_model="claude-opus-5",
    claude_effort="high",       # reasoning roles
    apply_effort="medium",      # rephraser=claude apply calls (stateless, parallel)
    gemini_model="gemini-pro-latest",   # rephraser=gemini apply calls
    # rephraser=qwen runs pod-side: Qwen3.5-9B (rules_loop_jobs.py, not a knob here)

    # evaluation sample: 96 bases per eval set (train / val_held / val8)
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
ENV = dict(
    RULES_APPLY_WORKERS="8",    # parallel claude/gemini apply calls
)

# --- quick-run: end-to-end shakeout with real LLMs + pod, minutes not hours --
def quick_overrides(p):
    p = dict(p)
    p.update(
        run_id="quick",
        rephrasers="gemini",    # one cheap pass exercises every stage
        claude_effort="medium",
        sample_n=12,
        patience=1,
        max_iters=2,
        max_probes=6,
        max_train_tasks=20,     # cap the pool so evidence files stay small
    )
    return p


def build(p):
    cmd = [".venv/bin/python", "scripts/rules_loop_driver.py"]
    for k, v in p.items():
        cmd += [f"--{k.replace('_', '-')}", str(v)]
    return cmd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="quick-run overrides")
    ap.add_argument("--exec", dest="run", action="store_true",
                    help="launch the driver after printing")
    ap.add_argument("--dry-run", action="store_true",
                    help="append --dry-run (plumbing only, no LLMs/pod)")
    args = ap.parse_args()

    p = quick_overrides(P) if args.quick else dict(P)
    cmd = build(p)
    if args.dry_run:
        cmd.append("--dry-run")

    mode = "QUICK RUN" if args.quick else "REAL RUN"
    print(f"# rules loop launch config -- {mode}\n")
    for k, v in p.items():
        print(f"  {k:20s} {v}")
    for k, v in ENV.items():
        print(f"  {k:20s} {v}   (env)")
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
