#!/usr/bin/env python3
"""LIBERO v2: one gemini scene trace per (task, phrase) for the round-1 naturals.

The A39 recipe (scripts/gen_a39_traces.py) ported to pi0.5/LIBERO: CoVer-template
prompt via phrase_rl.cover_prompt, gemini-3.5-flash, temperature 0.4, scene image
(agent view at init 0) + build_user_prompt(phrase, 1), extract_trace. The trace is
generated FROM THE INCOMING PHRASE ONLY -- the canonical instruction never enters
the prompt. Leak gate (from scripts/gen_pi05_traces.py): a trace containing the
task's canonical string case-insensitively is regenerated once; if it persists the
row is kept and flagged leak=True (never silently dropped). The gate is skipped
(leak=False, no regeneration) when the incoming phrase itself contains the
canonical -- e.g. "Please put the black bowl on the plate" -- since the trace
legitimately echoes the instruction it was given (the base==canonical exemption
in gen_pi05_traces.py, widened to substring because the naturals producer only
drops the exact canonical).

Resume-safe: existing (task, phrase) rows in the output are kept and skipped;
checkpoint every 25 traces.

  FINAL_EVAL=1 GEMINI_API_KEY=... .venv/bin/python scripts/gen_libero_traces_v2.py
  FINAL_EVAL=1 .venv/bin/python scripts/gen_libero_traces_v2.py --dry-run
      (prints the exact system + user prompt for one phrase; no API call)
  -> results/phrase_artifacts/traces_libero_v2.parquet
     (suite, task_id, task "suite:task_id", phrase, trace, leak)
"""
import argparse
import json
import os
import pathlib
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

if os.environ.get("FINAL_EVAL") != "1":
    raise SystemExit("FINAL_EVAL=1 required (sealed-set generation)")

R = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R / "src"))
sys.path.insert(0, str(R / "scripts"))
from phrase_rl.cover_prompt import build_user_prompt, extract_trace, load_system_prompt  # noqa: E402

D = R / "results/analysis/pi05_bank"
IN = D / "naturals_v2.parquet"
OUT = R / "results/phrase_artifacts/traces_libero_v2.parquet"
MODEL = "gemini-3.5-flash"
TEMPERATURE = 0.4
MAX_TOKENS = 2000
COLS = ["suite", "task_id", "task", "phrase", "trace", "leak"]


def frame_path(suite, task_id):
    """Agent-view frame at init 0 for a task, trying the frame stores in order.

    human_eval/assets/task_NN/start.png holds libero_90 scenes only (every entry
    in its manifest.json is suite=libero_90; it covers the 19 libero_90 tasks
    missing from frames/), so it is a candidate for that suite alone -- for any
    other suite it would silently return a wrong-scene frame for ids 07/08.
    """
    cands = [
        D / "frames" / f"{suite}__{task_id:02d}.png",
        D / "eval_frames" / f"{suite}__{task_id:02d}.png",
    ]
    if suite == "libero_90":
        cands.append(D / "human_eval/assets" / f"task_{task_id:02d}" / "start.png")
    for p in cands:
        if p.exists():
            return p
    raise FileNotFoundError(f"no frame for {suite}:{task_id}; tried {[str(c) for c in cands]}")


def split_task(task):
    suite, tid = task.rsplit(":", 1)
    return suite, int(tid)


def has_leak(trace, canonical):
    return canonical.strip().lower() in str(trace).lower()


def gate_exempt(phrase, canonical):
    """The phrase itself carries the canonical, so the trace echoing it is not a leak."""
    return canonical.strip().lower() in str(phrase).lower()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="print the exact system + user prompt for one phrase; no API call")
    ap.add_argument("--input", default=str(IN))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    in_p, out_p = pathlib.Path(args.input), pathlib.Path(args.out)

    canon = json.load(open(D / "libero_tasks.json"))  # {suite: [canonical, ...]} by task_id

    if args.dry_run:
        # Show exactly what the model sees. Falls back to a placeholder phrase when
        # the naturals parquet has not been produced yet.
        if in_p.exists():
            r = pd.read_parquet(in_p).iloc[0]
            task, phrase = r.task, r.phrase
            suite, tid = split_task(task)
            print(f"# task {task}  frame {frame_path(suite, tid)}")
            print(f"# canonical (gate only, NOT in prompt): {canon[suite][tid]!r}")
        else:
            phrase = "<placeholder phrase: input parquet missing>"
            print(f"# {in_p} missing; using placeholder phrase")
        print(f"# model {MODEL} temperature {TEMPERATURE} max_tokens {MAX_TOKENS}; "
              f"contents = [image/png, user]")
        print("=== SYSTEM ===")
        print(load_system_prompt())
        print("=== USER ===")
        print(build_user_prompt(phrase, 1))
        return

    ph = pd.read_parquet(in_p)[["task", "phrase"]].drop_duplicates()
    done = pd.read_parquet(out_p) if out_p.exists() else pd.DataFrame(columns=COLS)
    have = set(zip(done.task, done.phrase))
    todo = ph[[(t, p) not in have for t, p in zip(ph.task, ph.phrase)]].reset_index(drop=True)
    print(f"traces: {len(done)} done, {len(todo)} to generate; model {MODEL}")
    for r in todo.itertuples():
        print(f"  PREFLIGHT {r.task}: {r.phrase!r}")
    # resolve every frame before spending a single API call
    imgs = {}
    for task in sorted(set(todo.task)):
        suite, tid = split_task(task)
        imgs[task] = frame_path(suite, tid).read_bytes()
    if not len(todo):
        print("nothing to do")
        return

    from google import genai
    from google.genai import types
    from phrase_rl.gemini_redteam_assets import call_with_retry
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    system = load_system_prompt()

    def one(r):
        suite, tid = split_task(r.task)
        canonical = canon[suite][tid]
        img = types.Part.from_bytes(data=imgs[r.task], mime_type="image/png")
        exempt = gate_exempt(r.phrase, canonical)
        trace, leak = None, False
        for attempt in range(2):  # regenerate once on a canonical leak
            raw = call_with_retry(client, types, MODEL, [img, build_user_prompt(r.phrase, 1)],
                                  system=system, temperature=TEMPERATURE,
                                  max_tokens=MAX_TOKENS)
            trace = extract_trace(raw) or raw[:1200]
            leak = False if exempt else has_leak(trace, canonical)
            if not leak:
                break
        if exempt:
            print(f"  GATE-EXEMPT (phrase contains canonical) {r.task}: {r.phrase!r}", flush=True)
        if leak:
            print(f"  LEAK kept+flagged {r.task}: {r.phrase!r}", flush=True)
        return dict(suite=suite, task_id=tid, task=r.task, phrase=r.phrase,
                    trace=trace, leak=bool(leak))

    lock = threading.Lock()
    rows = list(done.to_dict("records"))

    def save():
        with lock:
            df = pd.DataFrame(rows, columns=COLS)
        df["task_id"] = df.task_id.astype(int)
        df["leak"] = df.leak.astype(bool)
        df.to_parquet(out_p, index=False)
        return df

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for i, rec in enumerate(ex.map(one, todo.itertuples())):
            with lock:
                rows.append(rec)
            if (i + 1) % 25 == 0:
                save()  # checkpoint
                print(f"  {i + 1}/{len(todo)}", flush=True)
    df = save()
    print(f"traces -> {out_p} {len(df)} rows; {int(df.leak.sum())} flagged leak=True")


if __name__ == "__main__":
    main()
