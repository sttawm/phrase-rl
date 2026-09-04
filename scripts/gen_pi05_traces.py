#!/usr/bin/env python3
"""Per-base scene traces for the pi0.5/LIBERO bank — the bridge recipe
(CoVer USER_TEMPLATE, gemini-3.5-flash, temp 0.8, image + base phrase) with the
leak lesson promoted to a hard gate: the canonical instruction NEVER enters the
prompt, and a trace that contains the canonical string (case-insensitive) is
regenerated, then dropped if it persists.

Traces are generated for every bank (task, phrase) pair on train tasks plus the
15 val canonicals. Sealed-test tasks are never touched (asserted).

  GEMINI_API_KEY=... .venv/bin/python scripts/gen_pi05_traces.py --go
  -> results/phrase_artifacts/traces_pi05_v1.parquet
"""
import argparse
import concurrent.futures as cf
import hashlib
import json
import pathlib
import re
import sys
import threading
import time

import pandas as pd

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
from phrase_rl.cover_prompt import USER_TEMPLATE  # noqa: E402

D = REPO / "results/analysis/pi05_bank"
FRAMES_DIRS = [D / "eval_frames", D / "test_frames", D / "frames"]
FRAMES = D / "frames"
OUT = REPO / "results/phrase_artifacts/traces_pi05_v1.parquet"
OUT_EVAL = REPO / "results/phrase_artifacts/traces_pi05_eval.parquet"
MODEL = "gemini-3.5-flash"


def sanitize(text):
    i = str(text).rfind("Original Instruction:")
    return str(text)[:i].rstrip() if i >= 0 else str(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--go", action="store_true")
    ap.add_argument("--preflight", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--phrases", default=None,
                    help="explicit parquet of (task, phrase[, canonical]) to trace; "
                         "used for the sealed eval set instead of the bank")
    args = ap.parse_args()
    if not (args.go or args.preflight):
        ap.error("pass --preflight or --go")
    global OUT
    if args.phrases:
        OUT = OUT_EVAL

    if args.phrases:
        # sealed-eval mode: trace exactly the supplied (task, phrase) pairs.
        import pandas as _pd
        todo = _pd.read_parquet(args.phrases)
        ev = json.load(open(D / "eval_set.json"))["tasks"]
        if "canonical" not in todo:
            todo["canonical"] = todo.task.map(ev)
        todo["suite"] = todo.task.str.rsplit(":", n=1).str[0]
        todo["task_id"] = todo.task.str.rsplit(":", n=1).str[1].astype(int)
        todo = todo[["suite", "task_id", "canonical", "phrase"]].drop_duplicates(
            ["suite", "task_id", "phrase"]).reset_index(drop=True)
        globals()["_PRESET_TODO"] = todo
    splits = json.load(open(D / "splits.json"))
    test_keys = {(t["suite"], t["task_id"]) for t in splits["sealed_test"]}
    bank = pd.read_parquet(D / "bank.parquet")
    vals = pd.read_parquet(D / "val_canonicals.parquet")
    todo = globals().get("_PRESET_TODO")
    if todo is None:
      todo = pd.concat([
        bank[["suite", "task_id", "canonical", "phrase"]],
        vals[["suite", "task_id", "canonical", "phrase"]],
      ]).drop_duplicates(["suite", "task_id", "phrase"]).reset_index(drop=True)
    if args.phrases:
        # sealed-eval mode: tracing sealed tasks IS the point, but only under the
        # FINAL_EVAL gate that every other sealed generator honours.
        import os as _os
        if _os.environ.get("FINAL_EVAL") != "1":
            raise SystemExit("--phrases traces sealed tasks; FINAL_EVAL=1 required")
    else:
        assert not any((s, t) in test_keys for s, t in zip(todo.suite, todo.task_id)), \
            "sealed-test tasks in trace plan!"

    have = set()
    if OUT.exists():
        old = pd.read_parquet(OUT)
        have = {(r.suite, r.task_id, r.phrase) for r in old.itertuples()}
    todo = todo[[(r.suite, r.task_id, r.phrase) not in have
                 for r in todo.itertuples()]].reset_index(drop=True)
    if args.limit:
        todo = todo.head(args.limit)
    print(f"{len(todo)} traces to generate "
          f"({todo.groupby(['suite','task_id']).ngroups} tasks); model {MODEL}; "
          f"prompt sha {hashlib.sha1(USER_TEMPLATE.encode()).hexdigest()[:12]}")
    def _frame_path(s, t):
        for fd in FRAMES_DIRS:
            p = fd / f"{s}__{t:02d}.png"
            if p.exists(): return p
        return None
    missing = sorted({(s, t) for s, t in zip(todo.suite, todo.task_id)
                      if _frame_path(s, t) is None})
    if missing:
        raise SystemExit(f"missing frames for {len(missing)} tasks, e.g. {missing[:3]}")
    if args.preflight:
        return

    import os
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    imgs = {}
    for s, t in {(s, t) for s, t in zip(todo.suite, todo.task_id)}:
        imgs[(s, t)] = _frame_path(s, t).read_bytes()

    lock = threading.Lock()
    rows, dropped = [], []

    def one(r):
        prompt = USER_TEMPLATE.format(instruction=r.phrase, batch_number=16)
        part = types.Part.from_bytes(data=imgs[(r.suite, r.task_id)],
                                     mime_type="image/png")
        canon_l = str(r.canonical).strip().lower()
        base_l = str(r.phrase).strip().lower()
        for attempt in range(4):
            try:
                out = client.models.generate_content(
                    model=MODEL, contents=[part, prompt],
                    config=types.GenerateContentConfig(
                        temperature=0.8, max_output_tokens=2000)).text or ""
            except Exception:
                time.sleep(3 * (attempt + 1))
                continue
            tr = sanitize(out).strip()
            # leak gate: the canonical must not appear verbatim in the trace
            # (unless the base IS the canonical, where it is legitimate)
            if tr and (base_l == canon_l or canon_l not in tr.lower()):
                with lock:
                    rows.append({"suite": r.suite, "task_id": r.task_id,
                                 "canonical": r.canonical, "phrase": r.phrase,
                                 "trace": tr, "model": MODEL})
                return
            time.sleep(1)
        with lock:
            dropped.append((r.suite, r.task_id, r.phrase))

    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        done = 0
        for _ in ex.map(one, todo.itertuples()):
            done += 1
            if done % 100 == 0:
                print(f"  {done}/{len(todo)}", flush=True)
                with lock:
                    df = pd.DataFrame(rows)
                if OUT.exists():
                    df = pd.concat([pd.read_parquet(OUT), df]).drop_duplicates(
                        ["suite", "task_id", "phrase"])
                df.to_parquet(OUT, index=False)

    df = pd.DataFrame(rows)
    if OUT.exists():
        df = pd.concat([pd.read_parquet(OUT), df]).drop_duplicates(
            ["suite", "task_id", "phrase"])
    df.to_parquet(OUT, index=False)
    # final leak audit over the whole table
    leaks = sum(1 for r in df.itertuples()
                if str(r.phrase).strip().lower() != str(r.canonical).strip().lower()
                and str(r.canonical).strip().lower() in str(r.trace).lower())
    print(f"traces: {len(df)} rows; dropped {len(dropped)} persistent leakers; "
          f"final leak audit: {leaks} (must be 0)")
    assert leaks == 0


if __name__ == "__main__":
    main()
