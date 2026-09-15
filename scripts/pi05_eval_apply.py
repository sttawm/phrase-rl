#!/usr/bin/env python3
"""Apply stage of the pi0.5/LIBERO sealed grid (and round-1 reuse).

Arms = {gemini, claude} x {none, in_only_v1, ood_only_v1, in_plus_ood_v2},
each rewriting every base in eval_bases.parquet under the loop's own apply.md
(rulebook + per-base trace + the base phrase). "none" is the no-rules ANCHOR:
the same rephraser with an EMPTY rulebook, not the un-rephrased base -- the
un-rephrased bases are a separate baseline row and are never produced here.

  FINAL_EVAL=1 .venv/bin/python scripts/pi05_eval_apply.py --applier gemini
  FINAL_EVAL=1 .venv/bin/python scripts/pi05_eval_apply.py --applier claude
-> results/analysis/pi05_bank/eval_applies/{applier}__{book}.parquet

The qwen arm is NOT produced here: --applier qwen exits with an error. The
p_eval job queue only discovers <jid>.spec.json files (rules_loop_worker.sh)
and its kind=apply handler (rules_loop_jobs.py) rebuilds the prompt from the
spec's rules_text and the BRIDGE trace parquets, so a payload built from
--traces would never reach Qwen. Queue qwen applies with a script that writes a
spec (see scripts/gen_a39_applies.py) once the handler reads payload prompts.

Round-1 reuse: point --bases / --traces / --outdir elsewhere and replace the
rulebook set with --book-files ("none" stays available as the empty rulebook):

  FINAL_EVAL=1 .venv/bin/python scripts/pi05_eval_apply.py --applier gemini \\
      --bases results/analysis/pi05_bank/r1/bases.parquet \\
      --traces results/analysis/pi05_bank/r1/traces.parquet \\
      --outdir results/analysis/pi05_bank/r1/applies \\
      --book-files r1_v1=results/analysis/pi05_bank/r1/rulebooks/r1_v1.md
-> <outdir>/{applier}__{book}.parquet
"""
import argparse
import concurrent.futures as cf
import os
import pathlib
import re
import subprocess
import sys
import time

import pandas as pd

if os.environ.get("FINAL_EVAL") != "1":
    raise SystemExit("FINAL_EVAL=1 required (sealed grid)")

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
D = REPO / "results/analysis/pi05_bank"
# Sealed-grid defaults; --book-files replaces this dict wholesale ("none" is
# always re-added).
BOOKS = {"none": None,
         "in_only_v1": D / "rulebooks/in_only_v1.md",
         "ood_only_v1": D / "rulebooks/ood_only_v1.md",
         "in_plus_ood_v2": D / "rulebooks/in_plus_ood_v2.md"}
DEFAULT_BASES = D / "eval_bases.parquet"
DEFAULT_TRACES = REPO / "results/phrase_artifacts/traces_pi05_eval.parquet"
DEFAULT_OUTDIR = D / "eval_applies"
APPLY_MD = (REPO / "prompts/rules_loop/apply.md").read_text()


def rules_only(text):
    """RULES section only -- the rationale is written for us, not the applier."""
    m = re.search(r"^===\s*RULES\s*===\s*$(.*?)(?=^===|\Z)", text, re.M | re.S)
    return m.group(1).strip() if m else text.split("===RATIONALE===")[0].strip()


def prompt_for(rules, trace, phrase):
    t = APPLY_MD
    for k, v in (("rules", rules), ("trace", trace), ("phrase", phrase)):
        t = t.replace("{{" + k + "}}", str(v))
    left = sorted(set(re.findall(r"\{\{([a-z_]+)\}\}", t)))
    if left:
        raise KeyError(f"apply.md: unsubstituted {left}")
    return t


def parse_book_files(spec):
    """'name=path,name=path' -> {name: Path}; 'none' (empty rulebook) is always
    present and may not be rebound to a file."""
    books = {"none": None}
    for item in filter(None, (s.strip() for s in spec.split(","))):
        if "=" not in item:
            raise SystemExit(f"--book-files: expected name=path, got {item!r}")
        name, path = item.split("=", 1)
        name = name.strip()
        if name == "none":
            raise SystemExit("--book-files: 'none' is reserved for the empty rulebook")
        p = pathlib.Path(path.strip())
        if not p.is_absolute():
            p = REPO / p
        if not p.exists():
            raise SystemExit(f"--book-files: {name}: {p} does not exist")
        books[name] = p
    return books


def load_bases(path):
    bases = pd.read_parquet(path)
    for col in ("task", "phrase"):
        if col not in bases.columns:
            raise SystemExit(f"{path}: missing column {col!r}")
    if "kind" not in bases.columns:
        bases["kind"] = "natural"
    if "stratum" not in bases.columns:
        bases["stratum"] = ""
    return bases


def load_traces(path):
    tr = pd.read_parquet(path)
    if "task" not in tr.columns:
        tr["task"] = tr.suite + ":" + tr.task_id.astype(str)
    return {(r.task, r.phrase): r.trace for r in tr.itertuples()}


def call_gemini(prompt):
    from google import genai
    from google.genai import types
    cl = call_gemini.cl = getattr(call_gemini, "cl", None) or genai.Client()
    for attempt in range(5):
        try:
            # thinking_budget pinned to 16384, matching the bridge applier
            # (r1_sim config.json gemini_thinking_budget, decision 48b0384c).
            # Sending nothing here left pi0.5 on the MODEL DEFAULT, so the two
            # benchmarks' gemini arms were not the same instrument. pro-class
            # models are thinking-only and reject an explicit 0 (rules_loop_driver
            # :472-476); minimum explicit budget on pro is 128.
            r = cl.models.generate_content(
                model="gemini-pro-latest", contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=1.0,
                    thinking_config=types.ThinkingConfig(thinking_budget=16384),
                    http_options=types.HttpOptions(timeout=180_000)))
            if (r.text or "").strip():
                return r.text.strip()
        except Exception as e:
            msg = str(e)
            if "429" in msg and any(m in msg.lower() for m in
                                    ("spending cap", "depleted", "prepay", "check your plan", "limit: 0")):
                raise SystemExit(f"HARD STOP (billing): {msg[:200]}")
            print(f"  gemini retry {attempt}: {type(e).__name__} {msg[:120]}", flush=True)
        time.sleep(4 * (attempt + 1))
    return ""


CLAUDE_MODEL = os.environ.get("APPLY_CLAUDE_MODEL", "claude-opus-5")
CLAUDE_EFFORT = os.environ.get("APPLY_CLAUDE_EFFORT", "high")


def call_claude(prompt):
    for attempt in range(3):
        r = subprocess.run(["claude", "-p", prompt, "--output-format", "text",
                            "--model", CLAUDE_MODEL, "--effort", CLAUDE_EFFORT],
                           capture_output=True, text=True, timeout=900)
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout.strip()
        print(f"  claude retry {attempt}: rc={r.returncode}", flush=True)
        time.sleep(5)
    return ""


def main():
    global CLAUDE_MODEL, CLAUDE_EFFORT
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--applier", required=True, choices=["gemini", "claude", "qwen"],
                    help="qwen is refused (see module docstring)")
    ap.add_argument("--books", default=None,
                    help="comma-separated subset of book names (default: all)")
    ap.add_argument("--book-files", default=None,
                    help="name=path,... rulebooks; REPLACES the built-in set "
                         "('none' = empty rulebook is always available)")
    ap.add_argument("--bases", default=str(DEFAULT_BASES),
                    help="parquet with task ('suite:task_id'), phrase "
                         "[, kind, stratum]")
    ap.add_argument("--traces", default=str(DEFAULT_TRACES),
                    help="parquet with phrase, trace and task or suite+task_id")
    ap.add_argument("--outdir", default=str(DEFAULT_OUTDIR),
                    help="writes <outdir>/<applier>__<book>.parquet")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--claude-model", default=CLAUDE_MODEL, help="claude -p model (default claude-opus-5)")
    ap.add_argument("--claude-effort", default=CLAUDE_EFFORT)
    a = ap.parse_args()
    CLAUDE_MODEL, CLAUDE_EFFORT = a.claude_model, a.claude_effort
    if a.applier == "qwen":
        raise SystemExit("qwen apply not supported by this queue: the p_eval "
                         "worker only claims *.spec.json jobs and its apply "
                         "handler ignores payload prompts (see docstring)")

    books = parse_book_files(a.book_files) if a.book_files else BOOKS
    names = ([b.strip() for b in a.books.split(",") if b.strip()]
             if a.books else list(books))
    unknown = [b for b in names if b not in books]
    if unknown:
        raise SystemExit(f"unknown books {unknown}; available: {list(books)}")
    outd = pathlib.Path(a.outdir)
    outd.mkdir(parents=True, exist_ok=True)

    bases = load_bases(a.bases)
    traces = load_traces(a.traces)
    missing = [(r.task, r.phrase) for r in bases.itertuples()
               if (r.task, r.phrase) not in traces]
    if missing:
        raise SystemExit(f"{len(missing)} bases have no trace, e.g. {missing[:2]}")
    print(f"bases={a.bases} traces={a.traces} outdir={outd} books={names}",
          flush=True)
    for r in bases.itertuples():
        print(f"  PREFLIGHT {r.task}/{r.kind}: {r.phrase!r}", flush=True)

    for book in names:
        out_p = outd / f"{a.applier}__{book}.parquet"
        if out_p.exists():
            print(f"skip {out_p.name} (exists)", flush=True); continue
        rules = "" if books[book] is None else rules_only(books[book].read_text())
        jobs = [(r.task, r.phrase, r.kind, r.stratum,
                 prompt_for(rules, traces[(r.task, r.phrase)], r.phrase))
                for r in bases.itertuples()]

        fn = call_gemini if a.applier == "gemini" else call_claude
        rows = [None] * len(jobs)

        def one(i):
            t, p, k, s, pr = jobs[i]
            out = fn(pr)
            return i, {"task": t, "phrase": p, "kind": k, "stratum": s,
                       "rewrite": out.split("\n")[0].strip(),
                       "applier": a.applier, "book": book}

        with cf.ThreadPoolExecutor(max_workers=a.workers) as ex:
            done = 0
            for i, rec in ex.map(one, range(len(jobs))):
                rows[i] = rec
                done += 1
                if done % 25 == 0:
                    print(f"  {a.applier}/{book} {done}/{len(jobs)}", flush=True)
        d = pd.DataFrame(rows)
        d.to_parquet(out_p, index=False)
        n_empty = int((d.rewrite.str.len() == 0).sum())
        n_same = int((d.rewrite.str.strip().str.lower()
                      == d.phrase.str.strip().str.lower()).sum())
        print(f"{out_p.name}: {len(d)} rewrites, {n_same} unchanged, "
              f"{n_empty} EMPTY", flush=True)


if __name__ == "__main__":
    main()
