#!/usr/bin/env python3
"""Apply stage of the pi0.5/LIBERO sealed grid.

Twelve arms = {gemini, claude, qwen} x {none, in_only_v1, ood_only_v1,
in_plus_ood_v2}, each rewriting every base in eval_bases.parquet under the
loop's own apply.md (rulebook + per-base trace + the base phrase). "none" is the
no-rules ANCHOR: the same rephraser with an EMPTY rulebook, not the un-rephrased
base -- the un-rephrased bases are a separate baseline row and are never
produced here.

  FINAL_EVAL=1 .venv/bin/python scripts/pi05_eval_apply.py --applier gemini
  FINAL_EVAL=1 .venv/bin/python scripts/pi05_eval_apply.py --applier claude
  FINAL_EVAL=1 .venv/bin/python scripts/pi05_eval_apply.py --applier qwen --queue-only
-> results/analysis/pi05_bank/eval_applies/{applier}__{book}.parquet
"""
import argparse
import concurrent.futures as cf
import hashlib
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
BOOKS = {"none": None,
         "in_only_v1": D / "rulebooks/in_only_v1.md",
         "ood_only_v1": D / "rulebooks/ood_only_v1.md",
         "in_plus_ood_v2": D / "rulebooks/in_plus_ood_v2.md"}
OUTD = D / "eval_applies"
OUTD.mkdir(parents=True, exist_ok=True)
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
            print(f"  gemini retry {attempt}: {type(e).__name__}", flush=True)
        time.sleep(4 * (attempt + 1))
    return ""


def call_claude(prompt):
    for attempt in range(3):
        r = subprocess.run(["claude", "-p", prompt, "--output-format", "text",
                            "--model", "claude-opus-5", "--effort", "high"],
                           capture_output=True, text=True, timeout=900)
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout.strip()
        print(f"  claude retry {attempt}: rc={r.returncode}", flush=True)
        time.sleep(5)
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--applier", required=True, choices=["gemini", "claude", "qwen"])
    ap.add_argument("--books", default=",".join(BOOKS))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--queue-only", action="store_true")
    a = ap.parse_args()

    bases = pd.read_parquet(D / "eval_bases.parquet")
    tr = pd.read_parquet(REPO / "results/phrase_artifacts/traces_pi05_eval.parquet")
    tr["task"] = tr.suite + ":" + tr.task_id.astype(str)
    traces = {(r.task, r.phrase): r.trace for r in tr.itertuples()}
    missing = [(r.task, r.phrase) for r in bases.itertuples()
               if (r.task, r.phrase) not in traces]
    if missing:
        raise SystemExit(f"{len(missing)} bases have no trace, e.g. {missing[:2]}")

    for book in a.books.split(","):
        out_p = OUTD / f"{a.applier}__{book}.parquet"
        if out_p.exists():
            print(f"skip {out_p.name} (exists)", flush=True); continue
        rules = "" if BOOKS[book] is None else rules_only(BOOKS[book].read_text())
        jobs = [(r.task, r.phrase, r.kind, r.stratum,
                 prompt_for(rules, traces[(r.task, r.phrase)], r.phrase))
                for r in bases.itertuples()]

        if a.applier == "qwen":
            jd = REPO / "results/rules_runs/p_eval/jobs"
            jd.mkdir(parents=True, exist_ok=True)
            jid = f"a00qapply_{book}_" + hashlib.sha1(
                (book + rules).encode()).hexdigest()[:8]
            pd.DataFrame([{"task": t, "phrase": p, "kind": k, "stratum": s,
                           "prompt": pr} for t, p, k, s, pr in jobs]).to_parquet(
                jd / f"{jid}.payload.parquet", index=False)
            print(f"queued {jid} ({len(jobs)} bases)", flush=True)
            continue

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
