#!/usr/bin/env python3
"""Round-1 NATURAL test phrases for the pi0.5/LIBERO sealed set v2 (FINAL_EVAL=1).

10 phrases per sealed task (20 tasks -> 200), written by gemini-pro-latest at
temperature 1.0 with the task's init-0 agent-view frame attached. The prompt is
paper/prompts/prompt_generate_rephrases.txt with its ADVERSARIAL section and
reply line removed (NATURAL block only), plus a fixed-seed, register-balanced
block of 12 human phrases from the A39 WidowX study as STYLE examples only.

Parsing is mechanical: bullets/numbering/quotes stripped, empties, exact
duplicates (case-insensitive) and the canonical itself dropped. If fewer than
10 survive the model is re-queried once with the kept lines listed as already
written. NO semantic judge of any kind. Every kept line is preflight-printed
before anything is written. Resume-safe: tasks already complete in the output
parquet are skipped.

Usage:  FINAL_EVAL=1 .venv/bin/python scripts/gen_libero_naturals_v2.py
        FINAL_EVAL=1 .venv/bin/python scripts/gen_libero_naturals_v2.py --dry-run
            (prints the exact prompt for one task, no API call)
Output: results/analysis/pi05_bank/naturals_v2.parquet
            (task "suite:task_id", canonical, phrase, k 1..10, author)
        results/analysis/pi05_bank/naturals_v2_preflight.txt
"""
import argparse
import json
import os
import pathlib
import re
import sys
import time

import pandas as pd

if os.environ.get("FINAL_EVAL") != "1":
    raise SystemExit("FINAL_EVAL=1 required")

R = pathlib.Path(__file__).resolve().parents[1]
BANK = R / "results/analysis/pi05_bank"
TEMPLATE = R / "paper/prompts/prompt_generate_rephrases.txt"
HUMAN = R / "results/human_naturals/a39_human_phrases.parquet"
OUT = BANK / "naturals_v2.parquet"
PREFLIGHT = BANK / "naturals_v2_preflight.txt"

MODEL = "gemini-pro-latest"
N_PER_TASK = 10
N_STYLE_PER_REGISTER = 4
STYLE_SEED = 39
REGISTERS = ("adult", "kid", "robot")
COLS = ["task", "canonical", "phrase", "k", "author"]


def sealed_tasks():
    """[(task "suite:id", canonical)] for the accepted tasks of both halves."""
    draw = json.load(open(BANK / "sealed_v2_draw.json"))
    canon = json.load(open(BANK / "libero_tasks.json"))
    out = []
    for half in ("in_finetune", "out_of_finetune"):
        for key in draw["halves"][half]["accepted"]:
            suite, tid = key.split("/")
            out.append((f"{suite}:{int(tid)}", canon[suite][int(tid)]))
    return out


def frame_path(task):
    """Init-0 agent-view PNG for "suite:id": frames/, then eval_frames/, then
    the human_eval start frame. Raises if none exists."""
    suite, tid = task.split(":")
    stem = f"{suite}__{int(tid):02d}.png"
    cands = [BANK / "frames" / stem, BANK / "eval_frames" / stem,
             BANK / "human_eval/assets" / f"task_{int(tid):02d}" / "start.png"]
    for p in cands:
        if p.exists():
            return p
    raise FileNotFoundError(f"no scene frame for {task}; tried "
                            + ", ".join(str(c.relative_to(R)) for c in cands))


def natural_template():
    """Base prompt with the ADVERSARIAL section and reply line removed."""
    t = TEMPLATE.read_text()
    t, n1 = re.subn(r"## ADVERSARIAL.*?These should be HARD\.\n\n", "", t, flags=re.S)
    t, n2 = re.subn(r"\nADVERSARIAL\n<one instruction per line>\n", "", t)
    assert n1 == 1 and n2 == 1 and "ADVERSARIAL" not in t, "template drift"
    return t.rstrip("\n")


def style_examples():
    """12 human phrases, 4 per register, fixed seed -- the same block in every prompt."""
    h = pd.read_parquet(HUMAN)[["register", "phrase"]].drop_duplicates("phrase")
    parts = [h[h.register == r].sample(N_STYLE_PER_REGISTER, random_state=STYLE_SEED)
             for r in REGISTERS]
    ex = pd.concat(parts).sample(frac=1, random_state=STYLE_SEED)
    return [(r, str(p).strip()) for r, p in zip(ex.register, ex.phrase)]


def build_prompt(canonical, examples, already=()):
    """Frozen template (NATURAL block only; "two sets" -> "one set" since the
    ADVERSARIAL set is gone), with the style examples, the image note and the
    anti-repetition block inserted BEFORE the reply-format block so the reply
    spec is the last thing the model reads."""
    t = (natural_template().replace("{{instruction}}", canonical)
                           .replace("{{n_natural}}", str(N_PER_TASK))
                           .replace("Write two sets of rewordings", "Write one set of rewordings"))
    head, sep, reply = t.partition("Reply with exactly this")
    assert sep, "template drift: reply-format block not found"
    extra = ("STYLE EXAMPLES. The lines below were written by real people (adults, "
             "children, and\npeople talking to a robot) for a DIFFERENT robot -- a WidowX "
             "tabletop arm -- and\nfor different tasks. They show the register and "
             "sentence shapes people actually\nuse. Match that style; do NOT copy their "
             "objects, places, or wording. Your lines\nmust describe only the task above "
             "and the scene in the attached image.\n"
             + "\n".join(f"  ({r}) {ph}" for r, ph in examples)
             + "\n\nAn image of the robot's starting scene is attached. You may name objects "
             "the way\nthey look in it, but never contradict the instruction or mention "
             "anything you\ncannot see.")
    if already:
        extra += ("\n\nALREADY WRITTEN for this task -- do not repeat these, and do not "
                  "produce close\nvariants of them:\n"
                  + "\n".join(f"  {a}" for a in already))
    return head.rstrip("\n") + "\n\n" + extra + "\n\n" + sep + reply


def clean_line(s):
    s = s.strip()
    s = re.sub(r"^\s*(?:[-*•]+|\d+[.)]|\(\d+\))\s*", "", s)
    s = s.strip().strip('"“”\'`').strip()
    return s


def parse_natural(text):
    """Lines of the NATURAL block (stops at any ADVERSARIAL header just in case)."""
    out, kind = [], None
    for line in (text or "").splitlines():
        up = line.strip().strip("#* ").upper()
        if up.startswith("NATURAL"):
            kind = "natural"; continue
        if up.startswith("ADVERSARIAL"):
            kind = None; continue
        s = clean_line(line)
        if kind == "natural" and s:
            out.append(s)
    return out


def norm(s):
    return re.sub(r"\s+", " ", s.lower().rstrip(" .!")).strip()


def call_with_retry(client, contents, retries=5):
    from google.genai import types
    for attempt in range(retries):
        try:
            r = client.models.generate_content(
                model=MODEL, contents=contents,
                config=types.GenerateContentConfig(
                    temperature=1.0,
                    http_options=types.HttpOptions(timeout=180_000)))
            return r.text or ""
        except Exception as e:
            msg = str(e)
            if "429" in msg and any(m in msg.lower() for m in
                                    ("depleted", "prepay", "check your plan", "limit: 0")):
                raise SystemExit(f"HARD STOP (billing): {msg[:200]}")
            if attempt == retries - 1:
                raise
            m = re.search(r"retry.{0,20}?(\d+(?:\.\d+)?)\s*s", msg, re.I)
            delay = float(m.group(1)) if m else min(10 * 2 ** attempt, 60)
            print(f"  retry {attempt + 1}/{retries} in {delay:.0f}s: {msg[:120]}", flush=True)
            time.sleep(delay)


def generate_task(client, task, canonical, examples):
    from google.genai import types
    img = types.Part.from_bytes(data=frame_path(task).read_bytes(), mime_type="image/png")
    kept, seen = [], {norm(canonical)}
    for attempt in range(2):
        text = call_with_retry(client, [img, build_prompt(canonical, examples, kept)])
        for cand in parse_natural(text):
            if norm(cand) in seen:
                continue
            seen.add(norm(cand))
            kept.append(cand)
            if len(kept) >= N_PER_TASK:
                break
        if len(kept) >= N_PER_TASK:
            break
        print(f"  {task}: {len(kept)}/{N_PER_TASK} after query {attempt + 1}", flush=True)
    if len(kept) < N_PER_TASK:
        raise RuntimeError(f"{task}: only {len(kept)} phrases after re-query; not writing")
    return kept[:N_PER_TASK]


def write_preflight(df):
    lines = []
    for task, g in df.groupby("task", sort=False):
        lines.append(f"{task} | {g.canonical.iloc[0]}")
        lines += [f"  {k:2d}. {p}" for k, p in zip(g.k, g.phrase)]
        lines.append("")
    PREFLIGHT.write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="print the exact prompt for one task and exit (no API call)")
    ap.add_argument("--task", default=None,
                    help="task for --dry-run (suite:id); default first sealed task")
    ap.add_argument("--allow-partial", action="store_true",
                    help="run even if the sealed set does not have 20 accepted tasks")
    args = ap.parse_args()

    tasks = sealed_tasks()
    examples = style_examples()
    print(f"sealed v2: {len(tasks)} accepted tasks; style block: {len(examples)} human phrases")
    if len(tasks) != 20 and not args.allow_partial:
        raise SystemExit("expected 20 accepted tasks (10 + 10); pass --allow-partial to "
                         "generate for the accepted subset now (resume-safe)")

    if args.dry_run:
        task, canonical = next(((t, c) for t, c in tasks if t == args.task), tasks[0])
        print(f"--- {task} | {canonical} | frame {frame_path(task).relative_to(R)}\n")
        print(build_prompt(canonical, examples))
        return

    done = pd.read_parquet(OUT) if OUT.exists() else pd.DataFrame(columns=COLS)
    complete = {t for t, g in done.groupby("task") if len(g) >= N_PER_TASK}
    done = done[done.task.isin(complete)]
    todo = [(t, c) for t, c in tasks if t not in complete]
    print(f"{len(complete)} tasks done, {len(todo)} to generate -> {OUT.relative_to(R)}")
    for t, _ in tasks:
        frame_path(t)  # fail before any API call if a frame is missing

    from google import genai
    client = genai.Client()  # GEMINI_API_KEY from the environment
    for task, canonical in todo:
        phrases = generate_task(client, task, canonical, examples)
        for k, p in enumerate(phrases, 1):
            print(f"PREFLIGHT {task} {k:2d} | {p}", flush=True)
        new = pd.DataFrame([dict(task=task, canonical=canonical, phrase=p, k=k, author=MODEL)
                            for k, p in enumerate(phrases, 1)])
        done = pd.concat([done, new], ignore_index=True)[COLS]
        done.to_parquet(OUT, index=False)  # checkpoint per task
        write_preflight(done)
    print(f"naturals_v2: {len(done)} phrases over {done.task.nunique()} tasks -> {OUT.relative_to(R)}")


if __name__ == "__main__":
    main()
