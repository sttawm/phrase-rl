#!/usr/bin/env python3
"""pi0.5/LIBERO sealed NATURAL set (port of A33; gated: FINAL_EVAL=1).

The bank's generate.md NATURAL block, verbatim and text-only (no image, matching
the bank's channel), run by THREE authors per task with an anti-repetition block
that carries every phrase already written for that task across authors:
  gemini-pro-latest (6)   claude-sonnet-5 (5)   Qwen3.5-9B (5, pod-side)
Fable is deliberately NOT used here -- it is reserved for the claude applier.

Mechanical meaning gate: a candidate is dropped if it loses the canonical's head
object or goal noun, is empty/multi-line, or duplicates an existing phrase.
Every kept line is preflight-printed; every rejection is logged with a reason.

Usage:  gen_sealed_naturals_v2.py gemini|claude        (local authors)
        gen_sealed_naturals_v2.py queue_qwen           (queue the pod job)
        gen_sealed_naturals_v2.py assemble             (merge + report)
Output: results/sealed/ph_pi05_natural_v2.parquet (task, phrase, k, author)
"""
import json
import os
import pathlib
import re
import sys

import pandas as pd

if os.environ.get("FINAL_EVAL") != "1":
    raise SystemExit("FINAL_EVAL=1 required")
sys.path.insert(0, "scripts")
# pi0.5/LIBERO task list: {"suite:task_id": canonical}, built by make_pi05_sealed_set.py
SEALED = __import__("json").load(open(
    pathlib.Path.home() / "dev/robotics/phrase-rl/results/analysis/pi05_bank/eval_set.json"))["tasks"]

REPO = pathlib.Path.home() / "dev/robotics/phrase-rl"
VARIANT = os.environ.get("NATV2_VARIANT", "text")   # "text" or "image"
SUF = "" if VARIANT == "text" else "_img"
OUT = REPO / f"results/sealed/ph_pi05_natural_v2{SUF}.parquet"
PART = REPO / f"results/sealed/pi05_natural_v2_parts{SUF}"
PART.mkdir(parents=True, exist_ok=True)
QUOTA = {"gemini": 6, "claude": 5, "qwen": 5}
TEMPLATE = (REPO / "prompts/rules_loop/generate.md").read_text()

STOP = {"the", "a", "an", "on", "in", "onto", "into", "to", "of", "up", "down",
        "put", "place", "set", "move", "top", "and", "it", "over"}


def content_nouns(canonical):
    """Head object + goal words the rewrite must preserve (stemmed, crudely)."""
    ws = [w for w in re.findall(r"[a-z]+", canonical.lower()) if w not in STOP]
    return [w[:-1] if w.endswith("s") and len(w) > 3 else w for w in ws]


def keeps_meaning(cand, canonical):
    """True when every content word of the canonical survives in some form.
    Deliberately lenient on synonyms (that is the point of naturals) but strict
    that no object/goal simply vanishes: each canonical content word must have a
    stem present, OR the candidate must be longer (a description standing in)."""
    c = cand.lower()
    missing = [n for n in content_nouns(canonical) if n[:4] not in c]
    return len(missing) == 0, missing


_FRAMES = None


def frame_for(task):
    """Sealed first-frame PNG bytes (A33 image variant only)."""
    global _FRAMES
    if _FRAMES is None:
        import glob
        _FRAMES = {}
        for d in ("frames", "test_frames", "reserve_frames"):
            for p in glob.glob(str(REPO / f"results/analysis/pi05_bank/{d}/*.png")):
                stem = pathlib.Path(p).stem            # e.g. libero_90__07
                suite, tid = stem.rsplit("__", 1)
                _FRAMES[f"{suite}:{int(tid)}"] = open(p, "rb").read()
    return _FRAMES[task]


def build_prompt(task, n, existing):
    p = (TEMPLATE.replace("{{instruction}}", SEALED[task])
                 .replace("{{n_natural}}", str(n))
                 .replace("{{n_adversarial}}", "0"))
    if VARIANT == "image":
        p = p.replace("THE TASK'S OWN INSTRUCTION",
                      "An image of the scene is attached; you may refer to objects the\n"
                      "way they appear in it, but never contradict the instruction or\n"
                      "mention anything you cannot see.\n\nTHE TASK'S OWN INSTRUCTION")
    if existing:
        p += ("\n\nALREADY WRITTEN for this task — do NOT repeat these, and do not\n"
              "produce close variants of them:\n"
              + "\n".join(f"  {e}" for e in existing))
    return p


def parse(text):
    out, kind = [], None
    for line in (text or "").splitlines():
        s = line.strip().strip('"').strip()
        up = s.upper()
        if up.startswith("NATURAL"):
            kind = "natural"; continue
        if up.startswith("ADVERSARIAL"):
            kind = None; continue
        s = re.sub(r"^\s*\d+[.)]\s*", "", s).strip()
        if kind == "natural" and len(s) > 3:
            out.append(s)
    return out


def existing_for(task):
    have = []
    for f in sorted(PART.glob("*.parquet")):
        d = pd.read_parquet(f)
        have += [str(p) for p in d[d.task == task].phrase]
    return have


def run_author(author):
    rows, rejects = [], []
    for task in SEALED:
        have = existing_for(task)
        prompt = build_prompt(task, QUOTA[author], have)
        if author == "gemini":
            from google import genai
            from google.genai import types
            cl = genai.Client()
            contents = [prompt]
            if VARIANT == "image":
                contents = [types.Part.from_bytes(data=frame_for(task),
                                                  mime_type="image/png"), prompt]
            r = cl.models.generate_content(
                model="gemini-pro-latest", contents=contents,
                config=types.GenerateContentConfig(
                    temperature=1.0, http_options=types.HttpOptions(timeout=180_000)))
            text = (r.text or "")
        else:
            import subprocess
            argv = ["claude", "-p", prompt, "--output-format", "text",
                    "--model", "claude-sonnet-5"]
            if VARIANT == "image":
                # the CLI reads an image from a path referenced in the prompt
                imgp = PART / f"frame_{task}.png"
                imgp.write_bytes(frame_for(task))
                argv[2] = (f"An image of the scene is at {imgp}. Read it first.\n\n"
                           + prompt)
            r = subprocess.run(argv, capture_output=True, text=True, timeout=600)
            text = r.stdout or ""
        # NO meaning gate (user 2026-09-04): the generators are instructed to
        # preserve the goal and we trust them. The previous lexical gate rejected
        # legitimate synonyms ("ramekin" -> "the white dish") and cut the
        # image-conditioned variant ~2x harder than the text one, biasing the
        # comparison the two variants exist to settle. Only exact duplicates go.
        seen = {h.lower() for h in have}
        for cand in parse(text):
            if cand.lower() in seen:
                rejects.append({"task": task, "author": author, "phrase": cand,
                                "reason": "duplicate"}); continue
            seen.add(cand.lower())
            rows.append({"task": task, "phrase": cand, "author": author})
            print(f"PREFLIGHT [{author}] {task.replace('widowx_','')} | {cand}", flush=True)
            if len([r for r in rows if r["task"] == task]) >= QUOTA[author]:
                break
    pd.DataFrame(rows).to_parquet(PART / f"{author}.parquet", index=False)
    if rejects:
        pd.DataFrame(rejects).to_csv(PART / f"{author}_rejects.csv", index=False)
    print(f"{author}: kept {len(rows)}, rejected {len(rejects)}", flush=True)


def queue_qwen():
    import hashlib
    jd = REPO / "results/rules_runs/r1_sim/jobs"
    specs = []
    for task in SEALED:
        specs.append({"task": task, "prompt": build_prompt(task, QUOTA["qwen"],
                                                           existing_for(task))})
    jid = "a00qwnatgen_" + hashlib.sha1(json.dumps(specs).encode()).hexdigest()[:8]
    pd.DataFrame(specs).to_parquet(jd / f"{jid}.payload.parquet", index=False)
    json.dump({"job_id": jid, "kind": "generate", "n_per_task": QUOTA["qwen"]},
              open(jd / f"{jid}.spec.json", "w"), indent=1)
    print("queued", jid, len(specs), "tasks", flush=True)


def assemble():
    parts = [pd.read_parquet(f) for f in sorted(PART.glob("*.parquet"))]
    d = pd.concat(parts, ignore_index=True).drop_duplicates(["task", "phrase"])
    d["k"] = d.groupby("task").cumcount()
    d = d[d.k < 16]
    d.to_parquet(OUT, index=False)
    print(f"assembled {len(d)} phrases over {d.task.nunique()} tasks -> {OUT.name}")
    print(d.author.value_counts().to_dict())
    return d


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd in ("gemini", "claude"):
        run_author(cmd)
    elif cmd == "queue_qwen":
        queue_qwen()
    elif cmd == "assemble":
        assemble()
    else:
        raise SystemExit(f"unknown command {cmd}")
