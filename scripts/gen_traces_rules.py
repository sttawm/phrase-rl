#!/usr/bin/env python3
"""Per-BASE scene traces for the rules loop.

WHY: every one of the 2,000 traces in cover35_teacher_train.parquet was
generated from the ORIGINAL instruction. Handing one to a natural or adversarial
base leaks the canonical vocabulary into the very input the rulebook is supposed
to repair -- and the CoVer template's own
`<Meaning of the instruction in the context of the image>` section states that
meaning explicitly. These traces are generated FROM THE BASE PHRASE instead, so
an adversarial base gets the scene read through its own wording.

RECIPE matches build_probe8.stage_traces exactly (CoVer USER_TEMPLATE,
gemini-3.5-flash, temp 0.8, image+text), which is how cover35 was made -- so old
and new traces are one grade.

KEY is (task, phrase, episode, t). A trace describes a specific FRAME, so
(task, phrase) alone would collide across frames. Carrying the frame also lets
this one table serve RL later, which needs per-context traces, with no schema
change.

  export GEMINI_API_KEY=...            # never echoed
  .venv/bin/python scripts/gen_traces_rules.py --preflight     # print, spend nothing
  .venv/bin/python scripts/gen_traces_rules.py --go
"""
import argparse
import concurrent.futures as cf
import hashlib
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

OUT = REPO / "results/phrase_artifacts/traces_rules_v1.parquet"
MODEL = "gemini-3.5-flash"          # matches cover35_teacher_train
SHARD_EVERY = 100

ap = argparse.ArgumentParser()
ap.add_argument("--go", action="store_true", help="actually call the API")
ap.add_argument("--preflight", action="store_true", help="print the plan, spend nothing")
ap.add_argument("--splits", nargs="+", default=["train", "train_held", "val8"])
ap.add_argument("--run", default="live1", help="rules run whose splits.json defines the sets")
ap.add_argument("--workers", type=int, default=6)
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--frames-val8", default="results/phrase_artifacts/val8_canonical_frames.parquet")
args = ap.parse_args()
if not (args.go or args.preflight):
    ap.error("pass --preflight or --go")

sys.path.insert(0, str(REPO))
import config.params as P


# --------------------------------------------------------------- base sets
def load_bases():
    import json
    sp = json.load(open(REPO / f"results/rules_runs/{args.run}/splits.json"))
    bank = pd.read_parquet(REPO / "results/analysis/bank_to_score.parquet")
    bank["src"] = bank.source.str.split("|").str[0]
    gen = pd.read_parquet(REPO / "results/analysis/bank_generated.parquet")

    def pick(tasks, n, label):
        out = []
        for t in tasks:
            g = gen[gen.task == t]
            nat = g[g.kind == "natural"].phrase.drop_duplicates().tolist()
            adv = g[g.kind == "adversarial"].phrase.drop_duplicates().tolist()
            # orig tier = fine_exam on sim, search_boards/originals on native
            orig = bank[(bank.task == t) & (bank.src.isin(["fine_exam", "search_boards"]))]
            orig = orig.phrase.drop_duplicates().tolist()
            n_o, n_n, n_a = [max(1, round(n * f)) for f in P.SIZE.orig_nat_adv]
            for tier, pool, k in (("orig", orig, n_o), ("natural", nat, n_n),
                                  ("adversarial", adv, n_a)):
                for ph in pool[:k]:
                    out.append({"task": t, "phrase": ph, "split": label, "tier": tier})
        return pd.DataFrame(out)

    frames = []
    if "train" in args.splits:
        frames.append(pick(sp["train"], P.SIZE.n_train, "train"))
    if "train_held" in args.splits:
        frames.append(pick(sp["val_held"], P.SIZE.n_train_held, "train_held"))
    if "val8" in args.splits:
        frames.append(pick(sp["val8"], P.SIZE.n_sim, "val8"))
    return pd.concat(frames, ignore_index=True).drop_duplicates(["task", "phrase"])


# ------------------------------------------------------- canonical frames
def load_frames(bases):
    """One canonical frame per task. Arbitrary but RECORDED -- the previous
    convention was drop_duplicates() row order, stable only until someone
    re-sorts the parquet."""
    need = set(bases.task)
    out = {}
    for f, kind in ((REPO / "data/contexts_train_multit16.parquet", "bridge"),
                    (REPO / "data/contexts_val_multit16.parquet", "bridge")):
        if not f.exists():
            continue
        d = pd.read_parquet(f, columns=["episode_index", "instruction", "t"])
        d = d[d.instruction.isin(need - set(out))].drop_duplicates("instruction")
        for r in d.itertuples():
            out[r.instruction] = (kind, str(f), int(r.episode_index), int(r.t))
    vf = REPO / args.frames_val8
    if vf.exists():
        d = pd.read_parquet(vf)
        for r in d.itertuples():
            out[r.task] = ("sim", str(vf), int(r.episode_id), int(getattr(r, "t", 0)))
    return out


bases = load_bases()
if args.limit:
    bases = bases.head(args.limit)
frames = load_frames(bases)
bases["has_frame"] = bases.task.map(lambda t: t in frames)

have = pd.read_parquet(OUT)[["task", "phrase"]] if OUT.exists() else pd.DataFrame(columns=["task", "phrase"])
todo = bases.merge(have.assign(_h=1), on=["task", "phrase"], how="left")
todo = todo[todo._h.isna() & todo.has_frame].drop(columns=["_h"])

print("=" * 74)
print("  PER-BASE TRACE GENERATION")
print("=" * 74)
print(f"  model            {MODEL}  (matches cover35_teacher_train)")
print(f"  output           {OUT.relative_to(REPO)}")
print(f"  key              (task, phrase, episode, t)")
print()
print(f"  {'split':12} {'bases':>7} {'no frame':>9} {'already':>8} {'TO GEN':>8}")
for s in bases.split.unique():
    b = bases[bases.split == s]
    t = todo[todo.split == s]
    nf = int((~b.has_frame).sum())
    print(f"  {s:12} {len(b):>7} {nf:>9} {len(b)-nf-len(t):>8} {len(t):>8}")
print(f"  {'TOTAL':12} {len(bases):>7} {int((~bases.has_frame).sum()):>9} "
      f"{len(bases)-int((~bases.has_frame).sum())-len(todo):>8} {len(todo):>8}")
print(f"\n  estimated cost   ~${len(todo)*0.05:.0f}   (at ~$0.05/call)")
miss = sorted(set(bases[~bases.has_frame].task))
if miss:
    print(f"\n  !! {len(miss)} tasks have NO frame and are SKIPPED: {miss[:6]}")
    print(f"     (val8 frames come from {args.frames_val8}, which e6 must export)")
print("\n  SAMPLE of what goes to the API:")
for r in todo.head(8).itertuples():
    k, _, ep, t = frames[r.task]
    print(f"    [{r.split}/{r.tier}] {str(r.task)[:34]:34} ep={ep} t={t} | {str(r.phrase)[:52]}")

if args.preflight:
    print("\n  PREFLIGHT ONLY -- nothing spent. Re-run with --go.")
    sys.exit(0)
if not len(todo):
    print("\n  nothing to do.")
    sys.exit(0)

# ------------------------------------------------------------- generation
from google import genai
from google.genai import types
from phrase_rl.cover_prompt import USER_TEMPLATE
from phrase_rl.gemini_redteam_assets import call_with_retry

if not os.environ.get("GEMINI_API_KEY"):
    sys.exit("GEMINI_API_KEY not set")
client = genai.Client()
PROMPT_SHA = hashlib.sha1(USER_TEMPLATE.encode()).hexdigest()[:12]

_IMG = {}


def preload_images(tasks):
    """Read each source parquet ONCE, pulling only the rows we need.

    The first version called pd.read_parquet(path) inside image_for(), i.e. once
    per task -- 187 full reads of a 3 GB file, from 6 threads at a time. It made
    the machine unusable before a single trace landed. Row-group filtering keeps
    peak memory to the frames we actually want."""
    import pyarrow.parquet as pq
    by_path = {}
    for t in tasks:
        kind, path, ep, tt = frames[t]
        by_path.setdefault((path, kind), []).append((t, ep, tt))
    for (path, kind), want in by_path.items():
        keys = {(w[0], w[1], w[2]) for w in want}
        idcol = "task" if kind == "sim" else "instruction"
        epcol = "episode_id" if kind == "sim" else "episode_index"
        pf = pq.ParquetFile(path)
        cols = [idcol, epcol, "image_png"] + ([] if kind == "sim" else ["t"])
        got = 0
        for batch in pf.iter_batches(batch_size=256, columns=cols):
            d = batch.to_pandas()
            for r in d.itertuples():
                k = (getattr(r, idcol), int(getattr(r, epcol)),
                     int(getattr(r, "t", 0)) if kind != "sim" else 0)
                if k in keys and k[0] not in _IMG:
                    _IMG[k[0]] = bytes(r.image_png)
                    got += 1
            if got >= len(keys):
                break
        print(f"  [frames] {got}/{len(keys)} from {Path(path).name}", flush=True)
    missing = [t for t in tasks if t not in _IMG]
    if missing:
        print(f"  [frames] !! {len(missing)} tasks have no image, skipped: {missing[:4]}")
    return missing


def image_for(task):
    return _IMG[task]


def one(rec):
    task, phrase = rec["task"], rec["phrase"]
    prompt = USER_TEMPLATE.format(instruction=phrase, batch_number=16)
    part = types.Part.from_bytes(data=image_for(task), mime_type="image/png")
    text = call_with_retry(client, types, MODEL, [part, prompt],
                           temperature=0.8, max_tokens=2000)
    trace = text.split("Reworded Instructions")[0].rstrip().rstrip(":").rstrip()
    trace = re.sub(r"\*\*Description of the image:?\*\*:?",
                   "<Description of the image>", trace)
    trace = re.sub(r"\*\*Meaning of the instruction in the context of the image:?\*\*:?",
                   "<Meaning of the instruction in the context of the image>", trace)
    trace = trace.replace("**", "")
    if "<Description of the image>" not in trace:
        raise RuntimeError(f"unexpected trace format for {task!r}")
    kind, _, ep, t = frames[task]
    return {**rec, "frame_kind": kind, "episode": ep, "t": t, "trace": trace,
            "model": MODEL, "prompt_sha": PROMPT_SHA,
            "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}


def flush(rows):
    if not rows:
        return
    new = pd.DataFrame(rows)
    old = pd.read_parquet(OUT) if OUT.exists() else None
    both = pd.concat([old, new], ignore_index=True) if old is not None else new
    both = both.drop_duplicates(["task", "phrase", "episode", "t"], keep="last")
    tmp = OUT.with_suffix(".tmp.parquet")
    both.to_parquet(tmp, index=False)
    back = pd.read_parquet(tmp)                       # validate before replacing
    assert len(back) == len(both), "parquet readback mismatch"
    tmp.replace(OUT)
    print(f"    [flush] {len(both)} rows -> {OUT.name}", flush=True)


print("\n  preloading frames (one pass per source file)...")
_missing = set(preload_images(sorted(set(todo.task))))
todo = todo[~todo.task.isin(_missing)]
recs = todo.drop(columns=["has_frame"]).to_dict("records")
print(f"  {len(recs)} calls after frame preload\n", flush=True)
done, buf, failed = 0, [], 0
t0 = time.time()
with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
    futs = {ex.submit(one, r): r for r in recs}
    for fut in cf.as_completed(futs):
        try:
            buf.append(fut.result())
        except Exception as e:
            failed += 1
            print(f"    [fail] {futs[fut]['task']!r}: {str(e)[:90]}", flush=True)
        done += 1
        if len(buf) >= SHARD_EVERY:
            flush(buf); buf = []
        if done % 25 == 0:
            el = time.time() - t0
            print(f"  {done}/{len(recs)}  {failed} failed  {el/60:.1f}m  "
                  f"eta {el/done*(len(recs)-done)/60:.0f}m", flush=True)
flush(buf)
print(f"\nTRACE-GEN-DONE {done - failed}/{len(recs)} in {(time.time()-t0)/60:.1f}m, {failed} failed")
