#!/usr/bin/env python3
"""Driver for ALGORITHM-RULES-SEARCH-AND-DISTILLATION.md.

Local orchestrator: owns the run directory, the scored bank, splits, the
iteration state machine, early stopping, and verbatim logging of every LLM
exchange. GPU work (proxy scoring, Qwen rule application) is submitted as jobs
through git; a pod running scripts/rules_loop_worker.sh consumes them and
commits results back. LLM calls go through pluggable backends:

    distiller / judge : `claude -p` (headless CLI, no API key needed; model set by
                        --claude-model, default claude-opus-5)
    rephraser=claude  : `claude -p`
    rephraser=gemini  : google-genai (GEMINI_API_KEY)
    rephraser=qwen    : pod job (kind=apply), Qwen3.5-9B on a worker

Reproducibility: config.json freezes seeds, sample sizes, CRN context ids, and
prompt hashes at run start. Every artifact lives under
results/rules_runs/<run_id>/ -- prompts and responses verbatim, one directory
per iteration. Resume = rerun the same command; every step skips itself if its
artifact already exists.

Smoke test (no pods, no LLMs, seconds):
    .venv/bin/python scripts/rules_loop_driver.py --run-id smoke --dry-run \
        --rephrasers gemini --patience 2 --max-iters 3

Real round 1:
    .venv/bin/python scripts/rules_loop_driver.py --run-id r1 \
        --rephrasers qwen,claude,gemini --patience 3
"""
import argparse
import concurrent.futures as cf
import hashlib
import itertools
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
PROMPTS = REPO / "prompts" / "rules_loop"
APPLY_WORKERS = int(os.environ.get("RULES_APPLY_WORKERS", "8"))

# --- frozen constants -------------------------------------------------------
SEALED_STEMS = [
    "carrot_on_ramekin", "carrot_on_sponge", "coke_can_on_keyboard",
    "coke_can_on_wheel", "cube_on_plate", "eggplant_on_keyboard",
    "eggplant_on_sponge", "nut_on_plate", "nut_on_wheel",
    "orange_juice_on_plate", "pepsi_on_plate", "small_plate_on_green_cube",
]
VAL8_TASKS = [
    "widowx_spoon_on_towel", "widowx_carrot_on_plate", "widowx_stack_cube",
    "widowx_put_eggplant_in_basket", "widowx_carrot_on_keyboard_clean",
    "widowx_carrot_on_wheel_clean", "widowx_coke_can_on_ramekin_clean",
    "widowx_coke_can_on_plate_clean",
]
# calibrated proxy (fit_simple_success_reward.py): success = sigmoid(C + bz*z + bg*(-grip))
PROXY = {"C": 8.123, "bz": 0.4445, "bg": 11.3193}
# measurements imported from the historical banks predate n_ctx bookkeeping;
# they were scored at F=4 C>=20, so this is a conservative stand-in
FALLBACK_NCTX = 80


def proxy_success(z, grip):
    x = PROXY["C"] + PROXY["bz"] * np.asarray(z) + PROXY["bg"] * (-np.asarray(grip))
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


# What KIND of input a phrase is. The rules exist to repair inputs, so which
# input regime a measurement came from is load-bearing: our own ladder shows rules
# worth +5.5..+6.3 on adversarial text and only +0.4..+2.8 on natural text. A
# distiller that cannot see the label averages those two regimes and tunes for
# neither. "unknown" is a real value -- historical measurements predate the
# labelling and are not going to be guessed at.
KINDS = ("original", "natural", "adversarial", "rephrased", "search", "unknown")


def label_kinds(df):
    """Label what is honestly recoverable; leave the rest 'unknown'.

    Five kinds plus an honest escape hatch. A rulebook's output is `rephrased` --
    it is a distinct thing, neither the wording a person would produce nor the
    hostile one -- and it additionally carries `base_kind`, the regime of the
    instruction it was rewritten FROM. Both facts matter and neither substitutes
    for the other: `kind` says what the phrase is, `base_kind` says what it was
    repairing. Rules are judged on the second.

    Anything genuinely unrecoverable stays `unknown` rather than being guessed."""
    if "kind" not in df:
        df["kind"] = np.nan
    k = df["kind"].astype(object)

    def fill(mask, value):
        nonlocal k
        blank = k.isna() | (k == "") | (k == "unknown")
        k = k.mask(blank & mask, value)

    # the task's own canonical instruction, verbatim
    fill(df.phrase.astype(str).str.strip().str.lower()
         == df.task.astype(str).str.strip().str.lower(), "original")
    if "source" in df:
        fill(df.source.astype(str).str.startswith("loop_"), "rephrased")
        fill(df.source.eq("search_boards"), "search")
        fill(df.source.eq("probe"), "search")
    df["kind"] = k.mask(k.isna() | (k == ""), "unknown")
    if "base_kind" not in df:
        df["base_kind"] = np.nan
    return df


def rule_id(text):
    """A rule's identity IS its wording. Whitespace is normalised (reflowing a
    rule is not changing it); everything else -- a word, a number, a comma --
    makes it a different rule with a different id, because it is a different
    instruction to the applier and may measure differently."""
    return hashlib.sha1(" ".join(str(text).split()).encode()).hexdigest()[:12]


# Single-rule measurements are the loop's dominant cost (SINGLE_EDIT_N x R applies
# AND scorings per iteration) and most rules survive an iteration untouched, so
# they are cached across iterations, passes and runs. Keyed on
# (rephraser, rule_id, task, base phrase) -- the rewrite depends on all four.
# Raw channels are stored, never proxy, so a recalibration does not invalidate it.
RULE_CACHE = REPO / "results" / "rules_runs" / "_rule_cache.parquet"


def rule_cache_lookup(rephraser, rid, bases):
    if not RULE_CACHE.exists():
        return pd.DataFrame(), bases
    c = pd.read_parquet(RULE_CACHE)
    c = c[(c.rephraser == rephraser) & (c.rule_id == rid)]
    if not len(c):
        return pd.DataFrame(), bases
    key = set(zip(c.task, c.base))
    hit = bases[[(t, p) in key for t, p in zip(bases.task, bases.phrase)]]
    miss = bases[[(t, p) not in key for t, p in zip(bases.task, bases.phrase)]]
    got = c.merge(hit[["task", "phrase"]].rename(columns={"phrase": "base"}),
                  on=["task", "base"], how="inner")
    return got, miss


def rule_cache_store(rephraser, rid, rule_text, scored):
    """scored: task, base, phrase(rewrite), z, grip, n_ctx"""
    rows = scored.assign(rephraser=rephraser, rule_id=rid,
                         rule_text=" ".join(str(rule_text).split()))
    cols = ["rephraser", "rule_id", "rule_text", "task", "base", "phrase",
            "z", "grip", "n_ctx"]
    rows = rows[[c for c in cols if c in rows]]
    RULE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    if RULE_CACHE.exists():
        rows = pd.concat([pd.read_parquet(RULE_CACHE), rows], ignore_index=True)
    rows.drop_duplicates(["rephraser", "rule_id", "task", "base"], keep="last") \
        .to_parquet(RULE_CACHE, index=False)


def recompute_proxy(df):
    """The single place proxy is computed. Search-board rows carry grip but no
    verifier channel, so z is imputed to the column mean -- exactly as at seed
    time. Any second implementation of this drifts and blanks the evidence."""
    z = df.z.fillna(df.z.mean()) if df.z.notna().any() else df.z.fillna(0.0)
    g = df.grip.fillna(df.grip.mean()) if df.grip.notna().any() else df.grip.fillna(0.0)
    out = proxy_success(z, g)
    df["proxy_imputed"] = df.z.isna() | df.grip.isna()   # visible in the evidence
    return out


def is_sealed(task):
    return any(s in str(task) for s in SEALED_STEMS)


# --- small utilities --------------------------------------------------------
def sh(cmd, timeout=240, check=False):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    if check and r.returncode != 0:
        raise RuntimeError(f"cmd failed ({r.returncode}): {cmd}\n{r.stderr[:500]}")
    return r


def gitsync(push_paths=None, msg=""):
    for _ in range(3):
        sh("git -c rebase.autoStash=true pull --rebase -q origin main", timeout=200)
        if not push_paths:
            return True
        sh("git add " + " ".join(str(p) for p in push_paths))
        sh(f"git commit -q -m {json.dumps(msg)}")
        if sh("git push -q origin HEAD:main", timeout=200).returncode == 0:
            return True
        sh("git rebase --abort")
    raise RuntimeError("git push failed 3x")


def jread(p):
    return json.loads(Path(p).read_text())


def jwrite(p, obj):
    tmp = Path(str(p) + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str))
    os.replace(tmp, p)


# --- LLM backends -----------------------------------------------------------
_LOG_LOCK = threading.Lock()
_LOG_SEQ = itertools.count()


def _log_llm(run, tag, prompt, response):
    """Collision-proof under the parallel appliers: a glob-derived index raced and
    made pairs cross-contaminate (prompt of one call beside another's response)."""
    d = run.dir / "llm_log"
    with _LOG_LOCK:
        d.mkdir(exist_ok=True)
        i = next(_LOG_SEQ)
    stem = d / f"{tag}_{i:04d}_{os.getpid()}"
    stem.with_suffix(".prompt.txt").write_text(prompt)
    stem.with_suffix(".response.txt").write_text(response)


REASONING_ROLES = ("distill", "judge", "plan", "corpus")


def stage_for_agent(run):
    """Mirror the run's working files into the agent workspace and return the
    files that came back. Only what the prompts reference crosses over: small
    CSV and markdown artifacts, never the repo."""
    run.agent_dir.mkdir(parents=True, exist_ok=True)
    for f in run.dir.rglob("*"):
        if f.is_file() and f.suffix in (".csv", ".md", ".json") and ".sb" not in f.name:
            dst = run.agent_dir / f.relative_to(run.dir)
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists() or dst.stat().st_mtime < f.stat().st_mtime:
                dst.write_bytes(f.read_bytes())


def collect_from_agent(run):
    """Copy back anything the agent wrote (it authors corpus_stats.md itself)."""
    if not run.agent_dir.exists():
        return
    for f in run.agent_dir.rglob("*"):
        if f.is_file() and f.suffix in (".csv", ".md", ".json"):
            dst = run.dir / f.relative_to(run.agent_dir)
            if not dst.exists() or dst.stat().st_mtime < f.stat().st_mtime:
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_bytes(f.read_bytes())


def sandbox_wrapper(run):
    """macOS seatbelt profile: everything allowed EXCEPT reading this repo, with
    the run directory re-allowed even though it lives inside the repo.

    A tool allowlist cannot contain a shell, so the boundary belongs below the
    agent, not inside it. What this keeps out is the project's own prior
    rulebooks (results/analysis/b4_phrasing_rules_v*.md): an agent that reads
    those is recalling earlier conclusions rather than distilling from the
    evidence in front of it, and the run would look normal while meaning nothing.

    Returns a command prefix, or [] where seatbelt is unavailable (the run then
    proceeds unsandboxed and says so, rather than silently dropping the
    protection)."""
    if not shutil.which("sandbox-exec"):
        return []
    run.agent_dir.mkdir(parents=True, exist_ok=True)
    prof = run.agent_dir / ".sandbox.sb"
    prof.write_text(
        "(version 1)\n"
        "(allow default)\n"
        f'(deny file-read* (subpath "{REPO.resolve()}"))\n'
        f'(allow file-read* file-write* (subpath "{run.agent_dir.resolve()}"))\n')
    return ["sandbox-exec", "-f", str(prof)]


def call_llm(run, backend, prompt, tag, timeout=900, session=None, add_dir=None,
             effort="high"):
    """session: a uuid to pin/resume a conversation. Reasoning roles (distill,
    judge, plan) share ONE session per pass so the distiller can refer back to
    everything it has already seen and concluded. Apply calls pass session=None
    deliberately -- they must not see each other's phrases (cross-contamination),
    and being stateless is what lets them run in parallel."""
    if run.dry:
        response = f"[dry-run:{backend}] " + hashlib.sha1(prompt.encode()).hexdigest()[:12]
        if tag.endswith("distill") or "distill" in tag:
            response = ("===RULES===\n"
                        "1. Keep the original wording unless a rule below applies.\n"
                        "2. Use common household nouns.\n"
                        "3. Keep existing color adjectives.\n"
                        "===RATIONALE===\n1. dry-run stub; numbered here on purpose so the\n"
                        "   RULES/RATIONALE split is exercised by the smoke test.\n")
        _log_llm(run, tag, prompt, response)
        return response
    if backend == "claude":
        cmd = sandbox_wrapper(run) + ["claude", "-p", prompt, "--output-format", "text",
               "--model", run.claude_model, "--effort", effort]
        if add_dir:
            # The agent reads its working files itself, Claude-Code style, rather
            # than us pasting summaries into the prompt. Full tooling including a
            # shell: the evidence table runs to thousands of rows and aggregating
            # it properly beats skimming it. Isolation is enforced at the
            # FILESYSTEM layer (sandbox_wrapper + a workspace outside the repo) --
            # a tool allowlist cannot bound a shell anyway. The workspace is the
            # cwd, so nothing needs adding.
            cmd += ["--allowedTools", "Read", "Grep", "Glob", "Bash", "Write"]
        if session:
            marker = run.dir / f".session_{session}"
            started = marker.exists()
            cmd += (["--resume", session] if started else ["--session-id", session])
        stage_for_agent(run)
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                           cwd=str(run.agent_dir), stdin=subprocess.DEVNULL)
        if r.returncode != 0 and session and marker.exists():
            # a resume can fail if the session was never really created; fall back
            # to starting it fresh rather than wedging every later call
            marker.unlink(missing_ok=True)
            cmd = [c for c in cmd if c not in ("--resume", session)] + ["--session-id", session]
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                               cwd=str(run.agent_dir), stdin=subprocess.DEVNULL)
        blob = (r.stdout or "") + (r.stderr or "")
        if "Not logged in" in blob or "/login" in blob:
            raise RuntimeError(
                "the claude CLI is not authenticated for headless use. Run `claude` "
                "once interactively and sign in, then re-run -- the loop resumes "
                "from wherever it stopped. (Or pass --distiller gemini to run the "
                "reasoning prompts through the Gemini API instead.)")
        if r.returncode != 0:
            raise RuntimeError(f"claude CLI failed (rc={r.returncode}): "
                               f"{(r.stderr or r.stdout or '(no output)')[:300]}")
        collect_from_agent(run)
        if session:
            marker.write_text(session)      # only after a call actually succeeded
        response = r.stdout.strip()
    elif backend == "gemini":
        from google import genai
        from google.genai import types
        # An API backend has no filesystem: inline every working file the prompt
        # names, so the same prompts serve both paths without a second wording.
        for m in re.findall(r"^\s{2}([A-Za-z0-9_./-]+\.(?:csv|md|json))\s*$",
                            prompt, re.M):
            f = run.dir / m
            if f.exists():
                body = f.read_text()[:120_000]
                prompt = prompt.replace(f"  {m}",
                                        f"  {m} (contents below)\n\n```\n{body}\n```\n")
        client = genai.Client()
        resp = client.models.generate_content(
            model="gemini-pro-latest", contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.0, http_options=types.HttpOptions(timeout=timeout * 1000)))
        response = (resp.text or "").strip()
    else:
        raise ValueError(f"no local backend for {backend} (qwen goes through pod jobs)")
    _log_llm(run, tag, prompt, response)
    return response


_TRACES = {}


def load_traces():
    """Scene descriptions, keyed by instruction. Expensive to generate, so they
    are loaded once from the teacher bank and reused for every rule application."""
    global _TRACES
    if _TRACES:
        return _TRACES
    f = REPO / "results/phrase_artifacts/cover35_teacher_train.parquet"
    if f.exists():
        t = pd.read_parquet(f, columns=["instruction", "trace"]).drop_duplicates("instruction")
        _TRACES = dict(zip(t.instruction, t.trace))
    return _TRACES


def prompt_from(name, **kw):
    t = (PROMPTS / name).read_text()
    for k, v in kw.items():
        t = t.replace("{{" + k + "}}", str(v))
    left = sorted(set(re.findall(r"\{\{([a-z_]+)\}\}", t)))
    if left:
        raise KeyError(f"{name}: unsubstituted placeholder(s) {left} -- the template "
                       f"and its call site have drifted apart. Supplied: "
                       f"{sorted(kw)}")
    return t


# --- run dir / config -------------------------------------------------------
class Run:
    def __init__(self, run_id, dry, mock_scoring=False):
        self.id = run_id
        self.dry = dry
        self.mock_scoring = mock_scoring
        self.dir = REPO / "results" / "rules_runs" / run_id
        # The agent's workspace is OUTSIDE the repo. Running an agent from a cwd
        # inside a directory the sandbox denies makes the CLI die with EPERM as it
        # walks up looking for .git and settings -- and more importantly, "given
        # only what it needs" is a property of the workspace, not of a deny rule.
        self.agent_dir = Path(os.environ.get(
            "RULES_AGENT_DIR", "/tmp/rules_loop_agent")) / run_id
        self.session = None
        self.claude_model = "claude-opus-5"   # set from config at run start
        (self.dir / "jobs").mkdir(parents=True, exist_ok=True)
        self.cfg_path = self.dir / "config.json"

    def config(self, args):
        if self.cfg_path.exists():
            return jread(self.cfg_path)
        cfg = {
            "run_id": self.id, "created": time.strftime("%Y-%m-%d %H:%M"),
            "seed": args.seed, "patience": args.patience, "max_iters": args.max_iters,
            "sample_n": args.sample_n, "single_edit_n": args.single_edit_n,
            "score_budget": args.score_budget,
            "contexts_per_task": args.contexts_per_task,
            "frames_per_episode": args.frames_per_episode,
            "min_eps_per_task": args.min_eps_per_task,

            "rephrasers": args.rephrasers.split(","),
            "distiller": args.distiller, "judge": args.distiller,
            "claude_model": args.claude_model,
            "max_probes": args.max_probes,
            "init_rules_from": args.init_rules_from,
            "rollback_on_regress": not args.no_rollback,
            "proxy": PROXY,
            "prompt_hashes": {p.name: hashlib.sha1(p.read_bytes()).hexdigest()[:12]
                              for p in sorted(PROMPTS.glob("*.md"))},
        }
        jwrite(self.cfg_path, cfg)
        return cfg


# --- scored bank ------------------------------------------------------------
def seed_bank(run):
    """Everything already measured, sealed excluded. Sources: the fine-exam panel
    (proxy channels + real-rollout gt) and the search boards (213 training
    instructions x ~30 proxy-scored phrases)."""
    bank_path = run.dir / "bank.parquet"
    if bank_path.exists():
        return pd.read_parquet(bank_path)
    frames = []
    fe = pd.concat([pd.read_parquet(REPO / "results/analysis/fine_exam_features_native.parquet"),
                    pd.read_parquet(REPO / "results/analysis/fine_exam_features_oov.parquet")])
    agg = fe.groupby(["task", "phrase"]).agg(z=("z_row", "mean"), grip=("grip_row", "mean")).reset_index()
    pan = pd.read_parquet(REPO / "results/analysis/fine_exam_phrases.parquet")
    agg = agg.merge(pan[["task", "phrase", "gt_success"]], on=["task", "phrase"], how="left")
    agg["source"] = "fine_exam"
    frames.append(agg)
    boards = REPO / "results/analysis/search_boards.jsonl"
    if boards.exists():
        rows = []
        for line in boards.open():
            b = json.loads(line)
            task = b["instruction"]  # training tasks are keyed by instruction
            seen = {}
            for ph in b.get("final_board", []):
                seen[ph["phrase"]] = ph.get("grip", np.nan)
            for rnd in b.get("rounds", []):
                for ph in rnd.get("top", []):
                    seen.setdefault(ph["phrase"], ph.get("screen_grip", np.nan))
            for phrase, grip in seen.items():
                rows.append({"task": task, "phrase": phrase, "z": np.nan,
                             "grip": grip, "gt_success": np.nan,
                             "source": "search_boards"})
        if rows:
            frames.append(pd.DataFrame(rows))
    bank = pd.concat(frames, ignore_index=True)
    bank = bank[~bank.task.map(is_sealed)].drop_duplicates(["task", "phrase"])
    bank["proxy"] = recompute_proxy(bank)
    bank["iter_added"] = -1
    bank = label_kinds(bank)
    bank.to_parquet(bank_path, index=False)
    print(f"[{run.id}] phrase kinds: "
          + ", ".join(f"{k}={v}" for k, v in bank.kind.value_counts().items()))
    return bank


def bank_add(run, df):
    """Accumulate measurements for a phrase rather than discarding either copy.

    A phrase measured twice should end up MORE precisely known, not overwritten:
    n_ctx adds, and z/grip become n-weighted means. (The previous
    drop_duplicates(keep="first") silently threw away every re-measurement, which
    made re-scoring for significance impossible.)"""
    bank_path = run.dir / "bank.parquet"
    bank = pd.read_parquet(bank_path)
    both = pd.concat([bank, df], ignore_index=True)
    for c in ("n_ctx", "n_meas"):
        if c not in both:
            both[c] = np.nan
    both["n_ctx"] = both.n_ctx.fillna(FALLBACK_NCTX)
    both["n_meas"] = both.n_meas.fillna(1)

    def merge(g):
        if len(g) == 1:
            return g.iloc[0]
        # only independent draws add confidence: identical (draw) rows are the
        # same measurement seen twice and must not inflate n_ctx
        if "draw" in g:
            g = g.drop_duplicates("draw", keep="last")
        w = g.n_ctx.to_numpy(dtype=float)
        r = g.iloc[0].copy()
        for ch in ("z", "grip"):
            v = g[ch].to_numpy(dtype=float)
            ok = ~np.isnan(v) & (w > 0)
            r[ch] = float(np.average(v[ok], weights=w[ok])) if ok.any() else np.nan
        r["n_ctx"] = float(w.sum())
        r["n_meas"] = float(g.n_meas.sum())
        if "gt_success" in g:            # a real rollout number always wins
            gt = g.gt_success.dropna()
            r["gt_success"] = gt.iloc[0] if len(gt) else np.nan
        return r

    dup = both.duplicated(["task", "phrase"], keep=False)
    rows = [merge(g) for _, g in both[dup].groupby(["task", "phrase"], sort=False)]
    merged = pd.concat([both[~dup], pd.DataFrame(rows)], ignore_index=True) \
        if rows else both[~dup].copy()
    merged = label_kinds(merged)
    merged["proxy"] = recompute_proxy(merged)
    merged.to_parquet(bank_path, index=False)
    return merged


# --- pod jobs (score / apply) ----------------------------------------------
def run_job(run, kind, payload: pd.DataFrame, spec: dict, tag: str, timeout=7200):
    """Submit a job through git; block until the worker commits the result.
    Resume-safe: if the result already exists, return it without resubmitting.

    The id hashes the payload AND the spec (which carries the rules file's
    CONTENT hash). Hashing the payload alone made every apply job collide across
    iterations -- val bases never change, so iteration 5 would silently replay
    iteration 0's rewrites."""
    key = pd.util.hash_pandas_object(payload).values.tobytes() + \
        json.dumps(spec, sort_keys=True, default=str).encode()
    jid = f"{tag}_{hashlib.sha1(key).hexdigest()[:10]}"
    jdir = run.dir / "jobs"
    result = jdir / f"{jid}.result.parquet"
    if result.exists():
        return pd.read_parquet(result)
    if run.dry or run.mock_scoring:
        out = payload.copy()
        h = payload.phrase.map(lambda p: int(hashlib.sha1(p.encode()).hexdigest()[:6], 16) / 0xFFFFFF)
        if kind == "score":
            out["z"] = -8 + 3 * h
            out["grip"] = 0.55 - 0.25 * h
            out["proxy"] = proxy_success(out.z, out.grip)
        else:  # apply
            out["rewrite"] = out.phrase.map(lambda p: "put the " + p.split()[-1] + " on the target")
        out.to_parquet(result, index=False)
        return out
    payload.to_parquet(jdir / f"{jid}.payload.parquet", index=False)
    jwrite(jdir / f"{jid}.spec.json", {"job_id": jid, "kind": kind, **spec})
    gitsync([jdir / f"{jid}.payload.parquet", jdir / f"{jid}.spec.json"],
            f"rules-loop job {jid} ({kind})")
    deadline = time.time() + timeout
    while time.time() < deadline:
        time.sleep(60)
        sh("git -c rebase.autoStash=true pull --rebase -q origin main", timeout=200)
        if result.exists():
            return pd.read_parquet(result)
        fail = jdir / f"{jid}.failed.txt"
        if fail.exists():
            raise RuntimeError(f"job {jid} failed on worker: {fail.read_text()[:400]}")
    raise TimeoutError(f"job {jid} ({kind}) not returned in {timeout}s")


def score_phrases(run, cfg, df, tag, draw=0):
    """df: [task, phrase] -> adds z, grip, proxy. CRN: the worker samples
    cfg['contexts_per_task'] contexts per task with seed cfg['seed'] -- same
    contexts for every phrase, all run."""
    out = run_job(run, "score", df[["task", "phrase"]].drop_duplicates(), {
        "score_budget": cfg.get("score_budget", 64),
        "draw": draw,          # distinct draw -> different CRN contexts, so a
                               # re-measurement adds information instead of
                               # reproducing the first measurement exactly
        "contexts_per_task": cfg["contexts_per_task"],
        "frames_per_episode": cfg["frames_per_episode"],
        "pool_val8_stems": True, "seed": cfg["seed"] + 1009 * draw,
        "proxy": PROXY}, tag)
    keep = [c for c in ("task", "phrase", "z", "grip", "proxy", "n_ctx") if c in out]
    res = df.drop(columns=[c for c in ("z", "grip", "proxy", "n_ctx") if c in df],
                  errors="ignore").merge(out[keep], on=["task", "phrase"], how="left")
    res["n_meas"] = 1
    res["draw"] = draw
    return res


def apply_rules(run, cfg, rephraser, rules, bases: pd.DataFrame, tag, only_rule=None):
    """bases: [task, phrase(base instruction)] -> adds rewrite column."""
    if rephraser == "qwen" and not run.dry:
        # rules travel as a file for the pod, but their hash goes in the spec so
        # the job id changes when the rulebook does
        return run_job(run, "apply", bases[["task", "phrase"]], {
            "rules_file": str((run.dir / "current_rules.md").relative_to(REPO)),
            "rules_sha": hashlib.sha1(rules.encode()).hexdigest()[:12],
            "only_rule": only_rule}, tag)
    # Single-edit: the applier sees a ONE-RULE rulebook, not the whole book with
    # an instruction to ignore the others. Showing rules it is told not to apply
    # contaminates the very contrast the measurement exists to isolate.
    rules = f"===RULES===\n1. {only_rule}\n" if only_rule else rules_only(rules)
    traces = load_traces()
    jobs = []
    for r in bases.itertuples():
        jobs.append((r.task, r.phrase, prompt_from(
            "apply.md", rules=rules, phrase=r.phrase,
            trace=traces.get(r.task, traces.get(r.phrase,
                                                "(no scene description available)")))))
    rows = [None] * len(jobs)

    def one(i):
        task, phrase, p = jobs[i]
        out = call_llm(run, rephraser, p, f"{tag}_apply", session=None, effort="medium")
        return i, {"task": task, "phrase": phrase, "rewrite": out.split("\n")[0].strip()}

    # stateless -> safe to run concurrently; this is the loop's dominant cost
    with cf.ThreadPoolExecutor(max_workers=1 if run.dry else APPLY_WORKERS) as ex:
        for i, rec in ex.map(one, range(len(jobs))):
            rows[i] = rec
    return pd.DataFrame(rows)


def write_new_measurements(run, bank, tasks, path, since_iter):
    """Just what has been measured since the last distillation: the probes the
    distiller asked for, and the rewrites its own rulebook produced.

    Without this the answers to its experiments arrive as ~20 new rows inside a
    table of thousands and are effectively invisible. Splitting them out for one
    iteration closes the loop on plan.md -- the distiller proposed a question and
    here is what came back. They stay in the bank either way; only the
    presentation separates them, and only until the next iteration."""
    if "iter_added" not in bank:
        Path(path).write_text("task,phrase,kind,base_kind,proxy,n_ctx,source\n")
        return 0
    fresh = bank[bank.task.isin(tasks)
                 & (pd.to_numeric(bank.iter_added, errors="coerce") >= since_iter)].copy()
    cols = [c for c in ("task", "phrase", "kind", "base_kind", "proxy", "n_ctx", "source")
            if c in fresh]
    fresh.sort_values("proxy", ascending=False)[cols].to_csv(path, index=False)
    return len(fresh)


def write_evidence_file(run, bank, tasks, path):
    """The full spread per task as CSV. CSV over JSON deliberately: this table
    runs to thousands of rows, where JSON's repeated keys triple the tokens and
    make it harder to grep; and the agent has a shell, so a table it can sort and
    group beats a structure it must parse by eye. A short README sits beside it."""
    sub = bank[bank.task.isin(tasks)].copy()
    sub = sub.sort_values(["task", "proxy"])
    if "n_ctx" not in sub:
        sub["n_ctx"] = FALLBACK_NCTX
    # a phrase measured on few contexts is a weak claim; the reader needs to see
    # that to decide whether a gap is real or worth re-measuring
    sub["n_ctx"] = sub.n_ctx.fillna(FALLBACK_NCTX).round().astype(int)
    cols = ["task", "phrase", "kind", "base_kind", "proxy", "proxy_imputed",
            "n_ctx", "gt_success", "source"]
    sub[[c for c in cols if c in sub.columns]].to_csv(path, index=False)
    # per-task summary: the aggregation an agent would otherwise need a shell for
    agg = sub.groupby("task").agg(
        phrases=("phrase", "size"),
        best=("proxy", "max"), worst=("proxy", "min"),
        median=("proxy", "median"),
        rollout_measured=("gt_success", lambda s: int(s.notna().sum())),
        thin_evidence=("n_ctx", lambda s: int((pd.to_numeric(s, errors="coerce") < 40).sum())),
    ).reset_index()
    agg["spread"] = agg.best - agg.worst
    agg = agg.sort_values("spread", ascending=False)
    agg.to_csv(str(path).replace("evidence.csv", "evidence_summary.csv"), index=False)
    Path(str(path) + ".README.md").write_text(
        "# evidence.csv\n\n"
        f"{len(sub)} measured phrases across {sub.task.nunique()} tasks, sorted by\n"
        "task then estimated success.\n\n"
        "Columns:\n"
        "  task       the instruction/task the phrase was measured on\n"
        "  phrase     the exact wording measured\n"
        "  proxy      estimated success rate, 0-1 (a calibrated estimate)\n"
        "  kind       what the phrase IS. Five kinds:\n"
        "               original     the task's own canonical instruction -- the\n"
        "                            wording the policy was trained on\n"
        "               natural      a fluent rewording, the kind of thing a\n"
        "                            person would actually say\n"
        "               adversarial  a deliberately awkward, ornate or indirect\n"
        "                            rewording -- the hard case rules exist for\n"
        "               rephrased    the output of applying a rulebook\n"
        "               search       surfaced by automated phrasing search, or\n"
        "                            proposed as a probe; exploratory wordings\n"
        "             plus `unknown` for measurements that predate labelling.\n"
        "  proxy_imputed  True when one reward channel was missing for this row\n"
        "             and was filled with the column mean. Those estimates are\n"
        "             driven by the surviving channel alone -- weaker evidence\n"
        "             than a row with both.\n"
        "  base_kind  for `rephrased` rows only: the kind of the instruction it\n"
        "             was rewritten FROM. Blank otherwise. This is the column that\n"
        "             says what a rewrite was REPAIRING.\n\n"
        "             COMPARE WITHIN A REGIME. Rules exist to repair inputs, and\n"
        "             the regimes behave differently: a rule that rescues\n"
        "             adversarial wordings may do nothing at all for natural ones.\n"
        "             A single number averaged across them hides both effects. For\n"
        "             a rephrased row the regime is `base_kind`, not `kind` --\n"
        "             every rewrite is `rephrased`, so that column alone tells you\n"
        "             nothing about which problem the rule was solving. When the\n"
        "             evidence supports it, say which regime a rule is for.\n"
        "  n_ctx      how many scored contexts back that estimate. LOW n_ctx =\n"
        "             a weak claim. If two phrases differ but both have small\n"
        "             n_ctx, the difference may be noise -- you can propose\n"
        "             re-measuring them (see the experiment format).\n"
        "  gt_success real measured success rate, 0-100, blank if never rolled.\n"
        "             Where present, trust this over `proxy`.\n"
        "  source     where the measurement came from\n\n"
        "\nevidence_summary.csv sits beside it: one row per task with phrase count,\n"
        "best/worst/median estimate, SPREAD (best-worst), how many phrases carry a\n"
        "real rollout number, and how many rest on thin sampling. Sorted by spread,\n"
        "descending -- the tasks where wording matters most are at the top. Read it\n"
        "first, then go into evidence.csv for the tasks it points you to. The\n"
        "within-task spread is the signal, not the extremes.\n")
    return len(sub)


def write_eval_file(run, path, rules, per_rule, pairs, base_mean, rules_mean):
    # machine-readable twin: small and structured, so JSON is the right shape here
    jwrite(Path(str(path).replace(".md", ".json")),
           {"whole_rulebook": {"with_rules": rules_mean, "unrephrased": base_mean,
                               "delta": rules_mean - base_mean},
            "per_rule": per_rule, "samples": pairs})
    lines = ["# Previous rulebook: measured performance", "",
             f"Whole rulebook applied: estimated success {rules_mean:.3f} vs "
             f"{base_mean:.3f} for the unrephrased instruction "
             f"({rules_mean - base_mean:+.3f}).", "",
             "## Per-rule single-edit effects", "",
             "Each rule applied ALONE to the same base instructions, so the",
             "delta below is attributable to that rule and not to the rest of",
             "the rulebook.", ""]
    for k, v in (per_rule or {}).items():
        lines.append(f"### {k}")
        lines.append(f"text: {v['text']}")
        lines.append(f"delta vs base: {v['delta_proxy']:+.4f}   (n={v['n']} instructions)")
        for kk, kv in (v.get("by_kind") or {}).items():
            lines.append(f"    on {kk:12s} inputs: {kv['delta_proxy']:+.4f}  (n={kv['n']})")
        lines.append("")
    lines += ["## Sample rewrites from the whole rulebook", ""]
    for p in pairs:
        lines.append(f"  [{p['task']}]")
        lines.append(f"    in : {p['base']!r}")
        lines.append(f"    out: {p['rewrite']!r}")
    Path(path).write_text("\n".join(lines))


def build_vocabulary(src_dir, floor=5):
    """Count the corpus deterministically. This used to be asked of an agent, but
    a word count is not a judgement -- code gets it exactly right, reproducibly,
    and without inlining 17k instructions into a prompt (which read-timed out).
    The agent is left the part that IS judgement: shape, register, what is
    conspicuously absent."""
    import collections
    txt = [f for f in sorted(src_dir.iterdir()) if f.suffix == ".txt"]
    if not txt:
        return None, 0, {}
    lines = [l.strip() for f in txt for l in f.read_text(errors="replace").splitlines()
             if l.strip()]
    toks = collections.Counter(w for l in lines for w in re.findall(r"[a-z']+", l.lower()))
    above = {w: c for w, c in toks.items() if c >= floor}
    common = sorted(((w, c) for w, c in above.items() if c >= 50), key=lambda x: -x[1])
    present = sorted(w for w, c in above.items() if c < 50)
    body = (
        f"# Corpus vocabulary — exact membership test\n\n"
        f"{len(lines)} instructions, {len(toks)} distinct tokens, {len(above)} above a\n"
        f"floor of {floor} occurrences (rarer words excluded as noise). Tokenised as\n"
        f"lowercase runs of letters and apostrophes.\n\n"
        f"A word is corpus-present if and only if it appears below. Presence licenses a\n"
        f"token; it never makes a token good — frequency is not outcome.\n\n"
        f"**Common (n>=50):** " + ", ".join(f"{w} {c}" for w, c in common) + "\n\n"
        f"**Present (n={floor}-49):** " + ", ".join(present) + "\n")
    return body, len(lines), {"distinct": len(toks), "above_floor": len(above),
                              "common": len(common), "present": len(present)}


def ensure_corpus_file(run, cfg):
    """Deterministic vocabulary table + an agent-written qualitative section."""
    out = run.dir / "corpus_stats.md"
    if out.exists():
        return out
    src_dir = REPO / "results/analysis/b4_rules_inputs"
    if run.dry or not src_dir.exists():
        out.write_text("# Corpus vocabulary\n\n(dry-run placeholder)\n")
        return out

    # stage only corpus material; anything naming rules or their outcomes is out
    staged = run.dir / "corpus_inputs"
    staged.mkdir(exist_ok=True)
    copied = []
    for f in sorted(src_dir.iterdir()):
        if f.is_file() and not any(x in f.name.lower() for x in ("rules", "readme")):
            (staged / f.name).write_bytes(f.read_bytes())
            copied.append(f.name)
    jwrite(run.dir / "corpus_inputs_manifest.json",
           {"copied": copied,
            "excluded": [f.name for f in sorted(src_dir.iterdir()) if f.name not in copied]})

    vocab, n_instr, stats = build_vocabulary(staged)
    if vocab is None:
        out.write_text("# Corpus vocabulary\n\n(no instruction list found)\n")
        return out
    print(f"[{run.id}] vocabulary computed: {n_instr} instructions, "
          f"{stats['common']} common + {stats['present']} present "
          f"({stats['distinct']} distinct before the floor)")

    # the agent adds only what requires judgement, over a SAMPLE, not the corpus
    sample = "\n".join((staged / copied[0]).read_text(errors="replace").splitlines()[:400])
    p = prompt_from("corpus.md", vocab_summary=vocab[:6000],
                    n_instructions=n_instr, sample=sample)
    try:
        qual = call_llm(run, cfg.get("distiller", "claude"), p, "corpus",
                        effort="high", timeout=900)
    except Exception as e:
        print(f"[{run.id}] corpus commentary failed ({type(e).__name__}); "
              f"vocabulary table still written")
        qual = "(commentary unavailable)"
    out.write_text(vocab + "\n---\n\n" + qual + "\n")
    return out


# --- rule parsing / evaluation ----------------------------------------------
def section(text, name):
    """Text between ===NAME=== and the next ===...=== delimiter (or the end)."""
    m = re.search(rf"^===\s*{re.escape(name)}\s*===\s*$(.*?)(?=^===|\Z)", text,
                  re.M | re.S)
    return m.group(1).strip() if m else ""


def rules_only(rules_text):
    """The rulebook with the RATIONALE stripped. The rationale is written for us,
    not for the applier -- shipping it would waste context and, worse, feed the
    applier commentary about rules it is supposed to follow literally."""
    body = section(rules_text, "RULES")
    if body:
        return body
    return rules_text.split("===RATIONALE===")[0].strip()


def parse_rules(rules_text):
    """Numbered rules from the RULES section ONLY. The RATIONALE section also
    contains numbered lines; counting those would invent phantom rules and
    corrupt every single-edit measurement, so a missing RULES delimiter is a
    hard failure rather than a silent fallback over the whole text."""
    body = section(rules_text, "RULES")
    if not body:
        head = rules_text.split("===RATIONALE===")[0].strip()
        if head == rules_text.strip():
            raise ValueError("distilled rules have no ===RULES=== section and no "
                             "===RATIONALE=== delimiter -- refusing to guess which "
                             "numbered lines are rules")
        body = head
    return [" ".join(m.group(1).split()) for m in re.finditer(
        r"^[ \t]*\d+[.)][ \t]+(.*(?:\n(?![ \t]*\d+[.)]|===)[ \t]+\S.*)*)", body, re.M)]


def baseline_proxy(run, cfg, bases, tag):
    """Mean proxy of the UNREPHRASED base phrases -- scored once per split and
    cached, since the bases are fixed for the whole run."""
    bh = hashlib.sha1(pd.util.hash_pandas_object(
        bases[["task", "phrase"]]).values.tobytes()).hexdigest()[:8]
    cache = run.dir / f"baseline_{tag}_{bh}.json"
    if cache.exists():
        return jread(cache)["mean"]
    sc = score_phrases(run, cfg, bases[["task", "phrase"]], f"baseline_{tag}")
    jwrite(cache, {"mean": float(sc.proxy.mean()), "n": int(len(sc))})
    return float(sc.proxy.mean())


def eval_rules(run, cfg, itdir, rephraser, rules, bases, tag, single_edit=False):
    """Returns (mean proxy score of rewrites, scored df, rules_eval_summary)."""
    rsha = hashlib.sha1(rules.encode()).hexdigest()[:10]
    art = itdir / f"eval_{tag}_{rsha}.json"
    scored_art = itdir / f"eval_{tag}_{rsha}.parquet"
    if art.exists():
        return jread(art)["score"], pd.read_parquet(scored_art), jread(art).get("summary")
    rewrites = apply_rules(run, cfg, rephraser, rules, bases, f"{tag}")
    rw = rewrites.rename(columns={"phrase": "base", "rewrite": "phrase"})[["task", "phrase", "base"]]
    scored = score_phrases(run, cfg, rw, f"{tag}_sc")
    score = float(scored.proxy.mean())
    summary = None
    if single_edit:
        rule_list = parse_rules(rules)
        sample = bases.sample(min(cfg["single_edit_n"], len(bases)), random_state=cfg["seed"])
        base_scored = score_phrases(run, cfg, sample[["task", "phrase"]], f"{tag}_base")
        per_rule = {}
        cache_hits = 0
        for k, rtext in enumerate(rule_list, 1):
            rid = rule_id(rtext)
            cached, todo = rule_cache_lookup(rephraser, rid, sample)
            cache_hits += len(cached)
            fresh = pd.DataFrame()
            if len(todo):
                rw = apply_rules(run, cfg, rephraser, rules, todo, f"{tag}_r{k}",
                                 only_rule=rtext)
                sc = score_phrases(run, cfg,
                                   rw[["task", "rewrite"]].rename(columns={"rewrite": "phrase"}),
                                   f"{tag}_r{k}sc")
                fresh = sc.merge(rw.rename(columns={"phrase": "base", "rewrite": "phrase"})
                                   [["task", "base", "phrase"]],
                                 on=["task", "phrase"], how="left")
                rule_cache_store(rephraser, rid, rtext, fresh)
            rs = pd.concat([c for c in (cached, fresh) if len(c)], ignore_index=True)
            if not len(rs):
                continue
            rs["proxy"] = recompute_proxy(rs)
            by_kind = {}
            if "kind" in sample:
                merged = rs.merge(sample[["task", "kind"]].drop_duplicates("task"),
                                  on="task", how="left")
                bk = base_scored.merge(sample[["task", "kind"]].drop_duplicates("task"),
                                       on="task", how="left")
                for kk in merged.kind.dropna().unique():
                    a = merged[merged.kind == kk].proxy.mean()
                    b = bk[bk.kind == kk].proxy.mean()
                    if pd.notna(a) and pd.notna(b):
                        by_kind[str(kk)] = {"delta_proxy": float(a - b),
                                            "n": int((merged.kind == kk).sum())}
            per_rule[f"rule_{k}"] = {
                "text": rtext,
                "delta_proxy": float(rs.proxy.mean() - base_scored.proxy.mean()),
                "n": int(len(rs)),
                "by_kind": by_kind,   # the interaction: a rule can help one regime
            }                          # and do nothing for another
        eval_file = itdir / "rules_eval.md"
        # headline must compare like with like: the rulebook's score is over the
        # FULL base set, so its baseline must be too (base_scored is the 8-base
        # single-edit sample and belongs only inside the per-rule deltas)
        write_eval_file(run, eval_file, rules, per_rule,
                        [{"task": r.task, "base": r.phrase, "rewrite": r.rewrite}
                         for r in rewrites.head(40).itertuples()],
                        baseline_proxy(run, cfg, bases, tag), score)
        if rule_list:
            print(f"    single-edit: {cache_hits}/{len(rule_list) * len(sample)} "
                  f"measurements served from the rule cache")
        judge_p = prompt_from("judge.md", rules=rules_only(rules), eval_file=eval_file)
        judgement = call_llm(run, cfg["judge"], judge_p, f"{tag}_judge",
                             session=run.session, add_dir=run.dir, effort="high")
        (itdir / "judge.md").write_text(judgement)
        summary = {"per_rule_perf": per_rule, "judge": judgement,
                   "rule_notes": section(judgement, "RULE NOTES"),
                   "suggestions": section(judgement, "SUGGESTIONS")}
    scored.to_parquet(scored_art, index=False)
    base_mean = baseline_proxy(run, cfg, bases, tag)
    rec = {"score": score, "base": base_mean, "delta": score - base_mean,
           "n": int(len(scored)), "rules_sha": rsha, "summary": summary}
    jwrite(art, rec)
    jwrite(itdir / f"eval_{tag}.json", rec)   # stable alias: latest for this tag
    return score, scored, summary


def plot_progress(run, pdir, rephraser):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    its = sorted(pdir.glob("iter_*/scores.json"))
    if not its:
        return
    rows = [jread(p) for p in its]
    xs = list(range(len(rows)))
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    for key, col in [("train", "#a0aec0"), ("val_held", "#2b6cb0"),
                     ("val8", "#805ad5"), ("val_avg", "#0d9488")]:
        ax.plot(xs, [r[key] for r in rows], "o-", color=col,
                lw=2.4 if key == "val_avg" else 1.4, label=key)
    # the decision-relevant view: improvement over the unrephrased instruction
    for key, col in [("train", "#a0aec0"), ("val_held", "#2b6cb0"), ("val8", "#805ad5")]:
        ys = [r.get("delta", {}).get(key) for r in rows]
        if all(y is not None for y in ys):
            ax2.plot(xs, ys, "o-", color=col, lw=1.6, label=key)
    ax2.axhline(0, ls="--", color="#718096", lw=1)
    ax2.set_xlabel("iteration"); ax2.set_ylabel("proxy gain over unrephrased")
    ax2.set_title("do the rules beat saying nothing?", fontsize=10)
    ax2.legend(fontsize=8); ax2.grid(alpha=0.25)
    ax.set_xlabel("iteration"); ax.set_ylabel("mean proxy success (estimated rate)")
    ax.set_title(f"{run.id} / {rephraser} -- early stop on val_avg")
    ax.legend(fontsize=8); ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(pdir / "progress.png", dpi=130)
    plt.close(fig)


# --- main loop ---------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--dry-run", action="store_true",
                    help="no LLM calls and no scoring; plumbing only")
    ap.add_argument("--mock-scoring", action="store_true",
                    help="REAL agents and rephraser, synthetic scores -- smoke-tests "
                         "prompts, sessions, parsing and the cache without a GPU pod")
    ap.add_argument("--rephrasers", default="qwen,claude,gemini")
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--max-iters", type=int, default=12)
    ap.add_argument("--sample-n", type=int, default=24)
    ap.add_argument("--single-edit-n", type=int, default=8)
    ap.add_argument("--score-budget", type=int, default=64,
                    help="target F*C forward passes per phrase per task")
    ap.add_argument("--contexts-per-task", type=int, default=16,
                    help="C cap: episodes per task")
    ap.add_argument("--frames-per-episode", type=int, default=4,
                    help="F floor. Measured: F saturates at 4-5 (F=4->5 buys "
                         "0-1pp at C>=16), so 4 is the operating point, not a "
                         "compromise. The club table holds exactly 4 frames per "
                         "episode, so all 213 search tasks score at F=4 C=16. "
                         "F only rises above this for C-starved multit16 tasks: "
                         "F = clamp(budget/C, this, 16)")
    ap.add_argument("--min-eps-per-task", type=int, default=8,
                    help="training-task support floor (same episode-count criterion as the search)")
    ap.add_argument("--distiller", default="claude", choices=["claude", "gemini"],
                    help="backend for the distiller/judge/planner (claude needs an "
                         "authenticated CLI; gemini uses the API and inlines files)")
    ap.add_argument("--claude-model", default="claude-opus-5",
                    help="model for the distiller/judge/planner and for "
                         "rephraser=claude")
    ap.add_argument("--max-probes", type=int, default=20)
    ap.add_argument("--init-rules-from", default=None, metavar="RUN_ID",
                    help="start each pass from RUN_ID's best rulebook for the same "
                         "model (round 2 initializes from round 1 this way)")
    ap.add_argument("--no-rollback", action="store_true",
                    help="keep revising the latest rulebook even after a "
                         "validation regression (default: revise the best)")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    run = Run(args.run_id, args.dry_run, mock_scoring=args.mock_scoring)
    cfg = run.config(args)
    run.claude_model = cfg.get("claude_model", "claude-opus-5")
    rng = np.random.default_rng(cfg["seed"])
    bank = seed_bank(run)
    sb = sandbox_wrapper(run)
    print(f"[{run.id}] agent filesystem sandbox: "
          + ("ON (repo unreadable, run dir allowed)" if sb else
             "OFF -- agents can read the repo, including prior rulebooks"))
    print(f"[{run.id}] bank: {len(bank)} phrases, {bank.task.nunique()} tasks "
          f"({int(bank.gt_success.notna().sum())} with real-rollout gt)")

    # splits: by task. val8 fixed; val_held = held-out non-val8 tasks.
    # Training tasks are FILTERED BY EPISODE SUPPORT in the context bank (the
    # same criterion the phrase search used): scoring accuracy comes from C, so
    # a task the proxy cannot score at C>=min_eps is not admitted. val8 tasks
    # pool episodes across instructions of the same task stem (worker-side).
    support = {}
    cb = REPO / "data" / "contexts_train_multit16.parquet"
    if cb.exists():
        import pyarrow.parquet as pq
        names = pq.ParquetFile(cb).schema_arrow.names
        key = "task" if "task" in names else "instruction"
        d = pd.read_parquet(cb, columns=[key, "episode_index"])
        support = d.groupby(key).episode_index.nunique().to_dict()
    # the search's club table (data/contexts_club.parquet, pod-side) covers every
    # full-Bridge instruction with >=20 episodes -- the 213. Its per-instruction
    # counts are recorded in search_boards.jsonl, so support can include them
    # without the 1.4G payload being present locally.
    boards = REPO / "results/analysis/search_boards.jsonl"
    if boards.exists():
        for line in boards.open():
            b = json.loads(line)
            support[b["instruction"]] = max(support.get(b["instruction"], 0),
                                            int(b.get("club_eps", 0)))
    # only tasks we hold a scene description for: traces are expensive and we are
    # reusing the existing bank rather than generating more (187 of the 213)
    have_trace = set(load_traces())
    tasks = sorted(t for t in bank.task.unique() if t not in VAL8_TASKS
                   and support.get(t, 0) >= cfg["min_eps_per_task"]
                   and (t in have_trace or run.dry))
    dropped = bank.task.nunique() - len(tasks) - len(VAL8_TASKS)
    print(f"[{run.id}] task support floor >={cfg['min_eps_per_task']} eps: "
          f"{len(tasks)} training tasks admitted, {dropped} dropped (recorded in splits.json)")
    rng.shuffle(tasks)
    n_held = max(2, len(tasks) // 6)
    val_held_tasks, train_tasks = tasks[:n_held], tasks[n_held:]
    jwrite(run.dir / "splits.json", {"train": train_tasks, "val_held": val_held_tasks,
                                     "val8": VAL8_TASKS,
                                     "support_floor": cfg["min_eps_per_task"],
                                     "support": {t: int(support.get(t, 0)) for t in train_tasks + val_held_tasks}})

    # frozen at run start: bank grows every iteration with the loop's OWN rewrites,
    # and sampling from the live bank would make iteration N's bases be iteration
    # N-1's outputs -- the comparison would drift instead of holding still
    # Persisted, not recomputed: bank.parquet grows every iteration with the
    # loop's own rewrites, so re-deriving the pool on a resume would silently
    # change every base instruction -- the exact drift the freeze exists to stop.
    pool_path = run.dir / "base_pool.parquet"
    if pool_path.exists():
        base_pool = pd.read_parquet(pool_path)
    else:
        cols = ["task", "phrase"] + (["kind"] if "kind" in bank else [])
        base_pool = bank[cols].copy()
        base_pool.to_parquet(pool_path, index=False)

    def bases_for(task_list, n=None, seed=0):
        d = base_pool[base_pool.task.isin(task_list)]
        return d.sample(min(n, len(d)), random_state=seed) if n else d

    corpus_file = ensure_corpus_file(run, cfg)
    rules_per_model = {}

    for rephraser in cfg["rephrasers"]:
        pdir = run.dir / f"pass_{rephraser}"
        pdir.mkdir(exist_ok=True)
        # one conversation per pass: the distiller/judge/planner see their own
        # full history -- every prior rulebook, rationale, and audit
        run.session = str(uuid.UUID(hashlib.sha1(f"{run.id}:{rephraser}".encode()).hexdigest()[:32]))
        state_p = pdir / "state.json"
        init_rules = None
        if cfg.get("init_rules_from"):
            prev_best = (REPO / "results" / "rules_runs" / cfg["init_rules_from"]
                         / f"pass_{rephraser}" / "best_rules.md")
            if prev_best.exists():
                init_rules = prev_best.read_text()
                print(f"[{run.id}:{rephraser}] initialized from "
                      f"{cfg['init_rules_from']}'s best rulebook")
        st = jread(state_p) if state_p.exists() else {
            "iter": 0, "best_val": -1e9, "best_iter": -1, "since_best": 0,
            "last_val": -1e9,
            "rules": init_rules or ("===RULES===\n"
                      "1. Rewrite the instruction as a short, plain imperative that "
                      "keeps the same objects and goal.\n"
                      "===RATIONALE===\nno-rules baseline: the scaffold alone, measured "
                      "as iteration 0 so every distilled rulebook has an anchor to beat.\n")}
        # iteration 0 measures whatever we start from -- the no-rules scaffold in
        # round 1, the round-1 rulebook in round 2 -- so the anchor is always real
        while st["since_best"] < cfg["patience"] and st["iter"] < cfg["max_iters"]:
            it = st["iter"]
            itdir = pdir / f"iter_{it:02d}"
            itdir.mkdir(exist_ok=True)
            print(f"[{run.id}:{rephraser}] iter {it} (since_best={st['since_best']})")

            # 1. distill
            ev_file = itdir / "evidence.csv"        # written below; referenced by plan too
            rp = itdir / "rules.md"
            if rp.exists():
                rules = rp.read_text()
            elif it == 0:
                # iteration 0 measures the no-rules prompt itself -- no distillation
                rules = st["rules"]
                rp.write_text(rules)
            else:
                bank = pd.read_parquet(run.dir / "bank.parquet")
                n_ev = write_evidence_file(run, bank, train_tasks, ev_file)
                new_file = itdir / "new_measurements.csv"
                n_new = write_new_measurements(run, bank, train_tasks, new_file,
                                               since_iter=it - 1)
                # Two matched (rulebook, measurements) pairs -- never a rulebook paired
                # with another rulebook's numbers. The best pair is what to build from;
                # the regressed pair, when there is one, is what to avoid.
                none_eval = itdir / "no_eval.md"
                if not none_eval.exists():
                    none_eval.write_text("(no measurements -- first iteration)")

                def pair(i):
                    r = pdir / f"rules_{i:02d}.md"
                    e = pdir / f"iter_{i:02d}" / "rules_eval.md"
                    return (r.read_text() if r.exists() else st["rules"],
                            e if e.exists() else none_eval)

                bi = st["best_iter"]
                best_rules, best_eval = pair(bi) if bi >= 0 else (st["rules"], none_eval)

                regressed_block, regressed_eval = (
                    "(no regression: the last attempt was the best so far)", none_eval)
                if cfg.get("rollback_on_regress", True) and st["since_best"] > 0 and bi >= 0:
                    rr, regressed_eval = pair(it - 1)
                    regressed_block = (
                        f"REGRESSED RULEBOOK (iteration {it - 1}, scored "
                        f"{st['last_val']:.4f} on validation against the best rulebook's "
                        f"{st['best_val']:.4f}). Do NOT build from this one -- diff it "
                        f"against the best rulebook above and work out which change cost "
                        f"the ground:\n{rr}\n")

                def rel(p):
                    try:
                        return str(Path(p).resolve().relative_to(run.dir.resolve()))
                    except ValueError:
                        return str(p)

                dp = prompt_from("distill.md", best_rules=best_rules,
                                 regressed_block=regressed_block,
                                 corpus_file=rel(corpus_file), evidence_file=rel(ev_file),
                                 new_file=rel(new_file),
                                 best_eval_file=rel(best_eval),
                                 regressed_eval_file=rel(regressed_eval))
                print(f"    distilling over {n_ev} measured phrases "
                      f"({len(train_tasks)} tasks), {n_new} newly measured ...")
                rules = None
                for attempt in range(3):
                    cand = call_llm(run, cfg["distiller"], dp, f"{rephraser}_distill",
                                    session=run.session, add_dir=run.dir, effort="high",
                                    timeout=1800)
                    try:
                        if parse_rules(cand):
                            rules = cand
                            break
                    except ValueError as e:
                        print(f"    unparseable rulebook (attempt {attempt + 1}): {e}")
                    (itdir / f"rules.rejected.{attempt}.md").write_text(cand)
                if rules is None:
                    raise RuntimeError("distiller produced no parseable rulebook in 3 "
                                       "attempts; rejected replies saved beside this "
                                       "iteration")
                rp.write_text(rules)        # only a parseable rulebook is persisted
                prev_ids = {rule_id(r) for r in parse_rules(best_rules)} \
                    if bi >= 0 else set()
                new_ids = {rule_id(r) for r in parse_rules(rules)}
                if prev_ids:
                    print(f"    rulebook churn: {len(prev_ids & new_ids)} unchanged, "
                          f"{len(new_ids - prev_ids)} new/reworded, "
                          f"{len(prev_ids - new_ids)} dropped "
                          f"(unchanged rules reuse cached measurements)")
            (pdir / f"rules_{it:02d}.md").write_text(rules)   # flat, browsable history
            cur = run.dir / "current_rules.md"
            cur.write_text(rules)
            if not run.dry and "qwen" in cfg["rephrasers"]:
                # the pod reads this file; it must exist on origin before any
                # apply job referencing it is submitted
                gitsync([cur], f"rules-loop {run.id}/{rephraser} iter {it} rulebook")

            # 2. eval on train sample (with per-rule single edits)
            # fixed across iterations (seed 0) so the train curve is a comparable
            # series -- new evidence enters through probes, not through resampling
            tb = bases_for(train_tasks, cfg["sample_n"], seed=0)
            train_score, ev_train, summary = eval_rules(
                run, cfg, itdir, rephraser, rules, tb, "train", single_edit=True)
            # a rewrite inherits the kind of the instruction it was rewritten FROM
            kmap = dict(zip(tb.phrase, tb.kind)) if "kind" in tb else {}
            applied = set(tuple(x) for x in st.get("bank_applied", []))
            akey = (rephraser, it, "train")
            bank = pd.read_parquet(run.dir / "bank.parquet") if akey in applied else bank_add(run, ev_train.assign(
                source=f"loop_{rephraser}", iter_added=it,
                base_kind=ev_train.base.map(kmap) if "base" in ev_train else np.nan))
            if akey not in applied:
                st["bank_applied"] = [list(x) for x in applied | {akey}]
                jwrite(state_p, st)      # record the mutation BEFORE the long val evals

            # 3. eval on both validation sets
            vh, _, _ = eval_rules(run, cfg, itdir, rephraser, rules,
                                  bases_for(val_held_tasks, cfg["sample_n"], seed=1), "val_held")
            v8, _, _ = eval_rules(run, cfg, itdir, rephraser, rules,
                                  bases_for(VAL8_TASKS, cfg["sample_n"], seed=2), "val8")
            val = (vh + v8) / 2
            deltas = {k: jread(itdir / f"eval_{k}.json").get("delta")
                      for k in ("train", "val_held", "val8")}
            print(f"    train={train_score:.3f} val_held={vh:.3f} val8={v8:.3f} avg={val:.3f}"
                  f"   | delta vs unrephrased: train {deltas['train']:+.3f} "
                  f"val_held {deltas['val_held']:+.3f} val8 {deltas['val8']:+.3f}")

            # 4. early-stopping bookkeeping
            st["last_val"] = val
            if val > st["best_val"]:
                st.update(best_val=val, best_iter=it, since_best=0)
            else:
                st["since_best"] += 1
            st["rules"] = rules
            st["iter"] = it + 1
            jwrite(state_p, st)
            jwrite(itdir / "scores.json", {"train": train_score, "val_held": vh,
                                           "val8": v8, "val_avg": val,
                                           "delta": deltas})
            plot_progress(run, pdir, rephraser)

            # 5. probe phrases the bank lacks
            if summary and (summary.get("suggestions") or "").strip():
                if not ev_file.exists():   # iteration 0 does not distil
                    write_evidence_file(run, pd.read_parquet(run.dir / "bank.parquet"),
                                        train_tasks, ev_file)
                # the per-task summary IS the permitted task list, with coverage
                # attached, so the planner can see where measurement is thin
                pp = prompt_from("plan.md", suggestions=summary["suggestions"],
                                 evidence_file=rel(ev_file),
                                 summary_file=rel(ev_file).replace(
                                     "evidence.csv", "evidence_summary.csv"),
                                 max_probes=cfg.get("max_probes", 20))
                planned = call_llm(run, cfg["distiller"], pp, f"{rephraser}_plan",
                                   session=run.session, add_dir=run.dir, effort="high")
                allowed = set(train_tasks)
                new, rejected = [], []
                for m in re.finditer(r"^\s*\[([^\]]+)\]\s+(.+)$", planned, re.M):
                    t = m.group(1).strip()
                    (new if t in allowed else rejected).append(
                        {"task": t, "phrase": m.group(2).strip()})
                if new:
                    print(f"    {len(new)} probes across "
                          f"{len({r['task'] for r in new})} task(s)")
                if rejected:
                    # a probe outside the training split would measure on a task
                    # the loop is not permitted to learn from
                    print(f"    dropped {len(rejected)} probe(s) naming tasks "
                          f"outside the training split")
                if new:
                    nd = score_phrases(run, cfg, pd.DataFrame(new), f"probe_i{it}",
                                       draw=it + 1)
                    bank = bank_add(run, nd.assign(source="probe", iter_added=it))

        best = pdir / f"iter_{st['best_iter']:02d}" / "rules.md"
        rules_per_model[rephraser] = best.read_text() if best.exists() else st["rules"]
        (pdir / "best_rules.md").write_text(rules_per_model[rephraser])
        print(f"[{run.id}:{rephraser}] done: best iter {st['best_iter']} val {st['best_val']:.3f}")

    jwrite(run.dir / "final.json", {
        "note": "per-model rulebooks -- deliberately NOT merged; each is tuned to "
                "its applier's rule-following capacity",
        "rulebooks": {m: f"pass_{m}/best_rules.md" for m in rules_per_model}})
    print(f"[{run.id}] all passes complete -> {run.dir}/final.json (per-model, unmerged)")


if __name__ == "__main__":
    sys.exit(main())
