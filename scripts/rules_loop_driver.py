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
# --- environment registry (user 2026-08-30): the (policy, dataset) pair the
# loop runs against, abstracted from the loop itself. Everything
# benchmark-specific lives here: bank seed sources, sealed tasks, the sim task
# list, context banks, traces, corpus inputs, and the rollout recipe for
# --phase sim. Adding pi0.5/LIBERO = filling in a second entry (+ worker-side
# data), not touching the loop.
ENVIRONMENTS = {
    "bridge_pi0": dict(
        sealed_stems="SEALED_STEMS",          # resolved below (module constants)
        sim_tasks="VAL8_TASKS",
        fine_exam_features=["results/analysis/fine_exam_features_native.parquet",
                            "results/analysis/fine_exam_features_oov.parquet"],
        fine_exam_panel="results/analysis/fine_exam_phrases.parquet",
        boards="results/analysis/search_boards.jsonl",
        worklist="results/analysis/bank_to_score.parquet",
        bank_scores_glob="bank_scores_*.parquet",
        support_bank="data/contexts_train_multit16.parquet",
        corpus_inputs="results/analysis/b4_rules_inputs",
        traces_per_base="results/phrase_artifacts/traces_rules_v1.parquet",
        traces_legacy="results/phrase_artifacts/cover35_teacher_train.parquet",
        # --phase sim scoring: phase0c_rollout.py on a render pod. episode_ids
        # 0-17 x 1 rep = n=18/phrase, the loop's historical sim-eval budget.
        rollout=dict(config="config/experiment/simpler/pi0_finetune_bridge_ev.yaml",
                     ckpt="juexzz/INTACT-pi0-finetune-rephrase-bridge",
                     episode_ids=list(range(18)), seed=42),
    ),
    "pi05_libero": None,   # TODO: banks + traces + rollout recipe (fourtier stack)
}


def resolve_env(name):
    e = ENVIRONMENTS.get(name)
    if e is None:
        raise SystemExit(f"environment {name!r} is not wired yet -- fill in its "
                         f"ENVIRONMENTS entry (banks, traces, corpus, rollout recipe) "
                         f"and the worker-side data first")
    e = dict(e)
    e["sealed_stems"] = SEALED_STEMS if e["sealed_stems"] == "SEALED_STEMS" else e["sealed_stems"]
    e["sim_tasks"] = VAL8_TASKS if e["sim_tasks"] == "VAL8_TASKS" else e["sim_tasks"]
    return e


# calibrated proxy (fit_simple_success_reward.py): success = sigmoid(C + bz*z + bg*(-grip))
PROXY = {"C": 8.123, "bz": 0.4445, "bg": 11.3193}
# measurements imported from the historical banks predate n_ctx bookkeeping;
# they were scored at F=4 C>=20, so this is a conservative stand-in
FALLBACK_NCTX = 80


def proxy_logit(z, grip):
    """The calibration's linear predictor, before the sigmoid."""
    return PROXY["C"] + PROXY["bz"] * np.asarray(z) + PROXY["bg"] * (-np.asarray(grip))


def proxy_success(z, grip):
    return 1.0 / (1.0 + np.exp(-np.clip(proxy_logit(z, grip), -30, 30)))


# The calibration was fit against SIM rollout success, where gripper error runs
# 0.5-0.9 because the scenes are out of distribution. On Bridge TRAINING frames,
# scored against ground-truth action chunks, grip is 0.04-0.05 -- an order of
# magnitude off the flat left edge of the sigmoid, where every phrase reads
# 0.9996+. Ranking survives (the sigmoid is monotone, so proxy order == logit
# order), but the NUMBER carries no spread, and a distiller reading a column of
# 1.000 learns nothing. So the evidence reports a within-task normalised score,
# and keeps the calibrated probability only where it is in range.
CALIB_GRIP_LO, CALIB_GRIP_HI = 0.30, 0.95


def in_calibration_range(grip):
    g = pd.to_numeric(pd.Series(grip), errors="coerce")
    return g.between(CALIB_GRIP_LO, CALIB_GRIP_HI)


def within_task_score(df):
    """Min-max the logit inside each task -> 0..1, where 1 is that task's best
    measured phrase. Monotone-equivalent to the proxy, but with usable spread on
    tasks where the calibrated probability saturates. Tasks with a single
    measured phrase, or no spread at all, get NaN rather than a fake 1.0."""
    lg = pd.Series(proxy_logit(df.z.astype(float), df.grip.astype(float)),
                   index=df.index)
    def mm(x):
        lo, hi = x.min(), x.max()
        return (x - lo) / (hi - lo) if np.isfinite(lo) and np.isfinite(hi) and hi > lo \
            else pd.Series(np.nan, index=x.index)
    return lg.groupby(df.task).transform(mm), lg


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
        # the generate_phrases arms: these are the input regimes the rules exist
        # to repair, and the distiller must be able to tell them apart
        fill(df.source.astype(str).str.startswith("generated_natural"), "natural")
        fill(df.source.astype(str).str.startswith("generated_adversarial"), "adversarial")
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


# Per-rule single-edit evaluation was REMOVED (2026-08-10, user decision): its
# cost scaled with the rule count and the measurement is of the book, not of its
# parts -- an LLM adherence judge reads the eval file instead. The historical
# cache artifact results/rules_runs/_rule_cache.parquet stays: its 66 rows are
# the measured 19.7% pass-through rate that config/params.py CachePolicy cites.


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
             effort=None):
    """session: a uuid to pin/resume a conversation. Reasoning roles (distill,
    judge, plan) share ONE session per pass so the distiller can refer back to
    everything it has already seen and concluded. Apply calls pass session=None
    deliberately -- they must not see each other's phrases (cross-contamination),
    and being stateless is what lets them run in parallel."""
    if effort is None:
        effort = run.claude_effort
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
        # transient network/server errors must not kill a multi-hour run: the
        # 2026-08-31 r1 launch died on a single unretried httpx.ReadTimeout
        def _gen_with_retry(**kw):
            import time as _t
            for attempt in range(6):
                try:
                    return client.models.generate_content(**kw)
                except Exception as e:
                    name = type(e).__name__
                    retryable = any(x in name for x in ("Timeout", "ServerError",
                                                        "ConnectError", "ReadError"))                         or "429" in str(e) or "500" in str(e) or "503" in str(e)
                    if attempt == 5 or not retryable:
                        raise
                    _t.sleep(min(5 * 2 ** attempt, 60))
        # thinking_budget > 0 is sent verbatim; <= 0 means "model default" --
        # pro-class Gemini models are thinking-only and reject an explicit 0
        # (400 INVALID_ARGUMENT), while flash-class models accept it. Minimum
        # explicit budget on pro models is 128.
        gcfg = dict(temperature=0.0,
                    http_options=types.HttpOptions(timeout=timeout * 1000))
        if run.gemini_thinking > 0:
            gcfg["thinking_config"] = types.ThinkingConfig(
                thinking_budget=run.gemini_thinking)
        resp = _gen_with_retry(
            model=run.gemini_model, contents=prompt,
            config=types.GenerateContentConfig(**gcfg))
        response = (resp.text or "").strip()
    else:
        raise ValueError(f"no local backend for {backend} (qwen goes through pod jobs)")
    _log_llm(run, tag, prompt, response)
    return response


_TRACES = {}


def _sanitize_trace(t):
    """Cut the trailing 'Original Instruction:' block that the trace-generation
    template appended (user-found leak, 2026-08-31): for LEGACY traces that
    block restates the CANONICAL instruction -- a verbatim answer key sitting
    inside the scene analysis, which the applier demonstrably copies from.
    Applied to both sources; on per-base traces it only removes a harmless
    echo of the base."""
    t = str(t)
    i = t.rfind("Original Instruction:")
    return t[:i].rstrip() if i >= 0 else t


def load_traces(env=None):
    """Scene descriptions for rule application. Expensive to generate, so they
    are loaded once and reused.

    TWO sources, and the distinction is the whole point:

      PER-BASE (traces_rules_v1.parquet, keyed (task, phrase)) -- generated FROM
      the base phrase itself. An adversarial base gets a scene description read
      through its own wording.

      LEGACY (cover35_teacher_train.parquet, keyed instruction) -- all 2,000 were
      generated from the ORIGINAL instruction, so handing one to a natural or
      adversarial base leaks the canonical vocabulary into the input the
      rulebook is supposed to repair. Kept only as a fallback for bases with no
      per-base trace yet.

    Returns a dict whose keys are BOTH (task, phrase) tuples and bare strings;
    trace_for() below encodes the precedence."""
    global _TRACES
    if _TRACES:
        return _TRACES
    legacy = REPO / (env or ENVIRONMENTS["bridge_pi0"])["traces_legacy"]
    per_base = REPO / (env or ENVIRONMENTS["bridge_pi0"])["traces_per_base"]
    out = {}
    if legacy.exists():
        t = pd.read_parquet(legacy, columns=["instruction", "trace"])
        t = t.drop_duplicates("instruction")
        out.update({i: _sanitize_trace(tr) for i, tr in zip(t.instruction, t.trace)})
    n_legacy = len(out)
    if per_base.exists():
        p = pd.read_parquet(per_base, columns=["task", "phrase", "trace"])
        p = p.drop_duplicates(["task", "phrase"])
        out.update({(str(r.task), str(r.phrase)): _sanitize_trace(r.trace)
                    for r in p.itertuples()})
        print(f"[traces] {len(p)} per-base + {n_legacy} legacy-by-instruction")
    else:
        print(f"[traces] {n_legacy} legacy-by-instruction ONLY -- every base will "
              f"be conditioned on a trace built from the ORIGINAL instruction, "
              f"which leaks canonical vocabulary into hostile inputs")
    _TRACES = out
    return _TRACES


def trace_for(traces, task, phrase):
    """Precedence: per-base first, then the legacy instruction-keyed fallbacks."""
    return (traces.get((str(task), str(phrase)))
            or traces.get(task)
            or traces.get(phrase)
            or "(no scene description available)")


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
        self.claude_effort = "high"
        self.apply_effort = "medium"
        self.gemini_model = "gemini-pro-latest"
        self.gemini_thinking = 0
        self.env = None            # resolved ENVIRONMENTS entry, set at run start
        (self.dir / "jobs").mkdir(parents=True, exist_ok=True)
        self.cfg_path = self.dir / "config.json"

    def config(self, args):
        if self.cfg_path.exists():
            return jread(self.cfg_path)
        cfg = {
            "run_id": self.id, "created": time.strftime("%Y-%m-%d %H:%M"),
            "seed": args.seed, "patience": args.patience, "max_iters": args.max_iters,
            "sample_n": args.sample_n,
            "score_budget": args.score_budget,
            "contexts_per_task": args.contexts_per_task,
            "frames_per_episode": args.frames_per_episode,
            "min_eps_per_task": args.min_eps_per_task,

            "env": args.env, "phase": args.phase,
            "rephrasers": args.rephrasers.split(","),
            "distiller": args.distiller, "judge": args.distiller,
            "claude_model": args.claude_model,
            "claude_effort": args.claude_effort,
            "apply_effort": args.apply_effort,
            "gemini_model": args.gemini_model,
            "gemini_thinking_budget": args.gemini_thinking_budget,
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
        cached = pd.read_parquet(bank_path)
        # The freeze is deliberate (iteration N's bases must not be iteration
        # N-1's outputs), so a cached bank is returned as-is even when its
        # inputs have moved on -- but SILENTLY doing so is how live1 nearly
        # distilled from a two-day-old bank with zero natural/adversarial
        # phrases. Warn loudly; deleting the file (fresh run) is the remedy.
        srcs = [REPO / run.env["worklist"],
                *sorted((REPO / "results/analysis").glob(run.env["bank_scores_glob"]))]
        newest = max((f.stat().st_mtime for f in srcs if f.exists()), default=0)
        stale = bank_path.stat().st_mtime < newest
        missing_kinds = "kind" in cached and not \
            cached.kind.isin(["natural", "adversarial"]).any()
        if stale or missing_kinds:
            print(f"[{run.id}] !! bank.parquet is FROZEN but "
                  + ("PREDATES its inputs" if stale else "")
                  + (" and " if stale and missing_kinds else "")
                  + ("holds no natural/adversarial phrases" if missing_kinds else "")
                  + " -- fine mid-run, wrong for a new run. Delete "
                  f"{bank_path} (or use a fresh run id) to rebuild.")
        return cached
    env = run.env
    frames = []
    fe = pd.concat([pd.read_parquet(REPO / f) for f in env["fine_exam_features"]])
    agg = fe.groupby(["task", "phrase"]).agg(z=("z_row", "mean"), grip=("grip_row", "mean")).reset_index()
    pan = pd.read_parquet(REPO / env["fine_exam_panel"])
    agg = agg.merge(pan[["task", "phrase", "gt_success"]], on=["task", "phrase"], how="left")
    agg["source"] = "fine_exam"
    frames.append(agg)
    boards = REPO / env["boards"]
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
    # The generated tiers (natural / adversarial bases) enter HERE. Until
    # 2026-08-10 seed_bank read only fine_exam + search_boards, so even a fresh
    # rebuild contained zero naturals or adversarials -- deleting a stale
    # bank.parquet did not actually fix it. bank_to_score.parquet is the full
    # scoring worklist (task, phrase, source); appended LAST so the richer
    # fine_exam/search rows above win the dedup, and the bank_scores merge
    # below attaches z/grip to whatever is measured.
    wl = REPO / env["worklist"]
    if wl.exists():
        w = pd.read_parquet(wl)[["task", "phrase", "source"]]
        frames.append(w.assign(z=np.nan, grip=np.nan, gt_success=np.nan))
    bank = pd.concat(frames, ignore_index=True)
    sealed = env["sealed_stems"]
    bank = bank[~bank.task.map(lambda t: any(x in str(t) for x in sealed))] \
        .drop_duplicates(["task", "phrase"])

    # Freshly measured channels supersede the seeded ones. The search boards
    # recorded grip ONLY, so 87% of the seed's z was imputed from a column mean
    # and the distiller would have been reading spread that was partly filled in.
    # bank_scores_*.parquet carries both channels, measured at F=4 C=16 -- the
    # same aggregation the calibration was fitted on.
    meas = [pd.read_parquet(f) for f in
            sorted((REPO / "results/analysis").glob(env["bank_scores_glob"]))]
    if meas:
        m = (pd.concat(meas, ignore_index=True)
             .dropna(subset=["z", "grip"])
             .drop_duplicates(["task", "phrase"], keep="last"))
        if "n_ctx" not in bank:
            bank["n_ctx"] = np.nan
        bank = bank.merge(m[["task", "phrase", "z", "grip", "n_ctx"]],
                          on=["task", "phrase"], how="left", suffixes=("", "_m"))
        hit = bank.z_m.notna() & bank.grip_m.notna()
        for c in ("z", "grip", "n_ctx"):
            bank.loc[hit, c] = bank.loc[hit, f"{c}_m"]
        # a SEPARATE column, not an append to `source`: label_kinds matches
        # source exactly ("search_boards"), so decorating it silently reclassified
        # those phrases as `unknown` and cost the distiller the kind label
        bank["remeasured"] = hit
        bank = bank.drop(columns=[c for c in bank.columns if c.endswith("_m")])
        print(f"[{run.id}] bank: {int(hit.sum())}/{len(bank)} phrases carry freshly "
              f"measured z+grip ({len(m)} measurements on disk)")

    bank["proxy"] = recompute_proxy(bank)
    bank["iter_added"] = -1
    bank = label_kinds(bank)
    bank.to_parquet(bank_path, index=False)
    print(f"[{run.id}] phrase kinds: "
          + ", ".join(f"{k}={v}" for k, v in bank.kind.value_counts().items()))
    return bank


import fcntl


def bank_add(run, df):
    """Accumulate measurements for a phrase rather than discarding either copy.

    A phrase measured twice should end up MORE precisely known, not overwritten:
    n_ctx adds, and z/grip become n-weighted means. (The previous
    drop_duplicates(keep="first") silently threw away every re-measurement, which
    made re-scoring for significance impossible.)"""
    bank_path = run.dir / "bank.parquet"
    # concurrent applier passes (user 2026-08-31) share one bank: serialize the
    # read-modify-write or simultaneous passes silently drop each other's rows
    lockf = open(run.dir / ".bank.lock", "w")
    fcntl.flock(lockf, fcntl.LOCK_EX)
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
    fcntl.flock(lockf, fcntl.LOCK_UN)
    lockf.close()
    return merged


# --- pod jobs (score / apply) ----------------------------------------------
def run_job(run, kind, payload: pd.DataFrame, spec: dict, tag: str, timeout=28800):
    # 8h: a score job runs ~4h and can queue behind another on the shared
    # worker fleet (7200 killed the gemini driver on 2026-08-31)
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
        if kind == "score" and spec.get("method") == "rollout":
            out["z"] = np.nan
            out["grip"] = np.nan
            out["gt_success"] = (100 * h).round(1)
            out["n_ctx"] = len(spec.get("rollout", {}).get("episode_ids", [])) or 18
        elif kind == "score":
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


_RVG = {"n": 0, "rhos": []}
# measured previously: the two-channel formula and gripper-only agree at rank
# correlation 0.645 within groups of 16 (76.3% pair agreement). A live run that
# falls far below that is not measuring what the calibration measured.
RVG_FLOOR = 0.50


def check_reward_vs_gripper(run, out, tag):
    """Assert the calibrated reward still ranks phrases roughly as gripper-only
    does. The proxy is a two-channel blend; if the verifier channel is broken or
    mis-scaled, the blend can look plausible while ranking nothing like the
    channel we know works. Checked on the first few real scoring jobs, then
    reported."""
    if run.dry or run.mock_scoring or "grip" not in out or "z" not in out:
        return
    rhos = []
    for t, g in out.groupby("task"):
        # rank on the LOGIT rather than the proxy: on training frames the sigmoid
        # sits flat near 1.0, and ties there would depress the correlation for a
        # numerical reason rather than a measurement one. The two are
        # rank-identical wherever the sigmoid is not saturated.
        g = g.dropna(subset=["z", "grip"]).assign(
            _lg=lambda d: proxy_logit(d.z.astype(float), d.grip.astype(float)))
        if len(g) < 4:
            continue
        r = g._lg.rank().corr((-g.grip).rank(), method="spearman")
        if pd.notna(r):
            rhos.append(float(r))
    if not rhos:
        return
    rho = float(np.mean(rhos))
    _RVG["n"] += 1
    _RVG["rhos"].append(rho)
    if _RVG["n"] <= 5:
        run_mean = float(np.mean(_RVG["rhos"]))
        print(f"    [reward check {_RVG['n']}/5] rank corr(calibrated reward, "
              f"gripper-only) = {rho:.3f} over {len(rhos)} task(s); "
              f"run mean {run_mean:.3f} (prior measurement 0.645)")
        jwrite(run.dir / "reward_vs_gripper.json",
               {"per_job": _RVG["rhos"], "mean": run_mean, "floor": RVG_FLOOR,
                "prior_measurement": 0.645})
        if _RVG["n"] == 3 and run_mean < RVG_FLOOR:
            raise RuntimeError(
                f"calibrated reward is ranking unlike gripper-only "
                f"(rank corr {run_mean:.3f} over the first 3 jobs, floor "
                f"{RVG_FLOOR}, previously 0.645). The verifier channel is the "
                f"likely culprit -- check reward_mode and the ensemble manifest "
                f"before trusting any number from this run.")


def score_phrases(run, cfg, df, tag, draw=0):
    """df: [task, phrase] -> measured channels. Method rides the phase:
    train -> proxy (adds z, grip; CRN contexts, worker samples
    cfg['contexts_per_task'] per task with seed cfg['seed']);
    sim   -> real rollouts via the env's rollout recipe (adds gt_success 0-100,
    n_ctx = episodes; z/grip stay NaN)."""
    method = "rollout" if cfg.get("phase") == "sim" else "proxy"
    spec = {"method": method, "draw": draw, "seed": cfg["seed"] + 1009 * draw,
            "proxy": PROXY}
    if method == "proxy":
        spec.update({"score_budget": cfg.get("score_budget", 64),
                     "contexts_per_task": cfg["contexts_per_task"],
                     "frames_per_episode": cfg["frames_per_episode"],
                     "pool_val8_stems": True})
    else:
        spec.update({"rollout": run.env["rollout"]})
    out = run_job(run, "score", df[["task", "phrase"]].drop_duplicates(), spec, tag)
    check_reward_vs_gripper(run, out, tag)
    keep = [c for c in ("task", "phrase", "z", "grip", "proxy", "gt_success",
                        "n_ctx") if c in out]
    res = df.drop(columns=[c for c in ("z", "grip", "proxy", "gt_success", "n_ctx")
                           if c in df],
                  errors="ignore").merge(out[keep], on=["task", "phrase"], how="left")
    res["n_meas"] = 1
    res["draw"] = draw
    return res


def apply_rules(run, cfg, rephraser, rules, bases: pd.DataFrame, tag):
    """bases: [task, phrase(base instruction)] -> adds rewrite column."""
    if rephraser == "qwen" and not run.dry:
        # rules travel as a file for the pod, but their hash goes in the spec so
        # the job id changes when the rulebook does
        return run_job(run, "apply", bases[["task", "phrase"]], {
            "rules_file": str((run.dir / f"current_rules_{rephraser}.md").relative_to(REPO)),
            "rules_sha": hashlib.sha1(rules.encode()).hexdigest()[:12]}, tag)
    rules = rules_only(rules)
    # persistent apply cache: a restarted driver must reuse rewrites, not
    # regenerate them -- temperature makes regenerated rewrites differ, which
    # re-keys the score job and orphans hours of scoring (2026-08-31)
    ck = hashlib.sha1((rephraser + "\x00" + rules + "\x00"
                       + "\x00".join(f"{r.task}\x01{r.phrase}"
                                     for r in bases.itertuples())).encode()).hexdigest()[:12]
    cache_p = run.dir / f"apply_cache_{rephraser}_{tag}_{ck}.parquet"
    if cache_p.exists():
        return pd.read_parquet(cache_p)
    traces = load_traces()
    jobs = []
    for r in bases.itertuples():
        jobs.append((r.task, r.phrase, prompt_from(
            "apply.md", rules=rules, phrase=r.phrase,
            trace=trace_for(traces, r.task, r.phrase))))
    rows = [None] * len(jobs)

    def one(i):
        task, phrase, p = jobs[i]
        out = call_llm(run, rephraser, p, f"{tag}_apply", session=None,
                       effort=run.apply_effort)
        return i, {"task": task, "phrase": phrase, "rewrite": out.split("\n")[0].strip()}

    # stateless -> safe to run concurrently; this is the loop's dominant cost
    with cf.ThreadPoolExecutor(max_workers=1 if run.dry else APPLY_WORKERS) as ex:
        for i, rec in ex.map(one, range(len(jobs))):
            rows[i] = rec
    out = pd.DataFrame(rows)
    out.to_parquet(cache_p, index=False)
    return out


def write_probe_results(run, bank, tasks, path, since_iter):
    """Answers to the distiller's own experiments: probe phrases proposed via
    plan.md, measured since the last distillation. Split from the rewrite
    outcomes -- "what did my questions return" and "what did my rulebook do"
    are different registers and are read differently."""
    hdr = "task,phrase,kind,score,logit,z,grip,n_ctx\n"
    if "iter_added" not in bank or "source" not in bank:
        Path(path).write_text(hdr)
        return 0
    fresh = bank[bank.task.isin(tasks)
                 & bank.source.eq("probe")
                 & (pd.to_numeric(bank.iter_added, errors="coerce") >= since_iter)].copy()
    if not len(fresh):
        Path(path).write_text(hdr)
        return 0
    fresh["score"], fresh["logit"] = within_task_score(fresh)
    cols = [c for c in ("task", "phrase", "kind", "score", "logit", "z", "grip",
                        "n_ctx") if c in fresh]
    fresh.sort_values(["task", "logit"], ascending=[True, False])[cols].to_csv(
        path, index=False)
    return len(fresh)


def write_rewrite_outcomes(run, bank, path, since_iter, same_draw=None):
    """The rulebook's own rewrites, PAIRED with the bases they rewrote: one row
    per (base -> rewrite) with both measured logits and the delta. This is the
    per-phrase outcome record of the distiller's last move -- the scalar delta
    says whether the book gained; these rows say WHERE, and base_kind says on
    which input regime."""
    hdr = ("task,base_kind,base,base_logit,rewrite,rewrite_logit,delta,n_ctx\n")
    if "iter_added" not in bank or "source" not in bank:
        Path(path).write_text(hdr)
        return 0
    rw = bank[bank.source.astype(str).str.startswith("loop_")
              & (pd.to_numeric(bank.iter_added, errors="coerce") >= since_iter)].copy()
    if not len(rw) or "base" not in rw:
        Path(path).write_text(hdr)
        return 0
    rw["rewrite_logit"] = proxy_logit(rw.z.astype(float), rw.grip.astype(float))
    base_lg = bank.dropna(subset=["z", "grip"]).copy()
    base_lg["base_logit"] = proxy_logit(base_lg.z.astype(float), base_lg.grip.astype(float))
    base_lg = base_lg.drop_duplicates(["task", "phrase"], keep="last")[
        ["task", "phrase", "base_logit"]].rename(columns={"phrase": "base"})
    rw = rw.merge(base_lg, on=["task", "base"], how="left")
    if same_draw:   # same-draw baseline beats bank history where available
        sd = rw.apply(lambda r: same_draw.get((str(r.task), str(r.base))), axis=1)
        rw["base_logit"] = sd.combine_first(rw.base_logit)
        rw["same_draw"] = sd.notna()
    rw["delta"] = rw.rewrite_logit - rw.base_logit
    out = rw.rename(columns={"phrase": "rewrite"})
    cols = [c for c in ("task", "base_kind", "base", "base_logit", "rewrite",
                        "rewrite_logit", "delta", "same_draw", "n_ctx") if c in out]
    for c in ("base_logit", "rewrite_logit", "delta"):
        out[c] = pd.to_numeric(out[c], errors="coerce").round(4)
    out.sort_values(["base_kind", "delta"], ascending=[True, True])[cols].to_csv(
        path, index=False)
    return len(out)


def write_evidence_file(run, bank, tasks, path):
    """The full spread per task as CSV. CSV over JSON deliberately: this table
    runs to thousands of rows, where JSON's repeated keys triple the tokens and
    make it harder to grep; and the agent has a shell, so a table it can sort and
    group beats a structure it must parse by eye. A short README sits beside it."""
    sub = bank[bank.task.isin(tasks)].copy()
    sub["score"], sub["logit"] = within_task_score(sub)
    # rollout-measured rows have no channels: rank them by gt within the task
    if "gt_success" in sub and sub.score.isna().any():
        gt = pd.to_numeric(sub.gt_success, errors="coerce")
        def mmgt(x):
            lo, hi = x.min(), x.max()
            return (x - lo) / (hi - lo) if pd.notna(lo) and pd.notna(hi) and hi > lo \
                else pd.Series(np.nan, index=x.index)
        gtr = gt.groupby(sub.task).transform(mmgt)
        sub["score"] = sub.score.fillna(gtr)
    sub = sub.sort_values(["task", "score"])
    if "n_ctx" not in sub:
        sub["n_ctx"] = FALLBACK_NCTX
    # a phrase measured on few contexts is a weak claim; the reader needs to see
    # that to decide whether a gap is real or worth re-measuring
    sub["n_ctx"] = sub.n_ctx.fillna(FALLBACK_NCTX).round().astype(int)
    for c in ("logit", "z", "grip"):
        if c in sub:
            sub[c] = pd.to_numeric(sub[c], errors="coerce").round(4)
    cols = ["task", "phrase", "kind", "base_kind", "score", "logit", "z", "grip",
            "proxy_imputed", "n_ctx", "gt_success", "source"]
    sub[[c for c in cols if c in sub.columns]].to_csv(path, index=False)
    # per-task summary: the aggregation an agent would otherwise need a shell for
    agg = sub.groupby("task").agg(
        phrases=("phrase", "size"),
        # on the LOGIT, not the normalised score: min-maxing inside a task makes
        # every task's spread exactly 1.0, which would flatten the very ranking
        # plan.md selects on. The logit is one common scale across tasks.
        best=("logit", "max"), worst=("logit", "min"),
        median=("logit", "median"),
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
        "  score      THE COLUMN TO RANK BY WITHIN A TASK. 0-1 within its task:\n"
        "             1.0 is the best phrase measured for that task, 0.0 the\n"
        "             worst. NOT comparable across tasks, and not a success\n"
        "             rate. Blank when a task has only one measured phrase.\n"
        "  logit      the raw proxy value that `score` normalises: a fixed\n"
        "             linear mixture of the two measured channels below, on one\n"
        "             common scale across tasks. Higher is better. It is NOT a\n"
        "             probability and NOT a success rate; treat differences,\n"
        "             not levels, as meaningful.\n"
        "  z          raw verifier-ensemble channel (higher = the trajectory\n"
        "             matches the instruction better, as judged by a frozen\n"
        "             learned verifier).\n"
        "  grip       raw gripper-error channel (LOWER is better: mean distance\n"
        "             between the policy's gripper action and the demonstrated\n"
        "             one). The two channels are independent measurements; a\n"
        "             phrase strong on one and weak on the other is a real\n"
        "             pattern worth noting, not an error.\n"
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
        "  proxy_imputed  True when one channel was missing for this row and\n"
        "             was filled with the column mean. Those rows are driven by\n"
        "             the surviving channel alone -- weaker evidence.\n"
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
        "             Where present, trust this over every estimated column.\n"
        "  source     where the measurement came from\n\n"
        "\nevidence_summary.csv sits beside it: one row per task with phrase count,\n"
        "best/worst/median logit, SPREAD (best-worst, in logit units), how many\n"
        "phrases carry a real rollout number, and how many rest on thin sampling.\n"
        "Sorted by spread, descending -- the tasks where wording matters most are\n"
        "at the top. Read it first, then go into evidence.csv for the tasks it\n"
        "points you to. The within-task spread is the signal, not the extremes.\n")
    return len(sub)


def write_eval_file(run, path, rules, pairs, base_mean, rules_mean):
    """Each sample pair carries its measured numbers (judge.md promises them):
    the rewrite's logit, the base's logit from the bank where measured, and the
    per-pair delta. A pair whose base was never measured shows the rewrite
    number alone rather than a fabricated delta."""
    bank = pd.read_parquet(run.dir / "bank.parquet")
    bank = bank.dropna(subset=["z", "grip"]).drop_duplicates(["task", "phrase"], keep="last")
    bank_bl = {(str(r.task), str(r.phrase)): float(proxy_logit(r.z, r.grip))
               for r in bank.itertuples()}
    same_draw = getattr(write_eval_file, "_same_draw", {})
    for p in pairs:
        key = (str(p["task"]), str(p["base"]))
        p["base_logit"] = same_draw.get(key, bank_bl.get(key))
        p["base_logit_same_draw"] = key in same_draw
        if p.get("rewrite_logit") is not None and p["base_logit"] is not None:
            p["delta"] = round(p["rewrite_logit"] - p["base_logit"], 4)
    jwrite(Path(str(path).replace(".md", ".json")),
           {"whole_rulebook": {"with_rules": rules_mean, "unrephrased": base_mean,
                               "delta": rules_mean - base_mean},
            "samples": pairs})
    lines = ["# Previous rulebook: measured performance", "",
             f"Whole rulebook applied: mean proxy logit {rules_mean:.3f} vs "
             f"{base_mean:.3f} for the unrephrased instruction "
             f"({rules_mean - base_mean:+.3f}; higher is better, not a "
             f"probability).", ""]
    lines += ["## Sample rewrites from the whole rulebook",
              "(logit: higher is better; delta = rewrite - base, positive means",
              "the rewrite beat leaving the instruction alone)", ""]
    for p in pairs:
        rl = p.get("rewrite_logit")
        blg = p.get("base_logit")
        d = p.get("delta")
        num = "    logit: " + (f"{rl:.3f}" if rl is not None else "unmeasured")
        if blg is not None:
            num += f"  (base {blg:.3f}" + (f", delta {d:+.3f})" if d is not None else ")")
        lines.append(f"  [{p['task']}]")
        lines.append(f"    in : {p['base']!r}")
        lines.append(f"    out: {p['rewrite']!r}")
        lines.append(num)
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


CORPUS_CACHE = REPO / "results/analysis/corpus_cache"


def corpus_cache_key(staged_dir):
    h = hashlib.sha1((PROMPTS / "corpus.md").read_bytes())
    for f in sorted(staged_dir.iterdir()):
        if f.is_file():
            h.update(f.name.encode())
            h.update(f.read_bytes())
    return h.hexdigest()[:12]


def ensure_corpus_file(run, cfg):
    """Deterministic vocabulary table + an agent-written qualitative section.
    The commentary is REUSED across runs (user 2026-08-31): cached under
    results/analysis/corpus_cache keyed on the staged inputs + prompt, so a new
    run (or the sim phase) pays zero LLM calls unless the corpus changed."""
    out = run.dir / "corpus_stats.md"
    if out.exists():
        staged = run.dir / "corpus_inputs"
        if staged.exists() and not run.dry:   # backfill the shared cache
            CORPUS_CACHE.mkdir(parents=True, exist_ok=True)
            c = CORPUS_CACHE / f"corpus_stats_{corpus_cache_key(staged)}.md"
            if not c.exists():
                c.write_bytes(out.read_bytes())
        return out
    src_dir = REPO / run.env["corpus_inputs"]
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

    CORPUS_CACHE.mkdir(parents=True, exist_ok=True)
    cached = CORPUS_CACHE / f"corpus_stats_{corpus_cache_key(staged)}.md"
    if cached.exists():
        out.write_bytes(cached.read_bytes())
        print(f"[{run.id}] corpus commentary reused from cache ({cached.name})")
        return out

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
                        timeout=900)
    except Exception as e:
        print(f"[{run.id}] corpus commentary failed ({type(e).__name__}); "
              f"vocabulary table still written")
        qual = "(commentary unavailable)"
    out.write_text(vocab + "\n---\n\n" + qual + "\n")
    cached.write_bytes(out.read_bytes())
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
    corrupt the novelty tracking that diffs rulebooks across iterations, so a
    missing RULES delimiter is a hard failure rather than a silent fallback
    over the whole text."""
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


def eval_metric(cfg, scored):
    if cfg.get("phase") == "sim":
        return float(np.nanmean(pd.to_numeric(scored.gt_success, errors="coerce"))) / 100.0
    return float(np.nanmean(proxy_logit(scored.z.astype(float), scored.grip.astype(float))))


def baseline_proxy(run, cfg, bases, tag):
    """Mean LOGIT of the UNREPHRASED base phrases -- scored once per split and
    cached, since the bases are fixed for the whole run. Cache name carries
    _logit so a resumed pre-change run can never mix units."""
    bh = hashlib.sha1(pd.util.hash_pandas_object(
        bases[["task", "phrase"]]).values.tobytes()).hexdigest()[:8]
    cache = run.dir / f"baseline_{tag}_{bh}_logit.json"
    rows = run.dir / f"baseline_{tag}_{bh}_rows.parquet"
    if cache.exists() and rows.exists():
        return jread(cache)["mean"]
    sc = score_phrases(run, cfg, bases[["task", "phrase"]], f"baseline_{tag}")
    # per-phrase rows persist so pair deltas compare SAME-DRAW measurements
    # (baseline and eval rewrites both score at draw 0): the bank's historical
    # values were measured under unrecorded, heterogeneous draws and mixing
    # them into a pair delta puts several logit units of context-set offset
    # into a number that should reflect only the text change.
    sc.to_parquet(rows, index=False)
    mean = eval_metric(cfg, sc)
    jwrite(cache, {"mean": mean, "n": int(len(sc))})
    return mean


def baseline_rows(run, tag, bases):
    """(task, phrase) -> same-draw base logit from the cached baseline job."""
    bh = hashlib.sha1(pd.util.hash_pandas_object(
        bases[["task", "phrase"]]).values.tobytes()).hexdigest()[:8]
    p = run.dir / f"baseline_{tag}_{bh}_rows.parquet"
    if not p.exists():
        return {}
    d = pd.read_parquet(p).dropna(subset=["z", "grip"])
    return {(str(r.task), str(r.phrase)): float(proxy_logit(r.z, r.grip))
            for r in d.itertuples()}


def eval_rules(run, cfg, itdir, rephraser, rules, bases, tag, judge=False):
    """Returns (mean proxy score of rewrites, scored df, rules_eval_summary).

    There is NO per-rule measurement. The book is measured, not its parts: the
    old single-edit path applied every rule alone to a sample (SINGLE_EDIT_N x R
    applies and scorings per iteration) and its cost scaled with the rule count.
    What replaced it is the `judge` flag -- an LLM reads the whole-rulebook
    numbers and sample rewrites and reports on ADHERENCE (were the rules
    followed), which is the part of the old measurement that was actually being
    consumed. If one rule needs isolating, that is a new measurement to request,
    not a standing cost."""
    rsha = hashlib.sha1(rules.encode()).hexdigest()[:10]
    art = itdir / f"eval_{tag}_{rsha}.json"
    scored_art = itdir / f"eval_{tag}_{rsha}.parquet"
    if art.exists():
        return jread(art)["score"], pd.read_parquet(scored_art), jread(art).get("summary")
    rewrites = apply_rules(run, cfg, rephraser, rules, bases, f"{tag}")
    rw = rewrites.rename(columns={"phrase": "base", "rewrite": "phrase"})[["task", "phrase", "base"]]
    scored = score_phrases(run, cfg, rw, f"{tag}_sc")
    # phase train: mean LOGIT, not mean sigmoid (user 2026-08-30) -- on training
    # frames the sigmoid pins 73% of phrases above 0.99. phase sim: the metric
    # IS measured success (0-1); no proxy anywhere.
    score = eval_metric(cfg, scored)
    summary = None
    if judge:
        eval_file = itdir / "rules_eval.md"
        rlg = {(str(r.task), str(r.phrase)): float(proxy_logit(r.z, r.grip))
               for r in scored.dropna(subset=["z", "grip"]).itertuples()}
        write_eval_file._same_draw = baseline_rows(run, tag, bases)
        write_eval_file(run, eval_file, rules,
                        [{"task": r.task, "base": r.phrase, "rewrite": r.rewrite,
                          "rewrite_logit": rlg.get((str(r.task), str(r.rewrite)))}
                         for r in rewrites.head(40).itertuples()],
                        baseline_proxy(run, cfg, bases, tag), score)
        judge_p = prompt_from("judge.md", rules=rules_only(rules), eval_file=eval_file)
        judgement = call_llm(run, cfg["judge"], judge_p, f"{tag}_judge",
                             session=run.session, add_dir=run.dir)
        (itdir / "judge.md").write_text(judgement)
        summary = {"judge": judgement,
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
    ax2.set_xlabel("iteration"); ax2.set_ylabel("logit gain over unrephrased")
    ax2.set_title("do the rules beat saying nothing?", fontsize=10)
    ax2.legend(fontsize=8); ax2.grid(alpha=0.25)
    ax.set_xlabel("iteration"); ax.set_ylabel("mean proxy logit")
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
    ap.add_argument("--claude-effort", default="high",
                    choices=["low", "medium", "high", "max"],
                    help="reasoning effort for the distiller/judge/planner/corpus")
    ap.add_argument("--apply-effort", default="medium",
                    choices=["low", "medium", "high", "max"],
                    help="effort for rephraser=claude apply calls")
    ap.add_argument("--gemini-model", default="gemini-pro-latest",
                    help="model for rephraser=gemini")
    ap.add_argument("--gemini-thinking-budget", type=int, default=128,
                    help="thinking budget for rephraser=gemini apply calls -- the "
                         "Gemini analog of effort. <=0 = model default; explicit "
                         "minimum on pro-class models is 128 (they reject 0)")
    ap.add_argument("--max-probes", type=int, default=20)
    ap.add_argument("--max-train-tasks", type=int, default=0,
                    help="cap the training pool (0 = all); the val splits are "
                         "unaffected")
    ap.add_argument("--init-rules-from", default=None, metavar="RUN_ID",
                    help="start each pass from RUN_ID's best rulebook for the same "
                         "model (round 2 initializes from round 1 this way)")
    ap.add_argument("--no-rollback", action="store_true",
                    help="keep revising the latest rulebook even after a "
                         "validation regression (default: revise the best)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--env", default="bridge_pi0", choices=sorted(ENVIRONMENTS),
                    help="which (policy, dataset) pair to run against")
    ap.add_argument("--phase", default="train", choices=["train", "sim"],
                    help="train: proxy-scored on the training corpus, val_held+val8 "
                         "splits, early stop on their mean. sim: ROLLOUT-scored on "
                         "the env's sim tasks, no validation split -- runs to "
                         "max_iters (seed rules via --init-rules-from)")
    args = ap.parse_args()

    run = Run(args.run_id, args.dry_run, mock_scoring=args.mock_scoring)
    cfg = run.config(args)
    run.env = resolve_env(cfg.get("env", "bridge_pi0"))
    phase = cfg.get("phase", "train")
    run.claude_model = cfg.get("claude_model", "claude-opus-5")
    run.claude_effort = cfg.get("claude_effort", "high")
    run.apply_effort = cfg.get("apply_effort", "medium")
    run.gemini_model = cfg.get("gemini_model", "gemini-pro-latest")
    run.gemini_thinking = int(cfg.get("gemini_thinking_budget", 0))
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
    have_trace = set(load_traces(run.env))
    sim_tasks = run.env["sim_tasks"]
    tasks = sorted(t for t in bank.task.unique() if t not in sim_tasks
                   and support.get(t, 0) >= cfg["min_eps_per_task"]
                   and (t in have_trace or run.dry))
    dropped = bank.task.nunique() - len(tasks) - len(VAL8_TASKS)
    print(f"[{run.id}] task support floor >={cfg['min_eps_per_task']} eps: "
          f"{len(tasks)} training tasks admitted, {dropped} dropped (recorded in splits.json)")
    rng.shuffle(tasks)
    # Prefer tasks whose phrases carry freshly measured channels. The loop exists
    # to reason about measured spread; a task still resting on imputed z hands the
    # distiller a table it cannot learn from. Python's sort is stable, so the
    # shuffle still randomises within each group.
    if "remeasured" in bank:
        measured = set(bank.loc[bank.remeasured.fillna(False), "task"])
        tasks.sort(key=lambda t: t not in measured)
        n_meas = sum(t in measured for t in tasks)
        print(f"[{run.id}] {n_meas}/{len(tasks)} admitted tasks carry fresh "
              f"measurements; those are drawn first")
    n_held = max(2, len(tasks) // 6)
    val_held_tasks, train_tasks = tasks[:n_held], tasks[n_held:]
    if args.max_train_tasks:
        train_tasks = train_tasks[:args.max_train_tasks]
        print(f"[{run.id}] training pool capped to {len(train_tasks)} tasks")
    if phase == "sim":
        # sim phase: TRAIN on the sim tasks by rollout; there is no held-out
        # split (user 2026-08-30) -- overfitting is bounded by max_iters, not by
        # validation. The sealed set stays the only certifier.
        train_tasks, val_held_tasks = list(sim_tasks), []
        print(f"[{run.id}] PHASE=sim: training on {len(train_tasks)} sim tasks "
              f"by ROLLOUT; no validation split; runs to max_iters={cfg['max_iters']}")
    jwrite(run.dir / "splits.json", {"train": train_tasks, "val_held": val_held_tasks,
                                     "val8": list(sim_tasks), "phase": phase,
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

    passes = [r for r in args.rephrasers.split(",") if r.strip()]
    cfg.setdefault("invocations", []).append(
        {"when": time.strftime("%Y-%m-%d %H:%M"), "rephrasers": passes})
    jwrite(run.cfg_path, cfg)
    for rephraser in passes:
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
                probe_file = itdir / "probe_results.csv"
                rw_file = itdir / "rewrite_outcomes.csv"
                n_new = write_probe_results(run, bank, train_tasks, probe_file,
                                            since_iter=it - 1)
                n_new += write_rewrite_outcomes(
                    run, bank, rw_file, since_iter=it - 1,
                    same_draw=baseline_rows(run, "train",
                                            bases_for(train_tasks, cfg["sample_n"], seed=0)))
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
                                 probe_file=rel(probe_file), rewrites_file=rel(rw_file),
                                 best_eval_file=rel(best_eval),
                                 regressed_eval_file=rel(regressed_eval))
                print(f"    distilling over {n_ev} measured phrases "
                      f"({len(train_tasks)} tasks), {n_new} newly measured ...")
                rules = None
                for attempt in range(3):
                    cand = call_llm(run, cfg["distiller"], dp, f"{rephraser}_distill",
                                    session=run.session, add_dir=run.dir,
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
            cur = run.dir / f"current_rules_{rephraser}.md"
            cur.write_text(rules)
            if not run.dry and "qwen" in cfg["rephrasers"]:
                # the pod reads this file; it must exist on origin before any
                # apply job referencing it is submitted
                gitsync([cur], f"rules-loop {run.id}/{rephraser} iter {it} rulebook")

            # 2. eval on train sample (judged for adherence)
            # fixed across iterations (seed 0) so the train curve is a comparable
            # series -- new evidence enters through probes, not through resampling
            tb = bases_for(train_tasks, cfg["sample_n"], seed=0)
            train_score, ev_train, summary = eval_rules(
                run, cfg, itdir, rephraser, rules, tb, "train", judge=True)
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

            # 3. eval on both validation sets (train phase). The sim phase has
            # no validation data: its "val" IS the train score, and stopping is
            # governed by max_iters alone.
            if phase == "sim":
                vh = v8 = val = train_score
                deltas = {"train": jread(itdir / "eval_train.json").get("delta"),
                          "val_held": None, "val8": None}
                print(f"    [sim] rollout success={train_score:.3f} "
                      f"(delta vs unrephrased {deltas['train']:+.3f}); "
                      f"iter cap {cfg['max_iters']}")
            else:
                vh, _, _ = eval_rules(run, cfg, itdir, rephraser, rules,
                                      bases_for(val_held_tasks, cfg["sample_n"], seed=1), "val_held")
                v8, _, _ = eval_rules(run, cfg, itdir, rephraser, rules,
                                      bases_for(list(sim_tasks), cfg["sample_n"], seed=2), "val8")
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
                # eagerly: the current-best rulebook is always readable at
                # pass_<applier>/best_rules.md, even mid-run -- this is the
                # artifact the sim phase seeds from (--init-rules-from)
                (pdir / "best_rules.md").write_text(rules)
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
                                   session=run.session, add_dir=run.dir)
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
