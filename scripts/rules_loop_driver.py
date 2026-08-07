#!/usr/bin/env python3
"""Driver for ALGORITHM-RULES-SEARCH-AND-DISTILLATION.md.

Local orchestrator: owns the run directory, the scored bank, splits, the
iteration state machine, early stopping, and verbatim logging of every LLM
exchange. GPU work (proxy scoring, Qwen rule application) is submitted as jobs
through git; a pod running scripts/rules_loop_worker.sh consumes them and
commits results back. LLM calls go through pluggable backends:

    distiller / judge : `claude -p` (headless CLI, no API key needed)
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
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
PROMPTS = REPO / "prompts" / "rules_loop"

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


def proxy_success(z, grip):
    x = PROXY["C"] + PROXY["bz"] * np.asarray(z) + PROXY["bg"] * (-np.asarray(grip))
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


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
def _log_llm(run, tag, prompt, response):
    d = run.dir / "llm_log"
    d.mkdir(exist_ok=True)
    n = len(list(d.glob(f"{tag}_*")))
    (d / f"{tag}_{n:03d}.prompt.txt").write_text(prompt)
    (d / f"{tag}_{n:03d}.response.txt").write_text(response)


def call_llm(run, backend, prompt, tag, timeout=600):
    if run.dry:
        response = f"[dry-run:{backend}] " + hashlib.sha1(prompt.encode()).hexdigest()[:12]
        if tag.startswith("distill"):
            response = ("RULES\n1. Keep the original wording unless a rule below applies.\n"
                        "2. Use common household nouns.\n3. Keep existing color adjectives.\n")
        _log_llm(run, tag, prompt, response)
        return response
    if backend == "claude":
        env = dict(os.environ)
        if tag.split("_")[-1].rstrip("0123456789") in ("distill", "judge", "plan"):
            env["MAX_THINKING_TOKENS"] = "16000"  # high effort where judgment lives
        r = subprocess.run(["claude", "-p", prompt, "--output-format", "text",
                            "--model", "claude-fable-5"],
                           capture_output=True, text=True, timeout=timeout, env=env)
        if r.returncode != 0:
            raise RuntimeError(f"claude CLI failed: {r.stderr[:300]}")
        response = r.stdout.strip()
    elif backend == "gemini":
        from google import genai
        from google.genai import types
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


def prompt_from(name, **kw):
    t = (PROMPTS / name).read_text()
    for k, v in kw.items():
        t = t.replace("{{" + k + "}}", str(v))
    return t


# --- run dir / config -------------------------------------------------------
class Run:
    def __init__(self, run_id, dry):
        self.id = run_id
        self.dry = dry
        self.dir = REPO / "results" / "rules_runs" / run_id
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
            "distiller_model": "claude-fable-5", "apply_model": "claude-fable-5",
            "rephrasers": args.rephrasers.split(","),
            "distiller": "claude", "judge": "claude",
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
    bank["proxy"] = proxy_success(bank.z.fillna(bank.z.mean()), bank.grip.fillna(bank.grip.mean()))
    bank["iter_added"] = -1
    bank.to_parquet(bank_path, index=False)
    return bank


def bank_add(run, df):
    bank_path = run.dir / "bank.parquet"
    bank = pd.read_parquet(bank_path)
    merged = pd.concat([bank, df], ignore_index=True).drop_duplicates(["task", "phrase"], keep="first")
    merged.to_parquet(bank_path, index=False)
    return merged


# --- pod jobs (score / apply) ----------------------------------------------
def run_job(run, kind, payload: pd.DataFrame, spec: dict, tag: str, timeout=7200):
    """Submit a job through git; block until the worker commits the result.
    Resume-safe: if the result already exists, return it without resubmitting."""
    jid = f"{tag}_{hashlib.sha1(pd.util.hash_pandas_object(payload).values.tobytes()).hexdigest()[:10]}"
    jdir = run.dir / "jobs"
    result = jdir / f"{jid}.result.parquet"
    if result.exists():
        return pd.read_parquet(result)
    if run.dry:
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


def score_phrases(run, cfg, df, tag):
    """df: [task, phrase] -> adds z, grip, proxy. CRN: the worker samples
    cfg['contexts_per_task'] contexts per task with seed cfg['seed'] -- same
    contexts for every phrase, all run."""
    out = run_job(run, "score", df[["task", "phrase"]].drop_duplicates(), {
        "score_budget": cfg.get("score_budget", 64),
        "contexts_per_task": cfg["contexts_per_task"],
        "frames_per_episode": cfg["frames_per_episode"],
        "pool_val8_stems": True, "seed": cfg["seed"],
        "proxy": PROXY}, tag)
    return df.drop(columns=[c for c in ("z", "grip", "proxy") if c in df], errors="ignore") \
             .merge(out[["task", "phrase", "z", "grip", "proxy"]], on=["task", "phrase"], how="left")


def apply_rules(run, cfg, rephraser, rules, bases: pd.DataFrame, tag, only_rule=None):
    """bases: [task, phrase(base instruction)] -> adds rewrite column."""
    if rephraser == "qwen" and not run.dry:
        return run_job(run, "apply", bases[["task", "phrase"]], {
            "rules_file": str((run.dir / "current_rules.md").relative_to(REPO)),
            "only_rule": only_rule}, tag)
    tmpl = "apply_single.md" if only_rule else "apply.md"
    rows = []
    for r in bases.itertuples():
        p = prompt_from(tmpl, rules=rules, phrase=r.phrase, task=r.task,
                        only_rule=only_rule or "")
        rows.append({"task": r.task, "phrase": r.phrase,
                     "rewrite": call_llm(run, rephraser, p, f"{tag}_apply").split("\n")[0].strip()})
    return pd.DataFrame(rows)


# --- rule parsing / evaluation ----------------------------------------------
def parse_rules(rules_text):
    return [m.group(1).strip() for m in re.finditer(r"^\s*\d+\.\s+(.+)$", rules_text, re.M)]


def eval_rules(run, cfg, itdir, rephraser, rules, bases, tag, single_edit=False):
    """Returns (mean proxy score of rewrites, scored df, rules_eval_summary)."""
    art = itdir / f"eval_{tag}.json"
    scored_art = itdir / f"eval_{tag}.parquet"
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
        for k, rtext in enumerate(rule_list, 1):
            rw = apply_rules(run, cfg, rephraser, rules, sample, f"{tag}_r{k}", only_rule=rtext)
            rs = score_phrases(run, cfg, rw.rename(columns={"rewrite": "phrase"})[["task", "phrase"]],
                               f"{tag}_r{k}sc")
            per_rule[f"rule_{k}"] = {
                "text": rtext,
                "delta_proxy": float(rs.proxy.mean() - base_scored.proxy.mean()),
                "n": int(len(rs)),
            }
        judge_p = prompt_from("judge.md", rules=rules,
                              pairs="\n".join(f"[{r.task}] {r.phrase!r} -> {r.rewrite!r}"
                                              for r in rewrites.head(40).itertuples()))
        judgement = call_llm(run, cfg["judge"], judge_p, f"{tag}_judge")
        summary = {"per_rule_perf": per_rule, "judge": judgement,
                   "suggestions": judgement.split("SUGGESTIONS")[-1].strip()
                   if "SUGGESTIONS" in judgement else ""}
    scored.to_parquet(scored_art, index=False)
    jwrite(art, {"score": score, "summary": summary})
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
    fig, ax = plt.subplots(figsize=(7, 4))
    for key, col in [("train", "#a0aec0"), ("val_held", "#2b6cb0"),
                     ("val8", "#805ad5"), ("val_avg", "#0d9488")]:
        ax.plot(xs, [r[key] for r in rows], "o-", color=col,
                lw=2.4 if key == "val_avg" else 1.4, label=key)
    ax.set_xlabel("iteration"); ax.set_ylabel("mean proxy success")
    ax.set_title(f"{run.id} / {rephraser} -- early stop on val_avg")
    ax.legend(fontsize=8); ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(pdir / "progress.png", dpi=130)
    plt.close(fig)


# --- main loop ---------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--dry-run", action="store_true")
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
                    help="F floor (the proxy calibration's F). When a task has "
                         "fewer episodes than the C cap, F rises to spend the "
                         "budget: F = clamp(budget/C, this, 16)")
    ap.add_argument("--min-eps-per-task", type=int, default=8,
                    help="training-task support floor (same episode-count criterion as the search)")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    run = Run(args.run_id, args.dry_run)
    cfg = run.config(args)
    rng = np.random.default_rng(cfg["seed"])
    bank = seed_bank(run)
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
    tasks = sorted(t for t in bank.task.unique() if t not in VAL8_TASKS
                   and support.get(t, 0) >= cfg["min_eps_per_task"])
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

    def bases_for(task_list, n=None, seed=0):
        d = bank[bank.task.isin(task_list)][["task", "phrase"]]
        return d.sample(min(n, len(d)), random_state=seed) if n else d

    corpus_summary = "(see results/analysis/b4_rules_inputs/ -- corpus stats, contrast pairs, case studies)"
    rules_per_model = {}

    for rephraser in cfg["rephrasers"]:
        pdir = run.dir / f"pass_{rephraser}"
        pdir.mkdir(exist_ok=True)
        state_p = pdir / "state.json"
        st = jread(state_p) if state_p.exists() else {
            "iter": 0, "best_val": -1e9, "best_iter": -1, "since_best": 0,
            "rules": "1. Rewrite the instruction as a short plain imperative.\n"}
        while st["since_best"] < cfg["patience"] and st["iter"] < cfg["max_iters"]:
            it = st["iter"]
            itdir = pdir / f"iter_{it:02d}"
            itdir.mkdir(exist_ok=True)
            print(f"[{run.id}:{rephraser}] iter {it} (since_best={st['since_best']})")

            # 1. distill
            rp = itdir / "rules.md"
            if rp.exists():
                rules = rp.read_text()
            else:
                bank = pd.read_parquet(run.dir / "bank.parquet")
                by = bank.sort_values("proxy")
                evidence = pd.concat([by.groupby("task").head(3),
                                      by.groupby("task").tail(3)]).drop_duplicates(["task", "phrase"])
                ev_txt = "\n".join(f"[{r.task}] {r.proxy:.2f}  {r.phrase!r}" +
                                   (f"  (gt={r.gt_success:.0f}%)" if pd.notna(r.gt_success) else "")
                                   for r in evidence.itertuples())
                prev_summary = ""
                prev = pdir / f"iter_{it - 1:02d}" / "eval_train.json"
                if prev.exists():
                    prev_summary = json.dumps(jread(prev).get("summary"), indent=1)[:4000]
                dp = prompt_from("distill.md", rephraser=rephraser, prev_rules=st["rules"],
                                 corpus_summary=corpus_summary, evidence=ev_txt,
                                 eval_summary=prev_summary)
                rules = call_llm(run, cfg["distiller"], dp, f"{rephraser}_distill")
                rp.write_text(rules)
            (run.dir / "current_rules.md").write_text(rules)

            # 2. eval on train sample (with per-rule single edits)
            tb = bases_for(train_tasks, cfg["sample_n"], seed=cfg["seed"] + it)
            train_score, ev_train, summary = eval_rules(
                run, cfg, itdir, rephraser, rules, tb, "train", single_edit=True)
            bank = bank_add(run, ev_train.assign(source=f"loop_{rephraser}", iter_added=it))

            # 3. eval on both validation sets
            vh, _, _ = eval_rules(run, cfg, itdir, rephraser, rules,
                                  bases_for(val_held_tasks, cfg["sample_n"], seed=1), "val_held")
            v8, _, _ = eval_rules(run, cfg, itdir, rephraser, rules,
                                  bases_for(VAL8_TASKS, cfg["sample_n"], seed=2), "val8")
            val = (vh + v8) / 2
            print(f"    train={train_score:.3f} val_held={vh:.3f} val8={v8:.3f} avg={val:.3f}")

            # 4. early-stopping bookkeeping
            if val > st["best_val"]:
                st.update(best_val=val, best_iter=it, since_best=0)
            else:
                st["since_best"] += 1
            st["rules"] = rules
            st["iter"] = it + 1
            jwrite(state_p, st)
            jwrite(itdir / "scores.json", {"train": train_score, "val_held": vh,
                                           "val8": v8, "val_avg": val})
            plot_progress(run, pdir, rephraser)

            # 5. probe phrases the bank lacks
            if summary and summary.get("suggestions"):
                pp = prompt_from("plan.md", suggestions=summary["suggestions"],
                                 tasks=", ".join(train_tasks[:40]))
                planned = call_llm(run, cfg["distiller"], pp, f"{rephraser}_plan")
                new = [{"task": m.group(1), "phrase": m.group(2).strip()}
                       for m in re.finditer(r"^\s*\[([^\]]+)\]\s+(.+)$", planned, re.M)]
                if new:
                    nd = score_phrases(run, cfg, pd.DataFrame(new), f"probe_i{it}")
                    bank = bank_add(run, nd.assign(source="probe", iter_added=it))

        best = pdir / f"iter_{st['best_iter']:02d}" / "rules.md"
        rules_per_model[rephraser] = best.read_text() if best.exists() else st["rules"]
        (pdir / "best_rules.md").write_text(rules_per_model[rephraser])
        print(f"[{run.id}:{rephraser}] done: best iter {st['best_iter']} val {st['best_val']:.3f}")

    jwrite(run.dir / "final.json", {m: f"pass_{m}/best_rules.md" for m in rules_per_model})
    print(f"[{run.id}] all passes complete -> {run.dir}/final.json")


if __name__ == "__main__":
    sys.exit(main())
