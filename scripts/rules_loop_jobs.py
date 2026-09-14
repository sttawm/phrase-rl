#!/usr/bin/env python3
"""Pod-side executor for rules-loop jobs (run by rules_loop_worker.sh).

  score  : proxy-score (task, phrase) rows. Extracts the SAME two channels the
           proxy was calibrated on (scripts/fit_simple_success_reward.py), in
           reward_mode="verifier" -- the ONLY mode in which the score server
           computes grip at all:
             z    = -mean over ensemble members of the returned array
             grip = the server's gripper-error sidecar, score_phrases.last_grips
           Consumed exactly as scripts/fine_exam_score.py does.
  apply  : Qwen3.5-9B applies a rules file to each incoming phrase.

Usage (from /workspace/phrase-rl on a pod):
  .venv-gen/bin/python scripts/rules_loop_jobs.py <spec.json>

The score kind requires a running score server (the worker boots one). Context
banks: contexts_train_multit16 + contexts_val_multit16 + contexts_club (the
search's 213-instruction table). CRN: contexts per task are a seeded draw, so
every phrase in a run faces identical contexts.
"""
import json
import os
import re
import sys
import types
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/workspace/phrase-rl")
sys.path.insert(0, str(REPO / "src"))

spec = json.loads(Path(sys.argv[1]).read_text())
jid = spec["job_id"]
jdir = Path(sys.argv[1]).parent
payload = pd.read_parquet(jdir / f"{jid}.payload.parquet")
result_path = jdir / f"{jid}.result.parquet"

# val8 tasks are simulator names ("widowx_coke_can_on_ramekin_clean"); the context
# banks key on natural-language instructions ("put the coke can on the ramekin").
# A substring test between the two can never match, so map each stem to a word
# pattern over the instruction text instead.
VAL8_PATTERNS = {
    "spoon_on_towel":      r"spoon.*(towel|cloth)",
    "carrot_on_plate":     r"carrot.*plate",
    "stack_cube":          r"(cube|block).*(cube|block)",
    "eggplant_in_basket":  r"eggplant.*(basket|rack)",
    "carrot_on_keyboard":  r"carrot.*keyboard",
    "carrot_on_wheel":     r"carrot.*wheel",
    "coke_can_on_ramekin": r"coke.*(ramekin|bowl)",
    "coke_can_on_plate":   r"coke.*plate",
}


def proxy_success(z, grip, P):
    x = P["C"] + P["bz"] * np.asarray(z) + P["bg"] * (-np.asarray(grip))
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


if spec["kind"] == "score" and spec.get("method") == "rollout":
    # --phase sim: REAL rollouts on a render pod (SIMPLER + INT-ACT stack).
    # Shells to phase0c_rollout.py with the env's recipe; every phrase rolls the
    # same episode_ids (shared initial states = the CRN analog for rollouts).
    import subprocess
    ro = spec["rollout"]
    int_act = os.environ.get("INT_ACT_ROOT", "/workspace/INT-ACT")
    pl = payload[["task", "phrase"]].drop_duplicates().assign(arm="rules_loop")
    # NW concurrent phase0c processes per job (the paper's sealed legs ran
    # NW=3 on the same GPU; one process leaves 2/3 of throughput unused).
    # Phrases are chunked; CRN determinism is per (task, episode, rep), so
    # chunking cannot change any episode's noise.
    nw = int(os.environ.get("ROLLOUT_NW", ro.get("nw", 3)))
    nw = max(1, min(nw, len(pl)))
    chunks = [pl.iloc[i::nw] for i in range(nw)]
    import time as _time
    procs, outs = [], []
    for ci, ch in enumerate(chunks):
        if ci:
            _time.sleep(45)   # stagger model-load memory spikes (run_sealed_leg.sh pattern)
        phr_path = (jdir / f"{jid}.rollphrases.{ci}.parquet").resolve()
        ch.to_parquet(phr_path, index=False)
        out_path = (jdir / f"{jid}.rollraw.{ci}.parquet").resolve()
        out_path.unlink(missing_ok=True)   # phase0c ACCUMULATES on an existing --out
        outs.append(out_path)
        cmd = [f"{int_act}/.venv/bin/python",
               str(REPO / "src/phrase_rl/phase0c_rollout.py"),
               "--int-act-root", int_act,
               "--config", ro["config"], "--ckpt", ro["ckpt"],
               "--phrases", str(phr_path),
               "--episode-ids", *[str(i) for i in ro["episode_ids"]],
               "--seed", str(ro.get("seed", 42)),
               "--repeats", str(ro.get("repeats", 1)),
               "--out", str(out_path)]
        print(f"rollout[{ci}/{nw}]:", " ".join(cmd), flush=True)
        procs.append(subprocess.Popen(cmd, cwd=int_act))
    fails = [ci for ci, pr in enumerate(procs) if pr.wait() != 0]
    if fails:
        raise SystemExit(f"rollout chunks failed: {fails}")
    raw = pd.concat([pd.read_parquet(o) for o in outs], ignore_index=True)
    agg = raw.groupby(["task", "phrase"]).success.agg(["mean", "size"]).reset_index()
    res = pl[["task", "phrase"]].merge(agg, on=["task", "phrase"], how="left")
    res["gt_success"] = 100.0 * res["mean"]
    res["n_ctx"] = res["size"]
    res["z"] = np.nan
    res["grip"] = np.nan
    res[["task", "phrase", "z", "grip", "gt_success", "n_ctx"]].to_parquet(
        result_path, index=False)
    print(f"rolled {len(res)} phrases x {len(ro['episode_ids'])} episodes")

elif spec["kind"] == "generate":
    # A33: Qwen3.5-9B writes natural rephrases from a per-task prompt carried in
    # the payload (prompt column). Text-only unless the payload carries an
    # image_png column, in which case the frame is attached (image variant).
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    mid = spec.get("model", "Qwen/Qwen3.5-9B")
    proc = AutoProcessor.from_pretrained(mid)
    model = AutoModelForImageTextToText.from_pretrained(
        mid, dtype=torch.bfloat16, device_map="cuda").eval()
    has_img = "image_png" in payload.columns
    rows = []
    for r in payload.itertuples():
        content = [{"type": "text", "text": str(r.prompt)}]
        images = None
        if has_img and r.image_png is not None:
            import io
            from PIL import Image
            images = [Image.open(io.BytesIO(bytes(r.image_png))).convert("RGB")]
            content = [{"type": "image"}, {"type": "text", "text": str(r.prompt)}]
        msgs = [{"role": "user", "content": content}]
        text = proc.apply_chat_template(msgs, tokenize=False,
                                        add_generation_prompt=True, enable_thinking=False)
        kw = {"text": [text], "return_tensors": "pt"}
        if images is not None:
            kw["images"] = images
        inp = proc(**kw).to(model.device)
        with torch.no_grad():
            g = model.generate(**inp, do_sample=True, temperature=1.0, max_new_tokens=256)
        out = proc.decode(g[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)
        if "</think>" in out:
            out = out.split("</think>")[-1]
        kind = None
        for line in out.splitlines():
            ln = line.strip().strip('"').strip()
            up = ln.upper()
            if up.startswith("NATURAL"):
                kind = "natural"; continue
            if up.startswith("ADVERSARIAL"):
                kind = None; continue
            ln = re.sub(r"^\s*\d+[.)]\s*", "", ln).strip()
            if kind == "natural" and len(ln) > 3:
                rows.append({"task": r.task, "phrase": ln, "author": "qwen"})
        print(f"[gen] {r.task}: {len([x for x in rows if x['task']==r.task])} lines", flush=True)
    pd.DataFrame(rows).to_parquet(result_path, index=False)
    print(f"generated {len(rows)} phrases over {payload.task.nunique()} tasks")

elif spec["kind"] == "score" and spec.get("method") == "libero_bank_eval":
    # pi05_libero: drive interactive-vlas bank_eval.py against a live
    # serve_policy --env LIBERO server. Task key convention: "suite:task_id".
    # Queue items roll each (suite, task_id, phrase, init) once; bank_eval is
    # resume-safe on its jsonl, so retries continue rather than re-roll.
    import subprocess
    ro = spec["rollout"]
    ipi = os.environ.get("INTERACTIVE_PI_ROOT", "/workspace/interactive-pi")
    py38 = os.environ.get("LIBERO_PY", f"{ipi}/.venv38/bin/python")
    pl = payload[["task", "phrase"]].drop_duplicates()
    # bank_eval.py's queue schema: ONE item per (task, phrase) carrying the
    # full inits list, plus canonical/arm bookkeeping fields it logs verbatim
    items = []
    for r in pl.itertuples():
        suite, tid = str(r.task).rsplit(":", 1)
        items.append({"suite": suite, "task_id": int(tid),
                      "canonical": "", "arm": "loop",
                      "phrase": str(r.phrase),
                      "inits": [int(i) for i in ro["inits"]]})
    qpath = (jdir / f"{jid}.queue.json").resolve()
    json.dump(items, open(qpath, "w"))
    out_jsonl = (jdir / f"{jid}.bankeval.jsonl").resolve()
    cmd = [py38, f"{ipi}/pi05_libero/eval/bank_eval.py",
           "--queue", str(qpath), "--out", str(out_jsonl),
           "--port", str(ro.get("port", 8000)),
           "--seed", str(ro.get("seed", 7))]
    env = dict(os.environ, MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl",
               PYTHONPATH=os.environ.get("LIBERO_PYTHONPATH", ""))
    print("libero_bank_eval:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=f"{ipi}/pi05_libero/eval", env=env)
    rows = [json.loads(x) for x in open(out_jsonl) if x.strip()]
    raw = pd.DataFrame(rows)
    raw["task"] = raw.suite.astype(str) + ":" + raw.task_id.astype(str)
    want = {(str(r.task), str(r.phrase)) for r in pl.itertuples()}
    raw = raw[[((t, p) in want) for t, p in zip(raw.task, raw.phrase.astype(str))]]
    agg = raw.groupby(["task", "phrase"]).success.agg(["mean", "size"]).reset_index()
    res = pl.merge(agg, on=["task", "phrase"], how="left")
    res["gt_success"] = 100.0 * res["mean"]
    res["n_ctx"] = res["size"]
    res["z"] = np.nan
    res["grip"] = np.nan
    res[["task", "phrase", "z", "grip", "gt_success", "n_ctx"]].to_parquet(
        result_path, index=False)
    # per-episode records travel with the aggregate: the worker commits
    # <jid>.episodes.parquet next to the result (pod-local jsonls die on terminate)
    raw[["task", "phrase", "init", "success", "steps"]].to_parquet(
        jdir / f"{jid}.episodes.parquet", index=False)
    print(f"bank-eval rolled {len(res)} phrases x {len(ro['inits'])} inits")

elif spec["kind"] == "score":
    from phrase_rl.phase2_train import score_phrases

    bank_files = [REPO / "data/contexts_train_multit16.parquet",
                  REPO / "data/contexts_val_multit16.parquet",
                  REPO / "data/contexts_club.parquet"]
    banks = pd.concat([pd.read_parquet(f) for f in bank_files if f.exists()],
                      ignore_index=True)
    bkey = "task" if "task" in banks.columns else "instruction"
    banks["_key"] = banks[bkey].astype(str)

    rng = np.random.default_rng(spec["seed"])
    F0 = int(spec.get("frames_per_episode", 4))
    budget = int(spec.get("score_budget", 64))
    ipc = Path(os.environ.get("IPC_DIR", "/workspace/ipc_rules"))
    # reward_mode MUST be "verifier": phase2_score_server only computes grip in
    # that branch (server lines 114-165) and writes NaN for every phrase in flow
    # or l2 mode. In verifier mode the returned (P, n_members) array is NEGATED
    # calibrated member logits, so z = -mean(L) recovers the logit -- exactly what
    # scripts/fine_exam_score.py does, and what the proxy was calibrated on.
    # The server must be started with --verifier-ensemble and --stats-contexts.
    args = types.SimpleNamespace(k=8, score_seed=int(spec["seed"]), tau_min=0.0,
                                 reward_mode="verifier", k_l2=4, score_timeout=3600,
                                 _reward_frames_map=None)

    out = []
    for task, grp in payload.groupby("task"):
        key = str(task)
        sub = banks[banks._key == key]
        if len(sub) == 0 and spec.get("pool_val8_stems"):
            pat = next((p for s, p in VAL8_PATTERNS.items() if s in key), None)
            if pat:
                sub = banks[banks._key.str.lower().str.contains(pat, regex=True, na=False)]
                if len(sub):
                    print(f"[val8] {key} -> /{pat}/ matched "
                          f"{sub._key.nunique()} instructions, "
                          f"{sub.episode_index.nunique()} episodes", flush=True)
        phrases = list(dict.fromkeys(grp.phrase.astype(str)))
        if len(sub) == 0:
            for p in phrases:
                out.append({"task": task, "phrase": p, "z": np.nan, "grip": np.nan,
                            "n_ctx": 0, "F": 0, "C": 0})
            print(f"[warn] no contexts for {key!r} -- {len(phrases)} phrases NaN", flush=True)
            continue

        eps = sorted(sub.episode_index.unique())
        C = min(int(spec.get("contexts_per_task", 16)), len(eps))
        # C-limited tasks spend the budget on frames instead
        F = int(np.clip(round(budget / max(C, 1)), F0, 16))
        pick = rng.choice(eps, size=C, replace=False)
        ctx_rows = []
        for e in sorted(pick):
            ep = sub[sub.episode_index == e]
            ep = ep.sort_values("t") if "t" in ep.columns else ep
            ctx_rows.extend(r for _, r in ep.head(F).iterrows())

        zs, gs = [], []
        CHUNK = 8  # frame-contexts per scoring job, matching fine_exam_score
        for i in range(0, len(ctx_rows), CHUNK):
            batch = ctx_rows[i:i + CHUNK]
            contexts = [(fr, phrases) for fr in batch]
            losses = score_phrases(ipc, f"rl_{jid}_{uuid.uuid4().hex[:8]}", contexts, args)
            grips = getattr(score_phrases, "last_grips", [None] * len(losses))
            for L, G in zip(losses, grips):
                zs.append(-np.asarray(L, dtype=np.float64).mean(axis=1))
                gs.append(np.asarray(G, dtype=np.float64) if G is not None
                          else np.full(len(phrases), np.nan))
        z_mean = np.nanmean(np.stack(zs), axis=0)
        g_mean = np.nanmean(np.stack(gs), axis=0)
        for p, z, g in zip(phrases, z_mean, g_mean):
            out.append({"task": task, "phrase": p, "z": float(z), "grip": float(g),
                        "n_ctx": len(ctx_rows), "F": F, "C": C})
        print(f"{key}: {len(phrases)} phrases x F={F} C={C} "
              f"({len(ctx_rows)} frame-contexts)", flush=True)

    res = pd.DataFrame(out)
    res["proxy"] = proxy_success(res.z, res.grip, spec["proxy"])
    res.to_parquet(result_path, index=False)
    print(f"scored {len(res)} rows ({int(res.z.isna().sum())} NaN)")

elif spec["kind"] == "apply":
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # rules_text in the spec is authoritative: reading the repo file raced a
    # lost rulebook push and silently applied a stale book (2026-09-01)
    rules = spec.get("rules_text") or (REPO / spec["rules_file"]).read_text()
    # strip the rationale: it is written for us, not for the applier, and feeding
    # a model commentary about rules it must follow literally is a live hazard
    if "===RULES===" in rules:
        body = rules.split("===RULES===", 1)[1]
        rules = "===RULES===" + body.split("===RATIONALE===", 1)[0]
    elif "===RATIONALE===" in rules:
        rules = rules.split("===RATIONALE===", 1)[0]
    only = spec.get("only_rule")
    if only:   # single-edit: a one-rule rulebook, so the contrast is clean
        rules = f"===RULES===\n1. {only}\n"
    tmpl = (REPO / "prompts/rules_loop/apply.md").read_text().replace("{{rules}}", rules)

    # scene descriptions are expensive to generate: load once, reuse per phrase
    traces = {}
    tf = REPO / "results/phrase_artifacts/cover35_teacher_train.parquet"
    def _san(x):
        i = str(x).rfind("Original Instruction:")
        return str(x)[:i].rstrip() if i >= 0 else str(x)
    if tf.exists():
        t = pd.read_parquet(tf, columns=["instruction", "trace"]).drop_duplicates("instruction")
        traces = {str(k): _san(v) for k, v in zip(t.instruction, t.trace)}
    pbf = REPO / "results/phrase_artifacts/traces_rules_v1.parquet"
    if pbf.exists():   # per-base traces take precedence, same as the driver
        p = pd.read_parquet(pbf, columns=["task", "phrase", "trace"]).drop_duplicates(["task", "phrase"])
        traces.update({(str(r.task), str(r.phrase)): _san(r.trace) for r in p.itertuples()})

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-9B")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3.5-9B", torch_dtype=torch.bfloat16, device_map="cuda")
    rows = []
    for r in payload.itertuples():
        trace = (traces.get((str(r.task), str(r.phrase)))
                 or traces.get(str(r.task))
                 or traces.get(str(r.phrase))
                 or "(no scene description available)")
        prompt = (tmpl.replace("{{trace}}", trace)
                      .replace("{{phrase}}", str(r.phrase))
                      .replace("{{task}}", str(r.task)))
        inp = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                      add_generation_prompt=True,
                                      enable_thinking=False,
                                      return_tensors="pt")
        # newer transformers returns a BatchEncoding here, older a bare tensor;
        # generate() needs the tensor (BatchEncoding.shape -> AttributeError)
        if hasattr(inp, "input_ids"):
            inp = inp["input_ids"]
        inp = inp.to(model.device)
        with torch.no_grad():
            g = model.generate(inp, attention_mask=torch.ones_like(inp),
                               do_sample=False, max_new_tokens=64)
        text = tok.decode(g[0][inp.shape[1]:], skip_special_tokens=True)
        # thinking-model guard: without it, every "rewrite" was the literal
        # preamble "Thinking Process:" (r1 iter 0, judge caught 0% adherence)
        if "</think>" in text:
            text = text.split("</think>", 1)[1]
        lines = [l.strip() for l in text.strip().split("\n")
                 if l.strip() and not l.strip().lower().startswith(("thinking", "<think"))]
        rows.append({"task": r.task, "phrase": r.phrase,
                     "rewrite": lines[0] if lines else str(r.phrase)})
    pd.DataFrame(rows).to_parquet(result_path, index=False)
    print(f"applied to {len(rows)} phrases")

else:
    raise SystemExit(f"unknown job kind {spec['kind']}")

print(f"JOB-DONE {jid} -> {result_path.name}")
