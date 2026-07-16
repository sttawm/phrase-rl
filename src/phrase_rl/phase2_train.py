"""Phase 2: advantage-weighted rephrase tuning of Qwen3.5-9B against frozen pi0.

PRIMARY experiment (EXPERIMENT.md "Training" section 1). Runs in .venv-gen
(transformers >= 5) on the generation GPU. The frozen pi0 reward lives in a
SEPARATE process (.venv, INTACT-era lerobot fork) — phrase_rl.phase2_score_server
— because the two dependency stacks cannot share a venv. All communication is
file-based through --ipc-dir:

  data    : {ipc_dir}/{job_id}.in.parquet — one row per (context, phrase), columns
              context_id, image_png (raw PNG bytes), state (8,), action_chunk (28,),
              phrase. Written BEFORE the req so the server never sees a half job.
  request : {ipc_dir}/{job_id}.req.json  (written to *.tmp then os.replace — atomic)
              {"in_parquet": <abs>, "out_parquet": <abs>, "k": K CRN draws,
               "seed": fixed CRN seed, "tau_min": tau clamp (0b finding)}
  response: server writes out_parquet (columns context_id, phrase, loss,
            loss_per_draw = list of K floats) then renames req.json ->
            {job_id}.done.json (atomic completion signal)
  error   : {job_id}.err.txt (traceback) + req.json -> {job_id}.failed.json;
            trainer raises ScoreServerError
The trainer polls every 0.5 s and deletes consumed done/failed/err/parquet
files. ONE job per training step covers the whole context group; each
context's rows are its surviving candidates PLUS the original instruction
(row 0 of its group), so the original's loss is logged as a paired baseline
under the exact same CRN draws.

Per training step (a group of --contexts-per-step contexts):
  1. GENERATE 16 candidates per context in one call with CoVer's verbatim
     template (inline reasoning then numbered list), temp 0.8, thinking OFF;
     parse_reworded + case-insensitive dedupe; <6 unique -> parse-fail, skip.
  2. GATE with the majority-of-3 faithfulness judge; drop goal_drift (0c: the
     confirmed reward-hacking class), keep renames (logged); <4 survivors -> skip.
  3. SCORE survivors + the original instruction in ONE CRN job (same draws,
     so the original is a paired baseline for free).
  4. R = -mean_k loss; z-normalize within the surviving group; keep A > 0.
  5. For each positive candidate: token-mean log-prob of the PHRASE TOKENS ONLY
     under the single-phrase prefix (CoVer template with batch_number=1 plus
     assistant continuation "1. "); loss_i = -A * mean_logprob + beta * KL to
     the base model (adapter disabled) at the same token positions. Candidate
     losses are averaged over the step's contributing positive candidates
     (mean-over-positives — see apply_update for the deviation note vs the
     spec's sum); backward runs in micro-batches of --accum candidates; one
     AdamW step (clip 1.0) per training step.

Phrase-token alignment: the candidate is appended to the assistant prefix text
and BOTH strings go through apply_chat_template; candidate token positions are
the suffix after the longest common prefix of the two id sequences. This keeps
BPE boundary behavior (space merging after "1. ") exactly as generation would
produce it — no manual space hacks.

Resumable: results/checkpoints/phase2/latest holds adapter + optimizer + RNG
states + trainer_state.json; --resume restores and fast-forwards the seeded
data order to the saved step. SIGTERM / Ctrl-C -> checkpoint, exit 0.
GateUnavailable -> checkpoint, exit 3. Score-server timeout (ScoreTimeout) ->
checkpoint, exit 4. Score-server-reported job failure (ScoreServerError) ->
checkpoint, exit 5.

Sibling interfaces (reconciled):
  cover_prompt.build_qwen_messages(image=..., instruction=..., batch_number=N)
      -> chat messages (image part + CoVer user template) for apply_chat_template
  cover_prompt.parse_reworded(text) -> list[str]
  cover_prompt.build_single_phrase_prefix(instruction=..., image=...)
      -> FULL chat messages whose LAST turn is assistant text "1. " (for
         continue_final_message=True log-prob conditioning)
  faithfulness_gate.FaithfulnessGate(votes=3).judge_many_sync([(orig, cands)])
      -> [[{"cls", "faithful", "votes", "cached"}]]; cls includes "judge_fail"
         (all ballots failed) which is fail-closed (faithful=False); raises
         GateUnavailable on unretryable judge-API states.

Usage (pod, via scripts/run_phase2.sh):
  .venv-gen/bin/python -m phrase_rl.phase2_train --ipc-dir /workspace/ipc --resume
"""

import argparse
import copy
import io
import itertools
import json
import os
import shutil
import signal
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from phrase_rl import cover_prompt
from phrase_rl.faithfulness_gate import FaithfulnessGate, GateUnavailable


class ScoreTimeout(Exception):
    """Score server did not answer within --score-timeout."""


class ScoreServerError(Exception):
    """Score server reported a job failure (.failed.json + .err.txt)."""


# Set by SIGTERM; checked at safe points -> checkpoint then clean exit 0.
STOP = {"flag": False}


def _on_sigterm(signum, frame):
    STOP["flag"] = True
    print("\nSIGTERM — will checkpoint and exit at the next safe point", flush=True)


# ---------------------------------------------------------------- utilities


def apply_template(processor, messages, **kw):
    """All chat-template calls go through here so enable_thinking=False is
    impossible to forget (thinking mode contaminates parses and log-probs)."""
    return processor.apply_chat_template(
        messages, tokenize=True, return_dict=True, return_tensors="pt",
        enable_thinking=False, **kw,
    )


def dedupe(cands: list[str]) -> list[str]:
    """Case/whitespace/trailing-period-insensitive dedupe, order-preserving."""
    seen, out = set(), []
    for c in cands:
        c = c.strip().strip('"').strip()
        key = " ".join(c.lower().split()).rstrip(".")
        if key and key not in seen:
            seen.add(key)
            out.append(c)
    return out


def context_groups(df, seed: int, cps: int, start_step: int):
    """Deterministic infinite stream of (step, rows) groups. Each epoch is a
    fresh seeded permutation, so resume = fast-forward by step count alone."""
    n = len(df)
    if n < cps:
        raise SystemExit(
            f"need at least contexts_per_step={cps} train contexts, got {n} — "
            "lower --contexts-per-step or provide more train contexts")
    step = 0
    for epoch in itertools.count():
        order = np.random.default_rng(seed * 1000 + epoch).permutation(n)
        for lo in range(0, n - cps + 1, cps):
            if step >= start_step:
                yield step, [df.iloc[int(j)] for j in order[lo : lo + cps]]
            step += 1


# ---------------------------------------------------------- score client (IPC)


def score_phrases(ipc_dir: Path, job_id: str, contexts: list, args) -> list[np.ndarray]:
    """One scoring job for a whole context group, via the server's
    json+parquet protocol (see phase2_score_server.py).

    contexts: list of (row, phrases) where phrases[0] is the ORIGINAL
    instruction of that context — scored in the same CRN group so its loss is
    logged as a paired baseline for free.

    Writes {job_id}.in.parquet (columns exactly context_id, image_png, state,
    action_chunk, phrase; one row per phrase) FIRST, then {job_id}.req.json
    via tmp-file + os.replace (atomic, absolute paths). Polls every 0.5 s for
    {job_id}.done.json (success -> read out_parquet) or {job_id}.failed.json
    (read .err.txt -> ScoreServerError).

    Returns one float32 (P, K) loss array per context, in input order (rows
    within a context preserve phrase order). Raises ScoreTimeout after
    --score-timeout (the first job's timeout also covers server model-load
    time), ScoreServerError on a server-reported failure, KeyboardInterrupt
    if SIGTERM arrives while waiting.
    """
    in_parquet = (ipc_dir / f"{job_id}.in.parquet").resolve()
    out_parquet = (ipc_dir / f"{job_id}.out.parquet").resolve()
    rollout = getattr(args, "reward_mode", "flow") == "rollout"
    frames_by_ep = getattr(args, "_reward_frames_map", None)
    recs = []
    for ci, (row, phrases) in enumerate(contexts):
        cid = str(row["context_id"]) if rollout else ci  # rollout server keys on "task|episode_id"
        # multi-frame reward: emit each frame of the episode with a frame_id;
        # the server averages calibrated logits across frames per phrase.
        frames = [row]
        if frames_by_ep is not None and not rollout:
            frames = frames_by_ep.get(int(row["episode_index"]), [row])
        for fi, fr in enumerate(frames):
            for p in phrases:
                rec = {"context_id": cid, "frame_id": fi,
                       "image_png": fr["image_png"], "phrase": str(p)}
                if not rollout:  # pi0 scoring needs proprioception + ground truth
                    rec["state"] = np.asarray(fr["state"], dtype=np.float32)
                    rec["action_chunk"] = np.asarray(fr["action_chunk"], dtype=np.float32)
                recs.append(rec)
    cols = ["context_id", "image_png", "phrase"] if rollout else         ["context_id", "frame_id", "image_png", "state", "action_chunk", "phrase"]
    pd.DataFrame(recs, columns=cols).to_parquet(in_parquet, index=False)

    req = ipc_dir / f"{job_id}.req.json"
    tmp = ipc_dir / f"{job_id}.req.json.tmp"  # .tmp never matches the server's *.req.json glob
    tmp.write_text(json.dumps({
        "in_parquet": str(in_parquet), "out_parquet": str(out_parquet),
        "k": args.k, "seed": args.score_seed, "tau_min": args.tau_min,
        "reward_mode": args.reward_mode, "k_l2": args.k_l2,
        "reps": getattr(args, "rollout_reps", 2),
    }))
    os.replace(tmp, req)

    done = ipc_dir / f"{job_id}.done.json"
    failed = ipc_dir / f"{job_id}.failed.json"
    err = ipc_dir / f"{job_id}.err.txt"
    deadline = time.time() + args.score_timeout
    while time.time() < deadline:
        if STOP["flag"]:
            raise KeyboardInterrupt("SIGTERM while waiting on score server")
        if failed.exists():
            msg = err.read_text() if err.exists() else "(no .err.txt found)"
            for p in (failed, err, in_parquet):
                p.unlink(missing_ok=True)
            raise ScoreServerError(f"score server error for {job_id}: {msg}")
        if done.exists():
            out = pd.read_parquet(out_parquet)
            losses = []
            for ci, (row, phrases) in enumerate(contexts):
                key = str(row["context_id"]) if rollout else ci  # rollout ids are "task|ep" strings
                g = out[out["context_id"] == key]
                if len(g) != len(phrases):
                    raise ScoreServerError(
                        f"{job_id}: context {ci} returned {len(g)} rows "
                        f"for {len(phrases)} phrases")
                losses.append(np.stack(
                    [np.asarray(r, dtype=np.float32) for r in g["loss_per_draw"]]))
            for p in (done, in_parquet, out_parquet):
                p.unlink(missing_ok=True)
            return losses
        time.sleep(0.5)
    req.unlink(missing_ok=True)
    in_parquet.unlink(missing_ok=True)
    raise ScoreTimeout(f"no response for {job_id} after {args.score_timeout}s")


# ------------------------------------------------- generate / gate / score


def process_context(model, processor, gate, row, args, min_survivors: int) -> dict:
    """Generate + gate for one context; shared by train steps and val.

    Result dict always has the counting fields; ok=True adds img/survivors.
    Scoring happens at the step level (score_phrases with the whole group),
    which then attaches r_orig/rewards to each ok result.
    """
    img = Image.open(io.BytesIO(row["image_png"])).convert("RGB")
    instruction = str(row["instruction"])  # the ORIGINAL: gate anchor + reward context
    source = instruction
    srcs = getattr(args, "_sources", {}).get((row["episode_index"], row["t"]))
    if srcs and args.source_aug > 0 and np.random.random() < args.source_aug:
        source = str(np.random.choice(srcs))  # robustness: condition on a synthetic phrasing
    res = {
        "instruction": instruction, "ok": False, "reason": None,
        "n_parsed": 0, "n_unique": 0, "n_judged": 0,
        "n_drift": 0, "n_rename": 0, "n_judge_fail": 0, "n_survivors": 0,
    }

    trace = getattr(args, "_traces", {}).get((row["episode_index"], row["t"])) or \
            getattr(args, "_val_traces", {}).get((row["episode_index"], row["t"]))
    res["trace"] = trace
    res["source"] = source  # used ONLY in the update (p_single) prompt — the inference-time
    res["source_augmented"] = source != instruction  # prompt; generation always farms candidates
    model.eval()
    _gen_t0 = time.time()
    if getattr(args, "gen_mode", "list") == "sample_single":
        # Arm B: TRUE SAMPLING from the deployment/update prompt (p_single, source-
        # augmented input) — on-policy GRPO estimator; NO dedupe (duplicates are
        # legitimate samples), no list parsing.
        msgs = cover_prompt.build_single_phrase_prefix(
            instruction=source, image=img, trace=trace)
        inputs = apply_template(processor, msgs, continue_final_message=True).to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs, do_sample=True, temperature=args.gen_temp,
                num_return_sequences=args.n_candidates, max_new_tokens=48,
            )
        plen = inputs["input_ids"].shape[1]
        cands = [processor.decode(o[plen:], skip_special_tokens=True).strip().split("\n")[0].strip()
                 for o in out]
        cands = [c for c in cands if c]
        res["gen_sec"] = round(time.time() - _gen_t0, 1)
        res.update(n_parsed=len(cands), n_unique=len(set(cands)))
    else:
        msgs = cover_prompt.build_qwen_messages(  # from the ORIGINAL for max pool quality
            image=img, instruction=instruction, batch_number=args.n_candidates, trace=trace)
        inputs = apply_template(processor, msgs, add_generation_prompt=True).to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs, do_sample=True, temperature=args.gen_temp,
                max_new_tokens=args.max_new_tokens,
            )
        text = processor.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        res["gen_sec"] = round(time.time() - _gen_t0, 1)
        parsed = cover_prompt.parse_reworded(text)
        unique = dedupe(parsed)
        cands = unique[: args.n_candidates]
        res.update(n_parsed=len(parsed), n_unique=len(unique))
    if len(cands) < args.min_parsed:
        res["reason"] = "parse_fail"
        return res

    if getattr(args, "no_gate", False):
        # L2-arm hypothesis (2026-07-10): decoded-action distance punishes goal drift
        # natively (wrong goal -> different trajectory -> large L2), so the judge is
        # redundant — and it's a full in-process generation per context.
        survivors = cands
        res.update(n_judged=len(cands), n_drift=0, n_rename=0, n_judge_fail=0,
                   n_survivors=len(cands), judge_sec=0.0)
    else:
        _judge_t0 = time.time()
        (verdicts,) = gate.judge_many_sync([(instruction, cands)], progress=False)
        res["judge_sec"] = round(time.time() - _judge_t0, 1)
        classes = [v["cls"] for v in verdicts]
        # v["faithful"] folds in judge_fail fail-closed semantics (faithful=False)
        survivors = [c for c, v in zip(cands, verdicts) if v["faithful"]]
        res.update(
            n_judged=len(cands),
            n_drift=sum(cl == "goal_drift" for cl in classes),
            n_rename=sum(cl == "rename" for cl in classes),
            n_judge_fail=sum(cl == "judge_fail" for cl in classes),
            n_survivors=len(survivors),
        )
    if len(survivors) < min_survivors:
        res["reason"] = "gate_fail"
        return res

    res.update(ok=True, img=img, survivors=survivors)
    return res


def score_group(ipc_dir: Path, job_id: str, pairs: list, args):
    """Score [(row, ok_res)] in ONE CRN job; attach r_orig/rewards to each res.

    Each context's phrase list is [original] + survivors — original at index 0
    is the paired baseline under the exact same CRN draws.
    """
    if not pairs:
        return
    contexts = [(row, [res["instruction"]] + res["survivors"]) for row, res in pairs]
    all_losses = score_phrases(ipc_dir, job_id, contexts, args)
    for (_, res), losses in zip(pairs, all_losses):
        rewards = -losses.mean(axis=1)
        res.update(r_orig=float(rewards[0]), rewards=rewards[1:])


# ------------------------------------------------------------ log-prob + KL

_LOGITS_TO_KEEP_OK = True  # flipped off once if this model's forward rejects it


def forward_logits(model, inputs, n_keep: int):
    """Logits for the last n_keep positions; uses logits_to_keep when the model
    supports it (saves a full-seq (L, vocab) tensor per forward)."""
    global _LOGITS_TO_KEEP_OK
    if _LOGITS_TO_KEEP_OK:
        try:
            out = model(**inputs, use_cache=False, logits_to_keep=n_keep)
            return out.logits[:, -n_keep:, :]
        except TypeError:
            _LOGITS_TO_KEEP_OK = False
    out = model(**inputs, use_cache=False)
    return out.logits[:, -n_keep:, :]


def _append_phrase(prefix_msgs: list, phrase: str) -> list:
    """Copy of the single-phrase prefix messages with the candidate appended to
    the trailing assistant text ('1. ' -> '1. <phrase>')."""
    msgs = copy.deepcopy(prefix_msgs)
    last = msgs[-1]
    assert last.get("role") == "assistant", "single-phrase prefix must end with an assistant turn"
    if isinstance(last["content"], str):
        last["content"] = last["content"] + phrase
    else:
        for part in reversed(last["content"]):
            if "text" in part:
                part["text"] = part["text"] + phrase
                break
        else:
            raise ValueError("no text part in the assistant prefix message")
    return msgs


def tokenize_phrase(processor, prefix_msgs, prefix_ids, phrase):
    """Tokenize prefix+phrase and locate the phrase tokens.

    Returns (inputs, start, n_new) or None if the phrase contributes no tokens
    beyond the prefix. Cheap (no forward pass) — apply_update runs it for every
    positive candidate up front so the objective divisor (count of contributing
    candidates) is known before any backward.
    """
    inputs = apply_template(
        processor, _append_phrase(prefix_msgs, phrase), continue_final_message=True)
    f_ids = inputs["input_ids"][0]
    n = min(prefix_ids.numel(), f_ids.numel())
    neq = (prefix_ids[:n] != f_ids[:n]).nonzero()
    start = int(neq[0]) if neq.numel() else n  # first token the phrase changed
    n_new = f_ids.numel() - start
    if n_new <= 0:
        return None
    return inputs, start, n_new


def phrase_logprob_and_kl(model, inputs, start, n_new, beta):
    """Token-mean log-prob of the phrase tokens only, plus token-mean full-vocab
    KL(current || base) at the same positions (base = adapter disabled).

    Returns (mean_logprob grad tensor, kl grad tensor).
    """
    f_ids = inputs["input_ids"][0]
    inputs = inputs.to(model.device)

    n_keep = n_new + 1  # need logits at positions start-1 .. L-2 to predict start .. L-1
    cur = forward_logits(model, inputs, n_keep)
    logp_cur = torch.log_softmax(cur[:, :-1, :].float(), dim=-1)  # (1, n_new, V)
    targets = f_ids[start:].to(model.device).view(1, -1, 1)
    mean_logp = logp_cur.gather(-1, targets).squeeze(-1).mean()

    kl = torch.zeros((), device=model.device)
    if beta > 0:
        with model.disable_adapter(), torch.no_grad():
            base = forward_logits(model, inputs, n_keep)
            logp_base = torch.log_softmax(base[:, :-1, :].float(), dim=-1)
        kl = (logp_cur.exp() * (logp_cur - logp_base)).sum(-1).mean()
    return mean_logp, kl


_BATCH_PARITY = {"checked": False, "use_batched": True}


def _left_pad_collate(chunk, pad_id, device):
    """LEFT-pad a chunk of processor outputs into one batch. Rows end aligned, so
    each row's phrase tokens are its last n_new positions. 1 image per row."""
    ids = [it[0]["input_ids"][0] for it in chunk]
    L = max(x.numel() for x in ids)
    input_ids = torch.full((len(ids), L), pad_id, dtype=ids[0].dtype)
    attn = torch.zeros((len(ids), L), dtype=torch.long)
    for i, x in enumerate(ids):
        input_ids[i, L - x.numel():] = x
        attn[i, L - x.numel():] = 1
    # NB: do NOT pass explicit position_ids — Qwen3.5 computes multi-axis rope
    # positions internally (vision mrope); overriding with 1D cumsum shifts logits
    # (caught by the arm-B parity check 2026-07-16).
    batch = {"input_ids": input_ids.to(device), "attention_mask": attn.to(device)}
    for key in chunk[0][0].keys():
        if key in ("input_ids", "attention_mask", "position_ids"):
            continue
        vals = [it[0][key] for it in chunk]
        if not torch.is_tensor(vals[0]):
            continue
        if vals[0].ndim == 2 and vals[0].shape[0] == 1 and vals[0].shape[1] == ids[0].numel():
            # sequence-aligned key (e.g. mm_token_type_ids): left-pad with zeros like input_ids
            padded = torch.zeros((len(vals), L), dtype=vals[0].dtype)
            for i, (v, x) in enumerate(zip(vals, ids)):
                padded[i, L - x.numel():] = v[0]
            batch[key] = padded.to(device)
        else:
            # per-sample stacked tensors (pixel patches, grid rows): concat on dim 0
            batch[key] = torch.cat(vals, dim=0).to(device)
    return batch


def phrase_logprob_and_kl_batched(model, chunk, beta, pad_id):
    """Batched equivalent of phrase_logprob_and_kl over a chunk of
    (inputs, start, n_new, a) items. One forward (plus one base forward for KL)
    for the whole chunk. Returns lists (mean_logp_i, kl_i) as grad tensors."""
    device = model.device
    batch = _left_pad_collate(chunk, pad_id, device)
    n_keep = max(it[2] for it in chunk) + 1
    cur = forward_logits(model, batch, n_keep)
    logp_cur = torch.log_softmax(cur[:, :-1, :].float(), dim=-1)  # (B, n_keep-1, V)
    if beta > 0:
        with model.disable_adapter(), torch.no_grad():
            base = forward_logits(model, batch, n_keep)
            logp_base = torch.log_softmax(base[:, :-1, :].float(), dim=-1)
    outs = []
    for i, (inputs, start, n_new, a) in enumerate(chunk):
        f_ids = inputs["input_ids"][0]
        targets = f_ids[start:].to(device)  # (n_new,)
        row = logp_cur[i, -(n_new):, :] if n_new < logp_cur.shape[1] else logp_cur[i]
        mean_logp = row.gather(-1, targets.view(-1, 1)).mean()
        kl_i = torch.zeros((), device=device)
        if beta > 0:
            rb = logp_base[i, -(n_new):, :] if n_new < logp_base.shape[1] else logp_base[i]
            kl_i = (row.exp() * (row - rb)).sum(-1).mean()
        outs.append((mean_logp, kl_i))
    return outs


def apply_update(model, processor, optimizer, trainable, ctx_results, args, do_step=True) -> dict:
    """Advantages -> positive-only candidate losses -> one AdamW step.

    Mutates each ok context's res with res["adv"]. Candidate loss:
      -A * mean_token_logprob(phrase) + beta * KL(current || base)
    backward runs in micro-batches of --accum candidates (mathematically
    identical to one big batch).

    Objective scaling note: the accumulated step loss is divided by the COUNT
    OF CONTRIBUTING (positive-advantage, non-empty-token) candidates — i.e.
    this is a MEAN over positives, a deliberate deviation from the spec's sum
    (L = -sum_i max(A_i, 0) * ...), chosen for lr stability: with a sum, a
    step with 1 positive would apply ~20x the per-candidate gradient of a step
    with 20. Recorded as "objective": "mean_over_positive_advantages" in
    metrics.json.
    """
    items = []
    for res in ctx_results:
        if not res["ok"]:
            continue
        r = res["rewards"]
        adv = (r - r.mean()) / (r.std() + 1e-6)
        res["adv"] = adv
        prefix_msgs = None
        grpo = getattr(args, "update_rule", "raft") == "grpo"
        ranked = sorted(zip(res["survivors"], adv), key=lambda t: -t[1])
        if not grpo and getattr(args, "max_pos_per_ctx", 0):
            ranked = ranked[: args.max_pos_per_ctx]  # top-K by advantage: sharper signal, ~half the update cost
        for cand, a in ranked:
            if not grpo and a <= 0:
                continue  # RAFT-style: imitate positives only. GRPO: signed advantages, full group.
            if prefix_msgs is None:  # build (and tokenize) the prefix once per context
                prefix_msgs = cover_prompt.build_single_phrase_prefix(
                    instruction=res.get("source", res["instruction"]), image=res["img"], trace=res.get("trace"))
                prefix_ids = apply_template(
                    processor, prefix_msgs, continue_final_message=True)["input_ids"][0]
            tok = tokenize_phrase(processor, prefix_msgs, prefix_ids, cand)
            if tok is None:  # phrase adds no tokens; cannot contribute
                continue
            items.append((id(res), *tok, float(a)))

    # n_pos = contributing candidates = the objective divisor (mean over positives)
    stats = {"n_pos": len(items), "update_loss": None, "kl": None,
             "logprob": None, "grad_norm": None}
    if not items:
        return stats

    model.train()
    n_tot = len(items)
    loss_sum = kl_sum = logp_sum = 0.0
    n_done = 0
    use_batched = getattr(args, "batched_update", True) and _BATCH_PARITY["use_batched"]
    pad_id = getattr(getattr(processor, "tokenizer", processor), "pad_token_id", None) or 0
    # context-local chunks: rows of a chunk share one context (=> identical image
    # tensors, safe collate); chunk size still bounded by args.accum
    _groups = {}
    for it in items:
        _groups.setdefault(it[0], []).append(it)
    chunks = []
    for _ci, _its in _groups.items():
        _its = sorted(_its, key=lambda x: x[3])  # by n_new: length-bucketed chunks -> minimal padding
        for _lo in range(0, len(_its), args.accum):
            chunks.append(_its[_lo:_lo + args.accum])
    for chunk in chunks:
        gloss = None
        chunk = [it[1:] for it in chunk]  # drop ctx ordinal -> (inputs, start, n_new, a)
        outs = None
        _t0 = time.time()
        if use_batched:
            try:
                outs = phrase_logprob_and_kl_batched(model, chunk, args.beta, pad_id)
            except Exception as e:  # NEVER kill training on a batching bug — fall back
                print(f"[batched-update] EXCEPTION ({type(e).__name__}: {e}) — permanent sequential fallback", flush=True)
                _BATCH_PARITY["use_batched"] = False
                use_batched = False
                outs = None
        if outs is not None and not _BATCH_PARITY["checked"]:  # one-time parity vs sequential
            _BATCH_PARITY["checked"] = True
            ok = True
            for (inputs, start_, n_new, a), (mlp_b, kl_b) in zip(chunk, outs):
                mlp_s, kl_s = phrase_logprob_and_kl(model, inputs, start_, n_new, args.beta)
                if abs(float(mlp_b) - float(mlp_s)) > 2e-2 or abs(float(kl_b) - float(kl_s)) > 2e-2:
                    ok = False
                    print(f"[batched-update] PARITY FAIL: {float(mlp_b):.4f} vs {float(mlp_s):.4f} "
                          f"kl {float(kl_b):.4f} vs {float(kl_s):.4f}", flush=True)
            if ok:
                print("[batched-update] parity check PASSED — using batched path", flush=True)
            else:
                print("[batched-update] falling back to SEQUENTIAL path", flush=True)
                _BATCH_PARITY["use_batched"] = False
                use_batched = False
                outs = None
        if outs is not None:
            for (inputs, start_, n_new, a), (mean_logp, kl) in zip(chunk, outs):
                li = -(a * mean_logp) + args.beta * kl
                gloss = li if gloss is None else gloss + li
                loss_sum += float(li.detach())
                kl_sum += float(kl.detach())
                logp_sum += float(mean_logp.detach())
                n_done += 1
        else:
            for inputs, start_, n_new, a in chunk:
                mean_logp, kl = phrase_logprob_and_kl(model, inputs, start_, n_new, args.beta)
                li = -(a * mean_logp) + args.beta * kl
                gloss = li if gloss is None else gloss + li
                loss_sum += float(li.detach())
                kl_sum += float(kl.detach())
                logp_sum += float(mean_logp.detach())
                n_done += 1
        stats["fwd_sec"] = stats.get("fwd_sec", 0.0) + round(time.time() - _t0, 2)
        _t1 = time.time()
        if gloss is not None:
            # extra /grad_accum_groups: gradients ACCUMULATE across steps until do_step,
            # so the aggregate update matches one step's scale (2026-07-10: standard-order
            # effective batches — e.g. accum 4 x cps 6 x ~28 phrases ~= 670 sequences/update)
            (gloss / n_tot / max(1, args.grad_accum_groups)).backward()
        stats["bwd_sec"] = stats.get("bwd_sec", 0.0) + round(time.time() - _t1, 2)

    if n_done:
        stats.update(update_loss=loss_sum / n_done, kl=kl_sum / n_done,
                     logprob=logp_sum / n_done)
    if do_step:
        if any(p.grad is not None for p in trainable):
            gn = torch.nn.utils.clip_grad_norm_(trainable, args.clip)
            optimizer.step()
            stats["grad_norm"] = float(gn)
        optimizer.zero_grad(set_to_none=True)
    return stats


# ------------------------------------------------------------------- val


def run_probes(model, processor, args) -> list[dict]:
    """Deployment-mode greedy phrase for the fixed probe contexts (longitudinal record).

    ~2-3s per probe; logged as {"type": "probe"} rows so phrase evolution over
    training is chartable from the train log alone, no extra passes needed.
    """
    out = []
    model.eval()
    for pr in getattr(args, "_probes", []):
        msgs = cover_prompt.build_single_phrase_prefix(pr["instruction"], pr["image"], trace=pr["trace"])
        inputs = apply_template(processor, msgs, continue_final_message=True).to(model.device)
        with torch.no_grad():
            gen = model.generate(**inputs, do_sample=False, max_new_tokens=48)
        text = processor.decode(gen[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        out.append({"task": pr["task"], "phrase": text.strip().split("\n")[0].strip()})
    return out


def run_val(model, processor, gate, val_df, args, ipc_dir: Path, step: int) -> dict:
    """Generate + gate + score on the val slice under a fixed sampling seed
    (training RNG state is snapshotted and restored, so val never perturbs
    the exact-resume guarantee)."""
    cpu_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    torch.manual_seed(args.val_seed)
    try:
        best, mean, orig = [], [], []
        n_failed = judged = survived = 0
        rows = val_df.head(args.val_n)
        for i, (_, row) in enumerate(tqdm(rows.iterrows(), total=len(rows),
                                          desc=f"val@{step}", leave=False)):
            if STOP["flag"]:
                raise KeyboardInterrupt("SIGTERM during val")
            res = process_context(model, processor, gate, row, args, min_survivors=1)
            judged += res["n_judged"]
            survived += res["n_survivors"]
            if not res["ok"]:
                n_failed += 1
                continue
            job = f"val{step:06d}i{i:03d}_{uuid.uuid4().hex[:8]}"
            score_group(ipc_dir, job, [(row, res)], args)
            best.append(float(res["rewards"].max()))
            mean.append(float(res["rewards"].mean()))
            orig.append(res["r_orig"])
        return {
            "step": step,
            "n_scored": len(best),
            "n_failed": n_failed,
            "mean_reward": float(np.mean(mean)) if mean else None,
            "mean_best_reward": float(np.mean(best)) if best else None,
            "mean_orig_reward": float(np.mean(orig)) if orig else None,
            "gate_pass_rate": survived / judged if judged else None,
        }
    finally:
        torch.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


# ------------------------------------------------------------- checkpoints


def _save_dir_atomic(path: Path, writer):
    """Write into <name>.tmp then swap, so a crash mid-save never corrupts the
    previous good checkpoint."""
    tmp, old = path.with_name(path.name + ".tmp"), path.with_name(path.name + ".old")
    for d in (tmp, old):
        if d.exists():
            shutil.rmtree(d)
    tmp.mkdir(parents=True)
    writer(tmp)
    if path.exists():
        path.rename(old)
    tmp.rename(path)
    if old.exists():
        shutil.rmtree(old)


def save_latest(model, optimizer, state, args, ckpt_dir: Path):
    # private runtime attrs (args._traces etc.) are tuple-keyed dicts — not JSON-safe
    state["args"] = {k: v for k, v in vars(args).items()
                     if not k.startswith("_") and isinstance(v, (str, int, float, bool, list, type(None)))}

    def write(d: Path):
        model.save_pretrained(str(d))
        torch.save(optimizer.state_dict(), d / "optimizer.pt")
        torch.save(
            {"torch": torch.get_rng_state(),
             "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []},
            d / "rng.pt",
        )
        (d / "trainer_state.json").write_text(json.dumps(state, indent=2))

    _save_dir_atomic(ckpt_dir / "latest", write)
    print(f"[ckpt] latest saved at step {state['step']}", flush=True)


def save_adapter(model, path: Path, extra: dict | None = None):
    def write(d: Path):
        model.save_pretrained(str(d))
        if extra is not None:
            (d / "info.json").write_text(json.dumps(extra, indent=2))

    _save_dir_atomic(path, write)
    print(f"[ckpt] adapter saved to {path}", flush=True)


def update_metrics_json(path: Path, state: dict, latest_val: dict | None):
    data = json.loads(path.read_text()) if path.exists() else {}
    data["phase2"] = {
        "step": state["step"],
        "best_val": {"mean_reward": state["best_val_mean"], "step": state["best_val_step"]},
        "latest_val": latest_val,
        # mean over positive-advantage candidates, not the spec's sum — see apply_update
        "objective": "mean_over_positive_advantages",
        "updated": datetime.now().isoformat(timespec="seconds"),
    }
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    os.replace(tmp, path)


# ------------------------------------------------------------------ setup


def build_model(args, ckpt_dir: Path):
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(args.model)
    base = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda")
    base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    base.enable_input_require_grads()  # required: frozen embeddings + LoRA + ckpt

    latest = ckpt_dir / "latest"
    resumed = args.resume and (latest / "adapter_config.json").exists()
    if resumed:
        model = PeftModel.from_pretrained(base, str(latest), is_trainable=True)
        print(f"resumed adapter from {latest}")
    elif getattr(args, "init_adapter", None):
        model = PeftModel.from_pretrained(base, args.init_adapter, is_trainable=True)
        print(f"initialized adapter from {args.init_adapter} (fresh optimizer/step)")
    else:
        model = get_peft_model(base, LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
            target_modules=[t.strip() for t in args.lora_targets.split(",")],
        ))
    model.print_trainable_parameters()
    return processor, model, resumed


def fresh_state() -> dict:
    return {
        "step": 0, "best_val_mean": None, "best_val_step": None,
        "counters": {"contexts": 0, "parse_fails": 0, "gate_fails": 0, "updates": 0},
        "val_history": [],
    }


# ------------------------------------------------------------- train loop


def log_jsonl(path: Path, rec: dict):
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")


def dedupe_train_log(path: Path, resume_step: int):
    """On --resume, drop records at/after the resumed step: those steps are
    about to be re-run (weights were restored to the checkpoint), so keeping
    the old records would double-count steps in any per-step log analysis.
    Runs once at startup, before anything is appended."""
    if not path.exists():
        return
    kept = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue  # partial trailing line from a crash mid-write
            if rec.get("step", -1) < resume_step:
                kept.append(line)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text("".join(ln + "\n" for ln in kept))
    os.replace(tmp, path)


def step_record(step: int, t0: float, ctx_results: list, upd: dict, args) -> dict:
    ok = [r for r in ctx_results if r["ok"]]
    advs = np.concatenate([r["adv"] for r in ok]) if ok else np.array([])
    n_parsed = sum(r["n_parsed"] for r in ctx_results)
    n_unique = sum(r["n_unique"] for r in ctx_results)
    n_judged = sum(r["n_judged"] for r in ctx_results)
    rec = {
        "step": step,
        "sec": round(time.time() - t0, 1),
        "n_ctx": len(ctx_results),
        "n_ok": len(ok),
        "parse_fail_rate": sum(r["reason"] == "parse_fail" for r in ctx_results) / len(ctx_results),
        "gate_fail_rate": sum(r["reason"] == "gate_fail" for r in ctx_results) / len(ctx_results),
        "dup_rate": 1 - n_unique / n_parsed if n_parsed else None,
        "gate_pass_rate": sum(r["n_survivors"] for r in ctx_results) / n_judged if n_judged else None,
        "rename_rate": sum(r["n_rename"] for r in ctx_results) / n_judged if n_judged else None,
        "judge_fail_rate": sum(r["n_judge_fail"] for r in ctx_results) / n_judged if n_judged else None,
        "adv_mean": float(advs.mean()) if advs.size else None,
        "adv_max": float(advs.max()) if advs.size else None,
        "cand_loss_mean": float(np.mean([-r["rewards"].mean() for r in ok])) if ok else None,
        "orig_loss_mean": float(np.mean([-r["r_orig"] for r in ok])) if ok else None,
        **{k: upd.get(k) for k in ("n_pos", "update_loss", "kl", "logprob", "grad_norm", "fwd_sec", "bwd_sec")},
    }
    if step % args.example_every == 0 and ok:
        rec["examples"] = [
            {"instruction": r["instruction"],
             "top_candidate": r["survivors"][int(np.argmax(r["rewards"]))],
             "top_reward": float(r["rewards"].max()),
             "orig_reward": r["r_orig"]}
            for r in ok
        ]
    return rec


def train_loop(model, processor, gate, optimizer, trainable, train_df, val_df,
               state, args, ckpt_dir: Path, ipc_dir: Path, metrics_path: Path) -> bool:
    """Returns True iff --max-steps completed (else interrupted upstream)."""
    log_path = ckpt_dir / "train_log.jsonl"
    pbar = tqdm(total=args.max_steps, initial=state["step"], desc="phase2 train")
    for step, rows in context_groups(train_df, args.seed, args.contexts_per_step, state["step"]):
        if step >= args.max_steps:
            break
        if STOP["flag"]:
            raise KeyboardInterrupt("SIGTERM at step boundary")
        t0 = time.time()

        ctx_results, ok_pairs = [], []
        for row in rows:
            if STOP["flag"]:
                raise KeyboardInterrupt("SIGTERM between contexts")
            res = process_context(model, processor, gate, row, args,
                                  min_survivors=args.min_survivors)
            ctx_results.append(res)
            if res["ok"]:
                ok_pairs.append((row, res))
            state["counters"]["contexts"] += 1
            if res["reason"] == "parse_fail":
                state["counters"]["parse_fails"] += 1
                print(f"step {step}: parse fail ({res['n_unique']} unique) "
                      f"for {res['instruction']!r}", flush=True)
            elif res["reason"] == "gate_fail":
                state["counters"]["gate_fails"] += 1
                print(f"step {step}: gate left {res['n_survivors']} survivors "
                      f"(<{args.min_survivors}) for {res['instruction']!r}", flush=True)

        # ONE score job per step: the whole context group shares a CRN batch
        _score_t0 = time.time()
        score_group(ipc_dir, f"s{step:06d}_{uuid.uuid4().hex[:8]}", ok_pairs, args)
        _score_sec = time.time() - _score_t0

        _upd_t0 = time.time()
        upd = apply_update(model, processor, optimizer, trainable, ctx_results, args,
                           do_step=((step + 1) % max(1, args.grad_accum_groups) == 0))
        _upd_sec = time.time() - _upd_t0
        if upd["n_pos"]:
            state["counters"]["updates"] += 1
        _rec = step_record(step, t0, ctx_results, upd, args)
        _rec["gen_sec"] = round(sum(r.get("gen_sec", 0) for r in ctx_results), 1)
        _rec["judge_sec"] = round(sum(r.get("judge_sec", 0) for r in ctx_results), 1)
        _rec["score_sec"] = round(_score_sec, 1)
        _rec["update_sec"] = round(_upd_sec, 1)
        log_jsonl(log_path, _rec)
        if args.kl_abort and _rec.get("kl"):
            _klh = getattr(args, "_kl_hist", [])
            _klh.append(_rec["kl"]); args._kl_hist = _klh[-20:]
            if len(args._kl_hist) >= 10 and sum(args._kl_hist) / len(args._kl_hist) > args.kl_abort:
                print(f"KL circuit breaker: rolling mean {sum(args._kl_hist)/len(args._kl_hist):.2f} > {args.kl_abort}; checkpoint + exit 6", flush=True)
                save_latest(model, optimizer, state, args, ckpt_dir)
                raise SystemExit(6)
        state["step"] = step + 1
        pbar.update(1)

        if args.probe_every and getattr(args, "_probes", None) and (step + 1) % args.probe_every == 0:
            log_jsonl(log_path, {"type": "probe", "step": step + 1,
                                 "probes": run_probes(model, processor, args)})

        if args.val_every and (step + 1) % args.val_every == 0:
            val = run_val(model, processor, gate, val_df, args, ipc_dir, step + 1)
            log_jsonl(log_path, {"type": "val", **val})
            print(f"\n[val@{step + 1}] mean_reward={val['mean_reward']} "
                  f"best_of_16={val['mean_best_reward']} orig={val['mean_orig_reward']}", flush=True)
            state["val_history"].append(val)
            # best_val by MARGIN over the original (2026-07-10): raw mean_reward is
            # composition-sensitive (which val context drops varies per val)
            margin = None
            if val["mean_reward"] is not None and val.get("mean_orig_reward") is not None:
                margin = val["mean_reward"] - val["mean_orig_reward"]
            if "best_val_margin" not in state:  # migrate old checkpoints: seed from history
                # EXCLUDE the val being processed (it was appended to history above) —
                # otherwise it seeds a bar equal to itself and fails the strictly-greater
                # test, silently dropping the checkpoint (bug: flow@600, 2026-07-10)
                hist = [v["mean_reward"] - v["mean_orig_reward"] for v in state.get("val_history", [])
                        if v.get("mean_reward") is not None and v.get("mean_orig_reward") is not None
                        and v.get("step") != step + 1]
                state["best_val_margin"] = max(hist) if hist else None
            if margin is not None and (
                    state["best_val_margin"] is None or margin > state["best_val_margin"]):
                state["best_val_margin"] = margin
                state["best_val_mean"] = val["mean_reward"]
                state["best_val_step"] = step + 1
                save_adapter(model, ckpt_dir / "best_val", {"val": val})
            save_latest(model, optimizer, state, args, ckpt_dir)
            update_metrics_json(metrics_path, state, val)
        elif (step + 1) % args.save_every == 0:
            save_latest(model, optimizer, state, args, ckpt_dir)
    pbar.close()
    return True


# ------------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--train-contexts", default="data/contexts_train.parquet")
    ap.add_argument("--traces", default=None, help="parquet(episode_index,t,trace): trace-conditioned primary (user 2026-07-08)")
    ap.add_argument("--val-traces", default=None)
    ap.add_argument("--probe-contexts", default=None,
                    help="parquet(task,episode_index,t,instruction,image_png): fixed contexts probed with a greedy deployment phrase every --probe-every steps (logged as type=probe)")
    ap.add_argument("--probe-traces", default=None, help="parquet(episode_index,t,trace) for the probe contexts")
    ap.add_argument("--probe-every", type=int, default=25)
    ap.add_argument("--source-aug", type=float, default=0.5,
                    help="prob of conditioning on a random teacher rephrase instead of the original (robustness; gate stays anchored to the original)")
    ap.add_argument("--val-contexts", default="data/contexts_val_0b.parquet")
    ap.add_argument("--ipc-dir", default="/workspace/ipc")
    ap.add_argument("--ckpt-dir", default="results/checkpoints/phase2")
    ap.add_argument("--metrics-json", default="results/checkpoints/metrics.json")
    # step composition
    ap.add_argument("--contexts-per-step", type=int, default=2)
    ap.add_argument("--n-candidates", type=int, default=16)
    ap.add_argument("--rollout-reps", type=int, default=2,
                    help="rollout reward: episodes per candidate (reward = success rate)")
    ap.add_argument("--reward-frames", type=int, default=1,
                    help=">1: score each candidate at N quartile frames of the episode (needs data/contexts_train_multit.parquet); server averages calibrated logits")
    ap.add_argument("--batched-update", action=argparse.BooleanOptionalAction, default=True,
                    help="batch candidate forwards in the update loop (parity-checked at startup)")
    ap.add_argument("--gen-mode", choices=["list", "sample_single"], default="list",
                    help="list: CoVer 16-list prompt (structural diversity). sample_single: "
                         "true sampling from the update prompt (on-policy GRPO), no dedupe")
    ap.add_argument("--update-rule", choices=["raft", "grpo"], default="raft",
                    help="raft: positive-only advantage-weighted SFT (v1-v3). grpo: full group, signed advantages — negatives get pushed down")
    ap.add_argument("--max-pos-per-ctx", type=int, default=0,
                    help="cap update to top-K positives per context by advantage (0 = all)")
    ap.add_argument("--grad-accum-groups", type=int, default=1,
                    help="accumulate gradients over N context-groups before optimizer.step() — standard-order effective batches at zero extra compute")
    ap.add_argument("--min-parsed", type=int, default=6, help="min unique candidates else parse-fail")
    ap.add_argument("--min-survivors", type=int, default=4, help="min gate survivors else skip")
    ap.add_argument("--gate-votes", type=int, default=1)  # flash-lite single vote; majority-of-3 only for offline precision
    ap.add_argument("--judge-model", default="gemini-3.1-flash-lite")
    ap.add_argument("--no-gate", action="store_true",
                    help="skip the faithfulness judge entirely (L2 arm: decoded-action reward punishes drift natively)")
    ap.add_argument("--judge-backend", choices=["qwen", "gemini"], default="qwen",
                    help="qwen = frozen base of the loaded model, local+free (user 2026-07-08); gemini = API")
    # generation
    ap.add_argument("--gen-temp", type=float, default=0.8)
    ap.add_argument("--max-new-tokens", type=int, default=1200,
                    help="CoVer template emits inline reasoning before the 16-list")
    # scoring (CRN)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--score-seed", type=int, default=0)
    ap.add_argument("--tau-min", type=float, default=0.25, help="0b: tau<0.25 is non-discriminative")
    ap.add_argument("--score-timeout", type=float, default=600.0)
    ap.add_argument("--reward-mode", choices=["flow", "l2", "rollout", "verifier"], default="flow",
                    help="flow: CRN flow-matching residual (multimodal-aware). l2: CRN decoded-action per-DoF-normalized L2 vs a* (clearer, unimodal). Both lower=better.")
    ap.add_argument("--k-l2", type=int, default=4, help="decode noise draws per phrase for reward_mode=l2 (each is a full denoise; keep small)")
    # optimization
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--beta", type=float, default=0.04, help="KL-to-base anchor weight")
    ap.add_argument("--kl-abort", type=float, default=0.0, help="if >0: checkpoint and exit 6 when rolling-20 KL exceeds this (degeneration circuit breaker, 2026-07-09)")
    ap.add_argument("--init-adapter", default=None, help="LoRA adapter dir to initialize from (fresh optimizer/step; e.g. a previous run's best_val)")
    ap.add_argument("--accum", type=int, default=4, help="candidates per backward micro-batch")
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.0)  # 0: the update-forward must score the SAME policy that sampled (dropout would perturb it and add KL noise)
    ap.add_argument("--lora-targets",
                    default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    # schedule
    ap.add_argument("--max-steps", type=int, default=1000)
    ap.add_argument("--val-every", type=int, default=100)
    ap.add_argument("--val-n", type=int, default=40)
    ap.add_argument("--val-seed", type=int, default=1234)
    ap.add_argument("--save-every", type=int, default=25)
    ap.add_argument("--example-every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    signal.signal(signal.SIGTERM, _on_sigterm)
    ipc_dir = Path(args.ipc_dir)
    ipc_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = Path(args.metrics_json)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_parquet(args.train_contexts)
    val_df = pd.read_parquet(args.val_contexts)
    print(f"contexts: {len(train_df)} train, {len(val_df)} val (using first {args.val_n})")
    TRACES, VAL_TRACES = {}, {}
    if args.traces:
        _t = pd.read_parquet(args.traces)
        TRACES = {(r.episode_index, r.t): str(r.trace) for r in _t.itertuples()}
        SOURCES = {(r.episode_index, r.t): [str(p) for p in r.rephrases]
                   for r in _t.itertuples() if hasattr(r, "rephrases")}
        train_df = train_df[train_df.apply(lambda r: (r["episode_index"], r["t"]) in TRACES, axis=1)]
    args._reward_frames_map = None
    if getattr(args, "reward_frames", 1) > 1:
        mt = pd.read_parquet("data/contexts_train_multit.parquet")
        fmap = {}
        for ep, g in mt.groupby("episode_index"):
            g = g.sort_values("t").reset_index(drop=True)
            idx = np.unique(np.linspace(0, len(g) - 1, args.reward_frames).round().astype(int))
            fmap[int(ep)] = g.iloc[idx].to_dict("records")
        args._reward_frames_map = fmap
        print(f"multi-frame reward: {len(fmap)} episodes x {args.reward_frames} frames")
        print(f"trace-conditioned: {len(TRACES)} traces, {len(train_df)} train contexts retained")
    if args.val_traces:
        _t = pd.read_parquet(args.val_traces)
        VAL_TRACES = {(r.episode_index, r.t): str(r.trace) for r in _t.itertuples()}
    args._traces, args._val_traces = TRACES, VAL_TRACES
    args._sources = SOURCES if args.traces else {}

    args._probes = []
    if args.probe_contexts:
        _pt = {}
        if args.probe_traces:
            _pdf = pd.read_parquet(args.probe_traces)
            _pt = {(r.episode_index, r.t): str(r.trace) for r in _pdf.itertuples()}
        for r in pd.read_parquet(args.probe_contexts).itertuples():
            args._probes.append({
                "task": str(getattr(r, "task", f"ep{r.episode_index}")),
                "instruction": str(r.instruction),
                "image": Image.open(io.BytesIO(r.image_png)).convert("RGB"),
                "trace": _pt.get((r.episode_index, r.t)),
            })
        print(f"probes: {len(args._probes)} contexts every {args.probe_every} steps")

    processor, model, resumed = build_model(args, ckpt_dir)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0)

    state = fresh_state()
    if resumed:
        latest = ckpt_dir / "latest"
        state = json.loads((latest / "trainer_state.json").read_text())
        opt_path = latest / "optimizer.pt"
        if opt_path.exists():
            optimizer.load_state_dict(torch.load(opt_path, map_location="cpu"))
        rng_path = latest / "rng.pt"
        if rng_path.exists():
            rng = torch.load(rng_path, weights_only=False)
            torch.set_rng_state(rng["torch"])
            if torch.cuda.is_available() and len(rng["cuda"]):
                torch.cuda.set_rng_state_all(rng["cuda"])
        dedupe_train_log(ckpt_dir / "train_log.jsonl", state["step"])
        print(f"resumed at step {state['step']} (best_val_mean={state['best_val_mean']})")
    else:
        torch.manual_seed(args.seed)

    if args.judge_backend == "qwen":
        def _local_judge(prompt_text: str) -> str:
            # frozen BASE model as judge: adapter disabled so the judge cannot
            # co-adapt with the policy being trained; text-only, no-think
            msgs = [{"role": "user", "content": [{"type": "text", "text": prompt_text}]}]
            inputs = apply_template(processor, msgs, add_generation_prompt=True).to(model.device)
            was_training = model.training
            model.eval()
            try:
                with torch.no_grad(), model.disable_adapter():
                    out = model.generate(**inputs, do_sample=True, temperature=0.3,
                                         max_new_tokens=400)  # verdict JSON is short; was 1000
            finally:
                if was_training:
                    model.train()
            return processor.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        gate = FaithfulnessGate(votes=args.gate_votes, generate_fn=_local_judge,
                                cache_path="data/gate_cache_qwen.json", include_reasons=False)
    else:
        # include_reasons=False in-loop: classes only (~3x fewer output tokens)
        gate = FaithfulnessGate(model=args.judge_model, votes=args.gate_votes,
                                include_reasons=False)

    completed, exit_code = False, 0
    try:
        completed = train_loop(model, processor, gate, optimizer, trainable,
                               train_df, val_df, state, args, ckpt_dir, ipc_dir, metrics_path)
    except KeyboardInterrupt as e:
        print(f"\ninterrupted ({e}) — checkpointing then exiting 0 (resume with --resume)")
    except GateUnavailable as e:
        print(f"\nFAITHFULNESS GATE UNAVAILABLE: {e}\n"
              "Checkpointing and exiting 3. Fix the judge API (quota/key) and "
              "re-run with --resume.")
        exit_code = 3
    except ScoreTimeout as e:
        print(f"\nSCORE SERVER TIMEOUT: {e}\n"
              "Checkpointing and exiting 4. Check the 'score' tmux session "
              "(crashed? still loading?) and re-run with --resume.")
        exit_code = 4
    except ScoreServerError as e:
        print(f"\nSCORE SERVER ERROR: {e}\n"
              "Checkpointing and exiting 5. See the traceback above (from the "
              "job's .err.txt) and the 'score' tmux session log, then re-run "
              "with --resume.")
        exit_code = 5

    save_latest(model, optimizer, state, args, ckpt_dir)
    update_metrics_json(metrics_path, state,
                        state["val_history"][-1] if state["val_history"] else None)
    if completed:
        save_adapter(model, ckpt_dir / "final",
                     {"step": state["step"], "best_val_mean": state["best_val_mean"],
                      "best_val_step": state["best_val_step"]})
        print(f"training complete at step {state['step']}")
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
