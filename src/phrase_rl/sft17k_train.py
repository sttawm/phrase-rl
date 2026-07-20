"""SFT-17k: supervised ERT->GT reconstruction (no RL, no reward model).

Trains Qwen3.5-9B (LoRA, v6 geometry: r16/a32, same 7 target modules) to map
hostile rephrasings back to the original Bridge instruction, text-only, over
~48k (variant -> gt) pairs spanning ~17.3k unique train-split instructions
(data/sft17k_pairs.parquet from pod4_gen_variants.py).

PRE-COMMITTED RECIPE (no checkpoint selection): 2 epochs, cosine LR 1e-4,
bs 16 x accum 4, loss on target tokens only; the ONLY validation is text
loss on a held-out 5% of pairs (never touches SIMPLER). final/ is THE
artifact; best_val/ (text loss) is archived per repo policy but not used
for selection. Deployment conditioning = build_msgs below, reused verbatim
by sft17k_generate_eval (continue_final_message + enable_thinking=False —
the no-reasoning prefill discipline of phase2_train.apply_template).

Resumable: out/latest/ holds adapter + optimizer/scheduler/position state,
saved every 100 optimizer steps; rerunning the same command resumes.

  .venv-gen/bin/python -m phrase_rl.sft17k_train \
      --pairs data/sft17k_pairs.parquet --out results/checkpoints/sft17k
"""
import argparse
import json
import math
import os
import random

import numpy as np
import pandas as pd
import torch

WRAPPER = ("This is a reworded robot instruction. Recover the original plain "
           "instruction it came from. Reply with ONLY that instruction.\n{variant}")
# prefill ends WITHOUT trailing space: the ':' boundary keeps prompt token ids
# a strict prefix of full-sequence token ids (a trailing space would merge into
# the target's first BPE token and shift the label mask)
PREFILL = "Canonical:"


def build_msgs(variant):
    return [{"role": "user", "content": [{"type": "text", "text": WRAPPER.format(variant=variant)}]},
            {"role": "assistant", "content": [{"type": "text", "text": PREFILL}]}]


def render_prompt(tok, variant):
    return tok.apply_chat_template(build_msgs(variant), tokenize=False,
                                   continue_final_message=True, enable_thinking=False)


def build_example(tok, variant, gt):
    prompt = render_prompt(tok, variant)
    full = prompt + " " + gt + tok.tokenizer.eos_token
    p_ids = tok.tokenizer(prompt, add_special_tokens=False)["input_ids"]
    f_ids = tok.tokenizer(full, add_special_tokens=False)["input_ids"]
    assert f_ids[:len(p_ids)] == p_ids, "prompt ids not a prefix — label mask would be wrong"
    labels = [-100] * len(p_ids) + f_ids[len(p_ids):]
    return f_ids, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="data/sft17k_pairs.parquet")
    ap.add_argument("--out", default="results/checkpoints/sft17k")
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--save-every", type=int, default=100, help="optimizer steps between latest/ saves")
    ap.add_argument("--val-every", type=int, default=200, help="optimizer steps between text-val evals")
    args = ap.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    df = pd.read_parquet(args.pairs).sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    # GROUPED holdout by GT (not by pair): every GT has ~3 variants, so a pair-level
    # split would score "new variant of a seen target" — memorizable. Holding out
    # whole GTs makes text-val a real memorization probe: reconstruct instructions
    # never seen as targets.
    gts = pd.Series(sorted(df["gt"].unique())).sample(frac=1.0, random_state=args.seed)
    hold = set(gts.iloc[:max(200, int(0.05 * len(gts)))])
    val_df = df[df["gt"].isin(hold)].reset_index(drop=True)
    train_df = df[~df["gt"].isin(hold)].reset_index(drop=True)
    print(f"pairs: train {len(train_df)} ({df['gt'].nunique() - len(hold)} gts), "
          f"heldout-text-val {len(val_df)} ({len(hold)} unseen gts)", flush=True)

    from transformers import AutoModelForImageTextToText, AutoProcessor
    from peft import LoraConfig, PeftModel, get_peft_model
    tok = AutoProcessor.from_pretrained(args.model)
    tok.tokenizer.padding_side = "right"
    base = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda")

    latest = os.path.join(args.out, "latest")
    state_path = os.path.join(latest, "trainer_state.pt")
    state = None
    if os.path.exists(state_path):
        model = PeftModel.from_pretrained(base, latest, is_trainable=True)
        state = torch.load(state_path, weights_only=False)
        print(f"RESUMING from {latest}: gstep {state['gstep']}, epoch {state['epoch']}, "
              f"micro {state['next_micro']}", flush=True)
    else:
        lcfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, task_type="CAUSAL_LM",
                          target_modules=["v_proj", "o_proj", "k_proj", "gate_proj",
                                          "q_proj", "up_proj", "down_proj"])
        model = get_peft_model(base, lcfg)
    model.print_trainable_parameters()
    trainable = [p for p in model.parameters() if p.requires_grad]

    def encode_batch(rows):
        exs = [build_example(tok, r.variant, r.gt) for r in rows.itertuples()]
        exs = [(i[:args.max_len], l[:args.max_len]) for i, l in exs]
        L = max(len(i) for i, _ in exs)
        pad = tok.tokenizer.pad_token_id or tok.tokenizer.eos_token_id
        ids = torch.tensor([i + [pad] * (L - len(i)) for i, _ in exs])
        lab = torch.tensor([l + [-100] * (L - len(l)) for _, l in exs])
        att = torch.tensor([[1] * len(i) + [0] * (L - len(i)) for i, _ in exs])
        return ids.cuda(), lab.cuda(), att.cuda()

    def val_loss():
        model.eval(); tot = cnt = 0
        with torch.no_grad():
            for s in range(0, min(len(val_df), 1024), args.bs):
                ids, lab, att = encode_batch(val_df.iloc[s:s + args.bs])
                out = model(input_ids=ids, labels=lab, attention_mask=att)
                tot += float(out.loss) * len(ids); cnt += len(ids)
        model.train()
        return tot / cnt

    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)
    steps_per_epoch = math.ceil(len(train_df) / (args.bs * args.accum))
    total = steps_per_epoch * args.epochs
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total)
    print(f"optimizer steps total: {total} ({steps_per_epoch}/epoch)", flush=True)

    gstep, start_epoch, start_micro, best = 0, 0, 0, float("inf")
    if state is not None:
        opt.load_state_dict(state["opt"]); sched.load_state_dict(state["sched"])
        gstep, start_epoch, start_micro = state["gstep"], state["epoch"], state["next_micro"]
        best = state.get("best_val", float("inf"))

    def save_latest(ep, next_micro):
        model.save_pretrained(latest)
        torch.save({"gstep": gstep, "epoch": ep, "next_micro": next_micro,
                    "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "best_val": best}, state_path)

    log = open(os.path.join(args.out, "sft_log.jsonl"), "a")
    model.train()
    for ep in range(start_epoch, args.epochs):
        order = train_df.sample(frac=1.0, random_state=args.seed + ep).reset_index(drop=True)
        micro0 = start_micro if ep == start_epoch else 0
        opt.zero_grad()
        for m in range(micro0, math.ceil(len(order) / args.bs)):
            ids, lab, att = encode_batch(order.iloc[m * args.bs:(m + 1) * args.bs])
            out = model(input_ids=ids, labels=lab, attention_mask=att)
            (out.loss / args.accum).backward()
            if (m + 1) % args.accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                opt.step(); sched.step(); opt.zero_grad()
                gstep += 1
                rec = {"step": gstep, "epoch": ep, "loss": float(out.loss),
                       "lr": sched.get_last_lr()[0]}
                if gstep % args.val_every == 0:
                    rec["val_loss"] = val_loss()
                    if rec["val_loss"] < best:
                        best = rec["val_loss"]
                        model.save_pretrained(os.path.join(args.out, "best_val"))
                        rec["best"] = True
                if gstep % args.save_every == 0:
                    save_latest(ep, m + 1)
                if gstep % 25 == 0 or "val_loss" in rec:
                    print(json.dumps(rec), flush=True)
                    log.write(json.dumps(rec) + "\n"); log.flush()
        vl = val_loss()
        if vl < best:
            best = vl
            model.save_pretrained(os.path.join(args.out, "best_val"))
        model.save_pretrained(os.path.join(args.out, f"epoch_{ep + 1}"))
        save_latest(ep + 1, 0)
        print(f"epoch {ep + 1} saved; val_loss={vl:.4f} (best {best:.4f})", flush=True)
    model.save_pretrained(os.path.join(args.out, "final"))
    json.dump({"wrapper": WRAPPER, "prefill": PREFILL, "recipe": vars(args), "best_val": best},
              open(os.path.join(args.out, "recipe.json"), "w"), indent=1)
    print("SFT-DONE", flush=True)


if __name__ == "__main__":
    main()
