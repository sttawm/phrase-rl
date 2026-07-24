"""Corpus-scale phrase-edit screening via the exam-validated grip channel.

For each Bridge episode context (real training frames + its GT instruction),
apply text-only edit transforms, score all variants with CRN-paired pi0
decodes (decode_verbose), and report the PAIRED per-context grip delta per
edit across episodes. DISCOVERY instrument only: signs are validated against
the certified rollout cells (clause family x8, six-edit matrix); rollout
certification remains ground truth (EXPERIMENT.md 2026-07-23 grip-screen).

Lower grip_err = closer to the demo gripper trajectory = "better" under the
screen. A NEGATIVE mean delta for an edit means the edit IMPROVED (reduced)
grip error vs the base instruction.

  .venv-gen/bin/python -m phrase_rl.phrase_grip_screen \
      --contexts data/contexts_train_multit16.parquet \
      --stats-contexts results/phrase_artifacts/chunk_stats.parquet \
      --episodes 200 --frames 4 --k 4 --out results/analysis/grip_screen_pilot.json
"""
import argparse
import io
import json
import re

import numpy as np
import pandas as pd
import torch
from PIL import Image

from phrase_rl.pi0_scoring import Pi0PhraseScorer, import_pi0_policy

CATEGORY_MAP = {
    "carrot": "vegetable", "eggplant": "vegetable", "corn": "vegetable",
    "banana": "fruit", "spoon": "utensil", "fork": "utensil", "knife": "utensil",
    "cup": "container", "mug": "container", "bowl": "container", "pot": "container",
    "can": "object", "bottle": "object", "block": "object", "cloth": "fabric",
    "towel": "fabric", "sushi": "food", "mushroom": "vegetable",
}

# noun-family substitution candidates for the substitutability table (cat 11)
FAMILY_MAP = {
    "towel": ["cloth", "rag"], "cloth": ["towel", "rag"], "rag": ["cloth", "towel"],
    "cup": ["mug", "bowl"], "mug": ["cup"], "bowl": ["dish", "basin"],
    "pot": ["pan", "saucepan"], "pan": ["pot", "skillet"],
    "block": ["cube"], "cube": ["block"], "basket": ["bin", "rack"],
    "plate": ["dish"], "spoon": ["scoop"], "knife": ["blade"],
    "bottle": ["flask"], "can": ["tin"], "lid": ["cover", "cap"],
    "drawer": ["tray"], "sink": ["basin"],
}

COLOR_WORDS = {"red", "blue", "green", "yellow", "orange", "purple", "white",
               "black", "brown", "pink", "silver", "gray", "grey", "teal"}


def trace_colors(trace: str):
    """noun -> color the Gemini trace assigns it (first 'color noun' bigram wins)."""
    out = {}
    for m in re.finditer(r"\b(" + "|".join(COLOR_WORDS) + r")\b\s+(?:toy\s+|metal\s+|plastic\s+)?(\w+)",
                         trace.lower()):
        color, noun = m.group(1), m.group(2)
        out.setdefault(noun, color)
    return out


def edits_for(instr: str, trace: str = ""):
    """Yield (edit_name, variant) for every transform applicable to instr.

    With a trace (Gemini scene description), adds trace-grounded categories:
    add_color (the color the TRACE assigns the noun — scene-true, not corpus
    prior), drop_color, and noun-family swaps (only for nouns present in the
    instruction; families reported per-pair by the analysis)."""
    out = {}
    low = instr.lower().strip().rstrip(".")
    if re.search(r"\bput\b", low):
        out["put_to_set"] = re.sub(r"\bput\b", "set", low, count=1)
        out["put_to_place"] = re.sub(r"\bput\b", "place", low, count=1)
    if re.search(r"\bon\b(?! top)", low):
        out["on_to_onto"] = re.sub(r"\bon\b(?! top)", "onto", low, count=1)
        out["on_to_on_top_of"] = re.sub(r"\bon\b(?! top)", "on top of", low, count=1)
    m = re.match(r"^put (the )?(?P<src>.+?) (?P<rel>on|in|into|onto) (?P<dst>.+)$", low)
    if m:
        out["clause_wrap"] = (f"pick up the {m.group('src')} and put it "
                              f"{m.group('rel')} {m.group('dst')}")
    tele = re.sub(r"\b(the|a|an)\b ", "", low)
    if tele != low:
        out["telegram"] = re.sub(r"\s+", " ", tele).strip()
    out["please"] = f"please {low}"
    for noun, cat in CATEGORY_MAP.items():
        if re.search(rf"\b{noun}\b", low):
            out["category_noun"] = re.sub(rf"\b{noun}\b", cat, low, count=1)
            break
    # --- trace-grounded categories ---
    toks = set(re.findall(r"[a-z]+", low))
    present_colors = toks & COLOR_WORDS
    if present_colors:
        c = sorted(present_colors)[0]
        out["drop_color"] = re.sub(rf"\b{c}\b ?", "", low, count=1).replace("  ", " ").strip()
    elif trace:
        tc = trace_colors(trace)
        for noun in toks:
            if noun in tc:
                out["add_color_traced"] = re.sub(rf"\b{noun}\b", f"{tc[noun]} {noun}", low, count=1)
                break
    for noun in sorted(toks):
        if noun in FAMILY_MAP:
            for alt in FAMILY_MAP[noun]:
                out[f"family:{noun}->{alt}"] = re.sub(rf"\b{noun}\b", alt, low, count=1)
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--contexts", required=True)
    ap.add_argument("--stats-contexts", required=True)
    ap.add_argument("--ckpt", default="juexzz/INTACT-pi0-finetune-rephrase-bridge")
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--frames", type=int, default=4, help="frames per episode (linspace subsample)")
    ap.add_argument("--k", type=int, default=4, help="CRN decode draws")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--phrase-sets", default=None,
                    help="json {episode: {gt, phrases:[...]}} — image-grounded per-scene "
                         "boards (partial-order mode) instead of auto text edits")
    ap.add_argument("--traces", default="results/phrase_artifacts/cover35_teacher_train.parquet",
                    help="episode-keyed Gemini trace parquet for trace-grounded edits ('' disables)")
    ap.add_argument("--ep-slice", default="", help="a:b slice of the (seeded) episode list for multi-pod sharding")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    tmap = {}
    if args.traces and not args.phrase_sets:
        tdf = pd.read_parquet(args.traces, columns=["episode_index", "trace"])
        tmap = {int(r.episode_index): str(r.trace) for r in tdf.itertuples()}
        print(f"traces loaded: {len(tmap)} episodes")

    chunks = np.stack(pd.read_parquet(args.stats_contexts, columns=["action_chunk"])
                      ["action_chunk"].map(np.asarray))
    dim_std = chunks.reshape(chunks.shape[0], -1, 7).std(axis=(0, 1)) + 1e-8

    df = pd.read_parquet(args.contexts)
    sets = None
    if args.phrase_sets:
        sets = {int(k): v for k, v in json.load(open(args.phrase_sets)).items()
                if not k.startswith("_")}
        df = df[df.episode_index.isin(sets)]
    else:
        eps = sorted(df.episode_index.unique())
        rng = np.random.default_rng(args.seed)
        eps = list(rng.choice(eps, size=min(args.episodes, len(eps)), replace=False))
        if args.ep_slice:
            a, b = (int(x) for x in args.ep_slice.split(":"))
            eps = eps[a:b]
            print(f"episode slice [{a}:{b}] -> {len(eps)} episodes")
        df = df[df.episode_index.isin(eps)]

    PI0Policy = import_pi0_policy()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy = PI0Policy.from_pretrained(args.ckpt).to(device).eval()
    scorer = Pi0PhraseScorer(policy, k=8, seed=args.seed, micro_batch=64)

    per_edit = {}       # edit -> list of per-episode mean paired deltas
    per_episode = {}    # episode -> {edit: delta} for post-hoc per-scene boards
    n_ctx = 0
    for ep, g in df.groupby("episode_index"):
        g = g.sort_values("t").reset_index(drop=True)
        idx = np.unique(np.linspace(0, len(g) - 1, args.frames).round().astype(int))
        g = g.iloc[idx]
        instr = str(g.iloc[0]["instruction"]).strip()
        if sets is not None:
            plist = sets[int(ep)]["phrases"]
            phrases = plist
            names = [f"p{i}" for i in range(1, len(plist))]  # vs phrases[0] as base
            phrases = [plist[0]] + plist[1:]
        else:
            variants = edits_for(instr, tmap.get(int(ep), ""))
            if not variants:
                continue
            phrases = [instr] + list(variants.values())
            names = list(variants.keys())
        frame_deltas = {nm: [] for nm in names}
        ok = True
        for _, row in g.iterrows():
            img = np.asarray(Image.open(io.BytesIO(row["image_png"]))).astype(np.float32) / 255.0
            try:
                _, _, grip = scorer.decode_verbose(
                    img.transpose(2, 0, 1), np.asarray(row["state"], dtype=np.float32),
                    np.asarray(row["action_chunk"], dtype=np.float32), phrases,
                    dim_std, k_l2=args.k)
            except Exception as e:  # noqa: BLE001 — skip broken context, keep the sweep
                print(f"[skip] ep{ep} t{row['t']}: {e}", flush=True)
                ok = False
                break
            base = grip[0].mean()
            for i, nm in enumerate(names):
                frame_deltas[nm].append(float(grip[i + 1].mean() - base))
        if not ok:
            continue
        n_ctx += 1
        per_episode.setdefault(int(ep), {"gt": instr})
        for nm in names:
            per_episode[int(ep)][nm] = round(float(np.mean(frame_deltas[nm])), 6)
        if sets is not None:
            # per-scene partial order: absolute mean grip per phrase across frames,
            # plus per-frame paired deltas vs base for chain SEs
            board = {"gt": instr, "phrases": {}}
            base_mean = float(np.mean([0.0]))  # base delta is 0 by construction
            board["phrases"][phrases[0]] = {"mean_delta_vs_base": 0.0, "se": 0.0}
            for i, nm in enumerate(names):
                d = np.asarray(frame_deltas[nm])
                board["phrases"][phrases[i + 1]] = {
                    "mean_delta_vs_base": round(float(d.mean()), 6),
                    "se": round(float(d.std(ddof=1) / np.sqrt(len(d))), 6) if len(d) > 1 else None,
                }
            per_edit[f"scene_{ep}"] = board
            print(f"[scene {ep}] done ({len(phrases)} phrases)", flush=True)
        else:
            for nm in names:
                per_edit.setdefault(nm, []).append(float(np.mean(frame_deltas[nm])))
            if n_ctx % 20 == 0:
                print(f"[{n_ctx}] episodes screened", flush=True)

    if sets is not None:
        out = {"contexts": args.contexts, "mode": "phrase_sets",
               "frames_per_ep": args.frames, "k": args.k,
               "note": "negative delta = phrase reduced grip error vs the scene's base "
                       "(row 0 of its set); per-scene boards sorted best-first",
               "scenes": {}}
        for key, board in per_edit.items():
            ordered = sorted(board["phrases"].items(), key=lambda kv: kv[1]["mean_delta_vs_base"])
            out["scenes"][key] = {"gt": board["gt"],
                                  "order_best_to_worst": [
                                      {"phrase": p, **v} for p, v in ordered]}
        json.dump(out, open(args.out, "w"), indent=1)
        for key, sc in out["scenes"].items():
            print(f"\n== {key}  (gt: {sc['gt']!r})")
            for r in sc["order_best_to_worst"]:
                print(f"  {r['mean_delta_vs_base']:+.5f} (se {r['se']})  {r['phrase'][:64]}")
        print(f"\nwrote {args.out}")
        return

    rows = []
    for nm, deltas in sorted(per_edit.items()):
        d = np.asarray(deltas)
        se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float("nan")
        rows.append({"edit": nm, "n_episodes": len(d),
                     "mean_delta_grip": round(float(d.mean()), 6),
                     "se": round(float(se), 6),
                     "z": round(float(d.mean() / se), 2) if se and not np.isnan(se) else None,
                     "frac_improved": round(float((d < 0).mean()), 3)})
    out = {"contexts": args.contexts, "episodes_screened": n_ctx,
           "frames_per_ep": args.frames, "k": args.k,
           "note": "negative mean_delta_grip = edit reduced grip error (screen-better)",
           "table": rows, "per_episode": per_episode}
    json.dump(out, open(args.out, "w"), indent=1)
    print(f"\n{'edit':18s} {'n':>5s} {'mean d(grip)':>13s} {'z':>7s} {'frac<0':>7s}")
    for r in rows:
        print(f"{r['edit']:18s} {r['n_episodes']:5d} {r['mean_delta_grip']:13.6f} "
              f"{str(r['z']):>7s} {r['frac_improved']:7.3f}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
