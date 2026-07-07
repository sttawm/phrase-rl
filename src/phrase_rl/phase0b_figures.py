"""Ranked-phrase figures for the Phase 0b writeup.

For each chosen context: the observation frame + rephrases ranked by frozen-pi0
flow loss (lower = policy prefers it), with the original Bridge instruction
highlighted so the reader sees where it lands. Ranks within one arm (gemini_pro
by default) for readability.

  python -m phrase_rl.phase0b_figures --arm gemini_pro --out results/charts
"""

import argparse
import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

# (episode_index, t, caption) — chosen to span the honesty range:
# original near-worst (typical), mid, and a case where original is strong.
EXAMPLES = [
    (45936, 5, "Original phrasing near-worst (typical case)"),
    (37862, 7, "Original mid-pack"),
    (None, None, "Original among the best (the ~5% case)"),
]
N_SHOW = 6  # top-k + bottom-k around the original


def panel(ax_img, ax_bar, img, ranked, orig_idx, caption):
    ax_img.imshow(img)
    ax_img.axis("off")
    ax_img.set_title(caption, fontsize=10, loc="left")

    n = len(ranked)
    # show best N_SHOW, worst 3, and always the original
    keep = sorted(set(list(range(N_SHOW)) + [n - 3, n - 2, n - 1] + [orig_idx]))
    labels, vals, colors = [], [], []
    prev = None
    for r in keep:
        if prev is not None and r != prev + 1:
            labels.append("…"); vals.append(np.nan); colors.append("none")
        p = ranked.iloc[r]
        tag = f"#{r + 1}  {p['phrase']}"
        labels.append(tag[:60])
        vals.append(p["loss"])
        colors.append("#D85A30" if p["arm"] == "original" else "#85B7EB")
        prev = r
    y = np.arange(len(labels))[::-1]
    ax_bar.barh(y, vals, color=colors)
    ax_bar.set_yticks(y); ax_bar.set_yticklabels(labels, fontsize=8, fontfamily="monospace")
    ax_bar.set_xlabel("frozen-π0 flow loss  (lower = policy prefers)", fontsize=9)
    ax_bar.set_title(f"original ranked #{orig_idx + 1} of {n}", fontsize=10, loc="left")
    for spine in ["top", "right"]:
        ax_bar.spines[spine].set_visible(False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default="data/scores_0b_rephrase.parquet")
    ap.add_argument("--contexts", default="data/contexts_val_0b.parquet")
    ap.add_argument("--arm", default="gemini_pro")
    ap.add_argument("--out", default="results/charts")
    args = ap.parse_args()

    sc = pd.read_parquet(args.scores)
    sc = sc[sc.arm.isin(["original", args.arm])]
    g = sc.groupby(["episode_index", "t", "arm", "phrase"])["loss"].mean().reset_index()
    ctx = pd.read_parquet(args.contexts).set_index(["episode_index", "t"])

    # resolve the "original is best" example automatically
    examples = list(EXAMPLES)
    for i, (ep, t, cap) in enumerate(examples):
        if ep is None:
            best = None
            for (e, tt), grp in g.groupby(["episode_index", "t"]):
                grp = grp.sort_values("loss").reset_index(drop=True)
                if "original" in grp.arm.values and grp.index[grp.arm == "original"][0] == 0 \
                        and 4 <= len(str(grp.iloc[0]["phrase"]).split()) <= 8:
                    best = (e, tt); break
            examples[i] = (best[0], best[1], cap)

    fig, axes = plt.subplots(len(examples), 2, figsize=(13, 4 * len(examples)),
                             gridspec_kw={"width_ratios": [1, 2.2]})
    for row, (ep, t, cap) in enumerate(examples):
        grp = g[(g.episode_index == ep) & (g.t == t)].sort_values("loss").reset_index(drop=True)
        orig_idx = int(grp.index[grp.arm == "original"][0])
        img = Image.open(io.BytesIO(ctx.loc[(ep, t), "image_png"]))
        panel(axes[row, 0], axes[row, 1], img, grp, orig_idx, cap)
    fig.suptitle(
        "Phase 0b — the frozen policy ranks rephrasings of the same instruction (val contexts)",
        fontsize=13, y=1.0,
    )
    fig.tight_layout()
    path = f"{args.out}/phase0b_ranked_phrases.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    print("wrote", path)


if __name__ == "__main__":
    main()
