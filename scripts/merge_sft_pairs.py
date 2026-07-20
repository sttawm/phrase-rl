"""Build the dead-simple SFT training set (user spec 2026-07-19):
Qwen hostile variants (styles 0-2) + pi0-seen benign paraphrases (style 3,
OXE_paraphrases Bridge slice, downsampled per key to keep a hostile
majority), everything mapped to the canonical train-clean GT string.

Train-split hygiene: benign pairs only for GTs in bridge_train_uniques
(the dictionary also covers 862 val/test-split keys — excluded).

  python scripts/merge_sft_pairs.py --out data/sft17k_pairs_merged.parquet
"""
import argparse
import unicodedata

import pandas as pd


def nk(s):
    return " ".join(unicodedata.normalize("NFKC", str(s)).casefold().split())


ap = argparse.ArgumentParser()
ap.add_argument("--hostile", default="data/sft17k_pairs.parquet")
ap.add_argument("--paraphrases", default="results/phrase_artifacts/oxe_paraphrases_bridge.parquet")
ap.add_argument("--inventory", default="results/phrase_artifacts/bridge_train_uniques.parquet")
ap.add_argument("--per-key", type=int, default=4)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default="data/sft17k_pairs_merged.parquet")
args = ap.parse_args()

host = pd.read_parquet(args.hostile)
canon = {nk(g): g for g in pd.read_parquet(args.inventory)["gt"]}

par = pd.read_parquet(args.paraphrases)
par["key"] = par["gt"].map(nk)
par = par[par["key"].isin(canon)]
par["gt"] = par["key"].map(canon)
par = par[par["paraphrase"].map(nk) != par["key"]]
par = (par.sample(frac=1.0, random_state=args.seed)
          .groupby("key", sort=True)
          .head(args.per_key))
benign = pd.DataFrame({"gt": par["gt"], "variant": par["paraphrase"], "style": 3})

merged = pd.concat([host[["gt", "variant", "style"]], benign], ignore_index=True)
merged["dk"] = merged["gt"].map(nk) + "||" + merged["variant"].map(nk)
merged = merged.drop_duplicates("dk").drop(columns="dk").reset_index(drop=True)
print(f"hostile {len(host)} + benign {len(benign)} -> merged {len(merged)} "
      f"({merged['gt'].nunique()} gts; per style: {merged['style'].value_counts().to_dict()})")
merged.to_parquet(args.out, index=False)
print(f"wrote {args.out}")
