"""One initial-scene frame (t=0) per unique SFT instruction — the image side
of v2's Qwen trace generation (user: jump directly to Qwen traces).

Representative episode per instruction key: lowest episode_index, preferring
hash-train episodes (episode_split_u >= 0.10). Downloads each mp4 straight to
a temp file and deletes it after decoding frame 0, so the HF cache never
grows. Shard-resumable: data/frames17k/shard_XXX.parquet, complete shards
skipped on rerun.

  .venv-gen/bin/python scripts/extract_frames_17k.py
"""
import io
import json
import os
import tempfile
import unicodedata
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import av
import pandas as pd

REPO = "IPEC-COMMUNITY/bridge_orig_lerobot"
IMAGE_KEY = "observation.images.image_0"
SHARD = 500
OUT_DIR = "data/frames17k"


def nk(s):
    return " ".join(unicodedata.normalize("NFKC", str(s)).casefold().split())


def u(i):
    return ((i * 2654435761) % (2**32)) / 2**32


def fetch_first_frame(ep_idx):
    url = (f"https://huggingface.co/datasets/{REPO}/resolve/main/"
           f"videos/chunk-{ep_idx // 1000:03d}/{IMAGE_KEY}/episode_{ep_idx:06d}.mp4")
    req = urllib.request.Request(url)
    tok = os.environ.get("HF_TOKEN")
    if tok:
        req.add_header("Authorization", f"Bearer {tok}")
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tf:
        tmp = tf.name
    try:
        last = None
        for attempt in range(3):  # bursty 429s under concurrency: back off and retry
            try:
                with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
                    f.write(r.read())
                break
            except Exception as e:
                last = e
                import time
                time.sleep(2 * 4 ** attempt)
        else:
            raise last
        with av.open(tmp) as c:
            for frame in c.decode(c.streams.video[0]):
                img = frame.to_image()
                break
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
    finally:
        os.unlink(tmp)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    from huggingface_hub import hf_hub_download
    mp = hf_hub_download(REPO, "meta/episodes.jsonl", repo_type="dataset")
    best = {}  # key -> (prefer_rank, episode_index, instruction)
    for line in open(mp):
        r = json.loads(line)
        t = r.get("tasks", [""])
        t = str(t[0] if isinstance(t, list) else t).strip()
        k = nk(t)
        if not k:
            continue
        cand = (0 if u(r["episode_index"]) >= 0.10 else 1, r["episode_index"], t)
        if k not in best or cand < best[k]:
            best[k] = cand
    inv = pd.read_parquet("results/phrase_artifacts/bridge_train_uniques.parquet")["gt"].tolist()
    todo = [(g, best[nk(g)][1]) for g in inv if nk(g) in best]
    print(f"{len(todo)} instructions to frame "
          f"({sum(1 for g, _ in todo if best[nk(g)][0] == 1)} fall back to non-train episodes)",
          flush=True)

    def run_batch(chunk, path, label):
        def one(item):
            gt, ep = item
            try:
                return {"gt": gt, "episode_index": ep, "image_png": fetch_first_frame(ep)}
            except Exception as e:
                return {"gt": gt, "episode_index": ep, "image_png": None, "err": str(e)[:120]}

        with ThreadPoolExecutor(5) as ex:
            rows = list(ex.map(one, chunk))
        ok = [r for r in rows if r.get("image_png")]
        errs = [r["err"] for r in rows if not r.get("image_png")]
        pd.DataFrame(ok).to_parquet(path, index=False)
        print(f"{label}: {len(ok)} ok, {len(errs)} failed"
              + (f" (e.g. {errs[0]})" if errs else ""), flush=True)

    n_shards = (len(todo) + SHARD - 1) // SHARD
    for si in range(n_shards):
        path = f"{OUT_DIR}/shard_{si:03d}.parquet"
        if os.path.exists(path):
            continue
        run_batch(todo[si * SHARD:(si + 1) * SHARD], path, f"shard {si + 1}/{n_shards}")

    # repair rounds: refetch whatever any shard dropped (rate-limit casualties)
    import glob
    for rnd in range(1, 4):
        have = set()
        for f in glob.glob(f"{OUT_DIR}/shard_*.parquet") + glob.glob(f"{OUT_DIR}/repair_*.parquet"):
            have.update(pd.read_parquet(f, columns=["gt"])["gt"])
        missing = [it for it in todo if it[0] not in have]
        if not missing:
            break
        print(f"repair round {rnd}: {len(missing)} missing", flush=True)
        for si in range(0, len(missing), SHARD):
            run_batch(missing[si:si + SHARD], f"{OUT_DIR}/repair_{rnd}_{si // SHARD:03d}.parquet",
                      f"repair {rnd}.{si // SHARD}")
    print("FRAMES-DONE", flush=True)


if __name__ == "__main__":
    main()
