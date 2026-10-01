#!/usr/bin/env python3
"""Side-by-side rollout grids for the project site: one clip per phrase pair.

Input: a directory of per-episode npz files (imgs = PNG bytes per step,
success) from phase0c_rollout --record-dir (SIMPLER) or libero_video_eval.py
(LIBERO), plus results/analysis/grid_video/grid_manifest.json.

Each clip: title card (both phrases + measured rates) -> split screen, the
winning phrase's 6x4 grid left, the losing phrase's right, all 48 cells playing
in lockstep from t=0; a cell freezes on its last frame with a check or cross the
moment its episode ends -> end card holds with the tallies.

  .venv/bin/python scripts/make_grid_video.py --frames results/analysis/grid_video/frames \
      --out results/charts/grid_video
"""
import argparse, io, json, pathlib, re
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import imageio.v2 as imageio

R = pathlib.Path(__file__).resolve().parents[1]
CELL = (160, 120)          # w, h of one cell
COLS, ROWS = 6, 4
GAP, PAD = 3, 16
GREEN, RED, INK, BG = (34, 197, 94), (239, 68, 68), (40, 44, 52), (250, 249, 246)
FPS, HOLD_S, TITLE_S = 12, 2.5, 2.5


def font(size):
    for f in ("/System/Library/Fonts/Helvetica.ttc", "/Library/Fonts/Arial.ttf"):
        try: return ImageFont.truetype(f, size)
        except Exception: pass
    return ImageFont.load_default()


def load_episode(p):
    z = np.load(p, allow_pickle=True)
    frames = [np.asarray(Image.open(io.BytesIO(b)).convert("RGB")) for b in z["imgs"]]
    return frames, bool(z["success"])


def episodes_for(frames_dir, bench, task, phrase, arm):
    """Match npz files to (task, phrase); layout/init index parsed from the name."""
    safe = "".join(c if c.isalnum() else "_" for c in phrase)[:40]
    eps = {}
    # SIMPLER frames live under <frames>/<arm>/ (phase0c names files by phrase only,
    # and "pepsi" vs "Pepsi" collide on a case-insensitive Mac filesystem)
    for p in sorted(pathlib.Path(frames_dir).rglob("*.npz")):
        n = p.name
        if bench == "simpler":
            if p.parent.name != arm: continue
            m = re.match(rf"{re.escape(task)}__ep(\d+)__{re.escape(safe)}\.npz$", n, re.I)
        else:
            suite, tid = task.split("/")
            m = re.match(rf"{suite}__{tid}__{arm}__{re.escape(safe)}__init(\d+)\.npz$", n)
        if m: eps[int(m.group(1))] = p
    return eps


def grid_frames(eps, order):
    """Yield composited grid images; per-cell freeze + mark after its episode ends."""
    loaded = {i: load_episode(eps[i]) for i in order if i in eps}
    if not loaded:
        raise SystemExit("no episodes found for this arm -- check the frames dir layout")
    T = max(len(f) for f, _ in loaded.values())
    gw = COLS * CELL[0] + (COLS - 1) * GAP
    gh = ROWS * CELL[1] + (ROWS - 1) * GAP
    fnt = font(26)
    for t in range(T + int(HOLD_S * FPS)):
        img = Image.new("RGB", (gw, gh), BG)
        d = ImageDraw.Draw(img)
        for k, i in enumerate(order):
            r, c = divmod(k, COLS)
            x, y = c * (CELL[0] + GAP), r * (CELL[1] + GAP)
            if i not in loaded:
                d.rectangle([x, y, x + CELL[0], y + CELL[1]], fill=(225, 225, 225)); continue
            frames, ok = loaded[i]
            fi = min(t, len(frames) - 1)
            img.paste(Image.fromarray(frames[fi]).resize(CELL, Image.BILINEAR), (x, y))
            if t >= len(frames) - 1:   # episode over: mark it
                col = GREEN if ok else RED
                d.rectangle([x, y, x + CELL[0] - 1, y + CELL[1] - 1], outline=col, width=3)
                # drawn badge (font glyphs for check/cross are unreliable)
                cx, cy, rr = x + CELL[0] - 18, y + 18, 13
                d.ellipse([cx - rr, cy - rr, cx + rr, cy + rr], fill=(255, 255, 255), outline=col, width=2)
                if ok:
                    d.line([(cx - 7, cy + 1), (cx - 2, cy + 6), (cx + 8, cy - 6)], fill=col, width=3, joint="curve")
                else:
                    d.line([(cx - 6, cy - 6), (cx + 6, cy + 6)], fill=col, width=3)
                    d.line([(cx - 6, cy + 6), (cx + 6, cy - 6)], fill=col, width=3)
        yield img, sum(ok for f, ok in loaded.values() if t >= len(f) - 1), len(loaded)


def text_card(size, lines, fnt_sizes):
    img = Image.new("RGB", size, BG); d = ImageDraw.Draw(img)
    y = size[1] // 2 - sum(fnt_sizes) * 0.9
    for line, fs in zip(lines, fnt_sizes):
        f = font(fs); w = d.textlength(line, font=f)
        d.text(((size[0] - w) / 2, y), line, fill=INK, font=f); y += fs * 1.6
    return img


def make_clip(pair, frames_dir, out, order):
    good, bad = pair["good"], pair["bad"]
    eg = episodes_for(frames_dir, pair["bench"], pair["task"], good["phrase"], "good")
    eb = episodes_for(frames_dir, pair["bench"], pair["task"], bad["phrase"], "bad")
    print(f"  {pair['task']}: {len(eg)} good / {len(eb)} bad episodes found")
    gw = COLS * CELL[0] + (COLS - 1) * GAP; gh = ROWS * CELL[1] + (ROWS - 1) * GAP
    W, H = 2 * gw + 3 * PAD, gh + 2 * PAD + 70
    W += W % 2; H += H % 2   # libx264 yuv420p needs even dimensions
    writer = imageio.get_writer(out, fps=FPS, codec="libx264", quality=8, macro_block_size=1)
    title = text_card((W, H), [f"“{good['phrase']}”  vs.  “{bad['phrase']}”",
                               f"{good.get('rate', '?')}%  vs.  {bad.get('rate', '?')}%  measured success",
                               "same scenes, same seeds — the wording is the only difference"], [30, 26, 20])
    for _ in range(int(TITLE_S * FPS)): writer.append_data(np.asarray(title))
    hdr = font(22)
    for (gi, gs, gn), (bi, bs, bn) in zip(grid_frames(eg, order), grid_frames(eb, order)):
        img = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(img)
        d.text((PAD, PAD), f"“{good['phrase']}”", fill=GREEN, font=hdr)
        d.text((2 * PAD + gw, PAD), f"“{bad['phrase']}”", fill=RED, font=hdr)
        img.paste(gi, (PAD, PAD + 40)); img.paste(bi, (2 * PAD + gw, PAD + 40))
        foot = font(20)
        d.text((PAD, PAD + 40 + gh + 6), f"{gs}/{gn} succeeded", fill=GREEN, font=foot)
        d.text((2 * PAD + gw, PAD + 40 + gh + 6), f"{bs}/{bn} succeeded", fill=RED, font=foot)
        writer.append_data(np.asarray(img))
    writer.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--only", default=None, help="substring of task to render")
    a = ap.parse_args()
    man = json.load(open(R / "results/analysis/grid_video/grid_manifest.json"))
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for k, pair in enumerate(man["pairs"], 1):
        if a.only and a.only not in pair["task"]: continue
        order = man["simpler_layouts"] if pair["bench"] == "simpler" else man["libero_inits"]
        dst = out / f"pair{k}_{pair['task'].replace('/', '_').replace('widowx_', '').replace('_clean', '')}.mp4"
        make_clip(pair, a.frames, dst, order)
        print("wrote", dst)


if __name__ == "__main__":
    main()
