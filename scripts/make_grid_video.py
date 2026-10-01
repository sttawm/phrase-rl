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
import argparse, difflib, io, json, pathlib, re
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import imageio.v2 as imageio

R = pathlib.Path(__file__).resolve().parents[1]
SCALE = 2                  # 1 = 1998px-wide canvas; 2 = 3996px (sharp when fullscreen)
CELL = (160 * SCALE, 120 * SCALE)   # w, h of one cell
COLS, ROWS = 6, 4
GAP, PAD = 3 * SCALE, 16 * SCALE
GREEN, RED, INK, BG = (34, 197, 94), (239, 68, 68), (40, 44, 52), (250, 249, 246)
HI_BG, LO_BG = (212, 241, 212), (250, 214, 214)      # highlight boxes (paper palette)
FPS, HOLD_S, TITLE_S, MAX_PLAY_S = 12, 2.5, 2.5, 12.0
QUALITY = 6.5              # imageio: crf = (10 - q) * 5
ORDER = "layout"           # 'layout': same scene in the same cell left/right (default); 'success': successes first


def font(size):
    size = int(size * SCALE)
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


def grid_frames(loaded, order, T, stride):
    """Yield composited grid images for t = 0..T (shared across both arms), showing
    every `stride`-th sim step; a cell freezes + gets marked once its episode ends.
    Cells are ordered successes-first; during the final hold a translucent
    green/red tint fades in over every cell."""
    gw = COLS * CELL[0] + (COLS - 1) * GAP
    gh = ROWS * CELL[1] + (ROWS - 1) * GAP
    fnt = font(26)
    order = [i for i in order if i in loaded]
    if ORDER == "success":
        order = sorted(order, key=lambda i: (not loaded[i][1], i))
    T_play = -(-T // stride)
    for tt in range(T_play + int(HOLD_S * FPS)):
        t = tt * stride
        tint = min(1.0, max(0.0, (tt - T_play) / (0.6 * FPS))) * 0.55   # fade-in during the hold
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
                d.rectangle([x, y, x + CELL[0] - 1, y + CELL[1] - 1], outline=col, width=3 * SCALE)
                # drawn badge (font glyphs for check/cross are unreliable)
                cx, cy, rr = x + CELL[0] - 18 * SCALE, y + 18 * SCALE, 13 * SCALE
                d.ellipse([cx - rr, cy - rr, cx + rr, cy + rr], fill=(255, 255, 255), outline=col, width=2 * SCALE)
                if ok:
                    d.line([(cx - 7 * SCALE, cy + 1 * SCALE), (cx - 2 * SCALE, cy + 6 * SCALE), (cx + 8 * SCALE, cy - 6 * SCALE)], fill=col, width=3 * SCALE, joint="curve")
                else:
                    d.line([(cx - 6 * SCALE, cy - 6 * SCALE), (cx + 6 * SCALE, cy + 6 * SCALE)], fill=col, width=3 * SCALE)
                    d.line([(cx - 6 * SCALE, cy + 6 * SCALE), (cx + 6 * SCALE, cy - 6 * SCALE)], fill=col, width=3 * SCALE)
        if tint > 0:
            ov = Image.new("RGBA", img.size, (0, 0, 0, 0)); od = ImageDraw.Draw(ov)
            for k, i in enumerate(order):
                r, c = divmod(k, COLS)
                x, y = c * (CELL[0] + GAP), r * (CELL[1] + GAP)
                col = GREEN if loaded[i][1] else RED
                od.rectangle([x, y, x + CELL[0] - 1, y + CELL[1] - 1], fill=col + (int(255 * tint),))
            img = Image.alpha_composite(img.convert("RGBA"), ov).convert("RGB")
            d = ImageDraw.Draw(img)
            for k, i in enumerate(order):   # redraw badges on top of the tint
                r, c = divmod(k, COLS)
                x, y = c * (CELL[0] + GAP), r * (CELL[1] + GAP)
                col = GREEN if loaded[i][1] else RED
                cx, cy, rr = x + CELL[0] - 18 * SCALE, y + 18 * SCALE, 13 * SCALE
                d.ellipse([cx - rr, cy - rr, cx + rr, cy + rr], fill=(255, 255, 255), outline=col, width=2 * SCALE)
                if loaded[i][1]:
                    d.line([(cx - 7 * SCALE, cy + 1 * SCALE), (cx - 2 * SCALE, cy + 6 * SCALE), (cx + 8 * SCALE, cy - 6 * SCALE)], fill=col, width=3 * SCALE, joint="curve")
                else:
                    d.line([(cx - 6 * SCALE, cy - 6 * SCALE), (cx + 6 * SCALE, cy + 6 * SCALE)], fill=col, width=3 * SCALE)
                    d.line([(cx - 6 * SCALE, cy + 6 * SCALE), (cx + 6 * SCALE, cy - 6 * SCALE)], fill=col, width=3 * SCALE)
        yield img, sum(ok for f, ok in loaded.values() if t >= len(f) - 1), len(loaded)


def diff_words(a, b):
    """Indices of words in a and in b that differ (word-level, case-sensitive)."""
    wa, wb = a.split(), b.split()
    ia, ib = set(), set()
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, wa, wb).get_opcodes():
        if op != "equal":
            ia.update(range(i1, i2)); ib.update(range(j1, j2))
    return ia, ib


def phrase_width(d, phrase, fnt):
    return d.textlength("“" + phrase + "”", font=fnt)


def draw_phrase(d, x, y, phrase, hl, fnt, color, box, prefix="", prefix_color=None):
    """Draw `phrase` word by word at (x, y); words whose index is in `hl` get a
    colored box behind them and colored text. `prefix` (e.g. the rate) goes first."""
    if prefix:
        d.text((x, y), prefix, fill=prefix_color or color, font=fnt)
        x += d.textlength(prefix, font=fnt)
    d.text((x, y), "“", fill=INK, font=fnt); x += d.textlength("“", font=fnt) + 3 * SCALE
    words = phrase.split(); sp = d.textlength(" ", font=fnt)
    asc, desc = fnt.getmetrics()
    # one box per run of adjacent highlighted words
    xs = []; cx = x
    for i, w in enumerate(words):
        ww = d.textlength(w, font=fnt); xs.append((cx, cx + ww)); cx += ww + sp
    i = 0
    while i < len(words):
        if i in hl:
            j = i
            while j + 1 < len(words) and (j + 1) in hl: j += 1
            d.rounded_rectangle([xs[i][0] - 4 * SCALE, y - 2 * SCALE, xs[j][1] + 4 * SCALE, y + asc + desc * 0.4],
                                radius=6 * SCALE, fill=box)
            i = j + 1
        else:
            i += 1
    for i, w in enumerate(words):
        d.text((xs[i][0], y), w, fill=color if i in hl else INK, font=fnt)
    x = xs[-1][1] + 2 * SCALE
    d.text((x, y), "”", fill=INK, font=fnt)
    return x


def fit_font(d, text, max_w, start, floor=22):
    size = start
    while size > floor and d.textlength(text, font=font(size)) > max_w:
        size -= 2
    return font(size)


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
    lg = {i: load_episode(eg[i]) for i in order if i in eg}
    lb = {i: load_episode(eb[i]) for i in order if i in eb}
    if not lg or not lb:
        raise SystemExit("no episodes found for an arm -- check the frames dir layout")
    T = max(len(f) for f, _ in list(lg.values()) + list(lb.values()))
    stride = max(1, round(T / (MAX_PLAY_S * FPS)))   # keep the longest episode within MAX_PLAY_S
    gw = COLS * CELL[0] + (COLS - 1) * GAP; gh = ROWS * CELL[1] + (ROWS - 1) * GAP
    W, H = 2 * gw + 3 * PAD, gh + 2 * PAD + 70 * SCALE
    W += W % 2; H += H % 2   # libx264 yuv420p needs even dimensions
    writer = imageio.get_writer(out, fps=FPS, codec="libx264", quality=QUALITY, macro_block_size=1)
    hg, hb = diff_words(good["phrase"], bad["phrase"])
    title = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(title)
    fmt = lambda r: f"{round(float(r))}%   " if r not in (None, "?") else "?%   "
    pg, pb = fmt(good.get("rate")), fmt(bad.get("rate"))
    longest = max(pg + "“" + good["phrase"] + "”", pb + "“" + bad["phrase"] + "”", key=len)
    big = fit_font(d, longest, W - 120 * SCALE, 54)
    lh = sum(big.getmetrics()) + 26 * SCALE
    y0 = H // 2 - lh - 10 * SCALE
    for k, (ph, hl, rate, col, box) in enumerate([(good["phrase"], hg, pg, GREEN, HI_BG), (bad["phrase"], hb, pb, RED, LO_BG)]):
        wline = d.textlength(rate, font=big) + phrase_width(d, ph, big)
        draw_phrase(d, (W - wline) / 2, y0 + k * lh, ph, hl, big, col, box, prefix=rate, prefix_color=col)
    small = font(22); note = "same scenes, same seeds — the wording is the only difference"
    d.text(((W - d.textlength(note, font=small)) / 2, y0 + 2 * lh + 18 * SCALE), note, fill=(110, 110, 110), font=small)
    for _ in range(int(TITLE_S * FPS)): writer.append_data(np.asarray(title))
    hdr = fit_font(d, "“" + max(good["phrase"], bad["phrase"], key=len) + "”", gw - 10 * SCALE, 24, 16)
    for (gi, gs, gn), (bi, bs, bn) in zip(grid_frames(lg, order, T, stride), grid_frames(lb, order, T, stride)):
        img = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(img)
        draw_phrase(d, PAD, PAD, good["phrase"], hg, hdr, GREEN, HI_BG)
        draw_phrase(d, 2 * PAD + gw, PAD, bad["phrase"], hb, hdr, RED, LO_BG)
        img.paste(gi, (PAD, PAD + 40 * SCALE)); img.paste(bi, (2 * PAD + gw, PAD + 40 * SCALE))
        foot = font(20)
        d.text((PAD, PAD + 40 * SCALE + gh + 6 * SCALE), f"{gs}/{gn} succeeded", fill=GREEN, font=foot)
        d.text((2 * PAD + gw, PAD + 40 * SCALE + gh + 6 * SCALE), f"{bs}/{bn} succeeded", fill=RED, font=foot)
        writer.append_data(np.asarray(img))
    writer.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--only", default=None, help="substring of task to render")
    ap.add_argument("--order", default="layout", choices=["layout", "success"])
    ap.add_argument("--suffix", default="", help="appended to the clip filename")
    a = ap.parse_args()
    global ORDER; ORDER = a.order
    man = json.load(open(R / "results/analysis/grid_video/grid_manifest.json"))
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for k, pair in enumerate(man["pairs"], 1):
        if a.only and a.only not in pair["task"]: continue
        order = man["simpler_layouts"] if pair["bench"] == "simpler" else man["libero_inits"]
        dst = out / f"pair{k}_{pair['task'].replace('/', '_').replace('widowx_', '').replace('_clean', '')}{a.suffix}.mp4"
        make_clip(pair, a.frames, dst, order)
        print("wrote", dst)


if __name__ == "__main__":
    main()
