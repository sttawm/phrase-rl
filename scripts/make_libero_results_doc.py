#!/usr/bin/env python3
"""results doc for the pi05_libero single-change phrase rounds.

Numbers only -- no interpretation. Every figure is n=50 on the full init
population (0-49), so there is no window-composition question. The repeat
floor at this n is 1.6pp mean / 4.0pp p90 (pilot A vs B, 1,050 paired eps).

Regenerate: .venv/bin/python scripts/make_libero_results_doc.py
Publish:    Artifact tool on /tmp/libero_results.html
"""
import base64, glob, html, io, json, re
import pandas as pd
from PIL import Image
from scipy import stats

R1 = glob.glob("results/analysis/pi05_bank/roll50/roll*.jsonl")
R2 = glob.glob("results/analysis/pi05_bank/round2/r2roll*.jsonl")
d = pd.DataFrame([json.loads(l) for f in R1 + R2 for l in open(f)])
d["task"] = d.suite + "/" + d.task_id.astype(str)
V = d.groupby(["task", "phrase"]).success.mean().mul(100).to_dict()
m1 = pd.read_parquet("results/analysis/pi05_bank/roll50_manifest.parquet")
m1["task"] = m1.suite + "/" + m1.tid.astype(str)
m2 = pd.read_parquet("results/analysis/pi05_bank/round2_manifest.parquet")
m2["task"] = m2.suite + "/" + m2.task_id.astype(str)
CAN = {**{t: g.canonical.iloc[0] for t, g in m1.groupby("task")},
       **{t: g.canonical.iloc[0] for t, g in m2.groupby("task")}}
NAN = float("nan")


def P(a, b, n=50):
    ka, kb = round(a / 100 * n), round(b / 100 * n)
    return stats.fisher_exact([[ka, n - ka], [kb, n - kb]])[1]


def diff(a, b):
    ta, tb = a.split(), b.split()
    k = lambda w: re.sub(r"[^a-z]", "", w.lower())
    p = 0
    while p < min(len(ta), len(tb)) and k(ta[p]) == k(tb[p]):
        p += 1
    s = 0
    while s < min(len(ta), len(tb)) - p and k(ta[-1 - s]) == k(tb[-1 - s]):
        s += 1
    return ((" ".join(ta[:p]), " ".join(ta[p:len(ta) - s]), " ".join(ta[len(ta) - s:])),
            (" ".join(tb[:p]), " ".join(tb[p:len(tb) - s]), " ".join(tb[len(tb) - s:])))


def rend(t, c):
    pre, mid, suf = t
    e = html.escape
    mk = '<mark class="%s">%s</mark>' % (c, e(mid) if mid else "&empty;")
    return " ".join(x for x in [e(pre), mk, e(suf)] if x)


_FC = {}
def frame(t):
    if t in _FC:
        return _FC[t]
    su, ti = t.split("/")
    sub = "eval_frames" if su in ("libero_spatial", "libero_object") else "frames"
    im = Image.open("results/analysis/pi05_bank/%s/%s__%02d.png" % (sub, su, int(ti)))
    im = im.convert("RGB").resize((130, 130), Image.LANCZOS)
    b = io.BytesIO(); im.save(b, "JPEG", quality=72)
    _FC[t] = base64.b64encode(b.getvalue()).decode()
    return _FC[t]


CSS = """<style>
:root{--bg:#fff;--fg:#111;--mut:#666;--line:#ddd;--bad:#b00;--ok:#060;--hlg:#d8f0d8;--hlb:#f6d6d6;--acc:#456}
@media(prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#141414;--fg:#e8e8e8;--mut:#999;--line:#333;--bad:#f88;--ok:#8d8;--hlg:#1e3a1e;--hlb:#3d1c1c;--acc:#9ab}}
:root[data-theme="dark"]{--bg:#141414;--fg:#e8e8e8;--mut:#999;--line:#333;--bad:#f88;--ok:#8d8;--hlg:#1e3a1e;--hlb:#3d1c1c;--acc:#9ab}
body{background:var(--bg);color:var(--fg);font:14px/1.55 ui-monospace,Menlo,Consolas,monospace;margin:0;padding:24px;max-width:1000px}
h1{font-size:18px;margin:0 0 4px}h2{font-size:15px;margin:0 0 2px;font-weight:600}
h3{font-size:13px;margin:30px 0 2px;font-weight:600;color:var(--acc);letter-spacing:.06em;border-bottom:2px solid var(--acc);padding-bottom:5px}
.hdr,.sub{color:var(--mut);font-size:13px}.hdr{margin:0 0 18px}.sub{margin:8px 0 0}
.task{display:flex;gap:14px;padding:16px 0;border-top:1px solid var(--line);align-items:flex-start}
img{width:130px;flex:none;border:1px solid var(--line)}
.body{min-width:0;flex:1}.canon{color:var(--mut);margin:0 0 10px;font-size:13px}
.pair{margin:0 0 12px;padding-left:10px;border-left:2px solid var(--line)}
.lab{color:var(--mut);font-size:12px;margin-bottom:3px}.lab b{color:var(--fg);font-weight:600}
.row{display:flex;gap:10px;align-items:baseline}
.pct{width:44px;text-align:right;flex:none;font-variant-numeric:tabular-nums}
.hi .pct{color:var(--ok)}.lo .pct{color:var(--bad)}.nu .pct{color:var(--mut)}
.pr{color:var(--mut);font-size:11px}
mark{background:var(--hlg);color:inherit;padding:0 3px;border-radius:2px}mark.b{background:var(--hlb)}
table{border-collapse:collapse;font-size:13px;font-variant-numeric:tabular-nums;margin:8px 0}
th{text-align:left;color:var(--mut);font-weight:400;font-size:11px;padding:4px 15px 4px 0}
td{padding:3px 15px 3px 0;white-space:nowrap}
td.b{color:var(--bad);font-weight:600}td.g{color:var(--ok)}td.q{color:var(--mut)}
</style>"""

o = ["<title>LIBERO Phrase Collapse Results</title>", CSS,
     "<h1>Single-change phrase results &mdash; pi05_libero / LIBERO</h1>",
     '<p class="hdr">90 phrases &times; 50 initial states = <b>4,500 episodes</b>, two rounds. '
     "Every phrase measured on the full init population (inits 0&ndash;49). "
     "Repeat-measurement floor at this n: 1.6&nbsp;pp mean, 4.0&nbsp;pp p90 "
     "(pilot A vs B, 1,050 paired episodes). p-values Fisher exact, n=50 per side.</p>"]


def block(title, sub, items, cls):
    o.append("<h3>" + title + "</h3>")
    if sub:
        o.append('<p class="sub">' + sub + "</p>")
    df = pd.DataFrame(items, columns=["task", "hi", "lo", "lab", "prior"])
    for t, g in df.groupby("task", sort=False):
        o.append('<div class="task"><img src="data:image/jpeg;base64,' + frame(t) + '" alt="' + t + '"><div class="body">')
        o.append("<h2>" + t + '</h2><p class="canon">canonical: ' + html.escape(str(CAN.get(t, ""))) + "</p>")
        for _, x in g.iterrows():
            a, b = V[(t, x.hi)], V[(t, x.lo)]
            A, B = diff(x.hi, x.lo)
            pr = ' &nbsp;<span class="pr">prior %+.0f pp</span>' % x.prior if x.prior == x.prior else ""
            o.append('<div class="pair"><div class="lab"><b>%s</b> &nbsp;%+.0f pp &nbsp;&middot;&nbsp; p=%.4f%s</div>' % (x.lab, b - a, P(a, b), pr))
            o.append('<div class="row hi"><span class="pct">%.0f%%</span><span>%s</span></div>' % (a, rend(A, "")))
            o.append('<div class="row %s"><span class="pct">%.0f%%</span><span>%s</span></div></div>' % (cls, b, rend(B, "b")))
        o.append("</div></div>")


# ---- collapses (>=18pp). "start the burners" is held back: the scene has ONE
# burner and the singular control "start the burner" is not yet measured.
CONF = [
    ("libero_goal/7", "power on the stove burner", "power on the hot plate", "destination noun", 100),
    ("libero_goal/7", "switch on the stove", "switch on the hot plate", "destination noun", 100),
    ("libero_goal/1", "place the grey bowl on the electric burner", "place the grey bowl on the hot plate", "destination noun", 80),
    ("libero_90/44", "fire up the stove", "turn on the stove", "verb phrase", 100),
    ("libero_goal/5", "slide that plate over to the front of the stove", "slide that dish over to the front of the stove", "object noun", 60),
    ("libero_goal/8", "put the bowl on the plate", "put the bowl on the dish", "destination noun", NAN),
    ("libero_90/10", "place the black bowl on top of the cabinet", "place the black bowl up onto the cabinet", "preposition", 60),
    ("libero_90/38", "pick up the right moka pot and place it on the stove", "lift the right moka pot and place it on the stove", "verb", 65),
    ("libero_goal/9", "put the wine bottle on the rack", "put the wine bottle on the stand", "destination noun", NAN),
]
block("COLLAPSES &ge;18 pp", "", CONF, "lo")

o.append('<h3>&ldquo;hot plate&rdquo; BY TASK</h3><table><tr><th>phrase</th><th>task</th><th>plate in scene</th><th>success</th><th>vs canonical</th></tr>')
for ph, t, pl in [("switch on the hot plate", "libero_goal/7", "yes"),
                  ("power on the hot plate", "libero_goal/7", "yes"),
                  ("place the grey bowl on the hot plate", "libero_goal/1", "yes"),
                  ("put the moka pot on the hot plate", "libero_90/19", "no"),
                  ("put the right moka pot on the hot plate", "libero_90/38", "no")]:
    v = V[(t, ph)]; c = V.get((t, CAN[t]))
    dd = "%+.0f pp" % (v - c) if c is not None else "&mdash;"
    o.append('<tr><td>%s</td><td>%s</td><td>%s</td><td class="%s">%.0f%%</td><td>%s</td></tr>'
             % (html.escape(ph), t, pl, "b" if v < 20 else "g", v, dd))
o.append('</table><p class="sub">KITCHEN_SCENE3 and KITCHEN_SCENE8 contain no plate.</p>')

o.append('<h3>SUBSTITUTES FOR &ldquo;STOVE&rdquo; &mdash; libero_goal/7</h3><table><tr><th>phrase</th><th>success</th><th>&Delta;</th></tr>')
for ph in ["turn on the stove", "switch on the stove", "switch on the range", "switch on the hob",
           "switch on the cooktop", "switch on the heating element", "switch on the griddle", "switch on the hot plate"]:
    v = V[("libero_goal/7", ph)]
    o.append('<tr><td>%s</td><td>%.0f%%</td><td class="%s">%+.0f</td></tr>' % (html.escape(ph), v, "b" if v <= 20 else "", v - 100))
o.append("</table>")

o.append('<h3>VERB &times; NOUN &mdash; libero_goal/7</h3>')
o.append('<p class="sub">The scene contains a <b>single</b> burner. Cells marked &mdash; are not yet measured.</p>')
o.append('<table><tr><th></th><th>the stove</th><th>the burner</th><th>the burners</th></tr>')
for verb, tmpl in [("turn on", "turn on the %s"), ("switch on", "switch on the %s"), ("start", "start the %s")]:
    cells = []
    for n in ["stove", "burner", "burners"]:
        v = V.get(("libero_goal/7", tmpl % n))
        cells.append('<td class="q">&mdash;</td>' if v is None
                     else '<td class="%s">%.0f%%</td>' % ("b" if v < 80 else "g", v))
    o.append("<tr><td>%s</td>%s</tr>" % (verb, "".join(cells)))
o.append('</table><p class="sub">&ldquo;Could you start the burners?&rdquo; measured 20% in the phrase bank against a 100% canonical.</p>')

o.append('<h3>TARGET vs DESTINATION RENAME &mdash; 8 tasks</h3>')
o.append("<table><tr><th>task</th><th>canonical</th><th>target renamed</th><th>&Delta;</th><th>destination renamed</th><th>&Delta;</th></tr>")
rr = []
for t, g in m2[m2.strand == "B_role"].groupby("task"):
    rr.append((t, V.get((t, CAN[t])),
               V[(t, g[g.label == "TARGET rename"].phrase.iloc[0])],
               V[(t, g[g.label == "DEST rename"].phrase.iloc[0])]))
for t, c, tg, ds in sorted(rr, key=lambda x: x[3] - x[1]):
    o.append('<tr><td>%s</td><td>%.0f%%</td><td>%.0f%%</td><td class="%s">%+.0f</td><td>%.0f%%</td><td class="%s">%+.0f</td></tr>'
             % (t, c, tg, "b" if tg - c <= -20 else "", tg - c, ds, "b" if ds - c <= -20 else "", ds - c))
mt = sum(x[2] - x[1] for x in rr) / len(rr); md = sum(x[3] - x[1] for x in rr) / len(rr)
o.append('<tr><td colspan=3 style="color:var(--mut)">mean</td><td>%+.1f</td><td></td><td>%+.1f</td></tr></table>' % (mt, md))
o.append('<div class="task"><img src="data:image/jpeg;base64,' + frame("libero_goal/8") + '" alt="goal/8"><div class="body">')
o.append('<h2>libero_goal/8</h2><p class="canon">the same substitute word in both slots</p>')
for lab, ph, cls in [("canonical", "put the bowl on the plate", "hi"),
                     ("target renamed", "put the dish on the plate", "hi"),
                     ("destination renamed", "put the bowl on the dish", "lo")]:
    o.append('<div class="pair"><div class="lab"><b>%s</b></div><div class="row %s"><span class="pct">%.0f%%</span><span>%s</span></div></div>'
             % (lab, cls, V[("libero_goal/8", ph)], html.escape(ph)))
o.append("</div></div>")

NULL = [("libero_90/9", "put the black bowl on the plate", "put the black bowl on the dish", "destination noun", NAN),
        ("libero_90/9", "put the black bowl on the plate", "put the dark bowl on the plate", "colour word", NAN),
        ("libero_spatial/5", "pick up the black bowl on the ramekin and place it on the plate", "pick up the black bowl on the ramekin and place it on the dish", "destination noun", NAN),
        ("libero_90/19", "put the moka pot on the stove", "just put the moka pot on the stove", "filler added", NAN),
        ("libero_90/72", "put the white mug on the plate", "put that white mug on the plate", "determiner", NAN),
        ("libero_90/70", "put the chocolate pudding to the right of the plate", "put the chocolate dessert to the right of the plate", "object noun", NAN)]
block("NULLS &mdash; representative", "24 of 28 cold-tested edits cost 10 pp or less.", NULL, "nu")

o.append("<h3>PRIORS THAT DID NOT HOLD</h3><table><tr><th>task</th><th>swap</th><th>prior (n=10&ndash;20)</th><th>n=50</th></tr>")
for t, hi, lo, sw, pri in [
        ("libero_90/28", "close the top drawer of the cabinet", "close the upper drawer of the cabinet", "top &rarr; upper", 70),
        ("libero_90/59", "grasp the green and orange can and put it in the wooden tray", "grasp the green and orange tin and put it in the wooden tray", "can &rarr; tin", 80),
        ("libero_90/59", "grasp the green and orange can and put it in the wooden tray", "grasp the orange and green can and put it in the wooden tray", "adjective order", 80),
        ("libero_90/82", "place the black book into the left slot of the brown caddy", "place the black book into the left compartment of the brown caddy", "slot &rarr; compartment", 60),
        ("libero_90/70", "set the brown pudding down to the right of that plate", "set the chocolate pudding down to the right of that plate", "brown &rarr; chocolate", 60),
        ("libero_90/82", "place the black book into the left slot of the brown caddy", "place the black book inside the left slot of the brown caddy", "into &rarr; inside", 60),
        ("libero_90/59", "grasp the tomato sauce can and place it in the wooden tray", "grasp the tomato sauce can and put it in the wooden tray", "place &rarr; put", 60),
        ("libero_90/28", "close the open top drawer of the white cabinet on the left", "push in the open top drawer of the white cabinet on the left", "push in &rarr; close", -65)]:
    o.append("<tr><td>%s</td><td>%s</td><td style=\"color:var(--mut)\">%+.0f pp</td><td>%+.0f pp</td></tr>"
             % (t, sw, -pri, V[(t, lo)] - V[(t, hi)]))
o.append("</table>")
o.append('<p class="hdr" style="margin-top:26px;border-top:1px solid var(--line);padding-top:12px">'
         'libero_90/44 canonical &ldquo;turn on the stove&rdquo; = 6%. libero_goal/7 canonical, same string, = 100%.<br>'
         "Data: roll50_results.parquet, round2_results.parquet &middot; per-episode rows in roll50/ and round2/.</p>")

open("/tmp/libero_results.html", "w").write("\n".join(o))
print("wrote /tmp/libero_results.html")
