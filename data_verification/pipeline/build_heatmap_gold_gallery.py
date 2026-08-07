#!/usr/bin/env python
"""Heatmap-gold review page: each hand's own eight views next to the words it was scored against.

The gold is per-hand now (score_A / score_B, overall = min), so a page that only shows the combined
picture and one number hides what the label was actually about. Here each half of a card is one
hand: its picture, its expected contact in words, and the score that hand got.

  python pipeline/build_heatmap_gold_gallery.py --dir quality_evaluation/claude_pilot_1000 \
      --out quality_evaluation/claude_pilot_1000/heatmap_gold.html
"""
import argparse, ast, html, json, os, struct, zipfile
from array import array
from collections import Counter

try:
    import numpy as np
except ModuleNotFoundError:  # gallery generation only needs two flat float arrays from each npz
    np = None

SC = {0: "bad", 1: "ok", 2: "good"}


def active_counts(path, threshold=.15):
    """Return (point count, active A, active B), with a stdlib fallback for light environments."""
    if np is not None:
        with np.load(path, allow_pickle=True) as d:
            return (len(d["scoreA"]), int((d["scoreA"] >= threshold).sum()),
                    int((d["scoreB"] >= threshold).sum()))

    def read_npy(zf, name):
        with zf.open(f"{name}.npy") as f:
            if f.read(6) != b"\x93NUMPY":
                raise ValueError(f"{path}:{name} is not an npy array")
            major, _minor = f.read(2)
            hlen = struct.unpack("<H" if major == 1 else "<I", f.read(2 if major == 1 else 4))[0]
            header = ast.literal_eval(f.read(hlen).decode("latin1"))
            shape, descr = header["shape"], header["descr"]
            if len(shape) != 1 or descr[-2:] not in ("f4", "f8"):
                raise ValueError(f"unsupported score array shape/dtype: {shape} {descr}")
            values = array("f" if descr[-2:] == "f4" else "d")
            values.frombytes(f.read())
            if descr[0] == ">":
                values.byteswap()
            return values

    with zipfile.ZipFile(path) as zf:
        a, b = read_npy(zf, "scoreA"), read_npy(zf, "scoreB")
    if len(a) != len(b):
        raise ValueError(f"score length mismatch in {path}")
    return len(a), sum(v >= threshold for v in a), sum(v >= threshold for v in b)


def rel(path, base):
    link = os.path.join(base, "imgs")
    if os.path.islink(link):
        under = os.path.relpath(os.path.realpath(path), os.path.realpath(link))
        if not under.startswith(".."):
            return os.path.join("imgs", under)
    return os.path.relpath(path, base)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--labels", default=None,
                    help="JSONL verdicts to display (default: DIR/heatmap_labels.gold.jsonl)")
    ap.add_argument("--allow-subset", action="store_true",
                    help="allow labels to contain a reviewed subset of the manifest (preview only)")
    ap.add_argument("--per-page", type=int, default=100)
    a = ap.parse_args()
    D, base = a.dir, os.path.dirname(os.path.abspath(a.out))

    man = {json.loads(l)["sample_id"]: json.loads(l) for l in open(f"{D}/manifest.jsonl")}
    labels_path = a.labels or f"{D}/heatmap_labels.gold.jsonl"
    gold = {json.loads(l)["sample_id"]: json.loads(l) for l in open(labels_path)}
    if set(gold) != set(man) and not (a.allow_subset and set(gold) < set(man)):
        raise ValueError(f"labels/manifest mismatch: labels={len(gold)}, manifest={len(man)}, "
                         f"missing={len(set(man)-set(gold))}, extra={len(set(gold)-set(man))}")
    models = sorted({g.get("model") for g in gold.values() if g.get("model")})
    prompt_versions = sorted({g.get("prompt_version") for g in gold.values()
                              if g.get("prompt_version")})
    provenance = ", ".join(models + prompt_versions) or os.path.basename(labels_path)

    rows = []
    for sid, g in gold.items():
        m = man[sid]
        meta = json.load(open(m["meta_path"]))
        ra, rb = meta["roles"][0], meta["roles"][1]
        n, na, nb = active_counts(m["scores_path"])
        rows.append({
            "sid": sid, "obj": meta["object_name"], "task": meta["task"],
            "A": g.get("score_A"), "B": g.get("score_B"), "o": g["overall_score"],
            "f": g.get("fault", "none"), "r": g.get("reason", ""),
            "ca": f'{ra["role"]} @ {ra["contact_region"]}',
            "cb": f'{rb["role"]} @ {rb["contact_region"]}',
            "aa": g.get("orange_alignment", ""), "ba": g.get("teal_alignment", ""),
            "ash": g.get("orange_shape", ""), "bsh": g.get("teal_shape", ""),
            "na": na, "nb": nb, "pa": 100 * na / n, "pb": 100 * nb / n,
            "ia": rel(m["heat_png_A"], base), "ib": rel(m["heat_png_B"], base),
        })
    rows.sort(key=lambda r: (r["o"], min(r["A"], r["B"]), r["sid"]))

    def card(r):
        def half(tag, colour, score, contact, img, npts, pct, alignment, shape):
            quality = ""
            if alignment or shape:
                quality = f'<div class="quality">{html.escape(alignment)} · {html.escape(shape)}</div>'
            return f"""<div class="hand">
  <div class="hh"><b>{tag}</b> <span class="sw {colour}"></span>
    <span class="pill s{score}">gold {score}</span>
    <span class="pts">{npts} pts &middot; {pct:.2f}%</span></div>
  <div class="ct">{html.escape(contact)}</div>
  {quality}<img loading="lazy" src="{img}"></div>"""
        return f"""<div class="card" data-o="{r['o']}" data-f="{r['f']}">
  <div class="hd"><b>{html.escape(r['obj'])}</b> &mdash; {html.escape(r['task'])}
    <span class="pill s{r['o']}">overall {r['o']}</span>
    <span class="pill tag">{html.escape(r['f'])}</span>
    <span class="sid">{html.escape(r['sid'])}</span></div>
  <div class="rs">{html.escape(r['r'])}</div>
  <div class="two">{half('HAND A', 'oa', r['A'], r['ca'], r['ia'], r['na'], r['pa'], r['aa'], r['ash'])}
{half('HAND B', 'ob', r['B'], r['cb'], r['ib'], r['nb'], r['pb'], r['ba'], r['bsh'])}</div>
</div>"""

    dist = Counter(r["o"] for r in rows)
    da = Counter(r["A"] for r in rows); db = Counter(r["B"] for r in rows)
    faults = Counter(r["f"] for r in rows)
    css = """
body{font-family:system-ui;margin:18px;background:#f4f4f6;color:#111}
h1{font-size:19px;margin:0 0 6px} .dim{color:#777;font-weight:400;font-size:13px}
.summary{background:#fff;border:1px solid #ddd;border-radius:8px;padding:10px 14px;margin-bottom:12px;font-size:13.5px}
.summary code{background:#f0f0f3;padding:1px 5px;border-radius:4px}
.bar{display:flex;height:12px;border-radius:6px;overflow:hidden;margin:4px 0 8px;max-width:640px}
.b0{background:#ef4444}.b1{background:#f59e0b}.b2{background:#22c55e}
.card{background:#fff;border:1px solid #ddd;border-radius:9px;padding:12px 14px;margin:14px 0}
.hd{font-size:15px;margin-bottom:4px}
.sid{color:#999;font-size:11.5px;margin-left:8px;font-family:ui-monospace,monospace}
.rs{color:#555;font-size:13px;margin-bottom:10px}
.two{display:flex;gap:14px}.hand{flex:1;min-width:0}
.hh{font-size:13px;margin-bottom:2px}
.ct{font-size:12.5px;color:#444;background:#fafafa;border-left:3px solid #ddd;padding:4px 8px;margin-bottom:5px;min-height:2.2em}
.sw{display:inline-block;width:10px;height:10px;border-radius:2px;vertical-align:middle}
.oa{background:#F5A623}.ob{background:#2ED8CE}
.pts{color:#b45309;font-size:11.5px;font-variant-numeric:tabular-nums}
.quality{color:#555;font-size:11.5px;margin:-2px 0 4px;font-family:ui-monospace,monospace}
img{width:100%;border:1px solid #eee;border-radius:5px;display:block}
.pill{display:inline-block;padding:1px 8px;border-radius:10px;font-size:12px;margin-left:6px}
.s0{background:#fde8e8;color:#b91c1c}.s1{background:#fef3c7;color:#92400e}
.s2{background:#dcfce7;color:#166534}.tag{background:#e0e7ff;color:#3730a3}
.controls{position:sticky;top:0;background:#f4f4f6;padding:8px 0;z-index:5;font-size:13px}
select{font-size:13px;padding:2px 4px}
.nav{margin:10px 0;font-size:13px}.nav a{margin-right:8px}
"""
    def bar(c):
        t = sum(c.values()) or 1
        return "".join(f'<div class="b{k}" style="width:{100*c[k]/t}%"></div>' for k in (0, 1, 2))

    pages = [rows[i:i + a.per_page] for i in range(0, len(rows), a.per_page)] or [[]]
    stem = os.path.splitext(os.path.basename(a.out))[0]

    first_name = os.path.basename(a.out)

    def nav(cur, n):
        return '<div class="nav">' + "".join(
            (f"<b>{i+1}</b> " if i == cur else
             f'<a href="{first_name if i == 0 else f"{stem}_p{i+1}.html"}">{i+1}</a>')
            for i in range(n)) + "</div>"

    head = f"""<!doctype html><meta charset="utf-8"><title>heatmap verification</title><style>{css}</style>
<h1>Heatmap verification &mdash; per-hand scores on the split renders
<span class="dim">(localization first; target-relative shape completeness second; overall = min(A,B))</span></h1>
<div class="summary">
judge: <code>{html.escape(provenance)}</code><br>
overall 0/1/2 = <code>{dist[0]}/{dist[1]}/{dist[2]}</code>&nbsp;&nbsp;
hand A = <code>{da[0]}/{da[1]}/{da[2]}</code>&nbsp;&nbsp;hand B = <code>{db[0]}/{db[1]}/{db[2]}</code><br>
<div class="bar">{bar(dist)}</div>
faults: {", ".join(f"<code>{k}</code> {v}" for k, v in faults.most_common())}<br>
<span class="dim">"pts" is how many point-cloud points that hand marks; it is displayed for review,
not used as a quality threshold. Scores come from the visual judge's precision-first,
target-relative completeness rubric.</span>
</div>
<div class="controls">overall <select id="fo"><option value="">all</option>
<option value="0">bad</option><option value="1">ok</option><option value="2">good</option></select>
&nbsp;fault <select id="ff"><option value="">all</option>{"".join(f'<option>{k}</option>' for k in faults)}</select>
&nbsp;<span class="dim">sorted worst-first</span></div>"""
    js = """<script>
const fo=document.getElementById('fo'),ff=document.getElementById('ff');
function ap(){document.querySelectorAll('.card').forEach(c=>{
  c.style.display=((!fo.value||c.dataset.o===fo.value)&&(!ff.value||c.dataset.f===ff.value))?'':'none';});}
fo.onchange=ff.onchange=ap;</script>"""

    for i, page in enumerate(pages):
        out = a.out if i == 0 else os.path.join(base, f"{stem}_p{i+1}.html")
        with open(out, "w") as f:
            f.write(head + nav(i, len(pages)) + "".join(card(r) for r in page)
                    + nav(i, len(pages)) + js)
    print(f"-> {a.out} + {len(pages)-1} paged files ({len(rows)} cards)")


if __name__ == "__main__":
    main()
