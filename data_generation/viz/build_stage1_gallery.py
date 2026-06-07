#!/usr/bin/env python
"""
Build a styled HTML gallery of stage1 query-ready records (task proposals + role decomposition).

  python viz/build_stage1_gallery.py --in outputs/stage1_dataset.json \
      --out galleries/stage1_gallery.html
"""
import os
import io
import json
import math
import base64
import argparse
import html as _html
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


def view_tag(view_path):
    """Short viewpoint tag from a gObjaverse view's camera json (TOP / UNDER / elev/az)."""
    d = os.path.dirname(view_path)
    idx = os.path.basename(d)
    try:
        o = json.load(open(os.path.join(d, f"{idx}.json"))).get("origin")
        r = math.sqrt(sum(t * t for t in o)) or 1.0
        elev = math.degrees(math.asin(o[2] / r))
        az = math.degrees(math.atan2(o[1], o[0])) % 360
    except Exception:
        return ""
    if elev > 60:
        return "TOP-DOWN"
    if elev < -60:
        return "UNDERSIDE"
    return f"~{round(elev)}° / az {round(az)}°"


def views_strip(views, h=150, limit=8):
    """All views as crisp, individually-labeled thumbnails laid out left-to-right."""
    out = []
    for p in views[:limit]:
        try:
            im = Image.open(p).convert("RGB")
            w0, h0 = im.size
            im = im.resize((max(1, round(w0 * h / h0)), h))
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=82)
            out.append((base64.b64encode(buf.getvalue()).decode(), view_tag(p)))
        except Exception:
            pass
    return out


def esc(x):
    return _html.escape(str(x if x is not None else ""))


CSS = """
:root{--bg:#0f1216;--card:#fff;--ink:#1a1d23;--muted:#6b7280;--line:#e6e8ec;
--inter:#2563eb;--intra:#059669;--chip:#f3f4f6;--accent:#7c3aed}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",Arial,"PingFang TC","Noto Sans CJK TC",sans-serif}
header{padding:28px 32px;color:#fff}
header h1{margin:0 0 4px;font-size:22px;font-weight:700;letter-spacing:.2px}
header .meta{color:#9aa3af;font-size:13px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(430px,1fr));gap:20px;padding:0 32px 48px}
.card{background:var(--card);border-radius:16px;overflow:hidden;box-shadow:0 1px 3px rgba(0,0,0,.25),0 12px 28px rgba(0,0,0,.18)}
.views{display:flex;gap:6px;overflow-x:auto;padding:8px 0 6px;margin:0 0 4px}
.views figure{margin:0;flex:none;text-align:center}
.views img{height:150px;width:auto;display:block;border-radius:8px;border:1px solid var(--line);background:#fafafa}
.views figcaption{font:10.5px/1.3 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;color:var(--muted);margin-top:3px}
.noimg{color:#6b7280;padding:24px 0;font-size:13px}
.body{padding:16px 18px 18px}
.title{font-size:17px;font-weight:700}
.sub{color:var(--muted);font-size:12.5px;margin:2px 0 12px}
.sub code{background:var(--chip);padding:1px 6px;border-radius:6px;font-size:11.5px}
.query{border:1px solid var(--line);border-radius:12px;padding:11px 13px;margin-bottom:10px;background:#fcfcfd}
.qhead{display:flex;align-items:center;gap:8px;font-weight:650;font-size:14.5px}
.badge{font-size:10.5px;font-weight:700;letter-spacing:.4px;text-transform:uppercase;
padding:3px 8px;border-radius:999px;color:#fff;flex:none}
.badge.inter{background:var(--inter)}.badge.intra{background:var(--intra)}
.pattern{color:var(--muted);font-size:12.5px;margin:5px 0 9px}
.pattern b{color:var(--accent);font-weight:700}
.roles{display:flex;flex-direction:column;gap:6px;margin-bottom:9px}
.role{display:flex;gap:8px;align-items:baseline;font-size:13px}
.role .rid{font-weight:700;color:#9ca3af;width:14px;flex:none}
.role .verb{font-weight:700}
.role .at{color:var(--muted)}
.role .reg{color:#111;background:#eef2ff;border:1px solid #e0e7ff;border-radius:6px;padding:0 6px}
.role .fn{color:var(--muted);font-style:italic}
.molmo{font:12px/1.45 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
background:#0f172a;color:#cbd5e1;border-radius:7px;padding:6px 9px;margin-top:5px;word-break:break-word}
.molmo .pt{color:#38bdf8}
footer{color:#6b7280;text-align:center;padding:24px;font-size:12px}
"""


def role_html(r):
    return (f'<div class="role"><span class="rid">{esc(r.get("id"))}</span>'
            f'<span class="verb">{esc(r.get("role"))}</span>'
            f'<span class="at">→ {esc(r.get("target"))} @</span>'
            f'<span class="reg">{esc(r.get("contact_region"))}</span>'
            f'<span class="fn">{esc(r.get("function"))}</span></div>')


def molmo_html(m):
    s = esc(m)
    s = s.replace("Point to", '<span class="pt">Point to</span>', 1)
    return f'<div class="molmo">{s}</div>'


def query_html(q):
    cat = q.get("category", "")
    roles = "".join(role_html(r) for r in q.get("roles", []))
    molmos = "".join(molmo_html(m) for m in q.get("molmo_queries", []))
    return (f'<div class="query"><div class="qhead">'
            f'<span class="badge {esc(cat)}">{esc(cat)}</span>{esc(q.get("query"))}</div>'
            f'<div class="pattern"><b>{esc(q.get("pattern"))}</b> · {esc(q.get("relation"))}</div>'
            f'<div class="roles">{roles}</div>{molmos}</div>')


def card_html(o):
    strip = views_strip(o.get("views_used", []))
    if strip:
        figs = "".join(
            f'<figure><img src="data:image/jpeg;base64,{b}">'
            f'<figcaption>{esc(tag)}</figcaption></figure>'
            for b, tag in strip)
        img = f'<div class="views">{figs}</div>'
    else:
        img = '<div class="noimg">no image</div>'
    parts = ", ".join(esc(c.get("name")) for c in o.get("components", []))
    queries = "".join(query_html(q) for q in o.get("queries", []))
    return (f'<div class="card"><div class="body">'
            f'<div class="title">{esc(o.get("object_name"))}</div>'
            f'{img}'
            f'<div class="sub"><code>{esc(str(o.get("object_id"))[:12])}</code> · '
            f'parts: {parts}</div>{queries}</div></div>')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="outputs/stage1_dataset.json")
    ap.add_argument("--out", default="galleries/stage1_gallery.html")
    ap.add_argument("--title", default="Stage 1 — Task proposals & role decomposition")
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    dg = os.path.dirname(here)
    inp = args.inp if os.path.isabs(args.inp) else os.path.join(dg, args.inp)
    out = args.out if os.path.isabs(args.out) else os.path.join(dg, args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)

    objs = [o for o in json.load(open(inp)) if o.get("queries")]
    n_q = sum(len(o.get("queries", [])) for o in objs)
    cards = "".join(card_html(o) for o in objs)
    doc = (f"<!doctype html><html><head><meta charset='utf-8'>"
           f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
           f"<title>{esc(args.title)}</title><style>{CSS}</style></head><body>"
           f"<header><h1>{esc(args.title)}</h1>"
           f"<div class='meta'>{len(objs)} objects · {n_q} queries · "
           f"inter = whole-object · intra = part-level · each role: verb → target @ contact_region</div>"
           f"</header><div class='grid'>{cards}</div>"
           f"<footer>generated from {esc(os.path.basename(inp))}</footer></body></html>")
    with open(out, "w") as f:
        f.write(doc)
    print(f"wrote {out}  ({len(objs)} objects, {n_q} queries)")


if __name__ == "__main__":
    main()
