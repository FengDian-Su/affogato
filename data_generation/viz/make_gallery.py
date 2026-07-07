"""
make_gallery.py — 把 stage1 JSON 輸出轉成 HTML gallery

用法：
    python make_gallery.py --in outputs/stage1_redesign_26b_test_0.json --out outputs/gallery.html
    python make_gallery.py --in outputs/stage1_redesign_26b.json --out outputs/gallery.html --max 100
"""

import argparse
import base64
import io
import json
import os
import sys
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    print("需要 Pillow: pip install Pillow")
    sys.exit(1)

# ──────────────────────────────────────────
# CSS
# ──────────────────────────────────────────
CSS = """
:root{--bg:#0f1216;--card:#fff;--ink:#1a1d23;--muted:#6b7280;--line:#e6e8ec;
--inter:#2563eb;--intra:#059669;--chip:#f3f4f6;--accent:#7c3aed}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif}
header{padding:28px 32px;color:#fff}
header h1{margin:0 0 4px;font-size:22px;font-weight:700;letter-spacing:.2px}
header .meta{color:#9aa3af;font-size:13px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(430px,1fr));gap:20px;padding:0 32px 48px}
.card{background:var(--card);border-radius:16px;overflow:hidden;
box-shadow:0 1px 3px rgba(0,0,0,.25),0 12px 28px rgba(0,0,0,.18)}
.views{display:flex;gap:6px;overflow-x:auto;padding:8px 12px 6px;margin:0 0 4px}
.views figure{margin:0;flex:none;text-align:center}
.views img{height:150px;width:auto;display:block;border-radius:8px;
border:1px solid var(--line);background:#fafafa}
.views figcaption{font:10.5px/1.3 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
color:var(--muted);margin-top:3px}
.noimg{color:#6b7280;padding:24px 12px;font-size:13px}
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

# ──────────────────────────────────────────
# 工具函式
# ──────────────────────────────────────────

def img_to_base64(path, max_height=150):
    try:
        img = Image.open(path).convert("RGB")
        if img.height > max_height:
            ratio = max_height / img.height
            img = img.resize((int(img.width * ratio), max_height), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=72)
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception:
        return None


def view_label(path):
    name = Path(path).stem
    try:
        idx = int(name)
        az = (idx // 4) * 60 % 360
        if name == "00026":
            return "top-down"
        if name == "00027":
            return "underside"
        return f"~24° / az {az}°"
    except ValueError:
        return name


def esc(s):
    return (str(s)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;"))


# ──────────────────────────────────────────
# HTML 生成
# ──────────────────────────────────────────

def render_views(views):
    if not views:
        return "<div class='noimg'>No images</div>"
    parts = ["<div class='views'>"]
    for v in views:
        if not os.path.exists(v):
            continue
        b64 = img_to_base64(v)
        if b64 is None:
            continue
        label = esc(view_label(v))
        parts.append(
            f"<figure>"
            f"<img src='data:image/jpeg;base64,{b64}' loading='lazy'>"
            f"<figcaption>{label}</figcaption>"
            f"</figure>"
        )
    parts.append("</div>")
    return "".join(parts)


def render_role(role):
    rid  = esc(role.get("id", "?"))
    verb = esc(role.get("role", ""))
    reg  = esc(role.get("contact_region", role.get("target", "")))
    fn   = esc(role.get("function", ""))
    return (
        f"<div class='role'>"
        f"<span class='rid'>{rid}</span>"
        f"<span class='verb'>{verb}</span>"
        f"<span class='at'>@</span>"
        f"<span class='reg'>{reg}</span>"
        f"<span class='fn'>{' — ' + fn if fn else ''}</span>"
        f"</div>"
    )


def render_molmo_query(mq):
    mq_esc = esc(mq)
    mq_esc = mq_esc.replace("Point to", "<span class='pt'>Point to</span>", 1)
    return f"<div class='molmo'>{mq_esc}</div>"


def render_query(q):
    cat    = q.get("category", "inter")
    task   = esc(q.get("task", ""))
    pattern = esc(q.get("pattern", ""))
    roles  = q.get("roles", [])
    molmos = q.get("molmo_queries", [])

    roles_html  = "".join(render_role(r) for r in roles)
    molmo_html  = "".join(render_molmo_query(m) for m in molmos)
    pattern_html = f"<div class='pattern'>pattern: <b>{pattern}</b></div>" if pattern else ""

    return (
        f"<div class='query'>"
        f"<div class='qhead'>"
        f"<span class='badge {esc(cat)}'>{esc(cat)}</span>"
        f"{task}"
        f"</div>"
        f"{pattern_html}"
        f"<div class='roles'>{roles_html}</div>"
        f"{molmo_html}"
        f"</div>"
    )


def render_card(obj):
    name    = esc(obj.get("object_name", "Unknown"))
    oid     = esc(obj.get("object_id", ""))
    views   = obj.get("views_used", [])
    queries = obj.get("queries", [])

    views_html   = render_views(views)
    queries_html = "".join(render_query(q) for q in queries)

    return (
        f"<div class='card'>"
        f"{views_html}"
        f"<div class='body'>"
        f"<div class='title'>{name}</div>"
        f"<div class='sub'><code>{oid}</code></div>"
        f"{queries_html}"
        f"</div>"
        f"</div>"
    )


def make_gallery(records, out_path, title="Stage 1 — Task proposals & 2-role decomposition"):
    n_obj = len(records)
    n_q   = sum(len(r.get("queries", [])) for r in records)

    cards_html = "".join(render_card(r) for r in records)

    html = f"""<!doctype html>
<html><head>
<meta charset='utf-8'>
<meta name='viewport' content='width=device-width,initial-scale=1'>
<title>{esc(title)}</title>
<style>{CSS}</style>
</head><body>
<header>
  <h1>{esc(title)}</h1>
  <div class='meta'>{n_obj} objects · {n_q} queries · inter = whole-object · intra = part-level · each role: verb → target @ contact_region</div>
</header>
<div class='grid'>
{cards_html}
</div>
<footer>generated by make_gallery.py</footer>
</body></html>"""

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[INFO] Saved gallery ({n_obj} objects, {n_q} queries) -> {out_path}")


# ──────────────────────────────────────────
# main
# ──────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in",  dest="inp", required=True,  help="stage1 JSON 輸出檔案")
    ap.add_argument("--out", dest="out", required=True,  help="輸出 HTML 路徑")
    ap.add_argument("--max", type=int,   default=None,   help="最多顯示幾筆（預設全部）")
    ap.add_argument("--title", default="Stage 1 — Task proposals & 2-role decomposition")
    args = ap.parse_args()

    with open(args.inp, encoding="utf-8") as f:
        records = json.load(f)

    records = [r for r in records if r.get("queries")]
    if args.max:
        records = records[:args.max]

    print(f"[INFO] {len(records)} objects loaded from {args.inp}")
    make_gallery(records, args.out, title=args.title)


if __name__ == "__main__":
    main()