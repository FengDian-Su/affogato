#!/usr/bin/env python
"""stage2_full review v2 — full-fidelity clone of bimanual_grounding_en:
dark page, white cards with badges + FULL texts, static cloud PNG, lazy
interactive plotly (FINAL/raw), and per-role <details> 40-view strips of
Molmo point overlays (written to views/... like v1's relative tree).
SAM per-view masks are not stored by the v2 runner; strips note that.
Run: ~/miniconda3/envs/mm/bin/python stage2full_review_v2.py [start end]
"""
import os, sys, json, io, base64
import html as _html
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

DG = "/home/michaellee/mclee/affogato/data_generation"
ROOT = f"{DG}/outputs/stage2_full"
OUTD = f"{DG}/outputs/stage2_full_review"
REF = f"{DG}/outputs/bimanual_grounding_en/index.html"
PER_PAGE = 100
THR = 0.15
UP, HOR = 1, [0, 2]
A_COL, B_COL = "#F28E2B", "#0ABAB5"

LAZY_JS = """
<script>
const obs = new IntersectionObserver(entries => {
  for (const e of entries) {
    const holder = e.target, el = holder.querySelector('.plot3d');
    if (e.isIntersecting && !holder.dataset.live) {
      const spec = JSON.parse(holder.querySelector('.plotspec').textContent);
      if (holder._cam) spec.layout.scene.camera = holder._cam;
      if (holder._vis) spec.data.forEach((t, i) => { t.visible = holder._vis[i]; });
      Plotly.newPlot(el, spec.data, spec.layout, {displayModeBar: false});
      holder.dataset.live = "1";
    } else if (!e.isIntersecting && holder.dataset.live) {
      try {
        holder._cam = JSON.parse(JSON.stringify(el._fullLayout.scene.camera));
        holder._vis = el.data.map(t => t.visible === undefined ? true : t.visible);
      } catch (err) {}
      Plotly.purge(el); delete holder.dataset.live;
    }
  }
}, {rootMargin: '600px 0px'});
document.querySelectorAll('.plotdiv').forEach(d => obs.observe(d));
</script>"""

EXTRA_CSS = """<style>
.plotdiv{background:#fff;padding:2px 10px 8px}
.roledet{margin:0 10px 8px;background:#fff}
.roledet summary{cursor:pointer;font-size:12.5px;color:#374151;padding:5px 8px;
 border-left:3px solid #ddd;background:#f8f9fb;border-radius:4px}
.views{display:flex;gap:8px;overflow-x:auto;padding:6px 4px 10px}
.vcell{margin:0;text-align:center;flex:0 0 auto}
.vcell img{height:120px;border-radius:4px;display:block}
.vcell figcaption{font-size:10px;color:#9ca3af}
.title,.query,.molmo,.reg{white-space:normal!important;overflow:visible!important;
 text-overflow:clip!important}
.note{font-size:11px;color:#9ca3af;padding:0 8px 6px}
</style>"""


def main():
    import plotly.graph_objects as go
    from plotly.offline import get_plotlyjs
    os.makedirs(OUTD, exist_ok=True)
    ref = open(REF, encoding="utf-8").read()
    css = ref[ref.index("<style>"): ref.index("</style>") + 8]
    plotlyjs = "<script>" + get_plotlyjs() + "</script>"

    recs, roots = [], {}
    for part in ("stage1_part0.json", "stage1_part1.json"):
        for r in json.load(open(f"{DG}/outputs/stage1_v2/{part}")):
            if not r.get("error"):
                recs.append(r)
                roots[r["object_id"]] = "/".join(r["views_used"][0].split("/")[:-2])
    done = []
    for r in recs:
        od = os.path.join(ROOT, r["object_id"])
        if os.path.isdir(od):
            qds = [q for q in sorted(os.listdir(od))
                   if os.path.exists(os.path.join(od, q, "meta.json"))
                   and os.path.exists(os.path.join(od, q, "scores.npz"))]
            if qds:
                done.append((r, qds))
    BYNAME = os.environ.get("BYNAME") == "1"
    PREFIX = "byname_p" if BYNAME else "index_p"
    if BYNAME:
        done.sort(key=lambda t: (str(t[0].get("object_name", "")).lower(), t[0]["object_id"]))
    s = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    e = int(sys.argv[2]) if len(sys.argv) > 2 else min(len(done), 100)
    print(f"completed objects: {len(done)}; rendering [{s}:{e}]")
    esc = _html.escape
    n_pages_total = (len(done) + PER_PAGE - 1) // PER_PAGE

    if BYNAME and s == 0:
        groups = []
        for gi, (r, _) in enumerate(done):
            nm = str(r.get("object_name", "?"))
            if not groups or groups[-1][0] != nm:
                groups.append([nm, gi // PER_PAGE + 1, r["object_id"], 0])
            groups[-1][3] += 1
        rows = "".join(
            f"<tr class='r'><td><a style='color:#7c9cff' href='{PREFIX}{pg:03d}.html#{oid}'>"
            f"{esc(nm)}</a></td><td>{cnt}</td><td>p{pg:03d}</td></tr>"
            for nm, pg, oid, cnt in groups)
        toc = (f"<!DOCTYPE html>\n<html><head><meta charset=\"utf-8\"/><title>stage2_full by-name TOC</title>"
               f"<style>body{{background:#0f1216;color:#e6e8ec;font:14px/1.6 sans-serif;padding:24px 40px}}"
               f"table{{border-collapse:collapse}}td{{padding:2px 18px 2px 0;border-bottom:1px solid #232830}}"
               f"a{{text-decoration:none}}input{{background:#1a1f27;color:#e6e8ec;border:1px solid #333;"
               f"padding:6px 10px;border-radius:6px;width:320px;margin-bottom:14px}}</style></head><body>"
               f"<h1>Stage 2 FULL — 物件名稱目錄({len(groups)} 組 / {len(done)} 物件)</h1>"
               f"<input id='q' placeholder='輸入名稱過濾…' oninput=\"const v=this.value.toLowerCase();"
               f"document.querySelectorAll('tr.r').forEach(t=>t.style.display="
               f"t.textContent.toLowerCase().includes(v)?'':'none')\"/>"
               f"<table><tr><th style='text-align:left'>object_name</th><th>數量</th><th>頁</th></tr>{rows}</table>"
               f"</body></html>")
        open(f"{OUTD}/byname_index.html", "w", encoding="utf-8").write(toc)
        print(f"{OUTD}/byname_index.html ({len(groups)} name groups)")

    def nav(cur):
        first = max(0, cur - 6)
        links = [(f"<b>{j+1}</b>" if j == cur else
                  f"<a style='color:#7c9cff' href='{PREFIX}{j+1:03d}.html'>{j+1}</a>")
                 for j in range(first, min(n_pages_total, first + 13))]
        extra = " · <a style='color:#7c9cff' href='byname_index.html'>名稱目錄</a>" if BYNAME else ""
        return f"<div style='padding:6px 32px;color:#9aa3af;font-size:13px'>頁 {' '.join(links)} / {n_pages_total}{extra}</div>"

    def cloud_png(xyz, sA, sB):
        sub = np.random.RandomState(0).choice(len(xyz), min(7000, len(xyz)), replace=False)
        X = xyz[sub]; a, b = sA[sub] > THR, sB[sub] > THR
        fig = plt.figure(figsize=(5.2, 2.6))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(X[:, HOR[0]], X[:, HOR[1]], X[:, UP], s=0.7, c="lightgrey", alpha=0.45, linewidths=0)
        ax.scatter(X[a & ~b, HOR[0]], X[a & ~b, HOR[1]], X[a & ~b, UP], s=2.2, c=A_COL, linewidths=0)
        ax.scatter(X[b & ~a, HOR[0]], X[b & ~a, HOR[1]], X[b & ~a, UP], s=2.2, c=B_COL, linewidths=0)
        ax.view_init(elev=18, azim=-60)
        try: ax.set_box_aspect(np.ptp(X[:, [HOR[0], HOR[1], UP]], 0))
        except Exception: pass
        ax.set_axis_off(); fig.tight_layout(pad=0)
        buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=92); plt.close(fig)
        return base64.b64encode(buf.getvalue()).decode()

    def plot_spec(xyz, d):
        rng = np.random.RandomState(0)
        base = rng.choice(len(xyz), min(3000, len(xyz)), replace=False)
        # display in upright orientation: (x, z, y-up)
        def tr(idx): return xyz[idx][:, HOR[0]], xyz[idx][:, HOR[1]], xyz[idx][:, UP]
        fig = go.Figure()
        x0, y0, z0 = tr(base)
        fig.add_trace(go.Scatter3d(x=x0, y=y0, z=z0, mode="markers",
                                   marker=dict(size=1.2, color="#cccccc", opacity=0.3), hoverinfo="skip"))
        for vi2, (ta, tb) in enumerate([(d["scoreA"], d["scoreB"]), (d["scoreA_raw"], d["scoreB_raw"])]):
            a, b = ta > THR, tb > THR
            for mm, col in ((a & ~b, A_COL), (b & ~a, B_COL), (a & b, "#dc28a0")):
                idx = np.nonzero(mm)[0]
                if len(idx) > 3500:
                    idx = rng.choice(idx, 3500, replace=False)
                xx, yy, zz = tr(idx)
                fig.add_trace(go.Scatter3d(x=xx, y=yy, z=zz, mode="markers",
                                           marker=dict(size=1.8, color=col, opacity=0.92),
                                           visible=(vi2 == 0), hoverinfo="skip"))
        buttons = [dict(label=nm, method="update",
                        args=[{"visible": [True] + [i // 3 == k for i in range(6)]}])
                   for k, nm in enumerate(("FINAL", "raw"))]
        fig.update_layout(updatemenus=[dict(type="buttons", direction="right", x=0.5, xanchor="center",
                                            y=1.15, buttons=buttons, showactive=True)],
                          scene=dict(aspectmode="data", camera=dict(eye=dict(x=1.2, y=1.2, z=0.7)),
                                     xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False)),
                          width=500, height=360, margin=dict(l=0, r=0, t=30, b=0), showlegend=False)
        return fig.to_json().replace("</", "<\\/")

    def role_strip(oid, qd, ri, pts, renders, n_views, multi=None):
        col = (A_COL, B_COL)[ri]
        rdir = f"{OUTD}/views/{oid}/{qd}/role{'AB'[ri]}"
        os.makedirs(rdir, exist_ok=True)
        cells, hits = [], 0
        for vi in range(n_views):
            p = pts[vi]
            if not np.isfinite(p[0]):
                continue
            hits += 1
            fp = f"{rdir}/view{vi:02d}.jpg"
            if not os.path.exists(fp):
                im = renders[vi].copy()
                dr = ImageDraw.Draw(im)
                # 07-19 recipe: a view can carry several points (one per
                # instance); the mask is their union — draw ALL of them
                view_pts = ([(float(r_[1]), float(r_[2])) for r_ in multi if int(r_[0]) == vi]
                            if multi is not None and len(multi) else [(float(p[0]), float(p[1]))])
                for x, y in view_pts:
                    r = 10
                    for c2, w2 in (("white", 8), (col, 4)):
                        dr.line([x-r, y-r, x+r, y+r], fill=c2, width=w2)
                        dr.line([x-r, y+r, x+r, y-r], fill=c2, width=w2)
                im.thumbnail((256, 256))
                im.save(fp, "JPEG", quality=72)
            rel = f"views/{oid}/{qd}/role{'AB'[ri]}/view{vi:02d}.jpg"
            cells.append(f"<figure class='vcell'><a href='{rel}' target='_blank'>"
                         f"<img loading='lazy' src='{rel}' alt='v{vi}'></a>"
                         f"<figcaption>view {vi:02d}</figcaption></figure>")
        return (f"<details class='roledet'><summary style='border-left-color:{col}'>"
                f"Role {'AB'[ri]} views — {hits}/{n_views} hit(Molmo points;SAM 每視角 mask v2 未存檔)</summary>"
                f"<div class='views'>{''.join(cells)}</div></details>"), hits

    page_objs = []
    e = min(e, len(done))
    for gi in range(s, e):
        page_objs.append(done[gi])
        if len(page_objs) == PER_PAGE or gi == e - 1:
            page_no = (gi - len(page_objs) + 1) // PER_PAGE
            body, n_tasks = [], 0
            for rec, qlist in page_objs:
                oid = rec["object_id"]
                od = os.path.join(ROOT, oid)
                obj_root = roots[oid]
                renders, n_views = None, 0
                cards = []
                for qd in qlist:
                    try:
                        m = json.load(open(os.path.join(od, qd, "meta.json")))
                        d = np.load(os.path.join(od, qd, "scores.npz"), allow_pickle=True)
                    except Exception:
                        continue
                    if renders is None:
                        n_views = int(m.get("n_views", 40))
                        renders = []
                        for vi in range(n_views):
                            p = f"{obj_root}/{vi:05d}/{vi:05d}.png"
                            renders.append(Image.open(p).convert("RGB") if os.path.exists(p) else Image.new("RGB", (512, 512)))
                    xyz = d["xyz"].astype(np.float32)
                    spec = plot_spec(xyz, d)
                    qi = qd.split("_")[0]
                    roles_html = ""
                    for ri, col in ((0, A_COL), (1, B_COL)):
                        ro = m["roles"][ri]
                        mq = m.get("molmo_queries", ["", ""])[ri]
                        roles_html += (
                            f"<div class='role'><span class='rid' style='color:{col}'>{'AB'[ri]}</span>"
                            f"<span class='verb'>{esc(str(ro.get('role','')))}</span><span class='at'>&rarr;</span>"
                            f"<span class='reg' style='border-color:{col}66'>{esc(str(ro.get('contact_region','')))}</span>"
                            f"<span class='tgt'>target: {esc(str(ro.get('target','')))}</span></div>"
                            f"<div class='fn'>{esc(str(ro.get('function','')))}</div>"
                            f"<div class='molmo'><span class='pt'>Point&nbsp;&rarr;</span> {esc(mq)}</div>")
                    stripA, _ = role_strip(oid, qd, 0, d["ptsA"][:n_views], renders, n_views)
                    stripB, _ = role_strip(oid, qd, 1, d["ptsB"][:n_views], renders, n_views)
                    cards.append(
                        f"<div class=\"card\"><div class=\"body\">"
                        f"<div class=\"qhead\"><span class='badge inter'>{esc(str(m.get('coordination','')))}</span>"
                        f"<span class=\"qidx\">{esc(qi)}</span></div>"
                        f"<div style='font-size:15.5px;font-weight:750;color:#111;margin:4px 0 2px'>{esc(str(m.get('query', m['task'])))}</div>"
                        f"<div class=\"query\">task: {esc(m['task'])}</div>"
                        f"<div class=\"roles\">{roles_html}</div></div>"
                        f"<div class='plotdiv'><div class='plot3d' style='width:500px;height:360px;margin:auto'></div>"
                        f"<script type='application/json' class='plotspec'>{spec}</script></div>"
                        f"{stripA}{stripB}</div>")
                    n_tasks += 1
                body.append(
                    f"<div class='obj' id='{oid}'><div class='obj-head'><span class='name'>{esc(rec['object_name'])}</span>"
                    f"<span class='oid'>{oid}</span></div>"
                    f"<div class='grid'>{''.join(cards)}</div></div>")
            html = (f"<!DOCTYPE html>\n<html><head><meta charset=\"utf-8\"/>"
                    f"<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\"/>"
                    f"<title>stage2_full review — p{page_no+1}</title>\n{css}{EXTRA_CSS}{plotlyjs}</head><body>"
                    f"<header><h1>Stage 2 FULL — grounding review(頁 {page_no+1}/{n_pages_total})</h1>"
                    f"<div class=\"meta\">{len(page_objs)} objects · {n_tasks} tasks · "
                    f"mixed+neg → refine → partition → post-prune;卡片含互動 plotly(FINAL/raw)與 Role 40-view 展開</div>"
                    f"<div class=\"legend\"><span><i style=\"background:{A_COL}\"></i>Role A</span>"
                    f"<span><i style=\"background:{B_COL}\"></i>Role B</span>"
                    f"<span><i style=\"background:lightgrey\"></i>object cloud</span></div></header>"
                    + nav(page_no) + "".join(body) + nav(page_no) + LAZY_JS + "</body></html>")
            out = f"{OUTD}/{PREFIX}{page_no+1:03d}.html"
            open(out, "w", encoding="utf-8").write(html)
            print(f"{out} ({os.path.getsize(out)/1e6:.1f} MB, {len(page_objs)} objs, {n_tasks} tasks)", flush=True)
            page_objs = []


if __name__ == "__main__":
    main()
