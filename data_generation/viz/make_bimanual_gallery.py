#!/usr/bin/env python
"""
make_bimanual_gallery.py
========================

Build ONE self-contained HTML review page for the **stage-2 bimanual grounding**
results produced by ``pipeline/stage2_bimanual_grounding.py`` (Molmo -> SAM2 -> multi-view voting
-> two-role partition), so all 100 objects / 282 tasks can be eyeballed at once
for evaluation instead of opening each ``cloud_two_role.html`` by hand.

For every (object, task) under ``outputs/bimanual_grounding/`` we show, per query:

  1. the query metadata (object, task, query text, category, role A / role B with
     their target + contact_region + function, and the actual Molmo "Point to..."
     prompt) -- laid out like the stage-1 gallery (``outputs/gallery_0625.html``).
  2. Role A / Role B **Molmo points** + **SAM2 masks** per view, mirroring the
     notebook's ``show_points_on_views`` / ``show_heatmaps_on_views`` -- one cell
     per *hit* view (a view where Molmo actually placed a point), with the Molmo
     overlay on top and the SAM2 heatmap overlay directly below it so a point and
     the mask it grew into stay side-by-side.
  3. a **static 3D point-cloud** preview of the two grounded regions after the
     partition (Role A = orange, Role B = tiffany, object = grey), rendered at two
     fixed viewpoints from ``scores.npz`` + a link to the full interactive
     ``cloud_two_role.html`` (kept opt-in: 282 live WebGL plots on one page hang
     the browser, per earlier experience).

Design choices for a page this large (44,888 view PNGs = 6.6 GB on disk):
  * View PNGs are **referenced by relative path** (``../bimanual_grounding/...``),
    not base64-embedded -- the browser lazy-loads them. The gallery therefore must
    stay a sibling of ``outputs/bimanual_grounding/`` to resolve the images.
  * Only the small point-cloud previews (16 k points each) are base64-embedded.
  * Each role's view strip lives inside a collapsed ``<details>`` so nothing loads
    until you open it.

Usage (repo ``mm`` env)::

    python viz/make_bimanual_gallery.py
    python viz/make_bimanual_gallery.py --in outputs/bimanual_grounding \
        --out outputs/bimanual_grounding_en/index.html --workers 12
"""

import os
import re
import io
import sys
import json
import base64
import argparse
from glob import glob
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")                       # headless render in worker processes
import matplotlib.pyplot as plt

# colours ported from bimanual_grounding.two_role_fig
ORANGE, TIFFANY = "#F28E2B", "#0ABAB5"       # role A / role B solid hues
THR = 0.15                                   # score threshold for "grounded" points (matches notebook)


# ==============================================================================
# per-task heavy lifting (runs in worker processes)
# ==============================================================================

def detect_hit_views(molmo_dir):
    """Sorted view indices whose Molmo overlay has a red 'X' drawn on it.

    ``bimanual_grounding.save_view_overlays`` writes the ORIGINAL image (no marker)
    for views where Molmo declined to point, and the same image + a pure-red X
    where it did. So "a view with a point" == "the PNG contains pure-red pixels".
    We gate on a min pixel count (skip stray red) and a max fraction (a genuinely
    red-coloured object floods the frame with red -- that's not an X). The notebook
    shows exactly these hit views for both the Molmo and the SAM2 grids.
    """
    hits = []
    for f in sorted(glob(os.path.join(molmo_dir, "view*.png"))):
        img = cv2.imread(f)                  # BGR
        if img is None:
            continue
        b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        red = (r > 180) & (g < 70) & (b < 70)
        n = int(red.sum())
        if 15 < n and red.mean() < 0.15:
            m = re.search(r"view(\d+)\.png$", f)
            if m:
                hits.append(int(m.group(1)))
    return sorted(hits)


def render_cloud_b64(npz_path, thr=THR):
    """Two-viewpoint static 3D scatter of the partitioned regions -> base64 PNG.

    A = orange, B = tiffany over a faint grey object cloud, matching the interactive
    ``two_role_fig``. Points are stride-subsampled to keep each PNG ~100 KB.
    """
    z = np.load(npz_path, allow_pickle=True)
    xyz = z["xyz"].astype(np.float32)
    sA, sB = z["scoreA"].astype(np.float32), z["scoreB"].astype(np.float32)

    def sub(pts, cap):
        return pts[:: max(1, len(pts) // cap)] if len(pts) else pts

    obj = sub(xyz, 6000)
    mA, mB = sA > thr, sB > thr
    ptsA, ptsB = sub(xyz[mA], 4000), sub(xyz[mB], 4000)
    span = tuple(max(1e-6, float(np.ptp(xyz[:, i]))) for i in range(3))

    fig = plt.figure(figsize=(6.6, 3.4), dpi=72)
    for i, az in enumerate((-60, 30)):
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")
        ax.scatter(obj[:, 0], obj[:, 1], obj[:, 2], s=1.2, c="lightgrey",
                   alpha=0.12, linewidths=0)
        if len(ptsA):
            ax.scatter(ptsA[:, 0], ptsA[:, 1], ptsA[:, 2], s=5, c=ORANGE,
                       alpha=0.85, linewidths=0)
        if len(ptsB):
            ax.scatter(ptsB[:, 0], ptsB[:, 1], ptsB[:, 2], s=5, c=TIFFANY,
                       alpha=0.85, linewidths=0)
        ax.set_box_aspect(span)
        ax.view_init(elev=18, azim=az)
        ax.axis("off")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def process_task(task_dir):
    """Worker entry: (meta, cloud b64, roleA/B hit-view lists) for one task dir."""
    task_dir = str(task_dir)
    try:
        meta = json.load(open(os.path.join(task_dir, "meta.json")))
        cloud = render_cloud_b64(os.path.join(task_dir, "scores.npz"))
        hitsA = detect_hit_views(os.path.join(task_dir, "roleA", "molmo"))
        hitsB = detect_hit_views(os.path.join(task_dir, "roleB", "molmo"))
        return {"dir": task_dir, "meta": meta, "cloud": cloud,
                "hitsA": hitsA, "hitsB": hitsB, "n_views": count_views(task_dir)}
    except Exception as e:                   # keep one bad task from killing the build
        return {"dir": task_dir, "error": f"{type(e).__name__}: {e}"}


def count_views(task_dir):
    return len(glob(os.path.join(task_dir, "roleA", "molmo", "view*.png")))


# ==============================================================================
# discovery
# ==============================================================================

def find_tasks(root):
    """[(object_id, [task_dir, ...])] with tasks sorted by q-index, objects by id."""
    out = []
    for obj_dir in sorted(p for p in Path(root).iterdir() if p.is_dir()):
        tasks = [d for d in obj_dir.iterdir()
                 if d.is_dir() and (d / "scores.npz").exists() and (d / "meta.json").exists()]
        tasks.sort(key=lambda d: int(m.group(1)) if (m := re.match(r"q(\d+)", d.name)) else 999)
        if tasks:
            out.append((obj_dir.name, tasks))
    return out


# ==============================================================================
# HTML rendering (main process)
# ==============================================================================

def esc(s):
    return (str(s).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


def rel(img_path, out_dir):
    """Relative href from the gallery file's dir to an image on disk."""
    return esc(os.path.relpath(img_path, out_dir))


def role_block(r, mq, color):
    """One role: 'A  lift → contact_region · target · function' + its Molmo prompt."""
    rid = esc(r.get("id", ""))
    verb = esc(r.get("role", ""))
    region = esc(r.get("contact_region", ""))
    target = esc(r.get("target", ""))
    fn = esc(r.get("function", ""))
    return (
        f"<div class='role'>"
        f"<span class='rid' style='color:{color}'>{rid}</span>"
        f"<span class='verb'>{verb}</span>"
        f"<span class='at'>&rarr;</span>"
        f"<span class='reg' style='border-color:{color}66'>{region}</span>"
        + (f"<span class='tgt'>target: {target}</span>" if target else "")
        + "</div>"
        + (f"<div class='fn'>{fn}</div>" if fn else "")
        + (f"<div class='molmo'><span class='pt'>Point&nbsp;&rarr;</span> {esc(mq)}</div>" if mq else "")
    )


def view_strip(task_dir, out_dir, role, hits, color, label):
    """Horizontal scroll strip: one cell per hit view = Molmo overlay (top) + SAM2 (bottom)."""
    total = count_views(task_dir)
    if not hits:
        return (f"<div class='strip-head' style='border-color:{color}'>{esc(label)} "
                f"<span class='hitn'>0/{total} views — no Molmo point</span></div>")
    cells = []
    for vi in hits:
        mp = rel(os.path.join(task_dir, role, "molmo", f"view{vi:02d}.png"), out_dir)
        sp = rel(os.path.join(task_dir, role, "sam2", f"view{vi:02d}.png"), out_dir)
        cells.append(
            f"<figure class='vcell'>"
            f"<a href='{mp}' target='_blank'><img loading='lazy' src='{mp}' alt='molmo v{vi}'></a>"
            f"<a href='{sp}' target='_blank'><img loading='lazy' src='{sp}' alt='sam2 v{vi}'></a>"
            f"<figcaption>view {vi:02d}</figcaption></figure>")
    return (f"<div class='strip-head' style='border-color:{color}'>{esc(label)} "
            f"<span class='hitn'>{len(hits)}/{total} views hit · Molmo point (top) → SAM2 mask (bottom)</span></div>"
            f"<div class='views'>{''.join(cells)}</div>")


def render_card(rec, out_dir):
    meta = rec["meta"]
    roles = meta.get("roles", [{}, {}])
    mqs = meta.get("molmo_queries", ["", ""])
    qi = meta.get("query_idx", "")
    cat = esc(meta.get("category", ""))
    task_dir = rec["dir"]
    stats = meta.get("score_stats", {})

    def statline(key, color, name):
        s = stats.get(key, {})
        return (f"<span style='color:{color}'>{name} max {s.get('max', 0):.2f} · "
                f"mean {s.get('mean', 0):.2f}</span>") if s else ""

    badge = f"<span class='badge {cat}'>{cat}</span>" if cat else ""
    cov = meta.get("coverage", None)
    cov_s = f" · coverage {cov*100:.0f}%" if isinstance(cov, (int, float)) else ""
    interactive = rel(os.path.join(task_dir, "cloud_two_role.html"), out_dir)

    detA = (f"<details class='roledet'><summary style='border-left-color:{ORANGE}'>"
            f"Role A views — {len(rec['hitsA'])}/{rec['n_views']} hit</summary>"
            f"{view_strip(task_dir, out_dir, 'roleA', rec['hitsA'], ORANGE, 'Role A')}</details>")
    detB = (f"<details class='roledet'><summary style='border-left-color:{TIFFANY}'>"
            f"Role B views — {len(rec['hitsB'])}/{rec['n_views']} hit</summary>"
            f"{view_strip(task_dir, out_dir, 'roleB', rec['hitsB'], TIFFANY, 'Role B')}</details>")

    return f"""
      <div class="card">
        <div class="body">
          <div class="qhead">{badge}<span class="qidx">q{esc(qi)}</span>
            <span class="title">{esc(meta.get('task',''))}</span></div>
          <div class="query">{esc(meta.get('query',''))}</div>
          <div class="roles">
            {role_block(roles[0], mqs[0] if len(mqs) > 0 else '', ORANGE) if len(roles) > 0 else ''}
            {role_block(roles[1], mqs[1] if len(mqs) > 1 else '', TIFFANY) if len(roles) > 1 else ''}
          </div>
        </div>
        <div class="cloud">
          <img src="data:image/png;base64,{rec['cloud']}" alt="two-role point cloud"/>
          <div class="cloud-foot">
            <span>{statline('roleA', ORANGE, 'A')} &nbsp; {statline('roleB', TIFFANY, 'B')}{cov_s}</span>
            <a href="{interactive}" target="_blank">open interactive 3D &nearr;</a>
          </div>
        </div>
        {detA}
        {detB}
      </div>"""


CSS = """
:root{--bg:#0f1216;--card:#fff;--ink:#1a1d23;--muted:#6b7280;--line:#e6e8ec;
--inter:#2563eb;--intra:#059669;--chip:#f3f4f6;--accent:#7c3aed}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:#e6e8ec;
font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Arial,sans-serif}
header{padding:26px 32px 14px}
header h1{margin:0 0 4px;font-size:22px;font-weight:700}
header .meta{color:#9aa3af;font-size:13px}
.legend{display:flex;gap:18px;margin-top:10px;font-size:13px;color:#cbd5e1;flex-wrap:wrap}
.legend i{display:inline-block;width:11px;height:11px;border-radius:3px;margin-right:6px;vertical-align:middle}
.obj{padding:8px 32px 0}
.obj-head{display:flex;align-items:baseline;gap:10px;margin:22px 0 10px;
border-top:1px solid #232830;padding-top:18px}
.obj-head .name{font-size:18px;font-weight:700;color:#fff}
.obj-head .oid{font:12px ui-monospace,Menlo,Consolas,monospace;color:#6b7280}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(540px,1fr));gap:18px;padding-bottom:8px}
.card{background:var(--card);color:var(--ink);border-radius:14px;overflow:hidden;
box-shadow:0 1px 3px rgba(0,0,0,.25),0 12px 28px rgba(0,0,0,.18)}
.body{padding:15px 17px 6px}
.qhead{display:flex;align-items:center;gap:8px;flex-wrap:wrap}
.qhead .qidx{color:#9ca3af;font:12px ui-monospace,Menlo,Consolas,monospace}
.title{font-size:17px;font-weight:700}
.query{color:var(--muted);font-size:12.5px;margin:4px 0 10px}
.roles{display:flex;flex-direction:column;gap:5px}
.role{display:flex;gap:7px;align-items:baseline;font-size:13px;flex-wrap:wrap}
.role .rid{font-weight:800;width:14px;flex:none}
.role .verb{font-weight:700}
.role .at{color:var(--muted)}
.role .reg{color:#111;background:#f6f8ff;border:1px solid;border-radius:6px;padding:0 6px}
.role .tgt{color:var(--muted);font-size:11.5px}
.fn{color:var(--muted);font-style:italic;font-size:12px;margin:1px 0 2px 21px}
.molmo{font:11.5px/1.45 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
background:#0f172a;color:#cbd5e1;border-radius:7px;padding:6px 9px;margin:3px 0 6px;word-break:break-word}
.molmo .pt{color:#38bdf8}
.cloud{padding:4px 10px 6px;background:#fbfbfc;border-top:1px solid var(--line)}
.cloud img{width:100%;height:auto;display:block;border-radius:8px}
.cloud-foot{display:flex;justify-content:space-between;align-items:center;gap:10px;
font-size:11.5px;color:var(--muted);padding:5px 2px 2px;flex-wrap:wrap}
.cloud-foot a{color:var(--inter);text-decoration:none;font-weight:600;white-space:nowrap}
.roledet{border-top:1px solid var(--line)}
.roledet>summary{cursor:pointer;padding:9px 16px;font-size:13px;font-weight:700;color:#374151;
border-left:4px solid;list-style:none}
.roledet>summary::-webkit-details-marker{display:none}
.roledet>summary:before{content:"▸ ";color:#9ca3af}
.roledet[open]>summary:before{content:"▾ "}
.strip-head{font-size:11.5px;color:#374151;font-weight:600;padding:2px 16px 4px;
border-left:4px solid;margin-left:0}
.strip-head .hitn{color:var(--muted);font-weight:500}
.views{display:flex;gap:8px;overflow-x:auto;padding:6px 14px 12px}
.views figure{margin:0;flex:none;text-align:center}
.views img{height:132px;width:auto;display:block;border-radius:6px;border:1px solid var(--line);
background:#fafafa;margin-bottom:3px}
.views figcaption{font:10px/1.2 ui-monospace,Menlo,Consolas,monospace;color:var(--muted)}
.badge{font-size:10px;font-weight:700;letter-spacing:.4px;text-transform:uppercase;
padding:2px 7px;border-radius:999px;color:#fff}
.badge.inter{background:var(--inter)}.badge.intra{background:var(--intra)}
footer{color:#6b7280;text-align:center;padding:26px;font-size:12px}
"""


def build(root, out_path, workers):
    out_dir = os.path.dirname(os.path.abspath(out_path))
    objects = find_tasks(root)
    if not objects:
        print(f"[error] no (object, task) results found under {root}")
        return
    task_dirs = [(oid, str(t)) for oid, ts in objects for t in ts]
    n_obj, n_task = len(objects), len(task_dirs)
    print(f"[scan] {n_obj} objects · {n_task} tasks — rendering clouds + detecting hit views "
          f"({workers} workers)…")

    # heavy per-task work (cloud render + red-X hit detection) in parallel
    results = {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(process_task, td): td for _, td in task_dirs}
        done = 0
        for fut in as_completed(futs):
            r = fut.result()
            results[r["dir"]] = r
            done += 1
            if r.get("error"):
                print(f"  [warn] {r['dir']}: {r['error']}")
            if done % 25 == 0 or done == n_task:
                print(f"  {done}/{n_task} tasks processed")

    # assemble HTML grouped by object, preserving discovery order
    sections, n_ok = [], 0
    for oid, ts in objects:
        cards = []
        name = oid
        for t in ts:
            r = results.get(str(t))
            if not r or r.get("error"):
                continue
            name = r["meta"].get("object_name", oid) or oid
            cards.append(render_card(r, out_dir))
            n_ok += 1
        if not cards:
            continue
        sections.append(
            f"<div class='obj'><div class='obj-head'>"
            f"<span class='name'>{esc(name)}</span>"
            f"<span class='oid'>{esc(oid)}</span>"
            f"<span class='oid'>· {len(cards)} tasks</span></div>"
            f"<div class='grid'>{''.join(cards)}</div></div>")

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Bimanual grounding review — stage 2</title>
<style>{CSS}</style>
</head><body>
<header>
  <h1>Stage 2 — Bimanual Molmo→SAM2 grounding review</h1>
  <div class="meta">{n_obj} objects · {n_ok} bimanual tasks · Molmo point → SAM2 mask → multi-view vote → two-role partition</div>
  <div class="legend">
    <span><i style="background:{ORANGE}"></i>Role A region</span>
    <span><i style="background:{TIFFANY}"></i>Role B region</span>
    <span><i style="background:lightgrey"></i>object cloud</span>
    <span>· expand a role to inspect its per-view Molmo points &amp; SAM2 masks</span>
  </div>
</header>
{''.join(sections)}
<footer>view PNGs referenced from ../{esc(os.path.basename(os.path.normpath(root)))}/ · point clouds rendered from scores.npz · make_bimanual_gallery.py</footer>
</body></html>"""

    os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w") as f:
        f.write(html)
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"\n[saved] {out_path}  ({size_mb:.1f} MB)  {n_obj} objects / {n_ok} tasks")


def main():
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    DG = os.path.dirname(SCRIPT_DIR)
    os.chdir(DG)
    p = argparse.ArgumentParser(description="Stage-2 bimanual grounding review gallery")
    p.add_argument("--in", dest="root", default="outputs/bimanual_grounding")
    p.add_argument("--out", default="outputs/bimanual_grounding_en/index.html")
    p.add_argument("--workers", type=int, default=min(12, (os.cpu_count() or 4)))
    p.add_argument("--limit", type=int, default=0, help="only first N objects (debug)")
    args = p.parse_args()
    if args.limit:
        # debug: shrink the scan by temporarily limiting object dirs
        objs = sorted(p for p in Path(args.root).iterdir() if p.is_dir())[: args.limit]
        globals()["_LIMIT_SET"] = {p.name for p in objs}
        orig = find_tasks
        globals()["find_tasks"] = lambda r, _o=orig: [(o, t) for o, t in _o(r) if o in globals()["_LIMIT_SET"]]
    build(args.root, args.out, args.workers)


if __name__ == "__main__":
    main()
