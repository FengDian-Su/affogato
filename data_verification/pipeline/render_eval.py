#!/usr/bin/env python
"""Render one review PNG per sampled stage2 result: an RGB reference view (object recognition)
+ role-A and role-B per-vertex heatmaps, each from two azimuths (so occluded faces are visible).
The image is what a Claude labeler sees to score heatmap quality; the meta text carries task/roles."""
import os, json, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import matplotlib
from matplotlib.colors import LinearSegmentedColormap
HERE = os.path.dirname(os.path.abspath(__file__))
# role colormaps (Candidate A hues, DARK = HIGH score): low score = light, high score = deep
# saturated, so the hottest regions read as the strongest colour. A = orange, B = tiffany.
CMAP_A = LinearSegmentedColormap.from_list("A", ["#FCE3C0", "#F5A623", "#E8730A", "#7a3300"])
CMAP_B = LinearSegmentedColormap.from_list("B", ["#CFF3EF", "#4FD0C8", "#0A9C99", "#044f49"])
OBJECT_GREY = "#DBDBDB"          # light-grey object on white so the dark high-score points pop
THRESHOLD = 0.15
# 8 views MATCHING the object-scan coverage pattern: 6 orbit azimuths + top-down + underside, so a
# heatmap error on the top or bottom face can't hide. (exact per-view camera-pose alignment with the
# RGB scan isn't possible - the point cloud is the affogato frame, the RGB is the gObjaverse rig.)
VIEWS = [(24, 0, "az 0"), (24, 60, "az 60"), (24, 120, "az 120"), (24, 180, "az 180"),
         (24, 240, "az 240"), (24, 300, "az 300"), (88, 0, "top-down"), (-88, 0, "underside")]


def _norm(s, active):
    """amount normalized to this role's own active range [THRESHOLD, p98] so the full colormap is
    used -> visible gradient even when raw scores are narrow and high."""
    v = s[active].astype(np.float64)
    hi = max(float(np.percentile(v, 98)), THRESHOLD + 1e-3)
    return np.clip((v - THRESHOLD) / (hi - THRESHOLD), 0.0, 1.0)


def _heat_ax(fig, gs, idx, xyz, sA, sB, elev, az, label, span):
    """One combined A+B panel at (elev, az)."""
    ax = fig.add_subplot(gs[idx], projection="3d")
    active_a = (sA >= THRESHOLD) & (sA >= sB)          # overlap -> larger score
    active_b = (sB >= THRESHOLD) & (sB > sA)
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], s=1.0, c=OBJECT_GREY, alpha=.55,
               linewidths=0, depthshade=False)
    for active, cmap, sc in ((active_a, CMAP_A, sA), (active_b, CMAP_B, sB)):
        if active.any():
            p = xyz[active]; o = np.argsort(sc[active], kind="stable")
            ax.scatter(p[o, 0], p[o, 1], p[o, 2], s=3.2, c=cmap(_norm(sc, active)[o]),
                       alpha=.95, linewidths=0, depthshade=False)
    try:
        ax.set_box_aspect(span, zoom=1.15)
    except TypeError:
        ax.set_box_aspect(span)
    ax.set_axis_off(); ax.view_init(elev=elev, azim=az)
    ax.set_title(label, fontsize=7.5, color="#6B7280", pad=0)


def render(rec, out_path):
    base = rec["path"]
    m = json.load(open(f"{base}/meta.json"))
    d = np.load(f"{base}/scores.npz", allow_pickle=True)
    # teaser axis order: display (x, z, y) so +y is up (single_region_affordance UP_AXIS=1)
    xyz = d["xyz"].astype(np.float32)[:, [0, 2, 1]]
    sA, sB = d["scoreA"].astype(np.float32), d["scoreB"].astype(np.float32)
    rA, rB = m["roles"][0], m["roles"][1]
    span = np.maximum(xyz.max(0) - xyz.min(0), 1e-6)

    # 8-view turntable (2x4) so no face is hidden when judging heatmap localization
    fig = plt.figure(figsize=(14, 7), facecolor="white")
    gs = gridspec.GridSpec(2, 4, wspace=.0, hspace=.08)
    for i, (elev, az, lab) in enumerate(VIEWS):
        _heat_ax(fig, gs, (i // 4, i % 4), xyz, sA, sB, elev, az, lab, span)
    fig.suptitle(f'{m["object_name"]} — {m["task"]}   [{m.get("coordination")}]   cov={m.get("coverage")}\n'
                 f'A (orange): {rA["role"]} @ {rA["contact_region"]}    '
                 f'B (tiffany): {rB["role"]} @ {rB["contact_region"]}', fontsize=10)
    fig.savefig(out_path, dpi=82, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def rgb_strip(rec, out_path):
    """8-view object scan (all sampled views: elevation-spanning orbit + top-down + underside) laid
    out 2x4, so the task/role axis is judged from a full object scan, not the heatmap."""
    from PIL import ImageDraw
    views = (rec.get("views") or [])[:8]
    if len(views) < 5:
        return
    # label the last two (top-down / underside per the gObjaverse rig); the rest are orbit azimuths
    labs = [f"view {i}" for i in range(len(views))]
    if len(views) >= 2:
        labs[-2], labs[-1] = "top-down", "underside"
    cell, pad = 240, 18
    cols, rows = 4, (len(views) + 3) // 4
    strip = Image.new("RGB", (cell * cols, (cell + pad) * rows), (255, 255, 255))
    dr = ImageDraw.Draw(strip)
    for i, p in enumerate(views):
        try:
            im = Image.open(p).convert("RGB").resize((cell, cell))
        except Exception:
            im = Image.new("RGB", (cell, cell), (30, 30, 30))
        r, c = divmod(i, cols)
        x, y = c * cell, r * (cell + pad)
        strip.paste(im, (x, y + pad)); dr.text((x + 4, y + 3), labs[i], fill=(0, 0, 0))
    strip.save(out_path)


FORCE = os.environ.get("FORCE") == "1"


def _one(args):
    rec, out = args
    try:
        if FORCE or not os.path.exists(out):
            render(rec, out)
        rgb = out.replace("/renders/", "/rgb/")
        os.makedirs(os.path.dirname(rgb), exist_ok=True)
        if FORCE or not os.path.exists(rgb):
            rgb_strip(rec, rgb)
        return True
    except Exception as e:
        return f"{rec['object_id']}/{rec['query_dir']}: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", default=os.path.join(HERE, "daily_used", "sample1000.json"))
    ap.add_argument("--outdir", default=os.path.join(HERE, "daily_used", "renders"))
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    sample = json.load(open(args.sample))
    if args.limit:
        sample = sample[:args.limit]
    jobs = [(r, os.path.join(args.outdir, f"{i:04d}_{r['object_id'][:8]}_{r['query_dir'][:24]}.png"))
            for i, r in enumerate(sample)]
    # record the render path back onto the manifest for the labeler
    for (r, out), i in zip(jobs, range(len(jobs))):
        r["render"] = out; r["idx"] = i
    (None if args.limit else json.dump(sample, open(args.sample.replace(".json", "_manifest.json"), "w")))
    from multiprocessing import Pool
    errs = []
    with Pool(args.workers) as p:
        for k, res in enumerate(p.imap_unordered(_one, jobs, chunksize=4)):
            if res is not True:
                errs.append(res)
            if (k + 1) % 100 == 0:
                print(f"  {k+1}/{len(jobs)} rendered", flush=True)
    print(f"done: {len(jobs)-len(errs)}/{len(jobs)} ok, {len(errs)} errors")
    for e in errs[:10]:
        print("  ERR", e)


if __name__ == "__main__":
    main()
