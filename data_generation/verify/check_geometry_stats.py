#!/usr/bin/env python
"""
Deterministic geometry-stats detector — full-coverage, CPU-only complement to the model judge.

Targets the junk classes a VLM judge reads past (calibrated 2026-07-20 on 200 Claude-labeled
objects): renders are RGBA, so the alpha channel gives an exact silhouette per view. Per object it
inspects the TOP-DOWN and UNDERSIDE views (the two a name-anchored judge under-uses) plus ring
views and flags:

  standing_sheet : hairline silhouette in BOTH top-down and underside -> the whole object is one
                   vertical sheet / extruded flat sprite (fake "vending machine" panels, pixel-art
                   slabs). Thin PARTS (knife blades) don't trigger: a knife lying on its side shows
                   a full profile from above.
  ground_slab    : underside silhouette fills nearly the whole frame -> object fused into a
                   ground-plane slab (scan artifact "bowl on a floor square").
  empty_render   : almost no foreground in any sampled view -> broken/empty render.

Usage:
  python verify/check_geometry_stats.py                 # full kept set, writes geomflags.json
  python verify/check_geometry_stats.py --calib <file>  # any stage0-format record list
"""
import os
import json
import argparse
from multiprocessing import Pool

import numpy as np
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DEFAULT = os.path.join(os.path.dirname(HERE), "outputs", "stage0", "daily_used")

THIN = 0.04        # hairline threshold: min silhouette bbox dim / image size
SLAB = 0.85        # underside fg-area ratio above which it's a ground slab
EMPTY = 0.01


def view_stats(path):
    """(fg_ratio, min_bbox_dim_ratio) for one render; None if unreadable."""
    try:
        im = Image.open(path)
        a = np.array(im.split()[-1]) if "A" in im.mode else (255 - np.array(im.convert("L")))
    except Exception:
        return None
    fg = a > 10
    n = fg.sum()
    h, w = fg.shape
    if n == 0:
        return (0.0, 0.0)
    ys, xs = np.nonzero(fg)
    bh, bw = (ys.max() - ys.min() + 1) / h, (xs.max() - xs.min() + 1) / w
    return (n / (h * w), min(bh, bw))


def analyze(rec):
    views = rec.get("views_used") or []
    if len(views) < 5:
        return None
    ring = views[:-2][:3]                      # a few ring azimuths are enough
    top, under = views[-2], views[-1]
    st, su = view_stats(top), view_stats(under)
    srs = [s for s in (view_stats(v) for v in ring) if s]
    if not st or not su or not srs:
        return {"object_id": rec["object_id"], "object_name": rec.get("object_name"),
                "flags": ["views_unreadable"]}
    flags = []
    max_fg = max([st[0], su[0]] + [s[0] for s in srs])
    if max_fg < EMPTY:
        flags.append("empty_render")
    else:
        if st[1] < THIN and su[1] < THIN:
            flags.append("standing_sheet")
        if su[0] > SLAB:
            flags.append("ground_slab")
    if not flags:
        return None
    return {"object_id": rec["object_id"], "object_name": rec.get("object_name"), "flags": flags,
            "stats": {"top": [round(x, 3) for x in st], "under": [round(x, 3) for x in su],
                      "ring_max_fg": round(max(s[0] for s in srs), 3)}}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=OUT_DEFAULT)
    ap.add_argument("--calib", default=None, help="single stage0-format json instead of parts")
    ap.add_argument("--out", default=None)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    if args.calib:
        recs = [r for r in json.load(open(args.calib)) if r.get("keep")]
        out = args.out or os.path.splitext(args.calib)[0] + ".geomflags.json"
    else:
        import glob as _glob
        recs = []
        for f in sorted(_glob.glob(f"{args.dir}/stage0_part*.json")):
            if f.endswith(".kept.json"):
                continue
            recs += [r for r in json.load(open(f)) if r.get("keep")]
        out = args.out or os.path.join(args.dir, "geomflags.json")
    print(f"analyzing {len(recs)} kept objects with {args.workers} workers")

    with Pool(args.workers) as pool:
        results = [r for r in pool.imap_unordered(analyze, recs, chunksize=64) if r]
    from collections import Counter
    c = Counter(f for r in results for f in r["flags"])
    with open(out, "w") as f:
        json.dump(results, f, indent=1, ensure_ascii=False)
    print(f"flagged {len(results)}/{len(recs)}: {dict(c)} -> {out}")


if __name__ == "__main__":
    main()
