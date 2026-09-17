#!/usr/bin/env python
"""Delete query outputs that an interrupted stage2 worker left half-written.

stage2_v2 writes scores.npz with np.savez_compressed and meta.json with a plain
dump; neither is atomic, so a worker killed mid-write can leave a truncated
archive.  --skip_existing only tests that both files EXIST, so such a query
would be skipped forever and silently ship corrupt.

Only recently-touched dirs can be affected, so the scan is bounded by --since
minutes unless --all is given.  A dir is removed whole (both files) so the next
run regenerates it; the object's other queries are untouched.

  python pipeline/repair_stage2_partial.py --root outputs/stage2 --since 120
"""
import argparse
import json
import os
import shutil
import time

import numpy as np

KEYS = ("xyz", "scoreA", "scoreB", "scoreA_raw", "scoreB_raw",
        "ptsA", "ptsB", "ptsA_multi", "ptsB_multi", "pexistA", "pexistB")


def bad(qdir):
    scores, meta = os.path.join(qdir, "scores.npz"), os.path.join(qdir, "meta.json")
    if not (os.path.isfile(scores) and os.path.isfile(meta)):
        return "incomplete pair"
    try:
        with open(meta) as stream:
            if "engine" not in json.load(stream):
                return "meta without engine"
    except Exception as exc:
        return f"meta unreadable ({type(exc).__name__})"
    try:
        with np.load(scores, allow_pickle=False) as data:
            for key in KEYS:
                _ = data[key].shape          # forces decompression of each member
    except Exception as exc:
        return f"npz unreadable ({type(exc).__name__})"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="outputs/stage2")
    ap.add_argument("--since", type=float, default=120, help="only scan dirs modified in the last N minutes")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    cutoff = 0 if args.all else time.time() - args.since * 60

    removed = scanned = 0
    for cat in sorted(os.listdir(args.root)):
        cat_dir = os.path.join(args.root, cat)
        if not os.path.isdir(cat_dir):
            continue
        for obj in sorted(os.listdir(cat_dir)):
            obj_dir = os.path.join(cat_dir, obj)
            if not os.path.isdir(obj_dir) or os.path.getmtime(obj_dir) < cutoff:
                continue
            for q in sorted(os.listdir(obj_dir)):
                qdir = os.path.join(obj_dir, q)
                if not os.path.isdir(qdir):
                    continue
                scanned += 1
                reason = bad(qdir)
                if reason:
                    print(f"REMOVE {cat}/{obj}/{q}: {reason}", flush=True)
                    removed += 1
                    if not args.dry_run:
                        shutil.rmtree(qdir)
    print(f"repair: scanned {scanned} query dirs, removed {removed}", flush=True)


if __name__ == "__main__":
    main()
