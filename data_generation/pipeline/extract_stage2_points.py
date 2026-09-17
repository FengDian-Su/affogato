#!/usr/bin/env python
"""Freeze the Molmo points of an existing stage2 release into a compact archive.

The replay runner regenerates heatmaps from the SAME points the original run
used, so those points must survive the release they were stored in.  A full
scores.npz is 315 KB (xyz + four 16384-point score arrays); the points and the
existence scores are 2 KB of it, so the whole 310k-query release compresses to
~0.6 GiB and can be kept on local disk indefinitely.

One .npz per object, mirroring outputs/stage2/<category>/<object_id>/, with the
query directory name as the key prefix.  meta.json travels with it so the
archive alone re-identifies every query (task, roles, molmo_queries) and records
the recipe flags the source run used (exist_thr, body_select, overlap2d).

  python pipeline/extract_stage2_points.py --category daily_used
"""
import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_GEN = os.path.dirname(HERE)
KEYS = ("ptsA_multi", "ptsB_multi", "pexistA", "pexistB")


def extract_object(args):
    src_obj, dst_npz = args
    arrays, metas = {}, {}
    for qdir in sorted(os.listdir(src_obj)):
        qpath = os.path.join(src_obj, qdir)
        spath, mpath = os.path.join(qpath, "scores.npz"), os.path.join(qpath, "meta.json")
        if not (os.path.isfile(spath) and os.path.isfile(mpath)):
            continue
        with np.load(spath, allow_pickle=False) as data:
            for key in KEYS:
                arrays[f"{qdir}/{key}"] = data[key]
        with open(mpath) as stream:
            metas[qdir] = json.load(stream)
    if not metas:
        return dst_npz, 0
    os.makedirs(os.path.dirname(dst_npz), exist_ok=True)
    tmp = dst_npz + ".tmp.npz"
    np.savez_compressed(tmp, _meta=np.array(json.dumps(metas, ensure_ascii=False)), **arrays)
    os.replace(tmp, dst_npz)
    return dst_npz, len(metas)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", required=True)
    ap.add_argument("--src", default="outputs/stage2")
    ap.add_argument("--dst", default="outputs/stage2_points_20260803")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    os.chdir(DATA_GEN)

    src_root = os.path.join(args.src, args.category)
    dst_root = os.path.join(args.dst, args.category)
    objs = sorted(o for o in os.listdir(src_root) if os.path.isdir(os.path.join(src_root, o)))
    jobs = [(os.path.join(src_root, o), os.path.join(dst_root, f"{o}.npz")) for o in objs
            if not os.path.isfile(os.path.join(dst_root, f"{o}.npz"))]
    print(f"{args.category}: {len(objs)} objects, {len(jobs)} to extract -> {dst_root}", flush=True)

    n_q = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(extract_object, j) for j in jobs]
        for done, fut in enumerate(as_completed(futures), 1):
            n_q += fut.result()[1]
            if done % 2000 == 0 or done == len(jobs):
                print(f"  [{done}/{len(jobs)}] objects, {n_q} queries", flush=True)
    print(f"{args.category}: DONE {n_q} queries", flush=True)


if __name__ == "__main__":
    main()
