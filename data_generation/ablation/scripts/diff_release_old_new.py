#!/usr/bin/env python
"""Characterise how the regenerated release differs from the shipped one.

Compares outputs/stage2/<cat> against the frozen 20260803 backup query by
query.  Both sides used the SAME Molmo points (the replay reads them from the
archive), so the diff is aggregation + whatever SAM2 nondeterminism contributes;
the pilot measured that floor at ~3e-5 MAE, two orders below the aggregation
effect, so it does not drive these numbers.

Reports, over the whole population rather than a sample: per-query final-score
MAE, correlation, the fraction of canvas points that move, and how often a point
crosses the 0.15 support threshold that downstream consumers read.  Stratified by
single-point vs multi-point role-views, because the new denominator only changes
shape where a role/view had more than one mask.

  python ablation/scripts/diff_release_old_new.py --category electronics
"""
import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DG = os.path.dirname(os.path.dirname(HERE))
OLD_ROOT = "/nfs_drive/mclee/affogato_backup/20260803_stage2"
SUPPORT_THR = 0.15


def one_object(job):
    oid, new_dir, old_dir = job
    rows = []
    for q in sorted(os.listdir(new_dir)):
        np_, op = f"{new_dir}/{q}/scores.npz", f"{old_dir}/{q}/scores.npz"
        if not (os.path.isfile(np_) and os.path.isfile(op)):
            continue
        try:
            with np.load(np_, allow_pickle=False) as n, np.load(op, allow_pickle=False) as o:
                # the points must be identical or we are not comparing like with like
                same_pts = all(np.array_equal(n[k], o[k], equal_nan=True)
                               for k in ("ptsA_multi", "ptsB_multi"))
                for role in "AB":
                    a = n[f"score{role}"].astype(np.float64)
                    b = o[f"score{role}"].astype(np.float64)
                    if a.shape != b.shape:
                        continue
                    multi = n[f"pts{role}_multi"]
                    k_per_view = np.bincount(multi[:, 0].astype(int)) if len(multi) else np.array([0])
                    rows.append(dict(
                        oid=oid, q=q, role=role, same_pts=bool(same_pts),
                        mae=float(np.abs(a - b).mean()),
                        maxabs=float(np.abs(a - b).max()),
                        moved=float((np.abs(a - b) > 1e-3).mean()),
                        moved_big=float((np.abs(a - b) > 0.05).mean()),
                        flipped=float(((a >= SUPPORT_THR) != (b >= SUPPORT_THR)).mean()),
                        cov_new=float((a >= SUPPORT_THR).mean()),
                        cov_old=float((b >= SUPPORT_THR).mean()),
                        multi_views=int((k_per_view > 1).sum()),
                    ))
        except Exception as exc:
            rows.append(dict(oid=oid, q=q, role="?", error=f"{type(exc).__name__}: {exc}"))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", required=True)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    os.chdir(DG)

    new_root = f"outputs/stage2/{args.category}"
    old_root = f"{OLD_ROOT}/{args.category}"
    jobs = []
    for oid in sorted(os.listdir(new_root)):
        nd, od = f"{new_root}/{oid}", f"{old_root}/{oid}"
        if os.path.isdir(nd) and os.path.isdir(od):
            jobs.append((oid, nd, od))
    print(f"{args.category}: {len(jobs)} objects with both old and new", flush=True)

    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(one_object, j) for j in jobs]
        for i, f in enumerate(as_completed(futs), 1):
            rows.extend(f.result())
            if i % 1000 == 0:
                print(f"  [{i}/{len(jobs)}]", flush=True)

    errs = [r for r in rows if r.get("error")]
    ok = [r for r in rows if not r.get("error")]
    bad_pts = [r for r in ok if not r["same_pts"]]
    single = [r for r in ok if r["multi_views"] == 0]
    multi = [r for r in ok if r["multi_views"] > 0]

    def stat(rs, key):
        v = np.array([r[key] for r in rs]) if rs else np.array([0.0])
        return f"mean {v.mean():.5f}  p50 {np.median(v):.5f}  p95 {np.percentile(v,95):.5f}  max {v.max():.5f}"

    print(f"\n=== {args.category}: {len(ok)} role-heatmaps ({len(errs)} errors) ===")
    print(f"points identical old-vs-new : {len(ok)-len(bad_pts)}/{len(ok)}"
          f"{'  <-- MISMATCH, diff is NOT attributable to aggregation' if bad_pts else ''}")
    print(f"single-point role-views : {len(single)}   multi-point : {len(multi)}")
    for label, rs in (("ALL", ok), ("single-point", single), ("multi-point", multi)):
        if not rs:
            continue
        print(f"\n  [{label}]")
        for key in ("mae", "maxabs", "moved", "moved_big", "flipped"):
            print(f"    {key:10} {stat(rs, key)}")
        cn = np.array([r["cov_new"] for r in rs]); co = np.array([r["cov_old"] for r in rs])
        print(f"    coverage   new {cn.mean():.4f}  old {co.mean():.4f}  delta {(cn-co).mean():+.4f} pp/1")

    out = args.out or f"ablation/results/release_diff_{args.category}.json"
    with open(out, "w") as s:
        json.dump(dict(category=args.category, n=len(ok), errors=errs[:50],
                       points_mismatch=len(bad_pts), rows=rows), s)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
