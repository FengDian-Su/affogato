#!/usr/bin/env python
"""
visualize_comparison.py
=======================

Side-by-side 3D point-cloud heatmap comparison: OUR reproduced affordance heatmap
vs the original AFFOGATO GT, on the SAME 16384 points, per query.

Reads the `affordance_pred.npz` files written by `single_region_affordance.py`
(which already store pred, gt, xyz, counts, queries) — no GPU / re-run needed.

For each object it writes one PNG: rows = the 5 queries, columns = [Ours | GT].
Each panel is a 3D scatter of the points colored by the heatmap value (jet).
Both maps are min-max normalized per panel so the comparison is about the spatial
DISTRIBUTION/shape (the right comparison, since our scale differs from GT's).
Per-query AUC is printed in each row title.

Usage:
  python visualize_comparison.py --pred_dir output_single_region_40v --out viz_40v
  python visualize_comparison.py --pred_dir output_single_region_40v --object 02886b7e... --out viz_40v
"""

import os
import glob
import json
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score


def minmax(v):
    lo, hi = v.min(), v.max()
    return (v - lo) / (hi - lo) if hi > lo else np.zeros_like(v)


def per_query_auc(P, G, thr=0.5):
    y = (G >= thr).astype(int)
    if y.sum() == 0 or y.sum() == len(y) or P.std() == 0:
        return float("nan")
    return roc_auc_score(y, P)


def visualize_object(npz_path, out_dir, max_points=9000, elev=16, azim=-72):
    d = np.load(npz_path, allow_pickle=True)
    pred, gt, xyz, counts = d["pred"], d["gt"].astype(np.float32), d["xyz"], d["counts"]
    queries = list(d["queries"]) if "queries" in d else [f"q{j}" for j in range(pred.shape[1])]
    oid = os.path.basename(os.path.dirname(npz_path))
    K = min(pred.shape[1], gt.shape[1])

    # subsample for plotting speed (keep visible points preferentially)
    N = xyz.shape[0]
    idx = np.arange(N)
    if N > max_points:
        idx = np.random.default_rng(0).choice(N, max_points, replace=False)
    P3 = xyz[idx]

    cov = (counts > 0).mean()
    fig = plt.figure(figsize=(8.2, 3.1 * K))
    for j in range(K):
        pv = minmax(pred[idx, j])
        gv = minmax(gt[idx, j])
        auc = per_query_auc(pred[counts > 0, j], gt[counts > 0, j])
        q = str(queries[j]) if j < len(queries) else f"q{j}"

        for col, (vals, name) in enumerate([(pv, "Ours (reproduced)"), (gv, "AFFOGATO GT")]):
            ax = fig.add_subplot(K, 2, 2 * j + col + 1, projection="3d")
            order = np.argsort(vals)               # draw high values last (on top)
            ax.scatter(P3[order, 0], P3[order, 1], P3[order, 2],
                       c=vals[order], cmap="jet", s=3, vmin=0, vmax=1,
                       depthshade=False, linewidths=0)
            ax.set_axis_off()
            ax.view_init(elev=elev, azim=azim)
            if col == 0:   # left panel carries the query label (no overlap with plots)
                ax.set_title(f'Q{j}  AUC={auc:.2f}   "{q[:58]}"\nOurs (reproduced)',
                             fontsize=8, loc="left")
            else:
                ax.set_title("AFFOGATO GT", fontsize=8)

    fig.suptitle(f"{oid}   |   coverage {cov*100:.0f}%   |   "
                 f"color = per-panel min-max heatmap (jet: blue=low, red=high)", fontsize=9)
    fig.subplots_adjust(left=0.0, right=1.0, top=0.95, bottom=0.01, hspace=0.32, wspace=0.0)
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"{oid}.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", default="outputs/output_single_region_40v")
    ap.add_argument("--out", default="assets/viz_comparison")
    ap.add_argument("--object", default=None, help="single object_id; default = all")
    args = ap.parse_args()

    if args.object:
        paths = [os.path.join(args.pred_dir, args.object, "affordance_pred.npz")]
    else:
        paths = sorted(glob.glob(os.path.join(args.pred_dir, "*", "affordance_pred.npz")))
    print(f"{len(paths)} objects -> {args.out}/")
    for p in paths:
        if not os.path.exists(p):
            print(f"  [skip] {p} missing"); continue
        out = visualize_object(p, args.out)
        print(f"  wrote {out}")


if __name__ == "__main__":
    main()
