#!/usr/bin/env python
"""
render_heatmaps.py — static PNG views of affordance heatmaps (no browser needed).

For each affordance_pred.npz (from single_region_affordance.py), renders a 2xK grid:
  row 0 = AFFOGATO GT heatmap, row 1 = our predicted heatmap, one column per query,
on the affogato 16384 points, from a fixed 45-degree angled top-down view.

Usage:
  python render_heatmaps.py --pred_dir output_single_region_fixed --out images \
         --elev 45 --azim -60
"""
import os, json, glob, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")                      # no display / no browser
import matplotlib.pyplot as plt


def render_object(npz_path, out_dir, elev, azim, cmap, point_size):
    d = np.load(npz_path, allow_pickle=True)
    xyz, pred, gt, counts = d["xyz"], d["pred"], d["gt"].astype(np.float32), d["counts"]
    queries = list(d["queries"]) if "queries" in d else [f"q{j}" for j in range(pred.shape[1])]
    oid = os.path.basename(os.path.dirname(npz_path))
    K = min(pred.shape[1], gt.shape[1])

    # display orientation: affogato (x,y,z) with y up looks natural for these objects
    X, Y, Z = xyz[:, 0], xyz[:, 2], -xyz[:, 1]

    fig = plt.figure(figsize=(3.4 * K, 7.2))
    fig.suptitle(f"{oid}   (top=GT, bottom=ours;  view elev={elev}, azim={azim})", fontsize=11)

    for j in range(K):
        for row, (vals, label) in enumerate([(gt[:, j], "GT"), (pred[:, j], "ours")]):
            ax = fig.add_subplot(2, K, row * K + j + 1, projection="3d")
            vmax = max(float(np.percentile(vals, 99.5)), 1e-6)   # robust per-panel scale
            order = np.argsort(vals)                              # draw hot points last (on top)
            sc = ax.scatter(X[order], Y[order], Z[order], c=vals[order],
                            cmap=cmap, vmin=0, vmax=vmax, s=point_size, alpha=0.85,
                            linewidths=0, depthshade=False)
            ax.view_init(elev=elev, azim=azim)
            ax.set_axis_off()
            try:
                ax.set_box_aspect((np.ptp(X), np.ptp(Y), np.ptp(Z)))
            except Exception:
                pass
            if row == 0:
                q = str(queries[j]) if j < len(queries) else f"q{j}"
                ax.set_title(f"q{j}: {q[:42]}{'…' if len(q) > 42 else ''}", fontsize=7)
            else:
                cov = (counts > 0).mean()
                ax.set_title(f"{label}  (coverage {cov*100:.0f}%)", fontsize=7)

    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"{oid}.png")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", default="outputs/output_single_region_fixed")
    ap.add_argument("--out", default="assets/images")
    ap.add_argument("--elev", type=float, default=45.0)
    ap.add_argument("--azim", type=float, default=-60.0)
    ap.add_argument("--cmap", default="jet")
    ap.add_argument("--point_size", type=float, default=4.0)
    ap.add_argument("--limit", type=int, default=0, help="0 = all objects")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.pred_dir, "*", "affordance_pred.npz")))
    if args.limit:
        paths = paths[:args.limit]
    if not paths:
        print(f"no affordance_pred.npz under {args.pred_dir}"); return
    print(f"rendering {len(paths)} objects -> {args.out}/  (elev={args.elev}, azim={args.azim})")
    for p in paths:
        out = render_object(p, args.out, args.elev, args.azim, args.cmap, args.point_size)
        print(f"  saved {out}")


if __name__ == "__main__":
    main()
