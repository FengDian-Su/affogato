#!/usr/bin/env python
"""
visualize_pred_vs_gt.py
=======================

Render one object's per-query PRED vs GT affordance heatmaps as interactive 3D
point clouds, all in a single self-contained HTML (rows = queries, left = GT,
right = our reproduction). Unseen points (never projected by any view) are drawn
in light gray so coverage is visible.

Usage:
  python visualize_pred_vs_gt.py --npz output_single_region_smoke/<id>/affordance_pred.npz
  python visualize_pred_vs_gt.py --pred_dir output_single_region --object_id <id>
"""

import os
import argparse
import numpy as np
import plotly.graph_objects as go


def scatter_fig(xyz, vals, seen, title, colorbar_title, subsample=1):
    """One interactive 3D scatter: gray = unseen, Jet-colored = score on seen points."""
    idx = np.arange(0, len(xyz), subsample)
    x, y, z = xyz[idx, 0], -xyz[idx, 1], xyz[idx, 2]   # -y for an upright view
    v = vals[idx]
    s = seen[idx]

    fig = go.Figure()
    if (~s).any():
        fig.add_trace(go.Scatter3d(
            x=x[~s], y=y[~s], z=z[~s], mode="markers",
            marker=dict(size=1.5, color="lightgray", opacity=0.35),
            name="unseen", hoverinfo="skip",
        ))
    cmax = float(v[s].max()) if s.any() and v[s].max() > 0 else 1.0
    fig.add_trace(go.Scatter3d(
        x=x[s], y=y[s], z=z[s], mode="markers",
        marker=dict(size=2.2, color=v[s], colorscale="Jet", cmin=0, cmax=cmax,
                    colorbar=dict(title=colorbar_title, thickness=12, len=0.8), opacity=0.9),
        text=[f"{val:.3f}" for val in v[s]], hoverinfo="text", name="score",
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=12)),
        scene=dict(aspectmode="data", xaxis_title="", yaxis_title="", zaxis_title="",
                   xaxis=dict(showticklabels=False), yaxis=dict(showticklabels=False),
                   zaxis=dict(showticklabels=False)),
        margin=dict(l=0, r=0, t=28, b=0), height=380,
    )
    return fig


def build_html(npz_path, out_path, subsample=1):
    d = np.load(npz_path, allow_pickle=True)
    pred, gt, counts, xyz = d["pred"], d["gt"].astype(np.float32), d["counts"], d["xyz"]
    queries = list(d["queries"]) if "queries" in d else [f"q{j}" for j in range(pred.shape[1])]
    meta = {}
    if "meta_json" in d:
        import json
        meta = json.loads(str(d["meta_json"]))
    object_id = os.path.basename(os.path.dirname(npz_path))
    seen = counts > 0
    coverage = float(seen.mean())
    K = min(pred.shape[1], gt.shape[1])

    parts = [
        "<html><head><meta charset='utf-8'>",
        "<script src='https://cdn.plot.ly/plotly-2.35.2.min.js'></script>",
        "<style>body{font-family:sans-serif;margin:16px;background:#fafafa}"
        ".row{display:flex;gap:8px;margin-bottom:6px}.cell{flex:1;background:#fff;border:1px solid #ddd;border-radius:6px}"
        ".qhead{margin:18px 0 4px;padding:6px 10px;background:#222;color:#fff;border-radius:6px;font-size:14px}"
        "h2{margin:4px 0}</style></head><body>",
        f"<h2>{object_id} — PRED vs GT affordance heatmaps</h2>",
        f"<div>views used: {meta.get('n_views_used','?')} &nbsp;|&nbsp; coverage: "
        f"{coverage*100:.1f}% of {len(xyz)} GT points seen (gray = unseen)</div>",
    ]
    first = True
    for j in range(K):
        q = str(queries[j]) if j < len(queries) else f"q{j}"
        G, P = gt[:, j], pred[:, j]
        gt_pos = int((G >= 0.5).sum())
        parts.append(f"<div class='qhead'>Q{j}: {q} &nbsp; "
                     f"(GT mean={G.mean():.3f}, GT≥0.5 pts={gt_pos}; PRED max={P.max():.3f})</div>")
        fig_gt = scatter_fig(xyz, G, seen, f"GT — Q{j}", "GT", subsample)
        fig_pr = scatter_fig(xyz, P, seen, f"PRED — Q{j}", "pred", subsample)
        inc = "cdn" if first else False
        html_gt = fig_gt.to_html(full_html=False, include_plotlyjs=False, default_height="380px")
        html_pr = fig_pr.to_html(full_html=False, include_plotlyjs=False, default_height="380px")
        first = False
        parts.append(f"<div class='row'><div class='cell'>{html_gt}</div>"
                     f"<div class='cell'>{html_pr}</div></div>")
    parts.append("</body></html>")

    with open(out_path, "w") as f:
        f.write("\n".join(parts))
    print(f"[INFO] wrote {out_path}  ({K} queries, coverage {coverage*100:.1f}%)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", help="path to affordance_pred.npz")
    ap.add_argument("--pred_dir", help="dir containing <object_id>/affordance_pred.npz")
    ap.add_argument("--object_id", help="used with --pred_dir")
    ap.add_argument("--out", default=None, help="output html path (default: next to npz)")
    ap.add_argument("--subsample", type=int, default=1, help="point stride for lighter html")
    args = ap.parse_args()

    if args.npz:
        npz = args.npz
    else:
        npz = os.path.join(args.pred_dir, args.object_id, "affordance_pred.npz")
    out = args.out or os.path.join(os.path.dirname(npz), "pred_vs_gt.html")
    build_html(npz, out, args.subsample)


if __name__ == "__main__":
    main()
