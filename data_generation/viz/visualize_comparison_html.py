#!/usr/bin/env python
"""
visualize_comparison_html.py
============================

Interactive (plotly) side-by-side 3D heatmap comparison: OUR reproduced affordance
heatmap vs original AFFOGATO GT, on the SAME points, per query — rotatable/zoomable.

Reads the `affordance_pred.npz` files from `single_region_affordance.py`
(pred, gt, xyz, counts, queries). No GPU / re-run needed.

For each object -> one HTML: rows = the 5 queries, columns = [Ours | GT].
Each panel is a 3D scatter colored by the heatmap value (Jet). Color is min-max
normalized per panel (compare DISTRIBUTION/shape, since our scale differs from GT's);
hover shows the raw value. Per-query AUC is in each row's titles.
Also writes an index.html linking every object, sorted by mean AUC.

Usage:
  python visualize_comparison_html.py --pred_dir output_single_region_40v --out viz_html_40v
  python visualize_comparison_html.py --pred_dir output_single_region_150 --out viz_html_150
"""

import os
import glob
import argparse

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import roc_auc_score


def minmax(v):
    lo, hi = float(v.min()), float(v.max())
    return (v - lo) / (hi - lo) if hi > lo else np.zeros_like(v)


def per_query_auc(P, G, thr=0.5):
    y = (G >= thr).astype(int)
    if y.sum() == 0 or y.sum() == len(y) or P.std() == 0:
        return float("nan")
    return float(roc_auc_score(y, P))


def build_object_html(npz_path, out_dir, max_points=7000):
    d = np.load(npz_path, allow_pickle=True)
    pred, gt, xyz, counts = d["pred"], d["gt"].astype(np.float32), d["xyz"], d["counts"]
    queries = list(d["queries"]) if "queries" in d else [f"q{j}" for j in range(pred.shape[1])]
    oid = os.path.basename(os.path.dirname(npz_path))
    K = min(pred.shape[1], gt.shape[1])
    vis = counts > 0
    cov = float(vis.mean())

    N = xyz.shape[0]
    idx = np.arange(N)
    if N > max_points:
        idx = np.random.default_rng(0).choice(N, max_points, replace=False)
    P3 = xyz[idx]

    titles, aucs = [], []
    for j in range(K):
        a = per_query_auc(pred[vis, j], gt[vis, j]); aucs.append(a)
        q = str(queries[j]) if j < len(queries) else f"q{j}"
        qshort = q if len(q) < 52 else q[:49] + "..."
        titles += [f"Q{j} Ours — AUC={a:.2f}<br><sub>{qshort}</sub>", f"Q{j} AFFOGATO GT"]

    fig = make_subplots(
        rows=K, cols=2,
        specs=[[{"type": "scene"}, {"type": "scene"}] for _ in range(K)],
        subplot_titles=titles, horizontal_spacing=0.02, vertical_spacing=0.04,
    )

    for j in range(K):
        for col, vals in enumerate([pred[idx, j], gt[idx, j]]):
            raw = vals
            fig.add_trace(go.Scatter3d(
                x=P3[:, 0], y=P3[:, 1], z=P3[:, 2], mode="markers",
                marker=dict(size=1.6, color=minmax(raw), colorscale="Jet",
                            cmin=0, cmax=1, showscale=False, opacity=0.85),
                text=[f"{v:.3f}" for v in raw], hoverinfo="text", showlegend=False,
            ), row=j + 1, col=col + 1)

    # equalize 3D aspect for every scene
    scene_kw = dict(aspectmode="data", xaxis=dict(visible=False),
                    yaxis=dict(visible=False), zaxis=dict(visible=False))
    fig.update_layout(
        **{f"scene{i+1 if i else ''}": scene_kw for i in range(2 * K)},
        title=f"{oid} — coverage {cov*100:.0f}% — mean AUC {np.nanmean(aucs):.2f} "
              f"(color = per-panel min-max heatmap; hover = raw score)",
        height=360 * K, width=1100, margin=dict(l=0, r=0, t=70, b=0),
    )
    for ann in fig.layout.annotations:
        ann.font.size = 11

    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"{oid}.html")
    # inline plotly.js -> fully self-contained, works in VSCode Live Preview / offline
    fig.write_html(out, include_plotlyjs=True)
    return out, oid, float(np.nanmean(aucs)), cov


def write_index(out_dir, rows):
    rows = sorted(rows, key=lambda r: r[2], reverse=True)
    items = "\n".join(
        f'<tr><td><a href="{os.path.basename(h)}">{oid}</a></td>'
        f'<td>{auc:.3f}</td><td>{cov*100:.0f}%</td></tr>'
        for (h, oid, auc, cov) in rows
    )
    html = f"""<!doctype html><meta charset=utf-8>
<title>AFFOGATO reproduction — heatmap comparison</title>
<style>body{{font-family:sans-serif;margin:30px}}table{{border-collapse:collapse}}
td,th{{border:1px solid #ccc;padding:6px 12px;text-align:left}}th{{background:#eee}}</style>
<h2>AFFOGATO reproduction vs GT — {len(rows)} objects</h2>
<p>Each link: per-query 3D heatmap, Ours vs GT (rotatable). Sorted by mean AUC.</p>
<table><tr><th>object</th><th>mean AUC</th><th>coverage</th></tr>
{items}
</table>"""
    p = os.path.join(out_dir, "index.html")
    with open(p, "w") as f:
        f.write(html)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", default="outputs/output_single_region_40v")
    ap.add_argument("--out", default="galleries/viz_html")
    ap.add_argument("--object", default=None)
    args = ap.parse_args()

    if args.object:
        paths = [os.path.join(args.pred_dir, args.object, "affordance_pred.npz")]
    else:
        paths = sorted(glob.glob(os.path.join(args.pred_dir, "*", "affordance_pred.npz")))
    print(f"{len(paths)} objects -> {args.out}/")
    rows = []
    for p in paths:
        if not os.path.exists(p):
            continue
        out, oid, auc, cov = build_object_html(p, args.out)
        rows.append((out, oid, auc, cov))
        print(f"  wrote {out}  (mean AUC {auc:.2f}, cov {cov*100:.0f}%)")
    if rows:
        idx = write_index(args.out, rows)
        print(f"\nINDEX: {idx}")


if __name__ == "__main__":
    main()
