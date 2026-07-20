#!/usr/bin/env python
"""
vis_point_cloud.py — 由 object_id 找到它的 affogato 點雲，存成可互動旋轉的 3D HTML。

點雲來源：daily_used_to_affogato.json 的 object_id -> dst，讀 dst/xyzc.npy 的前三欄 (xyz)。
(xyzc.npy: 0-2 = xyz 座標, 3.. = GT affordance 通道，這裡只畫座標。)

輸出：<out_dir>/{object_id}_point_cloud.html，預設就存在本檔所在的 viz/ 底下。
用瀏覽器打開後可用滑鼠拖曳旋轉、滾輪縮放。

用法：
    # 單一物件 (可接短前綴)
    python viz/vis_point_cloud.py 849b4573e03a40b8b073a2209d1bc1d3
    python viz/vis_point_cloud.py 849b4573

    # 多個物件一次產生
    python viz/vis_point_cloud.py 849b4573 8fd4afdf

    # 換輸出目錄 / 調點的大小
    python viz/vis_point_cloud.py 849b4573 --out_dir /tmp --point_size 2
"""
import os
import json
import argparse

import numpy as np
import plotly.graph_objects as go

HERE = os.path.dirname(os.path.abspath(__file__))            # data_generation/viz/
DATA_GEN = os.path.dirname(HERE)                             # data_generation/


def _resolve(p):
    return p if os.path.isabs(p) else os.path.join(DATA_GEN, p)


def load_aff_map(mapping_path):
    """object_id -> affogato 目錄 (dst)。"""
    with open(mapping_path, "r") as f:
        return {e["object_id"]: e["dst"] for e in json.load(f) if e.get("dst")}


def lookup(query, aff_map):
    """精確比對優先；否則當成前綴比對 (貼短 id 時方便)。回傳符合的 object_id 清單。"""
    if query in aff_map:
        return [query]
    return sorted(o for o in aff_map if o.startswith(query))


def load_cloud(aff_dir):
    """回傳 (xyz [N,3], class_name)。class_name 讀不到就給空字串。"""
    xyz = np.load(os.path.join(aff_dir, "xyzc.npy")).astype(np.float32)[:, :3]
    name = ""
    try:
        name = json.load(open(os.path.join(aff_dir, "queries.json")))[0].get("class_name", "")
    except Exception:
        pass
    return xyz, name


def cloud_fig(xyz, title, size=1.2):
    """灰色點雲，只留物體本身 (可拖曳旋轉)。

    aspectmode='data' 保持真實長寬高比例，不會被拉伸成立方體；
    三軸 visible=False 一次隱藏軸線/刻度/標籤/網格線/背景牆。
    """
    s = slice(None, None, max(1, len(xyz) // 60000))     # 超大點雲才降採樣
    fig = go.Figure(go.Scatter3d(
        x=xyz[s, 0], y=xyz[s, 1], z=xyz[s, 2], mode="markers",
        marker=dict(size=size, color="lightslategrey", opacity=0.85),
        hoverinfo="skip"))
    fig.update_layout(title=title, width=900, height=700,
                      scene=dict(aspectmode="data",
                                 xaxis=dict(visible=False),
                                 yaxis=dict(visible=False),
                                 zaxis=dict(visible=False)),
                      margin=dict(l=0, r=0, t=40, b=0))
    return fig


def main():
    ap = argparse.ArgumentParser(description="由 object_id 產生可互動的點雲 HTML")
    ap.add_argument("ids", nargs="+", help="一個或多個 object_id (可短前綴)")
    ap.add_argument("--mapping", default="dataset/daily_used_to_affogato.json",
                    help="object_id -> affogato 目錄 (dst) 的對應表")
    ap.add_argument("--out_dir", default=HERE, help="輸出目錄 (預設 = 本檔所在的 viz/)")
    ap.add_argument("--point_size", type=float, default=1.2)
    args = ap.parse_args()

    aff_map = load_aff_map(_resolve(args.mapping))
    os.makedirs(args.out_dir, exist_ok=True)

    for q in args.ids:
        hits = lookup(q, aff_map)
        if not hits:
            print(f"[miss] {q}: 不在 mapping 裡")
            continue
        if len(hits) > 1:
            print(f"[ambiguous] {q}: 命中 {len(hits)} 個，請給更長的前綴")
            for h in hits[:5]:
                print(f"    {h}")
            continue

        oid = hits[0]
        aff_dir = aff_map[oid]
        if not os.path.exists(os.path.join(aff_dir, "xyzc.npy")):
            print(f"[miss] {oid}: 找不到 {aff_dir}/xyzc.npy")
            continue

        xyz, name = load_cloud(aff_dir)
        out = os.path.join(args.out_dir, f"{oid}_point_cloud.html")
        cloud_fig(xyz, f"{name or oid} — {len(xyz):,} pts", args.point_size).write_html(out)
        print(f"[ok] {name or '?':<20} {len(xyz):,} pts -> {out}")


if __name__ == "__main__":
    main()
