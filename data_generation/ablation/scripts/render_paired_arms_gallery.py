#!/usr/bin/env python
"""Fixed-scale gallery for the three-arm paired ablation on the new release.

Reuses render_mask_consolidation_gallery for the support view and the page
style, so the cameras, up axis, role colours and 0.15 support threshold are the
ones every earlier Affogato gallery used; a per-file re-implementation would
quietly drift from them.

Two views per role, because neither alone is honest:
  SUPPORT     score > 0.15, role-coloured — what downstream consumers read.
  CONTINUOUS  the score itself on a 0..1 colour scale FIXED across every arm
              and every panel on the page.
The handoff is explicit that a thresholded support map must not be presented as
a continuous score map, and that panels must not be normalised individually --
per-panel normalisation makes any two arms look equally confident.

  python ablation/scripts/render_paired_arms_gallery.py \
      ablation/results/paired_arms_electronics --n 24
"""
import argparse
import html
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import render_mask_consolidation_gallery as base   # noqa: E402  (cameras, colours, CSS)

ARMS = ("arm1_conditional", "arm2_iou", "arm3_conditional_repeat")
LABELS = ("條件平均 τ=0.1（採用）", "IoU 分組（原法）", "條件平均 τ=0.1（重跑對照）")
THR = base.THR


def render_continuous(path, xyz, score, cmap="magma"):
    """One role's score on a FIXED 0..1 scale, same cameras as the support view."""
    import matplotlib.pyplot as plt

    xyz = xyz[:, [0, 2, 1]]
    span = np.maximum(np.ptp(xyz, axis=0), 1e-6)
    center = (xyz.min(0) + xyz.max(0)) / 2
    fig = plt.figure(figsize=(12, 2.7), dpi=110)
    for col, az in enumerate(base.AZIMUTHS):
        ax = fig.add_subplot(1, 4, col + 1, projection="3d")
        ax.scatter(*xyz.T, s=.5, c="#e8e8e8", alpha=.18, linewidths=0)
        live = score > 0                      # 0 is "no evidence", not "cold"
        if live.any():
            # vmin/vmax pinned, NOT derived from this panel's own range
            ax.scatter(*xyz[live].T, s=1.6, c=score[live], cmap=cmap,
                       vmin=0.0, vmax=1.0, linewidths=0)
        for setter, mid, extent in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), center, span):
            setter(mid - extent * .53, mid + extent * .53)
        ax.set_box_aspect(span)
        ax.view_init(elev=18, azim=az)
        ax.set_axis_off()
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0)
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def collect(root, n, seed):
    """Stratified pick: the change only bites where a role/view had K>1 masks."""
    a1 = root / ARMS[0]
    rows = []
    for oid in sorted(os.listdir(a1)):
        if not (a1 / oid).is_dir():
            continue
        for q in sorted(os.listdir(a1 / oid)):
            if not all((root / arm / oid / q / "scores.npz").is_file() for arm in ARMS):
                continue
            with np.load(a1 / oid / q / "scores.npz", allow_pickle=False) as z:
                multi = sum(int((np.bincount(z[f"pts{r}_multi"][:, 0].astype(int)) > 1).sum())
                            if len(z[f"pts{r}_multi"]) else 0 for r in "AB")
            rows.append((multi, oid, q))
    rows.sort(reverse=True)
    single = [r for r in rows if r[0] == 0]
    multi = [r for r in rows if r[0] > 0]
    rng = np.random.default_rng(seed)
    take = lambda pool, k: [pool[i] for i in rng.permutation(len(pool))[:k]] if pool else []
    picked = take(multi, max(n - n // 3, 0)) + take(single, n // 3)
    return picked, len(multi), len(single)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    root = args.root.resolve()
    os.environ.setdefault("MPLCONFIGDIR", str(root / "matplotlib_cache"))
    import matplotlib
    matplotlib.use("Agg")

    picked, n_multi, n_single = collect(root, args.n, args.seed)
    if not picked:
        raise SystemExit("no paired queries found")
    assets = root / "gallery_assets"
    assets.mkdir(exist_ok=True)

    cards = []
    for i, (multi, oid, q) in enumerate(picked, 1):
        data, meta = {}, None
        for arm in ARMS:
            with np.load(root / arm / oid / q / "scores.npz", allow_pickle=False) as z:
                data[arm] = {k: z[k] for k in ("xyz", "scoreA", "scoreB", "scoreA_raw", "scoreB_raw")}
            if meta is None:
                meta = json.loads((root / arm / oid / q / "meta.json").read_text())
            # identical geometry is what makes the panels comparable at all
            np.testing.assert_array_equal(data[arm]["xyz"], data[ARMS[0]]["xyz"])

        imgs = {}
        for arm in ARMS:
            for stage, sfx in (("final", ""), ("raw", "_raw")):
                p = assets / f"{oid}_{q}_{arm}_{stage}_support.png"
                if not p.exists():
                    base.render_cloud(p, data[arm]["xyz"], data[arm][f"scoreA{sfx}"],
                                      data[arm][f"scoreB{sfx}"])
                imgs[(arm, stage, "support")] = p.name
                for role in "AB":
                    pc = assets / f"{oid}_{q}_{arm}_{stage}_cont{role}.png"
                    if not pc.exists():
                        render_continuous(pc, data[arm]["xyz"], data[arm][f"score{role}{sfx}"])
                    imgs[(arm, stage, f"cont{role}")] = pc.name

        # numbers next to the pictures, so a visual read can be checked
        stats = {}
        for stage, sfx in (("final", ""), ("raw", "_raw")):
            m = np.abs(data[ARMS[0]][f"scoreA{sfx}"] - data[ARMS[1]][f"scoreA{sfx}"]).mean()
            nz = np.abs(data[ARMS[0]][f"scoreA{sfx}"] - data[ARMS[2]][f"scoreA{sfx}"]).mean()
            cov = {arm: [float((data[arm][f"score{r}{sfx}"] > THR).mean()) for r in "AB"] for arm in ARMS}
            stats[stage] = dict(method_mae=float(m), noise_mae=float(nz), cov=cov)
        cards.append(dict(oid=oid, q=q, meta=meta, multi=multi, imgs=imgs, stats=stats))
        print(f"[{i}/{len(picked)}] {meta['object_name'][:30]} — {meta['task'][:40]}", flush=True)

    (root / "paired_arms_gallery.html").write_text(build_page(cards, n_multi, n_single), encoding="utf-8")
    print(f"\nGallery: {root / 'paired_arms_gallery.html'}")


def build_page(cards, n_multi, n_single):
    e = html.escape
    out = [f"<!DOCTYPE html><html lang='zh-Hant'><head><meta charset='utf-8'>"
           f"<title>三臂 paired ablation — 固定色階</title><style>{base.CSS}"
           ".cmp{overflow-x:auto}.arm{border-top:1px solid var(--line);padding:12px 22px}"
           ".arm h3{margin:0 0 8px;font-size:15px}.arm img{display:block;width:100%;max-width:1180px;"
           "border:1px solid var(--line);border-radius:8px;margin:4px 0}"
           ".nums{font:12px ui-monospace,monospace;color:#475467;margin:6px 0}"
           ".scale{display:flex;align-items:center;gap:8px;font-size:12px;color:#aeb7c3;margin:6px 0}"
           ".bar{height:10px;width:220px;border-radius:5px;background:linear-gradient("
           "to right,#000004,#3b0f70,#8c2981,#de4968,#fe9f6d,#fcfdbf)}"
           "</style></head><body><header><h1>三臂 paired ablation — 固定 0–1 色階</h1>"
           f"<p>取樣 {len(cards)} 個 query（多點 {n_multi} / 單點 {n_single} 可選)。"
           "三臂共用同一批凍結點位;每臂各自重跑 SAM2、consensus、2D overlap 與 3D。</p>"
           "<p><b>arm1 vs arm2</b> = aggregation 效應;<b>arm1 vs arm3</b> = 重跑雜訊底線"
           "(實測為 0,逐位元相同)。</p>"
           "<p>所有 continuous 面板共用 <b>vmin=0 / vmax=1</b>,未逐面板 normalize;"
           "support 面板為 score &gt; 0.15,與下游消費者讀到的一致。四欄為固定方位角 "
           "45°/135°/225°/315°,完整 canvas、相同點序與 marker 大小。</p>"
           "<div class='scale'><span>0</span><span class='bar'></span><span>1</span>"
           "<span>連續分數色階(magma)</span></div></header><main>"]
    for c in cards:
        m = c["meta"]
        roles = m.get("roles", [{}, {}])
        out.append(f"<div class='card'><div class='body'><div class='head'>"
                   f"<h2>{e(m.get('object_name',''))}</h2>"
                   f"<span class='badge'>{'多點 '+str(c['multi'])+' role-view' if c['multi'] else '單點對照'}</span>"
                   f"</div><div class='task'>{e(m.get('task',''))}</div><div class='roles'>"
                   f"<div class='role'><span class='rid' style='color:{base.COLORS[0]}'>A</span>"
                   f"{e(str(roles[0].get('role','')))} @ {e(str(roles[0].get('contact_region','')))}"
                   f"<span class='muted'>({e(str(roles[0].get('target','')))})</span></div>"
                   f"<div class='role'><span class='rid' style='color:{base.COLORS[1]}'>B</span>"
                   f"{e(str(roles[1].get('role','')))} @ {e(str(roles[1].get('contact_region','')))}"
                   f"<span class='muted'>({e(str(roles[1].get('target','')))})</span></div></div>"
                   f"<div class='id'>{e(c['oid'])} / {e(c['q'])}</div></div>")
        for stage in ("final", "raw"):
            s = c["stats"][stage]
            out.append(f"<div class='section-title'><span>{stage} 3D</span>"
                       f"<span class='muted'>方法差異 MAE {s['method_mae']:.5f} · "
                       f"雜訊 MAE {s['noise_mae']:.5f}</span></div>")
            for arm, label in zip(ARMS, LABELS):
                cov = s["cov"][arm]
                out.append(f"<div class='arm'><h3>{e(label)}</h3>"
                           f"<div class='nums'>support A {cov[0]*100:.2f}% · B {cov[1]*100:.2f}%</div>"
                           f"<img loading='lazy' src='gallery_assets/{c['imgs'][(arm,stage,'support')]}'>"
                           f"<div class='nums'>連續分數 — role A</div>"
                           f"<img loading='lazy' src='gallery_assets/{c['imgs'][(arm,stage,'contA')]}'>"
                           f"<div class='nums'>連續分數 — role B</div>"
                           f"<img loading='lazy' src='gallery_assets/{c['imgs'][(arm,stage,'contB')]}'>"
                           f"</div>")
        out.append("</div>")
    out.append("</main></body></html>")
    return "".join(out)


if __name__ == "__main__":
    main()
