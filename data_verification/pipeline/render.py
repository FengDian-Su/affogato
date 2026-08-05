"""Combined-heatmap rendering + stratified pilot sampling (README sections 4 and 6.3).

Six SEPARATE labelled images per sample -- never a stitched panel, never an
A-only/B-only sheet (plan section 7.4). The HTML gallery is only a viewer over
those same files, so what a human inspects is byte-identical to what the Stage 2
VLM will receive.

Mask = absolute threshold tau_vis, SHARED by A and B (README 4.1.2 item 5: no
per-hand threshold). **FROZEN 2026-07-28 at tau_vis = 0.20.** --tau still accepts
several values so thresholds can be re-compared over identical samples, cameras
and colours, but 0.20 is the value the released protocol uses.

Inside the mask, intensity is the ORIGINAL continuous score in [0,1]
(intensity_mapping = "original_score_[0,1]"), so the picture stays a continuous
heatmap rather than a binary segmentation. --intensity gamma is kept only as a
legibility comparison aid; it is not the frozen protocol.

    python render.py --stratified 40                 # frozen: tau=0.20, linear
    python render.py --stratified 40 --tau 0.2 0.5 --intensity gamma
"""
import argparse
import collections
import json
import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402
from matplotlib.patches import Patch     # noqa: E402

import common as C                       # noqa: E402

RENDER_VERSION = "dual-heatmap-render-v1.0"
TAU_VIS = 0.20                                # FROZEN 2026-07-28, shared by A and B
INTENSITY_MAPPING = "original_score_[0,1]"    # FROZEN 2026-07-28

# Fixed camera protocol (plan section 7.5). xyz is permuted to Y-up -> matplotlib Z-up.
VIEWS = {
    "front": (10, -90), "back": (10, 90), "left": (10, 180), "right": (10, 0),
    "upper-front": (45, -90), "upper-back": (45, 90),
}
VIEW_ORDER = ["front", "left", "upper-front", "back", "right", "upper-back"]

GREY = np.array([0.72, 0.72, 0.72])
RED = np.array([0.90, 0.08, 0.08])
BLUE = np.array([0.06, 0.16, 0.90])
PURPLE = np.array([0.62, 0.08, 0.72])
BG = "white"
INACTIVE_SIZE, INACTIVE_ALPHA = 2.4, 0.40     # object-shape haze
ACTIVE_SIZE, ACTIVE_ALPHA = 3.6, 0.98         # bigger + drawn last so it reads over the haze
ZOOM = 1.12
GAMMA = 0.5                                   # comparison aid only, NOT the frozen protocol
FLOOR = 0.35                                  # comparison aid only, NOT the frozen protocol
FIGSIZE, DPI = (2.6, 2.6), 130

LEGEND = [Patch(facecolor=RED, label="Hand A"), Patch(facecolor=BLUE, label="Hand B"),
          Patch(facecolor=PURPLE, label="A/B overlap"), Patch(facecolor=GREY, label="inactive")]


def _yup(xyz):
    return xyz[:, [0, 2, 1]]


def intensity(score, mask, mode="linear"):
    """FROZEN: the original continuous score in [0,1]. Never per-sample min-max
    (plan section 5.3). `gamma` exists only to re-check legibility near tau."""
    t = np.zeros_like(score, dtype=np.float64)
    if mask.any():
        s = np.clip(score[mask], 0, 1)
        t[mask] = s if mode == "linear" else FLOOR + (1 - FLOOR) * s ** GAMMA
    return t


def point_colors(a, b, tau, mode="linear"):
    ma, mb = a >= tau, b >= tau                      # SAME tau for both hands
    ta, tb = intensity(a, ma, mode), intensity(b, mb, mode)
    col = np.zeros((len(a), 3))
    only_a, only_b, both = ma & ~mb, mb & ~ma, ma & mb
    col[only_a] = GREY + (RED - GREY) * ta[only_a, None]
    col[only_b] = GREY + (BLUE - GREY) * tb[only_b, None]
    col[both] = GREY + (PURPLE - GREY) * np.maximum(ta, tb)[both, None]
    return col, (ma | mb)


def render_views(qdir, out_dir, tau, prefix, mode="linear"):
    """Six separate PNGs. Returns {view: path}."""
    with np.load(os.path.join(qdir, "scores.npz"), allow_pickle=False) as z:
        xyz = _yup(z["xyz"].astype(np.float32))
        a, b = z["scoreA"].astype(np.float64), z["scoreB"].astype(np.float64)
    col, active = point_colors(a, b, tau, mode)
    span = xyz.max(0) - xyz.min(0)
    lim = [(xyz[:, i].min(), xyz[:, i].max()) for i in range(3)]
    pad = [0.04 * (hi - lo + 1e-6) for lo, hi in lim]
    inact = ~active

    os.makedirs(out_dir, exist_ok=True)
    paths = {}
    for v in VIEW_ORDER:
        fig = plt.figure(figsize=FIGSIZE, dpi=DPI, facecolor=BG)
        ax = fig.add_subplot(111, projection="3d", facecolor=BG)
        ax.scatter(xyz[inact, 0], xyz[inact, 1], xyz[inact, 2], s=INACTIVE_SIZE, c=[GREY],
                   alpha=INACTIVE_ALPHA, linewidths=0, depthshade=False)
        ax.scatter(xyz[active, 0], xyz[active, 1], xyz[active, 2], s=ACTIVE_SIZE,
                   c=col[active], alpha=ACTIVE_ALPHA, linewidths=0, depthshade=False)
        for i, setl in enumerate((ax.set_xlim, ax.set_ylim, ax.set_zlim)):
            setl(lim[i][0] - pad[i], lim[i][1] + pad[i])
        try:
            ax.set_box_aspect(span, zoom=ZOOM)
        except TypeError:
            ax.set_box_aspect(span)
        ax.view_init(elev=VIEWS[v][0], azim=VIEWS[v][1])
        ax.set_axis_off()
        ax.set_title(v, fontsize=8, pad=1)
        p = os.path.join(out_dir, "%s_%s.png" % (prefix, v))
        fig.savefig(p, bbox_inches="tight", pad_inches=0.02, facecolor=BG)
        plt.close(fig)
        paths[v] = p
    return paths, float((a >= tau).mean()), float((b >= tau).mean())


# ------------------------------------------------------------------ sampling

def _stratum(sm, fa, fb):
    """Strata requested 2026-07-28. First match wins, so hold+body is exact."""
    ra, rb = sm["roles"][0], sm["roles"][1]
    tgt = [str(r.get("target", "")).strip().lower() for r in (ra, rb)]
    roles = [str(r.get("role", "")).strip().lower() for r in (ra, rb)]
    body = ["body" in t for t in tgt]
    if any(roles[i] == "hold" and body[i] for i in (0, 1)):
        return "hold+body"
    if roles[0] == roles[1]:
        return "symmetric_pair"
    if any(body):
        return "other_body_target"
    if max(fa, fb) >= 0.60:
        return "large_area"
    return "named_part"


def stratified(dataset, n, pool, tau_ref, seed):
    samples, _ = C.expected_samples(C.load_stage1(dataset), C.STAGE2_ROOT.format(ds=dataset))
    present = [s for s in samples if s["state"] == "present"
               and isinstance(s["roles"], list) and len(s["roles"]) == 2]
    random.seed(seed)
    cand = random.sample(present, min(pool, len(present)))

    buckets = collections.defaultdict(list)
    for sm in cand:
        try:
            with np.load(os.path.join(sm["qdir"], "scores.npz"), allow_pickle=False) as z:
                fa = float((z["scoreA"] >= tau_ref).mean())
                fb = float((z["scoreB"] >= tau_ref).mean())
        except Exception:
            continue
        sm["active_A"], sm["active_B"] = fa, fb
        buckets[_stratum(sm, fa, fb)].append(sm)
        # large_area is a cross-cutting property, so also collect it separately
        if max(fa, fb) >= 0.60:
            buckets["large_area"].append(sm)

    per = max(1, n // max(1, len(buckets)))
    out, seen = [], set()
    for k in sorted(buckets):
        for sm in random.sample(buckets[k], min(per, len(buckets[k]))):
            if sm["sample_id"] in seen:
                continue
            seen.add(sm["sample_id"])
            sm["stratum"] = k
            out.append(sm)
    print("[render] strata: %s" % {k: len(v) for k, v in sorted(buckets.items())})
    return out[:n]


# ------------------------------------------------------------------ gallery

def gallery(rows, taus, out_html, dataset):
    h = ["<meta charset='utf-8'><title>dual-affordance heatmap pilot</title>",
         "<style>body{font-family:system-ui;margin:18px;background:#fff}"
         "h2{font-size:15px;margin:22px 0 4px}.meta{font-size:12px;color:#444;margin-bottom:6px}"
         ".r{display:flex;gap:4px;flex-wrap:nowrap;margin-bottom:6px;align-items:flex-start}"
         ".r img{width:190px;border:1px solid #e3e3e3}"
         ".tau{font-size:12px;color:#0a58ca;width:64px;flex:0 0 64px;padding-top:60px}"
         "code{background:#f4f4f4;padding:1px 4px}"
         ".lg{font-size:12px;color:#555;margin-bottom:10px}</style>",
         "<h1 style='font-size:17px'>Combined heatmap pilot &mdash; %s</h1>" % dataset,
         "<div class=lg>A = red &middot; B = blue &middot; overlap = purple &middot; "
         "inactive = grey. Mask = <code>score &ge; &tau;</code>, same &tau; for both hands "
         "(frozen &tau;<sub>vis</sub> = %.2f); intensity = original continuous score [0,1]."
         "</div>" % TAU_VIS]
    for r in rows:
        h.append("<h2>%s <span style='color:#888;font-weight:400'>[%s]</span></h2>"
                 % (r["sample_id"], r["stratum"]))
        h.append("<div class=meta>%s &nbsp;|&nbsp; <b>A</b> %s &middot; %s &middot; %s"
                 " &nbsp;|&nbsp; <b>B</b> %s &middot; %s &middot; %s</div>"
                 % (r["task"], r["A"]["role"], r["A"]["target"], r["A"]["contact_region"],
                    r["B"]["role"], r["B"]["target"], r["B"]["contact_region"]))
        for t in taus:
            imgs = r["renders"][str(t)]
            frac = r["active"][str(t)]
            h.append("<div class=r><div class=tau>&tau;=%.2f<br><span style='color:#888'>"
                     "A %.2f<br>B %.2f</span></div>%s</div>"
                     % (t, frac[0], frac[1],
                        "".join("<img src='%s'>" % os.path.relpath(imgs[v], os.path.dirname(out_html))
                                for v in VIEW_ORDER)))
    with open(out_html, "w") as f:
        f.write("\n".join(h))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="daily_used")
    ap.add_argument("--stratified", type=int, default=40)
    ap.add_argument("--pool", type=int, default=1200, help="candidates scanned before stratifying")
    ap.add_argument("--tau", type=float, nargs="+", default=[TAU_VIS])
    ap.add_argument("--intensity", default="linear", choices=["linear", "gamma"],
                    help="linear = frozen protocol (original score); gamma = legibility check")
    ap.add_argument("--tau_ref", type=float, default=TAU_VIS, help="tau used only to define strata")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    out = a.out or os.path.join(C.REPO, "data_verification/quality_evaluation/pilot_render", a.dataset)
    os.makedirs(out, exist_ok=True)
    picks = stratified(a.dataset, a.stratified, a.pool, a.tau_ref, a.seed)
    print("[render] %d samples x %d tau x 6 views = %d images"
          % (len(picks), len(a.tau), len(picks) * len(a.tau) * 6))

    rows = []
    for i, sm in enumerate(picks):
        oid, qn = sm["sample_id"].split("/", 1)
        d = os.path.join(out, "img", oid)
        row = {"sample_id": sm["sample_id"], "stratum": sm["stratum"], "task": sm["task"],
               "A": sm["roles"][0], "B": sm["roles"][1], "renders": {}, "active": {}}
        for t in a.tau:
            paths, fa, fb = render_views(sm["qdir"], d, t,
                                         "%s_%s_t%03d" % (qn[:42], a.intensity, int(t * 100)),
                                         a.intensity)
            row["renders"][str(t)], row["active"][str(t)] = paths, (fa, fb)
        rows.append(row)
        if (i + 1) % 10 == 0:
            print("  %d/%d" % (i + 1, len(picks)))

    html = os.path.join(out, "gallery.html")
    gallery(rows, a.tau, html, a.dataset)
    with open(os.path.join(out, "samples.json"), "w") as f:
        json.dump([{k: r[k] for k in ("sample_id", "stratum", "task", "A", "B", "active")}
                   for r in rows], f, indent=1)
    with open(os.path.join(out, "render_protocol.json"), "w") as f:
        json.dump({"rendering_version": RENDER_VERSION, "mask_method": "absolute_threshold",
                   "threshold": TAU_VIS, "threshold_frozen": "2026-07-28",
                   "taus_rendered": a.tau, "per_hand_threshold": False,
                   "intensity_mapping": (INTENSITY_MAPPING if a.intensity == "linear"
                                         else {"type": "gamma+floor (comparison aid, NOT frozen)",
                                               "gamma": GAMMA, "floor": FLOOR}),
                   "camera_poses": VIEWS, "view_order": VIEW_ORDER,
                   "point_size": {"inactive": INACTIVE_SIZE, "active": ACTIVE_SIZE},
                   "alpha": {"inactive": INACTIVE_ALPHA, "active": ACTIVE_ALPHA},
                   "background": BG, "zoom": ZOOM, "figsize": FIGSIZE, "dpi": DPI,
                   "object_normalization": "none (xyz as stored); Y-up permutation [0,2,1]",
                   "crop_rule": "bbox + 4% pad, box_aspect=span",
                   "colors": {"A": RED.tolist(), "B": BLUE.tolist(),
                              "overlap": PURPLE.tolist(), "inactive": GREY.tolist()},
                   "software": "matplotlib %s" % matplotlib.__version__}, f, indent=1)
    print("[render] %s" % html)


if __name__ == "__main__":
    main()
