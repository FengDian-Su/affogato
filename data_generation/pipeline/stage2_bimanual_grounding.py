#!/usr/bin/env python
"""
stage2_bimanual_grounding.py  (stage2 **v1** — superseded as a runner)
=====================

STATUS: the production runner is ``pipeline/stage2_v2.py`` (Molmo2-8B on vLLM +
object-batched SAM2). This file stays alive as the shared LIBRARY for object/
scene loading — build_aff_map / resolve_object / load_canvas / load_scene are
imported by stage2_v2 and the notebook.

Batch **bimanual** 3D-affordance grounding, packaged from the step-by-step
notebook ``notebook/stage02_walkthrough.ipynb`` (formerly
``molmo_sam_grounding_mailbox.ipynb``) into structured, callable functions + a CLI.

For one object + one of its stage-1 *bimanual tasks* (a two-role coordinated
manipulation), this grounds **both hands' contact regions** on the affogato
canvas point cloud:

    for each role's "Point to ..." query:
        Molmo (point per view) -> SAM2 (mask -> sigmoid heatmap)
        -> multi-view voting onto the affogato canvas -> per-point [0,1] score
    then partition_two_roles() splits the two role heatmaps into two DISTINCT
    regions (symmetric co-lift -> union+split; different parts -> make disjoint).

All the validated geometry / Molmo / SAM2 / voting / partition code lives in
``single_region_affordance.py`` (imported as ``sra``); this file only adds object
resolution, batch orchestration, and **persistence of the intermediates** (the
notebook's ``.show()`` calls are replaced by saving images + arrays + HTML).

Per (object, task) it writes::

    <output_dir>/<object_id>/q<idx>_<task_slug>/
      meta.json               task/query text, roles, molmo_queries, cfg, coverage, score stats
      roleA/molmo/view00.png ...  per view: original image + red X at the Molmo point
      roleA/sam2/view00.png  ...  per view: original image + jet SAM2 heatmap overlay
      roleB/molmo/...  roleB/sam2/...
      scores.npz              xyz, gt, counts, scoreA_raw/scoreB_raw (pre-split),
                              scoreA/scoreB (post-split), molmo_queries, task
      cloud_two_role.html      two-role 3D figure AFTER partition (the two distinct hand regions)

Object resolution follows the notebook's validated path:
  * canvas dir (xyzc.npy + queries.json): daily_used_to_affogato.json, object_id -> dst
  * gObjaverse render dir:                 rec["views_used"][0], minus the last two path parts

Run with the repo's ``mm`` conda env, e.g.::

    python pipeline/stage2_bimanual_grounding.py --start 0 --end 5 --gpu 2
    python pipeline/stage2_bimanual_grounding.py --object_id 849b4573e03a40b8b073a2209d1bc1d3 --query_idx 0
"""

import os
# Must be set before cv2 (depth EXR) / torch CUDA init. Importing `sra` below pulls
# in cv2 + torch, so set these first. (sra also setdefault's them, kept here for clarity.)
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import re
import sys
import json
import argparse
import traceback
from dataclasses import dataclass, asdict

import numpy as np
import cv2
import plotly.graph_objects as go

# Resolve paths relative to the data_generation root regardless of cwd, and make
# `single_region_affordance` importable whether run as a script or a module.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))     # .../data_generation/pipeline
DG = os.path.dirname(SCRIPT_DIR)                            # .../data_generation
# single_region_affordance.py was moved out of pipeline/ into data_generation/single_region/;
# add that dir (and keep SCRIPT_DIR as a fallback) so the import resolves from either layout.
sys.path.insert(0, os.path.join(DG, "single_region"))
sys.path.insert(0, SCRIPT_DIR)

import single_region_affordance as sra      # noqa: E402  (validated geometry/Molmo/SAM2/voting fns)


# ==============================================================================
# Two-role 3D figure  (ported verbatim from the notebook, returns a fig to save)
# ==============================================================================

ORANGE,  TIFFANY  = "#F28E2B", "#0ABAB5"                # solid hues for the labels
ORANGE_SCALE  = [[0.0, "#FBDDB6"], [1.0, ORANGE]]       # light tint -> orange   (role A)
TIFFANY_SCALE = [[0.0, "#BFE9E6"], [1.0, TIFFANY]]      # light tint -> tiffany  (role B)


def two_role_fig(xyz, sA, sB, labelA, labelB, task_query, thr=0.15):
    """Both grounded hand regions on the canvas: A=orange, B=tiffany, object=grey."""
    s = slice(None, None, max(1, len(xyz) // 60000))
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=xyz[s, 0], y=xyz[s, 1], z=xyz[s, 2], mode="markers",
        marker=dict(size=1.2, color="lightgrey", opacity=0.25), name="object", hoverinfo="skip"))
    for sc, name, scale in [(sA, labelA, ORANGE_SCALE), (sB, labelB, TIFFANY_SCALE)]:
        m = sc[s] > thr
        fig.add_trace(go.Scatter3d(x=xyz[s][m, 0], y=xyz[s][m, 1], z=xyz[s][m, 2], mode="markers",
            marker=dict(size=1.2, color=sc[s][m], colorscale=scale, cmin=thr,
                        cmax=max(thr + 0.05, float(sc.max())), opacity=0.9), name=name))

    title = (f'<b>Task query:</b> "{task_query}"<br>'
             + f"<span style='color:{ORANGE}'>Role A - {labelA}</span><br>"
             + f"<span style='color:{TIFFANY}'>Role B - {labelB}</span>")
    fig.add_annotation(text=title, xref="paper", yref="paper",
                       x=0.5, xanchor="center", y=1.02, yanchor="bottom",
                       align="left", showarrow=False, font=dict(size=13))
    fig.update_layout(width=860, height=720, scene=dict(aspectmode="data"),
                      margin=dict(l=0, r=0, t=100, b=0), showlegend=False)
    return fig


# ==============================================================================
# Object resolution + canvas/scene loading
# ==============================================================================

@dataclass
class Canvas:
    xyz: np.ndarray          # [N,3] affogato canvas points (original frame, for plotting/partition)
    xyz_vote: np.ndarray     # [N,3] aligned to gObjaverse camera frame (for projecting/voting)
    gt: np.ndarray           # [N,K] affogato GT affordance channels (reference only)


@dataclass
class Scene:
    n_views: int
    view_images: list        # list[PIL.Image]
    view_images_np: list     # list[np.ndarray]
    cameras: list            # list[(c2w, w2c)]
    K_list: list
    depth_maps: list


def build_aff_map(mapping_path):
    """object_id -> affogato canvas dir (dst), from daily_used_to_affogato.json."""
    with open(mapping_path, "r") as f:
        mapping = json.load(f)
    return {e["object_id"]: e["dst"] for e in mapping}


def resolve_object(rec, aff_map):
    """(gObjaverse render dir, affogato canvas dir) for one stage-1 record."""
    obj_root = "/".join(rec["views_used"][0].split("/")[:-2])   # drop /<view>/<view>.png
    aff_dir = aff_map.get(rec["object_id"])
    return obj_root, aff_dir


def load_canvas(aff_dir):
    """Load xyzc.npy (+ align frame) from an affogato canvas dir."""
    xyzc = np.load(os.path.join(aff_dir, "xyzc.npy")).astype(np.float32)
    xyz, gt = xyzc[:, :3], xyzc[:, 3:]
    # affogato points use a different axis convention than the gObjaverse cameras;
    # align once (swap Y/Z, flip new Y) before projecting/voting -- matches
    # sra.process_object / point_cloud_from_depth.ipynb. gt[i] stays attached to point i.
    xyz_vote = sra.align_affogato_frame(xyz)
    return Canvas(xyz=xyz, xyz_vote=xyz_vote, gt=gt)


def load_scene(obj_root, num_views):
    """Camera params + RGB views for the (already clamped) view count."""
    cameras, K_list, depth_maps = sra.prepare_camera_params(obj_root, num_views)
    view_images, view_images_np = sra.load_view_images(obj_root, num_views)
    return Scene(n_views=num_views, view_images=view_images, view_images_np=view_images_np,
                 cameras=cameras, K_list=K_list, depth_maps=depth_maps)


# ==============================================================================
# Grounding one role + saving intermediates
# ==============================================================================

def ground_role(molmo_query, scene: Scene, xyz_vote, models, cfg):
    """Full chain for one 'Point to ...' phrase. Returns (pts_per_view, heatmaps, scores, counts)."""
    pts, heatmaps, scores, counts = sra.ground_single_query(
        molmo_query, scene.view_images, scene.view_images_np, xyz_vote,
        scene.cameras, scene.K_list, scene.depth_maps, models, cfg)
    hit = sum(len(p) > 0 for p in pts)
    print(f"  Molmo hit {hit}/{scene.n_views} | score [{scores.min():.3f}, {scores.max():.3f}] "
          f"mean {scores.mean():.3f} | coverage {(counts > 0).mean() * 100:.0f}%")
    return pts, heatmaps, scores, counts


def _point_overlay(img_rgb, pts):
    """Original image with a red X drawn at each Molmo point (native resolution, no chrome)."""
    out = img_rgb.copy()
    H, W = out.shape[:2]
    r = max(8, int(0.015 * max(H, W)))     # marker size scales with image
    t = max(2, int(0.005 * max(H, W)))
    for (x, y) in pts:
        x, y = int(round(float(x))), int(round(float(y)))
        cv2.line(out, (x - r, y - r), (x + r, y + r), (255, 0, 0), t, cv2.LINE_AA)
        cv2.line(out, (x - r, y + r), (x + r, y - r), (255, 0, 0), t, cv2.LINE_AA)
    return out


def _heatmap_overlay(img_rgb, hm, alpha=0.5):
    """Original image with the SAM2 [0,1] heatmap jet-overlaid (matches the notebook's alpha=0.5)."""
    h8 = (np.clip(hm, 0.0, 1.0) * 255).astype(np.uint8)
    jet_rgb = cv2.cvtColor(cv2.applyColorMap(h8, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(img_rgb, 1.0 - alpha, jet_rgb, alpha, 0.0)


def save_view_overlays(role_dir, scene: Scene, pts_per_view, heatmaps):
    """Per view, save two plain PNGs: <role>/molmo/viewNN.png and <role>/sam2/viewNN.png."""
    molmo_dir = os.path.join(role_dir, "molmo")
    sam2_dir = os.path.join(role_dir, "sam2")
    os.makedirs(molmo_dir, exist_ok=True)
    os.makedirs(sam2_dir, exist_ok=True)
    for vi in range(scene.n_views):
        img = scene.view_images_np[vi]
        molmo_img = _point_overlay(img, pts_per_view[vi])
        hm = heatmaps[vi]
        # empty heatmap (view with no Molmo point) -> save the plain image, not an all-blue tint
        sam2_img = _heatmap_overlay(img, hm) if (hm is not None and float(hm.max()) > 1e-6) else img
        cv2.imwrite(os.path.join(molmo_dir, f"view{vi:02d}.png"),
                    cv2.cvtColor(molmo_img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(sam2_dir, f"view{vi:02d}.png"),
                    cv2.cvtColor(sam2_img, cv2.COLOR_RGB2BGR))


def _slugify(s, maxlen=40):
    s = re.sub(r"[^a-zA-Z0-9]+", "_", str(s).strip().lower()).strip("_")
    return s[:maxlen] or "task"


def _stats(scores):
    return {"min": float(scores.min()), "max": float(scores.max()), "mean": float(scores.mean())}


# ==============================================================================
# Grounding one bimanual task (both roles + partition + persist)
# ==============================================================================

def ground_task(rec, qi, scene: Scene, canvas: Canvas, models, cfg, task_dir):
    """Ground both roles of one bimanual task, partition them, and save all intermediates."""
    query = rec["queries"][qi]
    roles, mqs = query["roles"], query["molmo_queries"]
    if len(roles) != 2 or len(mqs) < 2:
        print(f"  [skip] task {qi} is not two-role ({len(roles)} roles); bimanual grounding needs 2")
        return
    os.makedirs(task_dir, exist_ok=True)
    roleA, roleB = roles[0], roles[1]
    mqA, mqB = mqs[0], mqs[1]
    labelA = f"{roleA['role']} @ {roleA['contact_region']}"
    labelB = f"{roleB['role']} @ {roleB['contact_region']}"

    print(f"\n{'-'*60}\n[{rec['object_id']}] q{qi}: {query['task']}\n  A: {labelA}\n  B: {labelB}\n{'-'*60}")

    grounded = {}
    for tag, mq in (("roleA", mqA), ("roleB", mqB)):
        print(f"{tag}:", mq)
        pts, hm, score, counts = ground_role(mq, scene, canvas.xyz_vote, models, cfg)
        save_view_overlays(os.path.join(task_dir, tag), scene, pts, hm)
        grounded[tag] = (score, counts)
    scoreA, countsA = grounded["roleA"]
    scoreB, _ = grounded["roleB"]

    # split the two (independently grounded) role heatmaps into two distinct regions
    scoreA_p, scoreB_p = sra.partition_two_roles(scoreA, scoreB, roleA, roleB, canvas.xyz)

    # 3D figure of the two distinct hand regions (post-partition)
    two_role_fig(canvas.xyz, scoreA_p, scoreB_p, labelA, labelB, query["query"]).write_html(
        os.path.join(task_dir, "cloud_two_role.html"))

    # per-point arrays (counts is geometry-only -> identical across roles)
    np.savez_compressed(
        os.path.join(task_dir, "scores.npz"),
        xyz=canvas.xyz.astype(np.float32),
        gt=canvas.gt.astype(np.float32),
        counts=countsA.astype(np.int32),
        scoreA_raw=scoreA.astype(np.float32), scoreB_raw=scoreB.astype(np.float32),
        scoreA=scoreA_p.astype(np.float32), scoreB=scoreB_p.astype(np.float32),
        molmo_queries=np.array([mqA, mqB], dtype=object),
        task=query["task"],
    )

    meta = {
        "object_id": rec["object_id"],
        "object_name": rec.get("object_name", ""),
        "query_idx": qi,
        "task": query["task"],
        "query": query["query"],
        "category": query.get("category", ""),
        "roles": [{k: r.get(k) for k in ("id", "role", "target", "contact_region", "function")}
                  for r in roles],
        "molmo_queries": [mqA, mqB],
        "n_views_used": scene.n_views,
        "coverage": float((countsA > 0).mean()),
        "score_stats": {"roleA_raw": _stats(scoreA), "roleB_raw": _stats(scoreB),
                        "roleA": _stats(scoreA_p), "roleB": _stats(scoreB_p)},
        "config": asdict(cfg),
    }
    with open(os.path.join(task_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"  [saved] {task_dir}")


def process_object(rec, obj_root, aff_dir, models, cfg, base_out, query_idx=None, skip_existing=False):
    """Load this object's canvas + scene once, then ground the chosen task(s)."""
    canvas = load_canvas(aff_dir)
    avail = sra.count_available_views(obj_root)
    n_views = min(cfg.num_views, avail)
    if n_views == 0:
        raise RuntimeError(f"no usable views in {obj_root}")
    if n_views < cfg.num_views:
        print(f"  [warn] only {avail} views available; using {n_views}")
    scene = load_scene(obj_root, n_views)

    qidxs = [query_idx] if query_idx is not None else range(len(rec["queries"]))
    for qi in qidxs:
        if qi >= len(rec["queries"]):
            print(f"  [skip] query_idx {qi} out of range ({len(rec['queries'])} tasks)")
            continue
        task_dir = os.path.join(base_out, rec["object_id"],
                                f"q{qi}_{_slugify(rec['queries'][qi]['task'])}")
        if skip_existing and os.path.exists(os.path.join(task_dir, "scores.npz")):
            print(f"  [skip-existing] {task_dir}")
            continue
        ground_task(rec, qi, scene, canvas, models, cfg, task_dir)


# ==============================================================================
# CLI
# ==============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Batch bimanual 3D-affordance grounding")
    p.add_argument("--stage1", default="outputs/stage1_v2/stage1_part0.json",
                   help="stage-1 dataset json (bimanual tasks + per-role molmo_queries)")
    p.add_argument("--mapping", default="dataset/daily_used_to_affogato.json",
                   help="object_id -> affogato canvas dir mapping (dst)")
    p.add_argument("--output_dir", default="outputs/bimanual_grounding")
    # selection
    p.add_argument("--start", type=int, default=0, help="stage-1 record index (inclusive)")
    p.add_argument("--end", type=int, default=5, help="stage-1 record index (exclusive)")
    p.add_argument("--object_id", default=None, help="ground only this object (overrides start/end)")
    p.add_argument("--query_idx", type=int, default=None,
                   help="ground only this task per object (default: all tasks)")
    p.add_argument("--skip_existing", action="store_true",
                   help="skip a task if its scores.npz already exists")
    # runtime / cfg knobs
    p.add_argument("--gpu", type=int, default=2)
    p.add_argument("--num_views", type=int, default=40)
    p.add_argument("--depth_tolerance", type=float, default=0.15)
    p.add_argument("--use_gaussian", action="store_true",
                   help="multiply SAM2 sigmoid by a Gaussian around the Molmo point (off = notebook default)")
    p.add_argument("--gaussian_sigma", type=float, default=77.0)
    p.add_argument("--mask_select", default="best_iou", choices=["best_iou", "smallest"])
    p.add_argument("--molmo_model_id", default="allenai/MolmoPoint-8B")
    p.add_argument("--sam2_checkpoint", default="checkpoints/sam2.1_hiera_large.pt")
    p.add_argument("--sam2_model_cfg", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    return p.parse_args()


def main():
    args = parse_args()
    os.chdir(DG)   # so sra's relative checkpoints/ + configs/ + default data paths resolve

    # Pin the chosen physical GPU BEFORE any CUDA init (device order set at import).
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = sra.setup_device()
    cfg = sra.PipelineConfig(
        molmo_model_id=args.molmo_model_id,
        sam2_checkpoint=args.sam2_checkpoint,
        sam2_model_cfg=args.sam2_model_cfg,
        num_views=args.num_views,
        depth_tolerance=args.depth_tolerance,
        use_gaussian=args.use_gaussian,
        gaussian_sigma=args.gaussian_sigma,
        mask_select=args.mask_select,
        device=device,
    )
    print("Config:", json.dumps(asdict(cfg), indent=2))

    models = sra.load_models(cfg)
    aff_map = build_aff_map(args.mapping)
    with open(args.stage1, "r") as f:
        recs = json.load(f)

    if args.object_id:
        selected = [r for r in recs if r["object_id"] == args.object_id]
        if not selected:
            print(f"[error] object_id {args.object_id} not in {args.stage1}")
            return
    else:
        selected = recs[args.start:args.end]

    n_done = n_skip = n_fail = 0
    for rec in selected:
        oid = rec["object_id"]
        obj_root, aff_dir = resolve_object(rec, aff_map)
        # silently skip objects whose renders / canvas aren't downloaded locally
        if aff_dir is None or not (os.path.isdir(obj_root)
                                   and os.path.exists(os.path.join(aff_dir, "xyzc.npy"))):
            print(f"[skip] {oid}: missing renders or canvas locally")
            n_skip += 1
            continue
        print(f"\n{'='*70}\nObject {oid}  ({rec.get('object_name', '')})  "
              f"[done {n_done}]\n  renders: {obj_root}\n  canvas : {aff_dir}\n{'='*70}")
        try:
            process_object(rec, obj_root, aff_dir, models, cfg, args.output_dir,
                           args.query_idx, args.skip_existing)
            n_done += 1
        except Exception as e:
            traceback.print_exc()
            print(f"  [fail] {oid}: {e}")
            n_fail += 1

    print(f"\n===== DONE =====\n  objects done: {n_done}  skip: {n_skip}  fail: {n_fail}")


if __name__ == "__main__":
    main()
