#!/usr/bin/env python
"""
single_region_affordance.py — shared library for the stage2 grounding pipelines.

Consumers: pipeline/stage2_v2.py (the production runner),
pipeline/stage2_resegment.py, and notebook/stage02_walkthrough.ipynb. Treat every public function as having
external callers.

Contents:
  * PipelineConfig — SAM2 checkpoint, projection tolerance, device.
  * G-Objaverse geometry / IO (verbatim from run_pipeline.py — proven path):
    intrinsics, camera poses, EXR depth, view loading, affogato->camera frame
    alignment.
  * SAM2: load_sam2_model, run_sam2_object_queries (official set_image_batch /
    predict_batch, encoder once per object; per-query mask_selects + partner
    negative points — the "mixed+neg" prompt recipe).
  * Multi-view voting: precompute_projection + sample_heatmaps_projected.
  * Canvas refinement + two-role partition (validated 07-15 on 195 queries /
    50 category-diverse objects): knn_indices, cc_prune, refine_scores,
    partition_two_roles, prune_partitioned.
  * Stage2 object/scene loading: Canvas / Scene, build_aff_map, resolve_object,
    load_canvas, load_scene.

History: this file began as the standalone single-region AFFOGATO reproduction
CLI (MolmoPoint + SAM2 serial chain, optional gaussian shaping). That chain and
the CLI were retired 07-15 (production pointing is Molmo2-8B on vLLM in
single_region/molmo2_vllm/); see git history for the code and
notes/affogato_validation_*.md for the findings it produced.
"""

import os
# Must be set before importing cv2 so OpenEXR depth (*_nd.exr) can be read.
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
# Make CUDA device indices match `nvidia-smi` (PCI order, not fastest-first).
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import re
import json
from dataclasses import dataclass

import numpy as np
import cv2
import tqdm
from PIL import Image


# ==============================================================================
# Config
# ==============================================================================

@dataclass
class PipelineConfig:
    # --- models ---
    sam2_checkpoint: str = "checkpoints/sam2.1_hiera_large.pt"
    sam2_model_cfg: str = "configs/sam2.1/sam2.1_hiera_l.yaml"

    # --- views / geometry ---
    depth_tolerance: float = 0.15      # relative depth tol for visibility test

    # --- runtime ---
    device: str = "cuda:0"


# ==============================================================================
# Geometry / IO  (copied from run_pipeline.py — proven path)
# ==============================================================================

def get_intrinsic_matrix(height, width):
    """G-Objaverse fixed focal length fx=fy=1422.222 at 1024x1024, principal point = centre."""
    fx = fy = 1422.222
    res_raw = 1024
    f_x = f_y = fx * height / res_raw
    cx = width / 2
    cy = height / 2
    return np.array([[f_x, 0, cx], [0, f_y, cy], [0, 0, 1]], dtype=np.float32)


def read_camera_matrix(json_file):
    """Read camera-to-world (c2w); Y and Z axes are flipped to match the render convention."""
    with open(json_file, "r", encoding="utf8") as f:
        data = json.load(f)
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, 0] = np.array(data["x"])
    c2w[:3, 1] = -np.array(data["y"])
    c2w[:3, 2] = -np.array(data["z"])
    c2w[:3, 3] = np.array(data["origin"])
    return c2w


def convert_pose_flip_yz(c2w):
    """Flip Y and Z axes (coordinate-system convention)."""
    flip_yz = np.eye(4)
    flip_yz[1, 1] = -1
    flip_yz[2, 2] = -1
    return c2w @ flip_yz


def read_depth_from_exr(exr_path, camera_position):
    """*_nd.exr: channels 0-2 = world normal, channel 3 = depth. Near-plane invalid -> 0."""
    cam_distance = np.linalg.norm(camera_position)
    near = 0.867  # sqrt(3) * 0.5 ; object normalised into unit sphere
    near_distance = cam_distance - near
    normald = cv2.imread(exr_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
    depth = normald[..., 3]
    depth[depth < near_distance] = 0
    return depth, normald[..., :3]


def count_available_views(img_folder_path):
    """Number of consecutive views 0,1,2,... that have a full {png,nd.exr,json} triplet."""
    n = 0
    while True:
        d = os.path.join(img_folder_path, f"{n:05d}")
        png = os.path.join(d, f"{n:05d}.png")
        exr = os.path.join(d, f"{n:05d}_nd.exr")
        js = os.path.join(d, f"{n:05d}.json")
        if not (os.path.exists(png) and os.path.exists(exr) and os.path.exists(js)):
            break
        n += 1
    return n


def prepare_camera_params(img_folder_path, num_views):
    """Per-view (c2w, w2c), intrinsics K, and depth map — for voting aggregation."""
    cameras, K_list, depth_maps = [], [], []
    for view_idx in tqdm.trange(num_views, desc="Preparing camera matrices"):
        json_path = f"{img_folder_path}/{view_idx:05d}/{view_idx:05d}.json"
        depth_path = f"{img_folder_path}/{view_idx:05d}/{view_idx:05d}_nd.exr"

        c2w = read_camera_matrix(json_path)
        c2w_converted = convert_pose_flip_yz(c2w)
        w2c = np.linalg.inv(c2w_converted)

        camera_pos = c2w[:3, 3]
        depth, _ = read_depth_from_exr(depth_path, camera_pos)

        H, W = depth.shape
        K = get_intrinsic_matrix(H, W)

        cameras.append((c2w, w2c))
        K_list.append(K)
        depth_maps.append(depth)
    return cameras, K_list, depth_maps


def load_view_images(img_folder_path, num_views):
    """Load num_views RGB images as (list[PIL.Image], list[np.ndarray])."""
    view_images, view_images_np = [], []
    for view_idx in range(num_views):
        img_path = f"{img_folder_path}/{view_idx:05d}/{view_idx:05d}.png"
        img = Image.open(img_path).convert("RGB")
        view_images.append(img)
        view_images_np.append(np.array(img))
    return view_images, view_images_np


def align_affogato_frame(xyz):
    """AFFOGATO canvas -> G-Objaverse camera frame: swap Y/Z, then flip the new Y.

    Matches point_cloud_from_depth.ipynb (CELL 17). Row order is preserved, so
    gt[i] stays attached to point i."""
    out = xyz[:, [0, 2, 1]].copy()
    out[:, 1] *= -1
    return out


# ==============================================================================
# SAM2
# ==============================================================================

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def load_sam2_model(checkpoint, model_cfg, device):
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    print(f"Loading SAM2: {checkpoint} ...")
    sam2_model = build_sam2(model_cfg, checkpoint, device=device)
    return SAM2ImagePredictor(sam2_model)


def _select_mask_idx(masks, iou_scores, mask_select):
    """Pick one of SAM2's 3 candidate masks. Granularities by area:
    middle ~ part, largest ~ whole visible face; best_iou = SAM2's own
    predicted-quality argmax."""
    if mask_select in ("middle", "largest"):
        order = np.argsort([(m > 0).sum() for m in masks])
        return int(order[{"middle": 1, "largest": 2}[mask_select]])
    return int(np.argmax(iou_scores))


def run_sam2_object_queries(view_images_np, queries_points, sam2_predictor, cfg: PipelineConfig,
                            neg_points=None, mask_selects=None):
    """SAM2 for ALL queries of one view window, encoder run once per window.

    Official batch API: set_image_batch embeds every view a single time, then
    predict_batch (decoder-only, ~ms/view) runs per query.

    queries_points: list over queries of points_per_view (list[T] of [K,2] arrays).
    neg_points:     optional, same nesting: one negative point (or None) per
                    query per view, passed to SAM with label 0 (the partner
                    role's point separates e.g. mug body from handle masks).
    mask_selects:   per-query candidate selection (required).
    Returns: list over queries of heatmap lists (list[T] of [H,W] float32).
    """
    T = len(view_images_np)
    H, W = view_images_np[0].shape[:2]
    zero = np.zeros((H, W), dtype=np.float32)
    dummy_pt = np.zeros((1, 2), dtype=np.float32)   # empty views get a dummy prompt, output discarded
    sam2_predictor.set_image_batch(list(view_images_np))
    out = []
    for qi, points_per_view in enumerate(queries_points):
        coords, labels = [], []
        for vi, pts in enumerate(points_per_view):
            arr = np.array(pts, dtype=np.float32) if len(pts) else dummy_pt
            lab = np.ones(len(arr), dtype=np.int32)
            neg = neg_points[qi][vi] if (neg_points is not None and len(pts)) else None
            if neg is not None:
                arr = np.concatenate([arr, np.asarray(neg, dtype=np.float32).reshape(1, 2)])
                lab = np.concatenate([lab, np.zeros(1, dtype=np.int32)])
            coords.append(arr)
            labels.append(lab)
        masks_b, ious_b, _ = sam2_predictor.predict_batch(
            point_coords_batch=coords, point_labels_batch=labels,
            multimask_output=True, return_logits=True)
        select = mask_selects[qi]
        hms = []
        for vi in range(T):
            if len(points_per_view[vi]) == 0:
                hms.append(zero)
                continue
            best_idx = _select_mask_idx(masks_b[vi], ious_b[vi], select)
            hms.append(sigmoid(masks_b[vi][best_idx]).astype(np.float32))
        out.append(hms)
    sam2_predictor.reset_predictor()
    return out


# ==============================================================================
# Multi-view voting
# ==============================================================================

def precompute_projection(points, cameras, K_list, depths, depth_tolerance=0.05):
    """Per-object geometry cache for the voting projection: (u_int, v_int, valid)
    per view — depends only on the object, shared by every role/query."""
    N = len(points)
    points_homo = np.hstack([points, np.ones((N, 1))])
    proj = []
    for view_idx in range(len(depths)):
        _, w2c = cameras[view_idx]
        K = K_list[view_idx]
        depth_map = depths[view_idx]
        H, W = depth_map.shape
        cam_xyz = (w2c @ points_homo.T).T[:, :3]
        in_front = cam_xyz[:, 2] > 0
        pixel_homo = (K @ cam_xyz.T).T
        pixel_uv = pixel_homo[:, :2] / pixel_homo[:, 2:3]
        u, v = pixel_uv[:, 0], pixel_uv[:, 1]
        in_bounds = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        u_int = np.clip(u.astype(np.int32), 0, W - 1)
        v_int = np.clip(v.astype(np.int32), 0, H - 1)
        sampled_depth = depth_map[v_int, u_int]
        visible = (sampled_depth > 0) & (cam_xyz[:, 2] < sampled_depth * (1 + depth_tolerance))
        proj.append((u_int, v_int, in_front & in_bounds & visible))
    return proj


def merge_instance_masks(masks, thr=0.5):
    """Merge one role's per-point SAM masks for ONE view.

    Points on the SAME instance produce near-identical masks whose stray
    fringes differ — a plain max-union keeps every fringe and only ever
    grows (measured +28% named-part support). Points on DIFFERENT instances
    (straws) produce disjoint masks that must union. So: group masks by
    overlap (IoU>0.5 on the >thr support = same instance), MEAN within a
    group (consensus damps the fringes below threshold), MAX across groups.
    """
    if len(masks) == 1:
        return masks[0]
    sup = [m > thr for m in masks]
    groups = []
    for i, s in enumerate(sup):
        for g in groups:
            r = sup[g[0]]
            inter = (s & r).sum()
            if inter and inter / max((s | r).sum(), 1) > 0.5:
                g.append(i)
                break
        else:
            groups.append([i])
    merged = [np.mean([masks[i] for i in g], axis=0) for g in groups]
    return merged[0] if len(merged) == 1 else np.maximum.reduce(merged)


def resolve_2d_overlap(hmA, hmB, ptsA, ptsB, thr=0.5):
    """Cross-role exclusivity at the 2D mask level, IN PLACE, one view.

    Pixels claimed by BOTH roles go to the role whose own prompt point is
    nearer (min over that role's points in this view); the loser's heatmap is
    zeroed there, so the 3D vote never receives double-claimed evidence.
    No-op unless both roles actually prompted this view. Assignment is by
    prompt-point distance, not mask value (the two roles' sigmoid logits come
    from different candidate selections and are not mutually calibrated); a
    stray point can locally win pixels, so the existence gate + per-view point
    dedup upstream matter.
    """
    if not len(ptsA) or not len(ptsB):
        return
    both = (hmA > thr) & (hmB > thr)
    if not both.any():
        return
    ys, xs = np.nonzero(both)
    P = np.stack([xs, ys], 1).astype(np.float32)
    dA = np.min([np.linalg.norm(P - np.asarray(p, dtype=np.float32).reshape(1, 2), axis=1)
                 for p in ptsA], axis=0)
    dB = np.min([np.linalg.norm(P - np.asarray(p, dtype=np.float32).reshape(1, 2), axis=1)
                 for p in ptsB], axis=0)
    a_wins = dA <= dB
    hmB[ys[a_wins], xs[a_wins]] = 0.0
    hmA[ys[~a_wins], xs[~a_wins]] = 0.0


def sample_heatmaps_projected(proj, heatmaps):
    """Multi-view voting with a precomputed projection: average each point's
    heatmap samples over the views that see it."""
    N = len(proj[0][0])
    score_sum = np.zeros(N, dtype=np.float32)
    view_counts = np.zeros(N, dtype=np.int32)
    for (u_int, v_int, valid), heatmap in zip(proj, heatmaps):
        sampled = np.zeros(N, dtype=np.float32)
        sampled[valid] = heatmap[v_int[valid], u_int[valid]]
        score_sum += sampled
        view_counts += valid.astype(np.int32)
    final_scores = np.zeros(N, dtype=np.float32)
    seen = view_counts > 0
    final_scores[seen] = score_sum[seen] / view_counts[seen]
    return final_scores, view_counts


# ------------------------------------------------------------------------------
# 3D refinement + two-role partition
# (recipe validated 07-15 on 195 cached queries over 50 category-diverse
#  objects; measurement trail in the stage2 memory notes)
# ------------------------------------------------------------------------------

UP_AXIS = 1                        # gravity axis of the original canvas frame.
                                   # MEASURED 07-16: over the 43 cached pairs where both roles
                                   # declare opposing vertical terms, the declared-upper role's
                                   # score anchor is higher along axis 1 in 43/43 (axis 2: 47% =
                                   # chance). Also equals the camera-rig image-up mapped through
                                   # align_affogato_frame (vote-z == original y).
HOR_AXES = [0, 2]
HI_RE = re.compile(r"\b(upper|top|uppermost)\b", re.I)
LO_RE = re.compile(r"\b(lower|bottom|base|beneath|under|below)\b", re.I)
OPP_RE = re.compile(r"\b(opposite|other side|each side|both sides|two sides|either side)\b", re.I)
# partition gate for "body-level" targets; DELIBERATELY wider than the SAM
# mask-selection rule (stage2_v2.is_body_target: body/wall/surface only) —
# the two were calibrated separately, do not unify.
BODY_TARGET_RE = re.compile(r"\b(body|wall|surface|face|corner|side|bag)\b", re.I)


def _vert_term(role):
    """+1 / -1 / 0: vertical level a role's contact_region declares."""
    cr = str(role.get("contact_region", ""))
    hi, lo = bool(HI_RE.search(cr)), bool(LO_RE.search(cr))
    return 1 if hi and not lo else (-1 if lo and not hi else 0)


def knn_indices(xyz, k=8):
    """[N, k] nearest-neighbour indices (self excluded); the geometry cache for
    all canvas-space refinement, computed once per object."""
    from scipy.spatial import cKDTree
    return cKDTree(xyz).query(xyz, k=k + 1)[1][:, 1:]


def _support_components(s, idxNN, thr):
    """Connected components of the >thr support over the kNN subgraph.
    (The canvas has near-duplicate points, so radius-based CC fragments;
    the kNN subgraph is density-adaptive.) Returns (point_idx, label, mass)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    pi = np.nonzero(s > thr)[0]
    inv = -np.ones(len(s), dtype=np.int64)
    inv[pi] = np.arange(len(pi))
    dst = inv[idxNN[pi].ravel()]
    src = np.repeat(np.arange(len(pi)), idxNN.shape[1])
    ok = dst >= 0
    g = coo_matrix((np.ones(int(ok.sum())), (src[ok], dst[ok])),
                   shape=(len(pi), len(pi)))
    ncc, lab = connected_components(g, directed=False)
    return pi, lab, np.bincount(lab, weights=s[pi], minlength=ncc)


def cc_prune(s, idxNN, thr=0.15, keep_frac=0.10):
    """Drop satellite components holding < keep_frac of the largest one's mass."""
    if (s > thr).sum() < 5:
        return s
    pi, lab, mass = _support_components(s, idxNN, thr)
    out = s.copy()
    out[pi[mass[lab] < keep_frac * mass.max()]] = 0.0
    return out


def refine_scores(s, idxNN, thr=0.15, iters=2, alpha=0.6):
    """Pre-partition cleanup: satellite prune + kNN smoothing (fills interior
    holes, decays isolated specks)."""
    out = cc_prune(s, idxNN, thr).astype(np.float32)
    for _ in range(iters):
        out = alpha * out + (1 - alpha) * out[idxNN].mean(1)
    return out


def prune_partitioned(score, s_ref, idxNN, thr=0.15):
    """Post-partition satellite prune (clears winner-take-all remnants) with an
    erase guard: cleanup must not delete the signal, so pruning is skipped when
    it would leave <20% of the role's pre-partition support s_ref."""
    out = cc_prune(score, idxNN, thr)
    if (out > thr).sum() < 0.2 * max(1, (s_ref > thr).sum()):
        return score
    return out


def _score_anchor(s, xyz):
    """Weighted centroid of the top-5% scores (None if the score is empty)."""
    m = s >= np.percentile(s, 95)
    if not m.any() or s[m].sum() <= 0:
        return None
    return (xyz[m] * s[m, None]).sum(0) / s[m].sum()


def partition_two_roles(scoreA, scoreB, roleA, roleB, xyz, thr=0.15):
    """Resolve the two grounded role heatmaps into two DISJOINT regions.

    Molmo grounds each role independently, so symmetric co-lifts collapse onto
    one region and asymmetric grasps overlap at part boundaries. Three layers,
    all object-agnostic (query text + score geometry only):

    1) text-declared SYMMETRY -- same target, contact_regions carry
       opposite/other-side wording, no one-sided vertical relation:
       straight vertical cut through the region center along the canvas axis
       whose center plane crosses the least material (canvas frames are
       canonical, so box walls align with axes; angle scans tilt on count
       noise). Sides assigned by each role's own mass.
    2) text-declared VERTICAL relation -- horizontal (gravity) cut at the two
       roles' anchor-height midpoint, the declared-upper role above. Triggers
       on same-target pairs with a vertical relation, or on any pair where BOTH
       sides declare opposing terms AND both targets are body-level (votes on
       body-level targets are saturated and carry no part signal; named parts
       stay evidence-driven -- hard text planes chop them: measured 0.735 vs
       0.768 FINAL-AUC when ungated).
    3) EVIDENCE split -- same-target: vertical plane at the two score anchors'
       midpoint, each half to the role whose anchor sits on it; diff-target:
       per-point winner-take-all on the >thr overlap with per-role p99
       normalization (a small part's peak beats a dominant role's diffuse
       score, so it keeps its patch). Sub-threshold scores are left in place:
       exclusivity is required on the support, ranking information survives.

    Returns (scoreA', scoreB').
    """
    same = str(roleA.get("target", "")).strip().lower() == str(roleB.get("target", "")).strip().lower()
    vA, vB = _vert_term(roleA), _vert_term(roleB)
    crs = str(roleA.get("contact_region", "")) + " | " + str(roleB.get("contact_region", ""))
    diag = float(np.linalg.norm(xyz.max(0) - xyz.min(0)))

    # ---- 1) symmetric text: canvas-axis center cut --------------------------
    if same and OPP_RE.search(crs) and vA == vB:
        region = np.maximum(scoreA, scoreB)
        m = region > thr
        if m.sum() >= 10:
            Xh = xyz[m][:, HOR_AXES]
            w = region[m]
            mu = (Xh * w[:, None]).sum(0) / w.sum()
            # cut normal = the region's LONGEST horizontal axis: hands grab the
            # two far ends of the long axis, like humans do (percentile span
            # so a stray point cannot flip the choice; min-material-crossing
            # picked the short axis in 27% of measured co-lift cases)
            span = np.percentile(Xh, 98, axis=0) - np.percentile(Xh, 2, axis=0)
            d = np.eye(2)[int(span.argmax())]
            side = (xyz[:, HOR_AXES] - mu) @ d >= 0
            a_side = scoreA[side].sum() >= scoreA[~side].sum()
            return (np.where(side == a_side, region, 0.0),
                    np.where(side == a_side, 0.0, region))

    # ---- 2) vertical text: horizontal gravity cut ---------------------------
    bodyish = bool(BODY_TARGET_RE.search(str(roleA.get("target", "")))
                   and BODY_TARGET_RE.search(str(roleB.get("target", ""))))
    if (vA * vB == -1 and bodyish) or (same and vA != vB):
        hiA = vA > vB
        aA, aB = _score_anchor(scoreA, xyz), _score_anchor(scoreB, xyz)
        if aA is not None and aB is not None and abs(aA[UP_AXIS] - aB[UP_AXIS]) > 0.02 * diag:
            z0 = (aA[UP_AXIS] + aB[UP_AXIS]) / 2.0
            if (aA[UP_AXIS] > aB[UP_AXIS]) != hiA:   # anchors contradict the text -> trust text
                m = np.maximum(scoreA, scoreB) > thr
                if m.sum() >= 10:
                    z0 = float(np.median(xyz[m, UP_AXIS]))
        else:
            m = np.maximum(scoreA, scoreB) > thr
            if m.sum() < 10:
                return _partition_evidence(scoreA, scoreB, xyz, thr, same, diag)
            z0 = float(np.median(xyz[m, UP_AXIS]))
        upper = xyz[:, UP_AXIS] >= z0
        if same:
            region = np.maximum(scoreA, scoreB)
            return (np.where(upper == hiA, region, 0.0),
                    np.where(upper == hiA, 0.0, region))
        A, B = scoreA.copy(), scoreB.copy()          # diff-target: only the overlap changes hands
        both = (scoreA > 0) & (scoreB > 0)
        A[both & (upper != hiA)] = 0.0
        B[both & (upper == hiA)] = 0.0
        return A, B

    # ---- 3) evidence split ---------------------------------------------------
    return _partition_evidence(scoreA, scoreB, xyz, thr, same, diag)


def _partition_evidence(scoreA, scoreB, xyz, thr, same, diag):
    if same:                     # anchor-directed vertical cut of the union
        region = np.maximum(scoreA, scoreB)
        m = region > thr
        if m.sum() < 10:
            return scoreA, scoreB
        aA, aB = _score_anchor(scoreA, xyz), _score_anchor(scoreB, xyz)
        d = None
        if aA is not None and aB is not None:
            d = aB - aA
            d[UP_AXIS] = 0.0
            if np.linalg.norm(d) < 0.05 * diag:
                d = None
        # NB: a role with zero votes anchors to None and lands here -> the
        # union is still split half/half (same-target roles are interchangeable,
        # so the halves stay usable; the pre-port code NaN'd on this input).
        if d is None:            # anchors collapsed -> widest horizontal axis
            pts = xyz[m][:, HOR_AXES] - xyz[m][:, HOR_AXES].mean(0)
            ax2 = np.linalg.svd(pts, full_matrices=False)[2][0]
            d = np.zeros(3)
            d[HOR_AXES[0]], d[HOR_AXES[1]] = ax2[0], ax2[1]
            aA = aB = xyz[m].mean(0)
        d = d / np.linalg.norm(d)
        mid = (aA + aB) / 2.0
        # A gets its anchor's side: aA projects <= 0 along d = aB - aA (and the
        # SVD fallback has aA == mid), so the non-positive half is A's.
        on_a = (xyz - mid) @ d <= 0
        return np.where(on_a, region, 0.0), np.where(~on_a, region, 0.0)

    mA, mB = scoreA > thr, scoreB > thr              # diff-target: p99-normalized WTA
    if not (mA & mB).any():
        return scoreA, scoreB

    def p99(s):
        return max(float(np.percentile(s[s > 0], 99)) if (s > 0).any() else 1.0, 1e-6)

    nA = np.clip(scoreA / p99(scoreA), 0, 1)
    nB = np.clip(scoreB / p99(scoreB), 0, 1)
    A, B = scoreA.copy(), scoreB.copy()
    both = mA & mB
    winA = nA >= nB
    A[both & ~winA] = 0.0
    B[both & winA] = 0.0
    return A, B


# ==============================================================================
# Stage2 object/scene loading
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
    # point_cloud_from_depth.ipynb. gt[i] stays attached to point i.
    xyz_vote = align_affogato_frame(xyz)
    return Canvas(xyz=xyz, xyz_vote=xyz_vote, gt=gt)


def load_scene(obj_root, num_views):
    """Camera params + RGB views for the (already clamped) view count."""
    cameras, K_list, depth_maps = prepare_camera_params(obj_root, num_views)
    view_images, view_images_np = load_view_images(obj_root, num_views)
    return Scene(n_views=num_views, view_images=view_images, view_images_np=view_images_np,
                 cameras=cameras, K_list=K_list, depth_maps=depth_maps)
