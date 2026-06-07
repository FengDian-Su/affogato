#!/usr/bin/env python
"""
single_region_affordance.py
===========================

Formal, standalone **single-region** 3D-affordance heatmap generator, extracted
from `run_pipeline.py` (the bimanual pipeline) and reduced to exactly the AFFOGATO
single-region setting:

        input query  ->  Molmo (point)  ->  SAM2 (mask -> sigmoid heatmap)
                      ->  multi-view voting  ->  per-point [0,1] heatmap

The predicted heatmap is scored on the **original AFFOGATO ground-truth points**
(the 16384 points in `xyzc.npy`, cols 0-2), so it can be compared 1:1 against the
GT heatmaps (`xyzc.npy` cols 3..7) with no resampling / point matching.

Why this file exists
--------------------
`run_pipeline.py::main()` (the bimanual path) does two things that make a direct
GT comparison impossible:
  1. it queries Molmo with *bimanual role* queries (answer[0]/answer[1]), NOT
     AFFOGATO's own per-object queries;
  2. it computes `affordance_scores_gt` on the GT points but then **discards** it
     (only the reconstruction-point HTML plots are saved).
This file fixes both: it uses each object's own `queries.json` (AFFOGATO's 5
queries) and **saves** the per-point GT-aligned heatmaps to disk.

Output (per object), written to `<output_dir>/<object_id>/affordance_pred.npz`:
  - pred    : (N, K) float32  predicted heatmap, one column per query (queries.json order)
  - counts  : (N,)   int32    multi-view visibility count per point (same for all queries)
  - xyz     : (N, 3) float32  GT point coordinates (xyzc.npy cols 0-2)
  - gt       : (N, K) float32 AFFOGATO GT heatmaps (xyzc.npy cols 3..3+K), saved for convenience
  - queries : (K,)   <U...    the K query strings used
  - config_json : 0-d str     the PipelineConfig used (for provenance)
  - meta_json   : 0-d str     per-object notes (n_views, coverage, score ranges)

Known divergences from the *original* AFFOGATO data-generation engine
(arXiv:2506.12009) — keep these in mind when interpreting the comparison:
  * Mask model: original used **MobileSAM**; here we use **SAM2** (`mask_select`
    picks one of SAM2's multimask outputs).
  * Heatmap: original = raw **sigmoid(mask logits)**, no spatial kernel; this
    pipeline multiplies by a **Gaussian weight** centred on the Molmo point
    (`use_gaussian=True`). Set `use_gaussian=False` to get the more
    paper-faithful raw-sigmoid heatmap.
  * Point model: original used `allenai/Molmo-7B-D-0924`; the team's pipeline
    uses `allenai/MolmoPoint-8B` (kept as the default here so we validate *our*
    reproduction).
  * Views: `num_views` of the G-Objaverse renders are used (default 24, matching
    run_pipeline.py); more views -> higher coverage of the GT points.

Run with the repo's `mm` conda env:
  python single_region_affordance.py --start 0 --end 5 --gpu 2 \
         --output_dir output_single_region

Most numerical functions are copied verbatim from run_pipeline.py so this file is
a faithful extraction of the path that actually ran.
"""

import os
# Must be set before importing cv2 so OpenEXR depth (*_nd.exr) can be read.
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
# Make CUDA device indices match `nvidia-smi` (PCI order, not fastest-first).
# Set before any torch.cuda call; --gpu then pins via CUDA_VISIBLE_DEVICES in main().
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import gc
import glob
import json
import argparse
from dataclasses import dataclass, asdict

import numpy as np
import cv2
import torch
import tqdm
from PIL import Image


# ==============================================================================
# Config
# ==============================================================================

@dataclass
class PipelineConfig:
    # --- models ---
    molmo_model_id: str = "allenai/MolmoPoint-8B"
    sam2_checkpoint: str = "checkpoints/sam2.1_hiera_large.pt"
    sam2_model_cfg: str = "configs/sam2.1/sam2.1_hiera_l.yaml"

    # --- views / geometry ---
    num_views: int = 24
    depth_tolerance: float = 0.15      # relative depth tol for visibility test
    align_gt_frame: bool = True        # apply swapYZ+flipY to AFFOGATO points before voting
                                       # (matches point_cloud_from_depth.ipynb; required!)

    # --- 2D heatmap shaping ---
    use_gaussian: bool = False          # multiply sigmoid mask by Gaussian around Molmo point
    gaussian_sigma: float = 77.0       # px; only used when use_gaussian
    mask_select: str = "best_iou"      # "best_iou" | "smallest"

    # --- molmo decoding ---
    molmo_max_new_tokens: int = 200

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


# ==============================================================================
# Models
# ==============================================================================

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def make_gaussian_weight(H, W, points, sigma=50):
    """Gaussian weight map centred on the Molmo point(s); union via max."""
    u = np.arange(W)[np.newaxis, :]
    v = np.arange(H)[:, np.newaxis]
    weight = np.zeros((H, W), dtype=np.float32)
    for pt in points:
        dx = u - pt[0]
        dy = v - pt[1]
        g = np.exp(-(dx ** 2 + dy ** 2) / (2 * sigma ** 2))
        weight = np.maximum(weight, g)
    return weight


def load_molmo_model(model_id):
    """Load MolmoPoint processor + model, and its `extract_image_points` helper."""
    from transformers import AutoProcessor, AutoModelForImageTextToText
    print(f"\nLoading Molmo model: {model_id} ...")
    molmo_processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    molmo_model = AutoModelForImageTextToText.from_pretrained(
        model_id, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="cuda",
    )
    _mod = __import__(type(molmo_model).__module__, fromlist=["extract_image_points"])
    extract_image_points = _mod.extract_image_points
    return molmo_processor, molmo_model, extract_image_points


def load_sam2_model(checkpoint, model_cfg, device):
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    print(f"Loading SAM2: {checkpoint} ...")
    sam2_model = build_sam2(model_cfg, checkpoint, device=device)
    return SAM2ImagePredictor(sam2_model)


@dataclass
class Models:
    molmo_processor: object
    molmo_model: object
    extract_image_points: object
    sam2_predictor: object


def load_models(cfg: PipelineConfig) -> Models:
    molmo_processor, molmo_model, extract_image_points = load_molmo_model(cfg.molmo_model_id)
    sam2_predictor = load_sam2_model(cfg.sam2_checkpoint, cfg.sam2_model_cfg, cfg.device)
    return Models(molmo_processor, molmo_model, extract_image_points, sam2_predictor)


def molmo_query_image(img, query_text, models: Models, max_new_tokens=200):
    """Single-image MolmoPoint query -> top-1 (x, y) point. (copied from run_pipeline.py)"""
    molmo_processor = models.molmo_processor
    molmo_model = models.molmo_model
    extract_image_points = models.extract_image_points

    messages = [{"role": "user", "content": [
        {"type": "image", "image": img},
        {"type": "text", "text": query_text},
    ]}]
    prompt_text = molmo_processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = molmo_processor(
        text=prompt_text, images=[img],
        return_pointing_metadata=True, return_tensors="pt",
    )
    metadata = inputs.pop("metadata")
    inputs = {k: v.to(molmo_model.device) for k, v in inputs.items()}

    with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        generated_ids = molmo_model.generate(**inputs, max_new_tokens=max_new_tokens)

    generated_tokens = generated_ids[0, inputs["input_ids"].size(1):]
    generated_text = molmo_processor.tokenizer.decode(generated_tokens, skip_special_tokens=False)

    extracted = extract_image_points(
        output_text=generated_text,
        pooling=metadata["token_pooling"],
        mappings=metadata["subpatch_mapping"],
        no_more_points_class=molmo_model.config.no_more_points_class,
        location=molmo_model.config.patch_location is not None,
        image_sizes=metadata["image_sizes"],
    )
    # Molmo decode order == confidence order; keep top-1.
    if len(extracted) > 0:
        _, _, x, y = extracted[0]
        pts = [np.array([x, y])]
    else:
        pts = []

    del inputs, generated_ids, generated_tokens
    gc.collect()
    torch.cuda.empty_cache()
    return pts, generated_text


# ==============================================================================
# Single-query pipeline stages
# ==============================================================================

def run_molmo_single_query(view_images, query, models: Models, cfg: PipelineConfig):
    """Molmo top-1 point per view for ONE query. Returns list[ list[np.array([x,y])] ]."""
    points_per_view = [[] for _ in range(len(view_images))]
    for view_idx in tqdm.trange(len(view_images), desc="MolmoPoint"):
        pts, _ = molmo_query_image(
            view_images[view_idx], query, models, cfg.molmo_max_new_tokens
        )
        points_per_view[view_idx] = pts
    n_with = sum(1 for p in points_per_view if len(p) > 0)
    print(f"  views with a point: {n_with}/{len(view_images)}")
    return points_per_view


def run_sam2_single_query(view_images_np, points_per_view, sam2_predictor, cfg: PipelineConfig):
    """SAM2 point-prompt -> sigmoid(mask logits) [-> Gaussian] per view. Returns list[[H,W]]."""
    heatmaps = []
    for view_idx in tqdm.trange(len(view_images_np), desc="SAM2 heatmaps"):
        points = points_per_view[view_idx]
        img_np = view_images_np[view_idx]
        H, W = img_np.shape[:2]

        if len(points) == 0:
            heatmaps.append(np.zeros((H, W), dtype=np.float32))
            continue

        sam2_predictor.set_image(img_np)
        point_coords = np.array(points, dtype=np.float32)
        point_labels = np.ones(len(points), dtype=np.int32)

        masks, iou_scores, _ = sam2_predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=True,
            return_logits=True,
        )

        if cfg.mask_select == "smallest":
            best_idx = int(np.argmin([(m > 0).sum() for m in masks]))
        else:
            best_idx = int(np.argmax(iou_scores))

        heatmap = sigmoid(masks[best_idx]).astype(np.float32)
        if cfg.use_gaussian:
            heatmap = heatmap * make_gaussian_weight(H, W, points, sigma=cfg.gaussian_sigma)

        heatmaps.append(heatmap)
        sam2_predictor.reset_predictor()
    return heatmaps


def project_and_sample_heatmaps(points, heatmaps, cameras, K_list, depths, depth_tolerance=0.05):
    """
    Project the 3D points into every view's 2D heatmap and average over views that
    see each point (multi-view voting). (copied from run_pipeline.py)

    Returns:
        scores: [N] aggregated affordance score in [0,1]
        counts: [N] number of views that see each point (geometry only)
    """
    N = len(points)
    num_views = len(heatmaps)
    score_sum = np.zeros(N, dtype=np.float32)
    view_counts = np.zeros(N, dtype=np.int32)
    points_homo = np.hstack([points, np.ones((N, 1))])

    for view_idx in range(num_views):
        _, w2c = cameras[view_idx]
        K = K_list[view_idx]
        heatmap = heatmaps[view_idx]
        depth_map = depths[view_idx]
        H, W = heatmap.shape

        cam_coords = (w2c @ points_homo.T).T
        cam_xyz = cam_coords[:, :3]
        in_front = cam_xyz[:, 2] > 0

        pixel_homo = (K @ cam_xyz.T).T
        pixel_uv = pixel_homo[:, :2] / pixel_homo[:, 2:3]
        u, v = pixel_uv[:, 0], pixel_uv[:, 1]
        in_bounds = (u >= 0) & (u < W) & (v >= 0) & (v < H)

        proj_depth = cam_xyz[:, 2]
        u_int = np.clip(u.astype(np.int32), 0, W - 1)
        v_int = np.clip(v.astype(np.int32), 0, H - 1)
        sampled_depth = depth_map[v_int, u_int]
        visible = (sampled_depth > 0) & (proj_depth < sampled_depth * (1 + depth_tolerance))

        valid_mask = in_front & in_bounds & visible
        sampled_scores = np.zeros(N, dtype=np.float32)
        sampled_scores[valid_mask] = heatmap[v_int[valid_mask], u_int[valid_mask]]

        score_sum += sampled_scores
        view_counts += valid_mask.astype(np.int32)

    final_scores = np.zeros(N, dtype=np.float32)
    seen = view_counts > 0
    final_scores[seen] = score_sum[seen] / view_counts[seen]
    return final_scores, view_counts


# ==============================================================================
# Orchestration
# ==============================================================================

def load_object_queries(pc_path):
    """Read AFFOGATO's own queries for this object from queries.json (same dir as xyzc.npy)."""
    qpath = os.path.join(os.path.dirname(pc_path), "queries.json")
    with open(qpath, "r") as f:
        data = json.load(f)
    # queries.json = [{"class_name": ..., "queries": [q0, q1, ...]}]
    entry = data[0]
    return entry.get("class_name", ""), list(entry["queries"])


def process_object(object_id, pc_path, img_folder_path, queries, models, cfg: PipelineConfig):
    """
    Run single-region affordance for every query on this object, scored on the
    AFFOGATO GT points. Returns a result dict (pred/counts/xyz/gt/queries/meta).
    """
    # GT point cloud + GT heatmaps (xyzc.npy: cols 0-2 xyz, cols 3.. heatmaps)
    xyzc = np.load(pc_path).astype(np.float32)
    xyz = xyzc[:, :3]
    gt = xyzc[:, 3:]
    N = xyz.shape[0]

    # AFFOGATO GT points use a different axis convention than the G-Objaverse
    # camera frame. We MUST align them before projecting/voting, otherwise each
    # point lands on the wrong pixel and samples the wrong heatmap value.
    # This matches point_cloud_from_depth.ipynb (CELL 17): swap Y/Z, flip new Y.
    # (run_pipeline.py omits this -> latent coordinate-frame bug.)
    # gt[i] stays attached to point i; only the coordinates change, so the
    # pred[i] <-> gt[i] correspondence used for the comparison is preserved.
    if cfg.align_gt_frame:
        xyz_vote = xyz[:, [0, 2, 1]].copy()
        xyz_vote[:, 1] *= -1
    else:
        xyz_vote = xyz

    # Clamp num_views to what's actually rendered for this object.
    avail = count_available_views(img_folder_path)
    n_views = min(cfg.num_views, avail)
    if n_views < cfg.num_views:
        print(f"  [warn] only {avail} views available; using {n_views}")
    if n_views == 0:
        raise RuntimeError(f"no usable views in {img_folder_path}")

    cameras, K_list, depth_maps = prepare_camera_params(img_folder_path, n_views)
    view_images, view_images_np = load_view_images(img_folder_path, n_views)

    K = len(queries)
    pred = np.zeros((N, K), dtype=np.float32)
    counts = np.zeros(N, dtype=np.int32)
    score_ranges = []

    for qi, q in enumerate(queries):
        print(f"\n--- query {qi+1}/{K}: {q!r} ---")
        pts_per_view = run_molmo_single_query(view_images, q, models, cfg)
        heatmaps = run_sam2_single_query(view_images_np, pts_per_view, models.sam2_predictor, cfg)
        scores, cnts = project_and_sample_heatmaps(
            xyz_vote, heatmaps, cameras, K_list, depth_maps, cfg.depth_tolerance
        )
        pred[:, qi] = scores
        counts = cnts  # visibility is geometry-only -> identical across queries
        score_ranges.append([float(scores.min()), float(scores.max()), float(scores.mean())])
        print(f"  score [{scores.min():.4f}, {scores.max():.4f}] mean {scores.mean():.4f}")

    coverage = float((counts > 0).mean())
    print(f"\n  coverage: {coverage*100:.1f}% of {N} GT points seen by >=1 view")

    meta = {
        "object_id": object_id,
        "n_views_used": n_views,
        "n_views_available": avail,
        "num_points": int(N),
        "num_queries": int(K),
        "num_gt_channels": int(gt.shape[1]),
        "coverage": coverage,
        "score_ranges_min_max_mean": score_ranges,
    }
    return {"pred": pred, "counts": counts, "xyz": xyz, "gt": gt, "queries": queries, "meta": meta}


def save_result(out_dir, object_id, result, cfg: PipelineConfig):
    obj_dir = os.path.join(out_dir, object_id)
    os.makedirs(obj_dir, exist_ok=True)
    save_path = os.path.join(obj_dir, "affordance_pred.npz")
    np.savez_compressed(
        save_path,
        pred=result["pred"].astype(np.float32),
        counts=result["counts"].astype(np.int32),
        xyz=result["xyz"].astype(np.float32),
        gt=result["gt"].astype(np.float32),
        queries=np.array(result["queries"], dtype=object),
        config_json=json.dumps(asdict(cfg)),
        meta_json=json.dumps(result["meta"]),
    )
    print(f"[INFO] saved {save_path}")
    return save_path


# ==============================================================================
# CLI
# ==============================================================================

def setup_device():
    """CUDA_VISIBLE_DEVICES is already pinned to the chosen GPU, so it is cuda:0 here."""
    if torch.cuda.is_available():
        dev = "cuda:0"
        print(f"Using device: {dev} ({torch.cuda.get_device_name(0)})")
        return dev
    print("Using device: cpu")
    return "cpu"


def iter_objects(mapping_path, start, end):
    """Yield (object_id, pc_path, img_folder_path) from the gobjaverse->affogato mapping."""
    with open(mapping_path, "r") as f:
        mapping = json.load(f)
    end = min(end, len(mapping))
    for i in range(start, end):
        entry = mapping[i]
        pc_path = f'{entry["dst"]}/xyzc.npy'
        img_folder_path = entry["src"]
        object_id = os.path.basename(entry["dst"])
        yield object_id, pc_path, img_folder_path


def main():
    parser = argparse.ArgumentParser(description="Single-region AFFOGATO affordance reproduction")
    parser.add_argument("--mapping", default="dataset/gobjaverse_to_affogato_nj.json",
                        help="gobjaverse->affogato mapping json (src renders, dst xyzc.npy)")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=5)
    parser.add_argument("--gpu", type=int, default=2)
    parser.add_argument("--output_dir", default="outputs/output_single_region")
    parser.add_argument("--num_views", type=int, default=24)
    parser.add_argument("--no_gaussian", action="store_true",
                        help="disable Gaussian weighting (raw sigmoid = closer to original AFFOGATO)")
    parser.add_argument("--gaussian_sigma", type=float, default=77.0)
    parser.add_argument("--mask_select", default="best_iou", choices=["best_iou", "smallest"])
    parser.add_argument("--molmo_model_id", default="allenai/MolmoPoint-8B")
    parser.add_argument("--sam2_checkpoint", default="checkpoints/sam2.1_hiera_large.pt")
    parser.add_argument("--sam2_model_cfg", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--max_objects", type=int, default=None,
                        help="stop after this many SUCCESSFUL objects (skips missing-render "
                             "entries silently); when set, --end is ignored (scans to end of mapping)")
    args = parser.parse_args()

    # Pin the chosen physical GPU (PCI order set at import) BEFORE any CUDA init,
    # so the only visible device is cuda:0 and Molmo's device_map="cuda" lands there.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = setup_device()
    cfg = PipelineConfig(
        molmo_model_id=args.molmo_model_id,
        sam2_checkpoint=args.sam2_checkpoint,
        sam2_model_cfg=args.sam2_model_cfg,
        num_views=args.num_views,
        use_gaussian=not args.no_gaussian,
        gaussian_sigma=args.gaussian_sigma,
        mask_select=args.mask_select,
        device=device,
    )
    print("Config:", json.dumps(asdict(cfg), indent=2))

    models = load_models(cfg)

    n_done = n_skip = n_fail = 0
    scan_end = args.end if args.max_objects is None else 10**9
    for object_id, pc_path, img_folder_path in iter_objects(args.mapping, args.start, scan_end):
        out_npz = os.path.join(args.output_dir, object_id, "affordance_pred.npz")
        if args.skip_existing and os.path.exists(out_npz):
            n_skip += 1; continue

        # silently skip entries whose renders/GT aren't downloaded locally (scattered)
        if not (os.path.exists(pc_path) and os.path.isdir(img_folder_path)):
            n_skip += 1; continue

        print(f"\n{'='*60}\nObject {object_id}  (done so far: {n_done})\n  pc:  {pc_path}\n  img: {img_folder_path}\n{'='*60}")
        try:
            class_name, queries = load_object_queries(pc_path)
            print(f"  class: {class_name} | {len(queries)} queries")
            result = process_object(object_id, pc_path, img_folder_path, queries, models, cfg)
            save_result(args.output_dir, object_id, result, cfg)
            n_done += 1
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"  [fail] {e}"); n_fail += 1

        if args.max_objects is not None and n_done >= args.max_objects:
            print(f"\n[INFO] reached --max_objects={args.max_objects}, stopping.")
            break

    print(f"\n===== DONE =====\n  done: {n_done}  skip: {n_skip}  fail: {n_fail}")


if __name__ == "__main__":
    main()
