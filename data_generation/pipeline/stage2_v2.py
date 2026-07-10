#!/usr/bin/env python
"""
Stage 2 v2 — bimanual grounding with BATCHED Molmo pointing (k-chunk x B-batch).

What changed vs stage2_bimanual_grounding.py (v1):
- Molmo pointing goes through single_region/molmo_batch (k=4 views per prompt for
  cross-view context, B=4 prompts per generate on the batch-patched model)
  -> ~5x faster than v1's per-view serial loop, and occluded views stop
  hallucinating because the model sees neighbouring views in the same prompt.
- The 6-hunk batch patch on the HF modules cache is VERIFIED at startup
  (ensure_batch_patch / assert_model_patched) - never runs silently unpatched.
- The official pointing logits processor is always on.
- Lean outputs: per query only scores.npz + meta.json (with per-view points);
  the notebook renders view overlays on demand, so no 160-PNG dumps.
- --stage1 defaults to the canonical v13.2 dataset (v1 still pointed at a
  June backup).

Everything heavy lives in the utils it imports:
  single_region/molmo_batch/molmo_point_batch.py  (patch mgmt + batched pointing)
  single_region/single_region_affordance.py       (SAM2, voting, partition)
  pipeline/stage2_bimanual_grounding.py           (scene/canvas loaders)

  CUDA_VISIBLE_DEVICES is set from --gpu BEFORE torch loads. Run with **mm** env:
  python pipeline/stage2_v2.py --start 0 --end 100 --gpu 3 --batch 4
"""
import os
import sys
import json
import time
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_GEN = os.path.dirname(HERE)
sys.path.insert(0, DATA_GEN)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage1", default="outputs/stage1_v2/stage1_part0.json")
    ap.add_argument("--mapping", default="dataset/daily_used_to_affogato.json")
    ap.add_argument("--output_dir", default="outputs/bimanual_grounding_v2")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--gpu", default="3")
    ap.add_argument("--num_views", type=int, default=40)
    ap.add_argument("--k", type=int, default=4)        # views per Molmo prompt
    ap.add_argument("--batch", type=int, default=4)    # prompts per generate (4 = shared-GPU safe)
    ap.add_argument("--skip_existing", action="store_true")
    return ap.parse_args()


def slugify(text, n=40):
    return "".join(c if c.isalnum() else "_" for c in text.lower())[:n].strip("_")


def ground_role(molmo_query, scene, xyz_vote, sam2_predictor, cfg, model, processor, k, batch):
    """ONE role: batched Molmo points -> SAM2 masks -> multi-view voting."""
    import numpy as np
    from single_region import single_region_affordance as sra
    from single_region.molmo_batch import molmo_point_batch as mb

    flat = mb.point_batch(scene.view_images, molmo_query, k=k, batch=batch,
                          model=model, processor=processor)   # [(x,y) | None] per view
    pts = [[np.array(p, dtype=np.float32)] if p is not None else [] for p in flat]
    heatmaps = sra.run_sam2_single_query(scene.view_images_np, pts, sam2_predictor, cfg)
    scores, counts = sra.project_and_sample_heatmaps(
        xyz_vote, heatmaps, scene.cameras, scene.K_list, scene.depth_maps, cfg.depth_tolerance)
    # per-view points as a [T,2] float array (NaN = no point) for the notebook overlays
    T = len(scene.view_images)
    pts_arr = np.full((T, 2), np.nan, dtype=np.float32)
    for vi, p in enumerate(flat):
        if p is not None:
            pts_arr[vi] = p
    return pts_arr, scores, counts


def process_query(rec, qi, query, scene, canvas, sam2_predictor, cfg,
                  model, processor, k, batch, out_dir):
    """Ground both roles of one query, partition, save scores.npz + meta.json."""
    import numpy as np
    from single_region import single_region_affordance as sra

    roles, mqs = query.get("roles", []), query.get("molmo_queries", [])
    if len(roles) != 2 or len(mqs) != 2:
        return None
    t0 = time.time()
    ptsA, scoreA_raw, counts = ground_role(mqs[0], scene, canvas.xyz_vote,
                                           sam2_predictor, cfg, model, processor, k, batch)
    ptsB, scoreB_raw, _ = ground_role(mqs[1], scene, canvas.xyz_vote,
                                      sam2_predictor, cfg, model, processor, k, batch)
    scoreA, scoreB = sra.partition_two_roles(scoreA_raw, scoreB_raw,
                                             roles[0], roles[1], canvas.xyz)

    qdir = os.path.join(out_dir, rec["object_id"], f"q{qi}_{slugify(query['task'])}")
    os.makedirs(qdir, exist_ok=True)
    np.savez_compressed(
        os.path.join(qdir, "scores.npz"),
        xyz=canvas.xyz, gt=canvas.gt, counts=counts,
        scoreA_raw=scoreA_raw, scoreB_raw=scoreB_raw, scoreA=scoreA, scoreB=scoreB,
        ptsA=ptsA, ptsB=ptsB,
        molmo_queries=np.array(mqs, dtype=object), task=str(query["task"]))
    meta = {
        "object_id": rec["object_id"], "object_name": rec["object_name"],
        "task": query["task"], "category": query.get("category"),
        "query": query.get("query"), "roles": roles, "molmo_queries": mqs,
        "pattern": query.get("pattern"), "coordination": query.get("coordination"),
        "n_views": scene.n_views,
        "hitA": int(np.isfinite(ptsA[:, 0]).sum()), "hitB": int(np.isfinite(ptsB[:, 0]).sum()),
        "coverage": float((counts > 0).mean()),
        "engine": {"k": k, "batch": batch, "logits_processor": True, "patched_model": True},
        "seconds": round(time.time() - t0, 1),
    }
    json.dump(meta, open(os.path.join(qdir, "meta.json"), "w"), indent=1)
    return meta


def main():
    args = parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
    os.chdir(DATA_GEN)

    # heavy imports AFTER the GPU pin
    from single_region import single_region_affordance as sra
    from single_region.molmo_batch import molmo_point_batch as mb
    from pipeline.stage2_bimanual_grounding import build_aff_map, resolve_object, load_canvas, load_scene

    cfg = sra.PipelineConfig(num_views=args.num_views)
    mb.ensure_batch_patch()                                   # loud failure if unpatchable
    model, processor = mb.load_official()                     # patched-model assert inside
    sam2_predictor = sra.load_sam2_model(cfg.sam2_checkpoint, cfg.sam2_model_cfg, cfg.device)

    recs = [r for r in json.load(open(args.stage1)) if not r.get("error")]
    aff_map = build_aff_map(args.mapping)
    end = args.end if args.end is not None else len(recs)
    todo = recs[args.start:end]
    print(f"stage2_v2: {len(todo)} objects [{args.start}:{end}] "
          f"k={args.k} batch={args.batch} views={args.num_views} -> {args.output_dir}")

    n_done = n_skip = 0
    for oi, rec in enumerate(todo):
        obj_root, aff_dir = resolve_object(rec, aff_map)
        if not aff_dir or not os.path.isdir(obj_root):
            print(f"[{args.start + oi}] {rec['object_name'][:28]:28} SKIP (missing renders/canvas)")
            n_skip += 1
            continue
        n_views = min(args.num_views, sra.count_available_views(obj_root))
        if n_views < 5:
            print(f"[{args.start + oi}] {rec['object_name'][:28]:28} SKIP (only {n_views} views)")
            n_skip += 1
            continue
        if args.skip_existing:
            qdirs = [q for qi, q in enumerate(rec.get("queries", []))]
            obj_out = os.path.join(args.output_dir, rec["object_id"])
            if os.path.isdir(obj_out) and len(os.listdir(obj_out)) >= len(qdirs) > 0:
                n_skip += 1
                continue
        t0 = time.time()
        canvas = load_canvas(aff_dir)
        scene = load_scene(obj_root, n_views)
        n_q = 0
        for qi, query in enumerate(rec.get("queries", [])):
            meta = process_query(rec, qi, query, scene, canvas, sam2_predictor, cfg,
                                 model, processor, args.k, args.batch, args.output_dir)
            n_q += meta is not None
        n_done += 1
        print(f"[{args.start + oi}] {rec['object_name'][:28]:28} {n_q} queries  {time.time() - t0:.0f}s")
    print(f"\nstage2_v2 done: {n_done} objects, {n_skip} skipped -> {args.output_dir}")


if __name__ == "__main__":
    main()
