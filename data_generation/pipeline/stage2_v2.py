#!/usr/bin/env python
"""
Stage 2 v2 — bimanual grounding: Molmo2-8B (vLLM) pointing + object-batched SAM2.

The production stage2 runner. What changed vs stage2_bimanual_grounding.py (v1)
and the retired MolmoPoint runner:
- Pointing engine = Molmo2-8B on vLLM (single_region/molmo2_vllm/engine.py).
  Precision parity with the best MolmoPoint arm on the 10-obj human-GT bench
  (k1 0.847 / k4-multi 0.842 vs MolmoPoint-k4 0.844), ~96-113 ms/view = ~1.6x
  faster than MolmoPoint k4xB12 and ~15x the old serial chain.
- Per OBJECT, every query x both roles goes into ONE llm.generate — vLLM
  continuous batching replaces any manual chunk/batch machinery.
- Default k=4 multi-image chunks ("Point to {label} in all images.", official
  template): same mean AUC as k=1, higher hit coverage (86% vs 78%), rescues
  sparse-hit objects; --k 1 for the steadier single-image mode.
- SAM2 runs object-batched (official set_image_batch: encoder once per object)
  with the voting projection geometry cached per object.
- Lean outputs: per query only scores.npz + meta.json (with per-view points);
  the stage02 notebook browser renders overlays from them on demand.
- 07-15 recipe (validated on 195 queries / 50 diverse objects): "mixed+neg"
  SAM prompts (body targets -> smallest candidate, named parts -> best_iou,
  partner point as negative beyond 30px), canvas refinement (satellite prune +
  kNN smoothing), partition v6 (text symmetry -> canvas-axis center cut, text
  vertical -> gravity cut, else evidence split with full overlap resolution),
  guarded post-prune. scoreA/scoreB are the final disjoint heatmaps;
  scoreA_raw/scoreB_raw stay the raw voted scores.

Run with the **molmo** conda env (vllm 0.16 + torch cu128 + SAM2 installed):
  ~/miniconda3/envs/molmo/bin/python pipeline/stage2_v2.py --start 0 --end 100 --gpu 3
"""
import os
import sys
import json
import time
import argparse

import numpy as np

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
    ap.add_argument("--k", type=int, default=4)          # views per prompt (1 = single-image mode)
    ap.add_argument("--gpu_mem", type=float, default=0.5)  # vLLM gpu_memory_utilization
    ap.add_argument("--skip_existing", action="store_true")
    return ap.parse_args()


def slugify(text, n=40):
    return "".join(c if c.isalnum() else "_" for c in text.lower())[:n].strip("_")


NEG_MIN_PX = 30      # partner point becomes a SAM negative only beyond this distance


def is_body_target(role):
    # SAM mask-selection rule; narrower than sra.BODY_TARGET_RE (the partition
    # gate) — calibrated separately, do not unify.
    tgt = str(role.get("target", "")).lower()
    return "body" in tgt or "wall" in tgt or "surface" in tgt


def build_sam_prompts(queries, per_query):
    """Per flat (query x role) job: candidate selection + partner-negative points.

    The validated mask recipe ("mixed+neg"): body-level targets take the
    smallest SAM candidate (best_iou bleeds to the whole object on smooth
    bodies), named parts take best_iou; the partner role's point in the same
    view is added as a negative prompt when the two points are >NEG_MIN_PX
    apart (separates e.g. mug handle from body masks).
    """
    mask_selects, neg_points = [], []
    for i, (_, q) in enumerate(queries):
        for r in range(2):
            mask_selects.append("smallest" if is_body_target(q["roles"][r]) else "best_iou")
            own, partner = per_query[i * 2 + r], per_query[i * 2 + (1 - r)]
            negs = []
            for p, np_ in zip(own, partner):
                use = (p is not None and np_ is not None
                       and float(np.linalg.norm(np.asarray(p) - np.asarray(np_))) > NEG_MIN_PX)
                negs.append(np.asarray(np_, dtype=np.float32) if use else None)
            neg_points.append(negs)
    return mask_selects, neg_points


def process_object(rec, scene, canvas, queries, per_query, heatmaps_all, proj,
                   idxNN, t_share, k, out_dir):
    """Voting + partition + save for every query; pointing AND SAM2 already done
    object-batched (encoder ran once, projection computed once).

    queries: [(qi, query_dict)] eligible queries; per_query / heatmaps_all are
    indexed i*2+r (query i, role r) — the flat order the pointing ran in.
    """
    from single_region import single_region_affordance as sra

    T = len(scene.view_images)
    n_done = 0
    for i, (qi, query) in enumerate(queries):
        roles, mqs = query["roles"], query["molmo_queries"]
        t0 = time.time()
        role_data = []
        for r in range(2):
            flat = per_query[i * 2 + r]
            scores, counts = sra.sample_heatmaps_projected(proj, heatmaps_all[i * 2 + r])
            pts_arr = np.full((T, 2), np.nan, dtype=np.float32)
            for vi, p in enumerate(flat):
                if p is not None:
                    pts_arr[vi] = p
            role_data.append((pts_arr, scores, counts))
        (ptsA, scoreA_raw, counts), (ptsB, scoreB_raw, _) = role_data
        # validated chain: refine (satellite prune + kNN smooth) -> partition
        # (text symmetry / text vertical / evidence) -> post prune (guarded)
        refA = sra.refine_scores(scoreA_raw, idxNN)
        refB = sra.refine_scores(scoreB_raw, idxNN)
        fA, fB = sra.partition_two_roles(refA, refB, roles[0], roles[1], canvas.xyz)
        scoreA = sra.prune_partitioned(fA, refA, idxNN)
        scoreB = sra.prune_partitioned(fB, refB, idxNN)

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
            "engine": {"name": "molmo2-vllm", "k": k, "sam2": "object-batched",
                       "recipe": "mixed+neg / refine / partition-v6 / post-prune"},
            "seconds": round(time.time() - t0 + t_share, 1),
        }
        json.dump(meta, open(os.path.join(qdir, "meta.json"), "w"), indent=1)
        n_done += 1
    return n_done


def main():
    args = parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
    os.chdir(DATA_GEN)

    # heavy imports AFTER the GPU pin; vLLM engine BEFORE SAM2 (it profiles GPU memory)
    from single_region.molmo2_vllm import engine as m2v
    llm, proc = m2v.load_engine(gpu_memory_utilization=args.gpu_mem, max_images=max(args.k, 1))
    from single_region import single_region_affordance as sra
    from single_region.single_region_affordance import (build_aff_map, resolve_object,
                                                        load_canvas, load_scene)

    cfg = sra.PipelineConfig(num_views=args.num_views)
    sam2_predictor = sra.load_sam2_model(cfg.sam2_checkpoint, cfg.sam2_model_cfg, cfg.device)

    recs = [r for r in json.load(open(args.stage1)) if not r.get("error")]
    aff_map = build_aff_map(args.mapping)
    end = args.end if args.end is not None else len(recs)
    todo = recs[args.start:end]
    print(f"stage2_v2: {len(todo)} objects [{args.start}:{end}] "
          f"k={args.k} views={args.num_views} -> {args.output_dir}", flush=True)

    n_done = n_skip = 0
    for oi, rec in enumerate(todo):
        obj_root, aff_dir = resolve_object(rec, aff_map)
        if not aff_dir or not os.path.isdir(obj_root):
            print(f"[{args.start + oi}] {rec['object_name'][:28]:28} SKIP (missing renders/canvas)", flush=True)
            n_skip += 1
            continue
        n_views = min(args.num_views, sra.count_available_views(obj_root))
        if n_views < 5:
            print(f"[{args.start + oi}] {rec['object_name'][:28]:28} SKIP (only {n_views} views)", flush=True)
            n_skip += 1
            continue
        queries = [(qi, q) for qi, q in enumerate(rec.get("queries", []))
                   if len(q.get("roles", [])) == 2 and len(q.get("molmo_queries", [])) == 2]
        if not queries:
            n_skip += 1
            continue
        if args.skip_existing:
            obj_out = os.path.join(args.output_dir, rec["object_id"])
            # complete = every eligible query dir has both output files (a dir
            # count would permanently skip an object interrupted mid-write)
            if all(os.path.exists(os.path.join(obj_out, f"q{qi}_{slugify(q['task'])}", fn))
                   for qi, q in queries for fn in ("scores.npz", "meta.json")):
                n_skip += 1
                continue

        t0 = time.time()
        canvas = load_canvas(aff_dir)
        scene = load_scene(obj_root, n_views)
        # ONE generate for every query x both roles of this object
        mqs_flat = [mq for _, q in queries for mq in q["molmo_queries"]]
        per_query, n_oob = m2v.ground_queries(llm, proc, scene.view_images, mqs_flat, k=args.k)
        t_point = time.time() - t0
        # SAM2 encoder once per object; projection geometry once per object
        queries_points = [[[np.array(p, dtype=np.float32)] if p is not None else []
                           for p in pq] for pq in per_query]
        mask_selects, neg_points = build_sam_prompts(queries, per_query)
        heatmaps_all = sra.run_sam2_object_queries(scene.view_images_np, queries_points,
                                                   sam2_predictor, cfg,
                                                   neg_points=neg_points,
                                                   mask_selects=mask_selects)
        t_sam = time.time() - t0 - t_point
        proj = sra.precompute_projection(canvas.xyz_vote, scene.cameras, scene.K_list,
                                         scene.depth_maps, cfg.depth_tolerance)
        idxNN = sra.knn_indices(canvas.xyz)
        n_q = process_object(rec, scene, canvas, queries, per_query, heatmaps_all, proj,
                             idxNN, (t_point + t_sam) / len(queries), args.k, args.output_dir)
        n_done += 1
        oob = f" oob={n_oob}" if n_oob else ""
        print(f"[{args.start + oi}] {rec['object_name'][:28]:28} {n_q} queries  "
              f"{time.time() - t0:.0f}s (point {t_point:.0f}s sam2 {t_sam:.0f}s){oob}", flush=True)
    print(f"\nstage2_v2 done: {n_done} objects, {n_skip} skipped -> {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
