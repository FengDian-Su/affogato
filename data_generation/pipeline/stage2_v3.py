#!/usr/bin/env python
"""
Stage 2 v3 — bimanual grounding with the Molmo2-8B vLLM pointing engine.

What changed vs stage2_v2.py (MolmoPoint + patched-transformers batching):
- Pointing engine = Molmo2-8B on vLLM (single_region/molmo2_vllm/engine.py).
  Precision parity with the best MolmoPoint arm on the 10-obj human-GT bench
  (k1 0.847 / k4-multi 0.842 vs MolmoPoint-k4 0.844), ~96-113 ms/view = ~1.6x
  faster than MolmoPoint k4xB12 and ~15x the old serial chain.
- Per OBJECT, every query x both roles goes into ONE llm.generate — vLLM
  continuous batching replaces the manual k-chunk x B-batch machinery.
- Default k=4 multi-image chunks ("Point to {label} in all images.", official
  template): same mean AUC as k=1, higher hit coverage (86% vs 78%), rescues
  sparse-hit objects; --k 1 for the steadier single-image mode.
- Output schema identical to v2 (scores.npz + meta.json per query), so the
  stage02 notebook browser reads both.

Run with the **molmo** conda env (vllm 0.16 + torch cu128 + SAM2 installed):
  ~/miniconda3/envs/molmo/bin/python pipeline/stage2_v3.py --start 0 --end 100 --gpu 3
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
    ap.add_argument("--output_dir", default="outputs/bimanual_grounding_v3")
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


def process_object(rec, scene, canvas, pts_per_role, sam2_predictor, cfg, k, out_dir):
    """SAM2 + voting + partition for every query of one object; pointing already done."""
    import numpy as np
    from single_region import single_region_affordance as sra

    n_done = 0
    for qi, query, t_point_share in pts_per_role["queries"]:
        roles, mqs = query["roles"], query["molmo_queries"]
        t0 = time.time()
        role_data = []
        for r in range(2):
            flat = pts_per_role["points"][(qi, r)]
            pts = [[np.array(p, dtype=np.float32)] if p is not None else [] for p in flat]
            heatmaps = sra.run_sam2_single_query(scene.view_images_np, pts, sam2_predictor, cfg)
            scores, counts = sra.project_and_sample_heatmaps(
                canvas.xyz_vote, heatmaps, scene.cameras, scene.K_list, scene.depth_maps,
                cfg.depth_tolerance)
            T = len(scene.view_images)
            pts_arr = np.full((T, 2), np.nan, dtype=np.float32)
            for vi, p in enumerate(flat):
                if p is not None:
                    pts_arr[vi] = p
            role_data.append((pts_arr, scores, counts))
        (ptsA, scoreA_raw, counts), (ptsB, scoreB_raw, _) = role_data
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
            "engine": {"name": "molmo2-vllm", "k": k},
            "seconds": round(time.time() - t0 + t_point_share, 1),
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
    from pipeline.stage2_bimanual_grounding import build_aff_map, resolve_object, load_canvas, load_scene

    cfg = sra.PipelineConfig(num_views=args.num_views)
    sam2_predictor = sra.load_sam2_model(cfg.sam2_checkpoint, cfg.sam2_model_cfg, cfg.device)

    recs = [r for r in json.load(open(args.stage1)) if not r.get("error")]
    aff_map = build_aff_map(args.mapping)
    end = args.end if args.end is not None else len(recs)
    todo = recs[args.start:end]
    print(f"stage2_v3: {len(todo)} objects [{args.start}:{end}] "
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
        if args.skip_existing:
            obj_out = os.path.join(args.output_dir, rec["object_id"])
            if os.path.isdir(obj_out) and len(os.listdir(obj_out)) >= len(queries) > 0:
                n_skip += 1
                continue
        if not queries:
            n_skip += 1
            continue

        t0 = time.time()
        canvas = load_canvas(aff_dir)
        scene = load_scene(obj_root, n_views)
        # ONE generate for every query x both roles of this object
        mqs_flat = [mq for _, q in queries for mq in q["molmo_queries"]]
        per_query, n_oob = m2v.ground_queries(llm, proc, scene.view_images, mqs_flat, k=args.k)
        t_point = time.time() - t0
        pts_per_role = {
            "points": {(qi, r): per_query[i * 2 + r] for i, (qi, _) in enumerate(queries) for r in range(2)},
            "queries": [(qi, q, t_point / len(queries)) for qi, q in queries],
        }
        n_q = process_object(rec, scene, canvas, pts_per_role, sam2_predictor, cfg,
                             args.k, args.output_dir)
        n_done += 1
        oob = f" oob={n_oob}" if n_oob else ""
        print(f"[{args.start + oi}] {rec['object_name'][:28]:28} {n_q} queries  "
              f"{time.time() - t0:.0f}s (point {t_point:.0f}s){oob}", flush=True)
    print(f"\nstage2_v3 done: {n_done} objects, {n_skip} skipped -> {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
