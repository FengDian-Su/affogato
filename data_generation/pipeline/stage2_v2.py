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
- SAM2 runs object-batched (official set_image_batch: encoder once per object)
  with the voting projection geometry cached per object.
- Lean outputs: per query only scores.npz + meta.json (with per-view points);
  the stage02 notebook browser renders overlays from them on demand.
- 07-17 recipe (validated on 195 queries / 50 diverse objects): "mixed+neg"
  SAM prompts (body targets -> middle candidate, named parts -> best_iou,
  partner point as negative beyond 30px), canvas refinement (satellite prune +
  kNN smoothing), partition v6 (text symmetry -> canvas-axis center cut, text
  vertical -> gravity cut, else evidence split with full overlap resolution),
  guarded post-prune. scoreA/scoreB are the final disjoint heatmaps;
  scoreA_raw/scoreB_raw stay the raw voted scores.
- 07-19 recipe (single path; multi-image k>1 mode deleted): view-by-view
  pointing — Molmo declines natively per view and emits its first-token
  existence confidence, thresholded by --exist_thr (calibrated on 500
  objects); ADAPTIVE multi-point (uncapped: the model's own instance count
  decides; one SAM mask per point, max-merged per role); 2D cross-role
  overlap resolution before the vote (--overlap2d). scores.npz gains
  ptsA_multi/ptsB_multi ((M,3) rows vi,x,y — ALL Molmo points, pre-gate) and
  pexistA/pexistB ((T,) P(exist) per view), so the existence threshold can be
  re-applied post hoc in either direction. scoreA_raw/scoreB_raw are the
  votes AFTER the 2D overlap resolution.

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
    ap.add_argument("--gpu_mem", type=float, default=0.5)  # vLLM gpu_memory_utilization
    ap.add_argument("--sam_chunk", type=int, default=40)   # views per SAM2 encoder pass;
                                                           # 40x1024^2 batch encode peaks >20GB,
                                                           # set 20 on <=48GB cards
    ap.add_argument("--skip_existing", action="store_true")
    # 07-18/19 recipe additions (each independently toggleable for ablation):
    ap.add_argument("--overlap2d", type=int, default=1)    # resolve A/B mask overlap per view
    ap.add_argument("--exist_thr", type=float, default=0.9)
    # first-token existence confidence gate (0 disables). 0.9, NOT 0.95: a real
    # thin part can sit uniformly at ~0.88 (chin strap: 81 points, med 0.879 ->
    # 0.95 emptied all 40 views; absent parts have the SAME per-view median, so
    # only spatial voting separates them — the gate must stay conservative).
    # P(exist) =
    # position-0 subset softmax '<points' vs 'There are none' — the model's own
    # trained existence decision. 500-obj calibration: AUROC 0.912; thr 0.9
    # keeps 94.3% of real-part views (97.4% of real roles keep >=1 view) and
    # kills 57% of hallucinated-target views on top of the ~54% native decline.
    # Raw scores are stored per view (pexistA/pexistB), so the threshold can be
    # re-applied later without re-inference.
    return ap.parse_args()


def slugify(text, n=40):
    return "".join(c if c.isalnum() else "_" for c in text.lower())[:n].strip("_")


NEG_MIN_PX = 30      # partner point becomes a SAM negative only beyond this distance


def is_body_target(role):
    # SAM mask-selection rule; narrower than sra.BODY_TARGET_RE (the partition
    # gate) — calibrated separately, do not unify.
    tgt = str(role.get("target", "")).lower()
    return "body" in tgt or "wall" in tgt or "surface" in tgt


def expand_sam_subqueries(queries, sam_pts):
    """SAM jobs from the (gated) per-view point lists, one sub-query per point
    slot: slot s of flat role j prompts each view's s-th point separately
    (multi-instance targets need one mask per instance, not one multi-positive
    blob); the caller max-merges slots back into the role's heatmaps.

    Per sub-query: candidate selection ("mixed" recipe: body-level targets ->
    MIDDLE SAM candidate, named parts -> best_iou; smallest retired 07-17 —
    it collapses onto paint/decal patches) and the partner role's first point
    as a negative prompt when it sits >NEG_MIN_PX from THIS slot's positive
    (separates e.g. mug handle from body masks without ever giving SAM a
    near-coincident positive/negative pair).

    Returns (sub_points, sub_owner, sub_selects, sub_negs), parallel lists.
    """
    selects = ["middle" if is_body_target(q["roles"][r]) else "best_iou"
               for _, q in queries for r in range(2)]
    sub_points, sub_owner, sub_selects, sub_negs = [], [], [], []
    for j, pq in enumerate(sam_pts):
        partner = sam_pts[j + 1 if j % 2 == 0 else j - 1]
        for s in range(max(1, max(len(p) for p in pq))):
            row_q, row_n = [], []
            for pts, n_list in zip(pq, partner):
                pos = np.asarray(pts[s], dtype=np.float32) if len(pts) > s else None
                row_q.append([pos] if pos is not None else [])
                neg = np.asarray(n_list[0], dtype=np.float32) if len(n_list) else None
                use = (pos is not None and neg is not None
                       and float(np.linalg.norm(pos - neg)) > NEG_MIN_PX)
                row_n.append(neg if use else None)
            sub_points.append(row_q)
            sub_owner.append(j)
            sub_selects.append(selects[j])
            sub_negs.append(row_n)
    return sub_points, sub_owner, sub_selects, sub_negs


def process_object(rec, scene, canvas, queries, per_query, heatmaps_all, proj,
                   idxNN, t_share, out_dir, p_exist, recipe_flags):
    """Voting + partition + save for every query; pointing AND SAM2 already done
    object-batched (encoder ran once, projection computed once).

    queries: [(qi, query_dict)] eligible queries; per_query / heatmaps_all are
    indexed i*2+r (query i, role r) — the flat order the pointing ran in.
    per_query values are per-view LISTS of points (multi-instance targets);
    p_exist: per flat role, per-view existence confidence (float | None) —
    persisted raw so the threshold can be re-tuned without re-inference.
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
            multi = []
            for vi, plist in enumerate(flat):
                if len(plist):
                    pts_arr[vi] = plist[0]
                    multi.extend([[float(vi), float(p[0]), float(p[1])] for p in plist])
            multi = np.array(multi, dtype=np.float32) if multi else np.zeros((0, 3), np.float32)
            role_data.append((pts_arr, multi, scores, counts))
        (ptsA, multiA, scoreA_raw, counts), (ptsB, multiB, scoreB_raw, _) = role_data
        # validated chain: refine (satellite prune + kNN smooth) -> partition
        # (text symmetry / text vertical / evidence) -> post prune (guarded)
        refA = sra.refine_scores(scoreA_raw, idxNN)
        refB = sra.refine_scores(scoreB_raw, idxNN)
        fA, fB = sra.partition_two_roles(refA, refB, roles[0], roles[1], canvas.xyz)
        scoreA = sra.prune_partitioned(fA, refA, idxNN)
        scoreB = sra.prune_partitioned(fB, refB, idxNN)

        qdir = os.path.join(out_dir, rec["object_id"], f"q{qi}_{slugify(query['task'])}")
        os.makedirs(qdir, exist_ok=True)
        pexist = {key: np.array([np.nan if p is None else p for p in p_exist[j]], dtype=np.float32)
                  for key, j in (("pexistA", i * 2), ("pexistB", i * 2 + 1))}
        np.savez_compressed(
            os.path.join(qdir, "scores.npz"),
            xyz=canvas.xyz, gt=canvas.gt, counts=counts,
            scoreA_raw=scoreA_raw, scoreB_raw=scoreB_raw, scoreA=scoreA, scoreB=scoreB,
            ptsA=ptsA, ptsB=ptsB, ptsA_multi=multiA, ptsB_multi=multiB,
            molmo_queries=np.array(mqs, dtype=object), task=str(query["task"]), **pexist)
        meta = {
            "object_id": rec["object_id"], "object_name": rec["object_name"],
            "task": query["task"], "category": query.get("category"),
            "query": query.get("query"), "roles": roles, "molmo_queries": mqs,
            "pattern": query.get("pattern"), "coordination": query.get("coordination"),
            "n_views": scene.n_views,
            "hitA": int(np.isfinite(ptsA[:, 0]).sum()), "hitB": int(np.isfinite(ptsB[:, 0]).sum()),
            "coverage": float((counts > 0).mean()),
            "engine": {"name": "molmo2-vllm", "k": 1, "sam2": "object-batched",
                       "recipe": "mixed+neg / refine / partition-v6 / post-prune",
                       **recipe_flags},
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
    llm, proc = m2v.load_engine(gpu_memory_utilization=args.gpu_mem)
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
          f"views={args.num_views} -> {args.output_dir}", flush=True)

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
        try:
            canvas = load_canvas(aff_dir)
            scene = load_scene(obj_root, n_views)
            # ONE generate for every query x both roles of this object
            mqs_flat = [mq for _, q in queries for mq in q["molmo_queries"]]
            per_query, p_exist = m2v.ground_queries(llm, proc, scene.view_images, mqs_flat)
            # existence gate: SAM sees only views the model itself believes in
            # (see --exist_thr); per_query stays UNFILTERED so the npz records
            # every Molmo point + its score and the threshold can be re-applied
            # in either direction without re-inference
            sam_pts = per_query
            if args.exist_thr > 0:
                sam_pts = [[pts if (p is None or p >= args.exist_thr) else []
                            for pts, p in zip(pq, pe)]
                           for pq, pe in zip(per_query, p_exist)]
            t_point = time.time() - t0
            sub_points, sub_owner, sub_selects, sub_negs = expand_sam_subqueries(queries, sam_pts)
            # per-image outputs are independent, so encoding the views in chunks is
            # equivalent; it only caps the encoder's peak memory
            heat_sub = None
            for v0 in range(0, n_views, args.sam_chunk):
                hm = sra.run_sam2_object_queries(scene.view_images_np[v0:v0 + args.sam_chunk],
                                                 [q[v0:v0 + args.sam_chunk] for q in sub_points],
                                                 sam2_predictor, cfg,
                                                 neg_points=[n[v0:v0 + args.sam_chunk] for n in sub_negs],
                                                 mask_selects=sub_selects)
                heat_sub = hm if heat_sub is None else [a + b for a, b in zip(heat_sub, hm)]
            # merge slot masks per role per view: same-instance masks average
            # (consensus, no fringe inflation), distinct instances union
            slots_of = [[] for _ in per_query]
            for si, j in enumerate(sub_owner):
                slots_of[j].append(si)
            heatmaps_all = [
                [sra.merge_instance_masks([heat_sub[si][vi] for si in slots
                                           if len(sub_points[si][vi])] or [heat_sub[slots[0]][vi]])
                 for vi in range(n_views)]
                for slots, j in zip(slots_of, range(len(per_query)))]
            proj = sra.precompute_projection(canvas.xyz_vote, scene.cameras, scene.K_list,
                                             scene.depth_maps, cfg.depth_tolerance)
            # consensus validation for NAMED-PART roles: vote once, then drop
            # any view whose mask grossly disagrees with the cross-view
            # consensus (of the visible canvas points the mask covers, <20%
            # lie in the consensus core). Catches SAM merging a grazing-angle
            # handle into the whole body (2/40 such views smear diffuse score
            # everywhere through mean-voting) WITHOUT any mask-size prior —
            # a plate seen face-on is huge but agrees with its own consensus.
            # Body-level roles legitimately mask the whole object: exempt.
            body_flags = [is_body_target(q["roles"][r]) for _, q in queries for r in range(2)]
            for j, hms in enumerate(heatmaps_all):
                if body_flags[j]:
                    continue
                score1, _ = sra.sample_heatmaps_projected(proj, hms)
                core = score1 > 0.3
                if core.sum() < 30:
                    core = score1 > 0.15
                    if core.sum() < 30:
                        continue
                for vi, h in enumerate(hms):
                    u, v, valid = proj[vi]
                    idx = np.where(valid)[0]
                    if not len(idx):
                        continue
                    covered = idx[h[v[idx], u[idx]] > 0.5]
                    if len(covered) >= 20 and core[covered].mean() < 0.2:
                        hms[vi] = np.zeros_like(h)
            if args.overlap2d:   # cross-role exclusivity BEFORE the 3D vote
                for i in range(len(queries)):
                    for vi in range(n_views):
                        sra.resolve_2d_overlap(heatmaps_all[2 * i][vi], heatmaps_all[2 * i + 1][vi],
                                               sam_pts[2 * i][vi], sam_pts[2 * i + 1][vi])
            t_sam = time.time() - t0 - t_point
            idxNN = sra.knn_indices(canvas.xyz)
            n_q = process_object(rec, scene, canvas, queries, per_query, heatmaps_all, proj,
                                 idxNN, (t_point + t_sam) / len(queries), args.output_dir,
                                 p_exist=p_exist,
                                 recipe_flags=dict(overlap2d=bool(args.overlap2d),
                                                   exist_thr=args.exist_thr))
        except KeyboardInterrupt:
            raise
        except Exception as e:
            # a corrupt render (truncated PNG etc.) must not kill a 5000-object
            # run; engine deaths still propagate so the outer wrapper restarts
            if "EngineDead" in type(e).__name__:
                raise
            print(f"[{args.start + oi}] {rec['object_name'][:28]:28} FAILED "
                  f"({type(e).__name__}: {str(e)[:80]}) — object skipped", flush=True)
            n_skip += 1
            continue
        n_done += 1
        print(f"[{args.start + oi}] {rec['object_name'][:28]:28} {n_q} queries  "
              f"{time.time() - t0:.0f}s (point {t_point:.0f}s sam2 {t_sam:.0f}s)", flush=True)
    print(f"\nstage2_v2 done: {n_done} objects, {n_skip} skipped -> {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
