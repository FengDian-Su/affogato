#!/usr/bin/env python
"""Re-segment stage2_full body roles from stored Molmo points — NO Molmo re-run.

Historical tool: it applied the 07-17 middle-candidate fix to the k=4 dataset.
For every query with >=1 body-like-target role (body/wall/surface): re-run
SAM2 from the stored Molmo points with mask_select="middle" for body-like
roles (named-part roles keep their stored raw — their selection is unchanged),
re-vote, refine, partition (long-axis rule), prune, and rewrite scores.npz
atomically. Queries with no body-like role are untouched.

Idempotent: skips queries whose npz already has sel_version >= 3.
Usage: stage2_resegment.py <gpu_id> <shard_idx> <num_shards> [sam_chunk=20]
"""
import os, sys, json
GPU, SHARD, NSHARD = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
CHUNK = int(sys.argv[4]) if len(sys.argv) > 4 else 20
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = GPU
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
DG = "/home/michaellee/mclee/affogato/data_generation"
sys.path.insert(0, DG)
FULL = f"{DG}/outputs/stage2_full"
NEG_MIN_PX = 30
SEL_VERSION = 3


def body_role(role):
    tgt = str(role.get("target", "")).lower()
    return "body" in tgt or "wall" in tgt or "surface" in tgt


def main():
    import numpy as np
    os.chdir(DG)
    from single_region import single_region_affordance as sra

    recs = {}
    for part in ("stage1_part0.json", "stage1_part1.json"):
        for r in json.load(open(f"outputs/stage1_v2/{part}")):
            if not r.get("error"):
                recs[r["object_id"]] = r
    mp_map = {e["object_id"]: e["dst"] for e in json.load(open("dataset/daily_used_to_affogato.json"))}

    oids = sorted(d for d in os.listdir(FULL)
                  if os.path.isdir(os.path.join(FULL, d)) and d != "logs")
    oids = [o for i, o in enumerate(oids) if i % NSHARD == SHARD]
    print(f"shard {SHARD}/{NSHARD} on GPU {GPU}: {len(oids)} objects", flush=True)

    cfg = sra.PipelineConfig()
    sam2 = sra.load_sam2_model(cfg.sam2_checkpoint, cfg.sam2_model_cfg, cfg.device)

    done = skipped = failed = 0
    for oi, oid in enumerate(oids):
        od = os.path.join(FULL, oid)
        try:
            jobs = []
            for qd in sorted(os.listdir(od)):
                sp = os.path.join(od, qd, "scores.npz")
                mpth = os.path.join(od, qd, "meta.json")
                if not (os.path.exists(sp) and os.path.exists(mpth)):
                    continue
                m = json.load(open(mpth))
                broles = [r for r in range(2) if body_role(m["roles"][r])]
                if not broles:
                    continue
                with np.load(sp, allow_pickle=True) as d:
                    if "sel_version" in d.files and int(d["sel_version"]) >= SEL_VERSION:
                        continue
                    if "ptsA_multi" in d.files:
                        # 07-19+ recipe output (multi-point + exist gate + 2D
                        # overlap resolution): this single-point re-segmentation
                        # would silently downgrade it — never touch
                        continue
                jobs.append((qd, m, broles))
            if not jobs:
                skipped += 1
                continue
            rec = recs.get(oid)
            if rec is None:
                failed += 1
                continue
            obj_root = "/".join(rec["views_used"][0].split("/")[:-2])
            T = min(sra.count_available_views(obj_root), 40)
            _, vi_np = sra.load_view_images(obj_root, T)
            cams, K_list, depths = sra.prepare_camera_params(obj_root, T)
            xyz = np.load(f"{mp_map[oid]}/xyzc.npy").astype(np.float32)[:, :3]
            proj = sra.precompute_projection(sra.align_affogato_frame(xyz), cams, K_list,
                                             depths, cfg.depth_tolerance)
            idxNN = sra.knn_indices(xyz)

            # flatten all body roles of all pending queries into ONE SAM call
            # (one encoder pass per object)
            qp, negs, owners = [], [], []
            data = {}
            for qd, m, broles in jobs:
                d = dict(np.load(f"{od}/{qd}/scores.npz", allow_pickle=True))
                data[qd] = d
                pts = {0: d["ptsA"][:T], 1: d["ptsB"][:T]}
                for r in broles:
                    qp.append([[np.asarray(p, dtype=np.float32)] if np.isfinite(p[0]) else []
                               for p in pts[r]])
                    row = []
                    for vi in range(T):
                        p, q = pts[r][vi], pts[1 - r][vi]
                        use = (np.isfinite(p[0]) and np.isfinite(q[0])
                               and np.linalg.norm(p - q) > NEG_MIN_PX)
                        row.append(np.asarray(q, dtype=np.float32) if use else None)
                    negs.append(row)
                    owners.append((qd, r))
            hms = [[] for _ in qp]
            for v0 in range(0, T, CHUNK):
                part = sra.run_sam2_object_queries(
                    vi_np[v0:v0 + CHUNK], [q[v0:v0 + CHUNK] for q in qp], sam2,
                    ["middle"] * len(qp),
                    neg_points=[n[v0:v0 + CHUNK] for n in negs])
                for qi, h in enumerate(part):
                    hms[qi].extend(h)
            for (qd, r), hm in zip(owners, hms):
                raw = sra.sample_heatmaps_projected(proj, hm)[0].astype(np.float32)
                data[qd]["scoreA_raw" if r == 0 else "scoreB_raw"] = raw
            for qd, m, broles in jobs:
                d = data[qd]
                rawA = d["scoreA_raw"].astype(np.float32)
                rawB = d["scoreB_raw"].astype(np.float32)
                refA, refB = sra.refine_scores(rawA, idxNN), sra.refine_scores(rawB, idxNN)
                fA, fB = sra.partition_two_roles(refA, refB, m["roles"][0], m["roles"][1], xyz)
                fA = sra.prune_partitioned(fA, refA, idxNN)
                fB = sra.prune_partitioned(fB, refB, idxNN)
                d["scoreA"], d["scoreB"] = fA.astype(np.float32), fB.astype(np.float32)
                d["sel_version"] = np.int8(SEL_VERSION)
                tmp = f"{od}/{qd}/.tmp{SHARD}_scores.npz"
                np.savez_compressed(tmp, **d)
                os.replace(tmp, f"{od}/{qd}/scores.npz")
            done += 1
        except Exception as e:
            failed += 1
            print(f"FAILED {oid}: {type(e).__name__}: {e}", flush=True)
        if (oi + 1) % 25 == 0:
            print(f"[{oi+1}/{len(oids)}] done {done} skipped {skipped} failed {failed}", flush=True)
    print(f"FINISHED shard {SHARD}: done {done} skipped {skipped} failed {failed}", flush=True)


if __name__ == "__main__":
    main()
