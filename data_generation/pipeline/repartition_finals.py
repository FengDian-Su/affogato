#!/usr/bin/env python
"""Re-materialize stage2 final heatmaps under the two-pose partition rule.

CPU-only: for every query npz, recompute refine -> partition_two_roles ->
prune_partitioned from the stored scoreA_raw/scoreB_raw (the exact production
inputs, stage2_v2.py:186-190) and overwrite scoreA/scoreB in place. Raws and
every other key are untouched, so the operation is idempotent; each npz is
replaced atomically (tmp + os.replace). Resume via the done-list file.

Run detached:  nohup python pipeline/repartition_finals.py >> <log> 2>&1
"""
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

DG = "/home/michaellee/mclee/affogato/data_generation"
sys.path.insert(0, DG)
os.chdir(DG)

ROOTS = ["outputs/stage2/daily_used", "outputs/stage2/electronics"]
DONE = "outputs/stage2/repartition_done.txt"
WORKERS = 24


def do_object(key):
    os.nice(10)
    od = key
    from single_region import single_region_affordance as sra
    idxNN = None
    n = 0
    for qd in sorted(os.listdir(od)):
        npf, mp = os.path.join(od, qd, "scores.npz"), os.path.join(od, qd, "meta.json")
        if not (os.path.exists(npf) and os.path.exists(mp)):
            continue
        try:
            z = dict(np.load(npf, allow_pickle=True))
            meta = json.load(open(mp))
            if idxNN is None:
                idxNN = sra.knn_indices(z["xyz"])
            refA = sra.refine_scores(z["scoreA_raw"].astype(np.float64), idxNN)
            refB = sra.refine_scores(z["scoreB_raw"].astype(np.float64), idxNN)
            fA, fB = sra.partition_two_roles(refA, refB, meta["roles"][0], meta["roles"][1],
                                             z["xyz"])
            z["scoreA"] = sra.prune_partitioned(fA, refA, idxNN).astype(z["scoreA"].dtype)
            z["scoreB"] = sra.prune_partitioned(fB, refB, idxNN).astype(z["scoreB"].dtype)
            tmp = npf + ".tmp.npz"
            np.savez_compressed(tmp, **z)
            os.replace(tmp, npf)
            n += 1
        except Exception as e:
            return key, n, f"{qd}: {e!r}"
    return key, n, None


def main():
    done = set()
    if os.path.exists(DONE):
        done = {ln.strip() for ln in open(DONE)}
    keys = []
    for root in ROOTS:
        for oid in sorted(os.listdir(root)):
            p = os.path.join(root, oid)
            if os.path.isdir(p) and p not in done:
                keys.append(p)
    print(f"pending objects: {len(keys)} (done: {len(done)})", flush=True)
    t0 = time.time()
    n_obj = n_q = n_err = 0
    with Pool(WORKERS) as pool, open(DONE, "a") as df:
        for key, n, err in pool.imap_unordered(do_object, keys, chunksize=8):
            n_obj += 1
            n_q += n
            if err:
                n_err += 1
                print(f"ERR {key} {err}", flush=True)
            else:
                df.write(key + "\n")
                df.flush()
            if n_obj % 1000 == 0:
                r = (time.time() - t0) / n_obj
                print(f"[{n_obj}/{len(keys)}] {n_q} queries, {n_err} err, "
                      f"{r:.2f}s/obj, eta {(len(keys)-n_obj)*r/3600:.1f}h", flush=True)
    print(f"DONE repartition: {n_obj} objects, {n_q} queries, {n_err} errors, "
          f"{(time.time()-t0)/3600:.1f}h", flush=True)


if __name__ == "__main__":
    main()
