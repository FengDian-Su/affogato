"""Diagnostic: where does Hand B's over-wide active region come from?

Stage 0 measured that B's active region is systematically far broader than A's
at every display threshold. A display threshold must not be used to paper over a
generation-side bias, so before any tau_vis / mask decision this script isolates
WHICH step of the stage2 chain produces the asymmetry:

    scoreA_raw --refine--> refA --partition--> fA --prune--> scoreA   (stored)

Only the endpoints are stored in scores.npz, so the two middle states are
recomputed from the stored xyz + *_raw using the generator's own functions
(single_region_affordance). The recomputed endpoint is compared against the
stored scoreA/scoreB as a faithfulness check -- if that match rate is not ~100%,
every number below is meaningless.

Freezes nothing. Reports only.

    python check_partition.py --n 300
"""
import argparse
import collections
import json
import os
import random
import sys

import numpy as np

import common as C

sys.path.insert(0, os.path.join(C.REPO, "data_generation"))
from single_region import single_region_affordance as sra  # noqa: E402

THR = sra.SUPPORT_THR      # 0.15 -- the generator's own "belongs to the region" cutoff


def _stage_stats(s, n):
    return {"active": float((s > THR).sum() / n), "mass": float(s.sum()), "zero": bool(not (s > 0).any())}


def analyse(sm, xyz, idxNN, z):
    n = xyz.shape[0]
    roles = sm["roles"]
    ra, rb = z["scoreA_raw"].astype(np.float32), z["scoreB_raw"].astype(np.float32)

    refA, refB = sra.refine_scores(ra, idxNN), sra.refine_scores(rb, idxNN)
    fA, fB = sra.partition_two_roles(refA, refB, roles[0], roles[1], xyz)
    pA, pB = sra.prune_partitioned(fA, refA, idxNN), sra.prune_partitioned(fB, refB, idxNN)

    # faithfulness: recomputed endpoint vs what stage2 actually wrote
    okA = np.allclose(pA, z["scoreA"], atol=1e-5)
    okB = np.allclose(pB, z["scoreB"], atol=1e-5)

    rec = {"sample_id": sm["sample_id"],
           "role_A": roles[0].get("role"), "role_B": roles[1].get("role"),
           "pair": " + ".join(sorted([str(roles[0].get("role")), str(roles[1].get("role"))])),
           "reproduced": bool(okA and okB)}
    for tag, sa, sb in (("raw", ra, rb), ("refined", refA, refB),
                        ("partitioned", fA, fB), ("final", pA, pB)):
        rec["A_" + tag] = _stage_stats(sa, n)
        rec["B_" + tag] = _stage_stats(sb, n)
    for h, seq in (("A", (ra, refA, fA, pA)), ("B", (rb, refB, fB, pB))):
        m0 = seq[0].sum()
        rec["%s_mass_retained" % h] = float(seq[-1].sum() / (m0 + C.EPS))
        rec["%s_mass_after_partition" % h] = float(seq[2].sum() / (m0 + C.EPS))
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="daily_used")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    root = C.STAGE2_ROOT.format(ds=a.dataset)
    samples, _ = C.expected_samples(C.load_stage1(a.dataset), root)
    present = [s for s in samples if s["state"] == "present"
               and isinstance(s["roles"], list) and len(s["roles"]) == 2]
    random.seed(a.seed)
    pick = random.sample(present, min(a.n, len(present)))
    # group by object: idxNN is per-object geometry, exactly as stage2 computes it once
    by_obj = collections.defaultdict(list)
    for s in pick:
        by_obj[s["object_id"]].append(s)
    print("[check] %d samples over %d objects" % (len(pick), len(by_obj)))

    recs = []
    for i, (oid, group) in enumerate(by_obj.items()):
        idxNN = xyz = None
        for sm in group:
            try:
                z = C.load_npz(os.path.join(sm["qdir"], "scores.npz"))
                if xyz is None:
                    xyz = z["xyz"].astype(np.float32)
                    idxNN = sra.knn_indices(xyz)
                recs.append(analyse(sm, xyz, idxNN, z))
            except Exception as e:
                print("  skip %s: %s" % (sm["sample_id"], e))
        if (i + 1) % 25 == 0:
            print("  %d/%d objects" % (i + 1, len(by_obj)))

    out = a.out or os.path.join(C.REPO, "data_verification/quality_evaluation/stage0",
                                a.dataset, "partition_check.jsonl")
    with open(out, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    report(recs)
    print("\n[check] wrote %s" % out)


def report(recs):
    n = len(recs)
    rep = sum(r["reproduced"] for r in recs)
    print("\n=== faithfulness ===")
    print("recomputed chain == stored scoreA/scoreB : %d/%d (%.1f%%)" % (rep, n, 100.0 * rep / n))
    if rep < n:
        print("  !! non-reproducing samples make the rest of this report unreliable")

    print("\n=== active ratio (points > %.2f), mean over %d samples ===" % (THR, n))
    print("stage          A       B      B/A")
    for tag in ("raw", "refined", "partitioned", "final"):
        a = np.mean([r["A_" + tag]["active"] for r in recs])
        b = np.mean([r["B_" + tag]["active"] for r in recs])
        print("%-12s %.3f   %.3f   %.2fx" % (tag, a, b, b / (a + 1e-9)))

    print("\n=== score mass retained (final / raw) ===")
    # medians only: a hand whose raw mass is ~0 divides by EPS and produces a
    # 1e9 ratio that destroys the mean.
    for h in "AB":
        v = np.array([r["%s_mass_retained" % h] for r in recs])
        p = np.array([r["%s_mass_after_partition" % h] for r in recs])
        keep = np.isfinite(v) & (v < 10)
        print("  %s: after partition %.3f -> after prune %.3f (median; n=%d, raw~0 dropped)"
              % (h, np.median(p[keep]), np.median(v[keep]), int(keep.sum())))

    print("\n=== all-zero introduced at each stage (%% of samples) ===")
    print("stage          A       B")
    for tag in ("raw", "refined", "partitioned", "final"):
        a = 100.0 * np.mean([r["A_" + tag]["zero"] for r in recs])
        b = 100.0 * np.mean([r["B_" + tag]["zero"] for r in recs])
        print("%-12s %5.1f%%  %5.1f%%" % (tag, a, b))

    for key, label in (("role_A", "A role"), ("role_B", "B role"), ("pair", "unordered pair")):
        g = collections.defaultdict(list)
        for r in recs:
            g[r[key]].append(r)
        print("\n=== final active ratio by %s (n>=5) ===" % label)
        rows = [(k, len(v), np.mean([x["A_final"]["active"] for x in v]),
                 np.mean([x["B_final"]["active"] for x in v])) for k, v in g.items() if len(v) >= 5]
        for k, c, a, b in sorted(rows, key=lambda x: -x[3]):
            print("  %-24s n=%-4d A=%.3f  B=%.3f" % (k, c, a, b))

    print("\n=== examples ===")
    wide = sorted([r for r in recs if r["reproduced"]],
                  key=lambda r: -(r["B_final"]["active"] - r["A_final"]["active"]))
    norm = sorted([r for r in recs if r["reproduced"]],
                  key=lambda r: abs(r["B_final"]["active"] - r["A_final"]["active"]))
    for label, rows in (("most B-over-wide", wide[:4]), ("most balanced", norm[:4])):
        print("  -- %s --" % label)
        for r in rows:
            print("     %-58s %-18s A=%.3f B=%.3f (raw A=%.3f B=%.3f)"
                  % (r["sample_id"][:58], r["pair"], r["A_final"]["active"], r["B_final"]["active"],
                     r["A_raw"]["active"], r["B_raw"]["active"]))
    zero = [r for r in recs if r["A_final"]["zero"] or r["B_final"]["zero"]]
    print("  -- final all-zero (%d) --" % len(zero))
    for r in zero[:4]:
        print("     %-58s %-18s A_raw=%.3f B_raw=%.3f -> A=%.3f B=%.3f"
              % (r["sample_id"][:58], r["pair"], r["A_raw"]["active"], r["B_raw"]["active"],
                 r["A_final"]["active"], r["B_final"]["active"]))


if __name__ == "__main__":
    main()
